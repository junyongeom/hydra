#!/usr/bin/env python
# coding: utf-8

# Reimplementation of the official baseline released by the dataset authors:
#
# > Wang, Lv, Zhu, Ding, Han. *XRF55: A Radio Frequency Dataset for Human Indoor Action Analysis.* Proc. ACM IMWUT 8(1), 2024.
# > Code: https://github.com/aiotgroup/XRF55-repo
#
# **Three ResNet18 trained jointly with Deep Mutual Learning (DML)**:
# - `resnet1d.resnet18_mutual()` - WiFi
# - `resnet1d_rfid.resnet18_mutual()` - RFID
# - `resnet2d.resnet18_mutual()` - mmWave
#
# Each model returns `(logits, vec)` where `vec` is aligned with BERT-encoded class text embeddings via L1.  We disable that auxiliary L1 alignment here because the BERT vector file (`bert_new_sentence_large_uncased.npy`) is auxiliary and not part of the core mutual-learning idea; we keep CE + pairwise KL divergence (the DML loss) which is the model's core contribution. The fusion prediction at evaluation time follows `dml_eval.py`: average of the three softmax outputs.
#
# **Hyperparameters (matching `opts.py` / `dml_train.py`)**:
# - epochs = 200, batch_size = 64
# - optimizer = Adam, lr = 1e-3
# - scheduler = MultiStepLR(milestones=[40, 80, 120, 160], gamma=0.5)
# - loss = CrossEntropy + sum KLDiv(student_i || teacher_j) / (M-1)
#
# **Data splits** are identical to the shared experiment protocol so numbers are directly comparable:
# 1. In-Domain (Scene1 samples 1-14 train / 15-20 test)
# 2. Cross-Subject 21-9 (overlap subjects in source)
# 3. Cross-Scene (S2/3/4, strict subject consistency)
# 4. Cross Scene & Subject (22 clean source subjects -> S2/3/4)
#

import os, gc, json, time, random, copy, socket
import numpy as np
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.amp
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# Paths & device  (match the shared experiment protocol)
ROOT = os.environ.get("XRF55_ROOT", "data/XRF55")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
HOSTNAME = socket.gethostname()

# Reproducibility
def set_reproducible_mode(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)

set_reproducible_mode(SEED)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def fresh_generator():
    g = torch.Generator(); g.manual_seed(SEED)
    return g

# Dataset  (identical to shared experiment protocol so splits match)
class XRF55Dataset(Dataset):
    """scene_filter: {scene: {'samples': [int], 'vols': [int] or None}}"""
    def __init__(self, root_dir, scene_filter):
        self.root_dir = Path(root_dir)
        self.scene_filter = scene_filter
        self.sensors = ['WiFi', 'mmWave', 'RFID']
        self.samples = []
        # Truncated/corrupted mmWave files in XRF55 -> skip silently
        self._EXPECTED_MMW_SIZE = 2_228_352
        self._build_metadata()

    def _build_metadata(self):
        grouped = defaultdict(dict)
        for scene, cfg in self.scene_filter.items():
            allowed_samples = set(cfg['samples'])
            allowed_vols = set(cfg['vols']) if cfg.get('vols') is not None else None
            for sensor in self.sensors:
                sensor_dir = self.root_dir / scene / sensor
                if not sensor_dir.exists(): continue
                for fp in sensor_dir.glob("*.npy"):
                    parts = fp.stem.split('_')
                    if len(parts) != 3: continue
                    v, a, s = map(int, parts)
                    if s not in allowed_samples: continue
                    if allowed_vols is not None and v not in allowed_vols: continue
                    grouped[(scene, v, a, s)][sensor] = str(fp)
        for key, paths in grouped.items():
            if len(paths) == len(self.sensors):
                self.samples.append({'activity': key[2], 'paths': paths})

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        wifi = torch.tensor(np.load(s['paths']['WiFi']), dtype=torch.float32)
        if wifi.dim() == 2: wifi = wifi.unsqueeze(0)
        mmw = torch.tensor(np.load(s['paths']['mmWave']), dtype=torch.float32)
        if mmw.dim() == 4: mmw = mmw.squeeze(0)
        mmw = torch.log1p(torch.abs(mmw))
        rfid = torch.tensor(np.load(s['paths']['RFID']), dtype=torch.float32)
        if rfid.dim() == 2: rfid = rfid.unsqueeze(0)
        return {'WiFi': wifi, 'mmWave': mmw, 'RFID': rfid}, s['activity']

def make_loader(root_path, scene_filter, BS=64, NUM_WORKERS=4, shuffle=False, generator=None):
    dataset = XRF55Dataset(root_path, scene_filter)
    if len(dataset) == 0:
        raise ValueError(
            f"No XRF55 samples found under {root_path!r} for scene_filter={scene_filter!r}. "
            "Check that --data-root points to the directory containing Scene1/Scene2/Scene3/Scene4."
        )
    return DataLoader(
        dataset,
        batch_size=BS, shuffle=shuffle, num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker, generator=generator)

def make_concat_loader(root_path, scene_filter_list, BS=64, NUM_WORKERS=4,
                        shuffle=True, generator=None):
    datasets = [XRF55Dataset(root_path, sf) for sf in scene_filter_list]
    datasets = [d for d in datasets if len(d) > 0]
    if len(datasets) == 0:
        raise ValueError("All datasets empty.")
    ds = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return DataLoader(ds, batch_size=BS, shuffle=shuffle,
                      num_workers=NUM_WORKERS,
                      pin_memory=torch.cuda.is_available(),
                      worker_init_fn=seed_worker, generator=generator)

def get_volunteers_in_scene(root_dir, scene):
    sensor_dir = Path(root_dir) / scene / 'WiFi'
    vols = set()
    if sensor_dir.exists():
        for fp in sensor_dir.glob("*.npy"):
            parts = fp.stem.split('_')
            if len(parts) == 3:
                vols.add(int(parts[0]))
    return sorted(vols)

def save_result(name, payload):
    path = f"reproc/results/{name}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"saved: {path}")


# =============================================================================
# Models - ResNet18 ports of the three backbones in aiotgroup/XRF55-repo:
#   resnet1d        -> WiFi   : 1D ResNet18 over time, input (B, 1, 270, 1000)
#                              we use 2D conv along (subcarrier, time) which
#                              is consistent with the dataset's CSI tensor.
#   resnet1d_rfid   -> RFID   : 1D ResNet18 over time, input (B, 1, 23, 148)
#                              with the 23 tags as input channels.
#   resnet2d        -> mmWave : 2D ResNet18 over (range, doppler/angle) per
#                              frame, input (B, 17, 256, 128).
#
# Each backbone returns (logits, vec); `vec` is a 1024-d embedding that the
# original paper aligns to BERT class-name vectors with an L1 loss.  We keep
# `vec` for completeness but DO NOT use the auxiliary L1 since the BERT
# vector file is not part of this baseline run - only the DML KL loss + CE.
# =============================================================================

VEC_DIM = 1024    # paper aligns vec to BERT large embedding (1024-d)

class _BasicBlock1d(nn.Module):
    expansion = 1
    def __init__(self, in_ch, ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(ch)
        self.downsample = downsample
    def forward(self, x):
        i = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + i, inplace=True)

class _BasicBlock2d(nn.Module):
    expansion = 1
    def __init__(self, in_ch, ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.downsample = downsample
    def forward(self, x):
        i = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + i, inplace=True)


# ---------- WiFi backbone (resnet1d in the repo) ----------
# WiFi input in the shared experiment protocol: (B, 1, 270, 1000)
# 270 = 9 links x 30 subcarriers.  We collapse subcarriers/links into the
# channel axis to feed a 1D ResNet18 over the time axis.
class WiFi_ResNet18Mutual(nn.Module):
    def __init__(self, num_classes=55, vec_dim=VEC_DIM):
        super().__init__()
        in_ch = 270
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 64, 7, 2, 3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, 2, 1),
        )
        self.in_ch = 64
        self.layer1 = self._make_layer(64,  2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head_cls = nn.Linear(512, num_classes)
        self.head_vec = nn.Linear(512, vec_dim)

    def _make_layer(self, ch, n, stride):
        downsample = None
        if stride != 1 or self.in_ch != ch:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_ch, ch, 1, stride, bias=False),
                nn.BatchNorm1d(ch))
        layers = [_BasicBlock1d(self.in_ch, ch, stride, downsample)]
        self.in_ch = ch
        for _ in range(1, n):
            layers.append(_BasicBlock1d(ch, ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 1, 270, 1000) -> (B, 270, 1000)
        x = x.squeeze(1)
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        feat = self.pool(x).flatten(1)        # (B, 512)
        logits = self.head_cls(feat)
        vec = self.head_vec(feat)
        return logits, vec


# ---------- RFID backbone (resnet1d_rfid in the repo) ----------
# RFID input in the shared experiment protocol: (B, 1, 23, 148)
# 23 tags -> channels; 148 -> time
class RFID_ResNet18Mutual(nn.Module):
    def __init__(self, num_classes=55, vec_dim=VEC_DIM):
        super().__init__()
        in_ch = 23
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 64, 7, 2, 3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, 2, 1),
        )
        self.in_ch = 64
        self.layer1 = self._make_layer(64,  2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head_cls = nn.Linear(512, num_classes)
        self.head_vec = nn.Linear(512, vec_dim)

    def _make_layer(self, ch, n, stride):
        downsample = None
        if stride != 1 or self.in_ch != ch:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_ch, ch, 1, stride, bias=False),
                nn.BatchNorm1d(ch))
        layers = [_BasicBlock1d(self.in_ch, ch, stride, downsample)]
        self.in_ch = ch
        for _ in range(1, n):
            layers.append(_BasicBlock1d(ch, ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.squeeze(1)   # (B, 23, 148)
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        feat = self.pool(x).flatten(1)
        logits = self.head_cls(feat)
        vec = self.head_vec(feat)
        return logits, vec


# ---------- mmWave backbone (resnet2d in the repo) ----------
# mmWave input in the shared experiment protocol: (B, 17, 256, 128)
# 17 frames -> channels (treated as the "depth" of a 2D ResNet18 input)
class MmWave_ResNet18Mutual(nn.Module):
    def __init__(self, num_classes=55, vec_dim=VEC_DIM, in_frames=17):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_frames, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.in_ch = 64
        self.layer1 = self._make_layer(64,  2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_cls = nn.Linear(512, num_classes)
        self.head_vec = nn.Linear(512, vec_dim)

    def _make_layer(self, ch, n, stride):
        downsample = None
        if stride != 1 or self.in_ch != ch:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_ch, ch, 1, stride, bias=False),
                nn.BatchNorm2d(ch))
        layers = [_BasicBlock2d(self.in_ch, ch, stride, downsample)]
        self.in_ch = ch
        for _ in range(1, n):
            layers.append(_BasicBlock2d(ch, ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        feat = self.pool(x).flatten(1)
        logits = self.head_cls(feat)
        vec = self.head_vec(feat)
        return logits, vec


# =============================================================================
# DML wrapper - three backbones trained jointly with mutual KL loss
# Forward returns logits per modality + averaged softmax (used at eval).
# =============================================================================

class XRF55_DML(nn.Module):
    def __init__(self, num_classes=55):
        super().__init__()
        self.wifi = WiFi_ResNet18Mutual(num_classes)
        self.rfid = RFID_ResNet18Mutual(num_classes)
        self.mmw  = MmWave_ResNet18Mutual(num_classes)

    def forward(self, wifi_x, mmw_x, rfid_x, apply_drop=False):
        # apply_drop kept for interface compatibility; unused (baseline).
        w_log, w_vec = self.wifi(wifi_x)
        r_log, r_vec = self.rfid(rfid_x)
        m_log, m_vec = self.mmw(mmw_x)
        # Eval-time fusion: average of softmax probs (mirrors dml_eval.py)
        probs = (F.softmax(w_log, dim=-1) + F.softmax(r_log, dim=-1) + F.softmax(m_log, dim=-1)) / 3.0
        return {
            'logits': torch.log(probs.clamp_min(1e-12)),     # for NLL compat
            'fused_prob': probs,
            'wifi_logits': w_log, 'rfid_logits': r_log, 'mmw_logits': m_log,
            'wifi_vec':    w_vec, 'rfid_vec':    r_vec, 'mmw_vec':    m_vec,
        }


MODEL_REGISTRY = {
    'XRF55': XRF55_DML,
}


# =============================================================================
# Train / Eval
#   Loss = sum_i [CE(logits_i, y) + 1/(M-1) sum_{j!=i} KL(logits_i || logits_j)]
#   Optimizer: Adam, lr=1e-3
#   Scheduler: MultiStepLR(milestones=[40,80,120,160], gamma=0.5)
# =============================================================================

@torch.no_grad()
def evaluate_model(model, loader, device):
    if len(loader) == 0: return 0.0
    c, t = 0, 0
    for data, labels in loader:
        wifi = data['WiFi'].to(device, non_blocking=True)
        mmw  = data['mmWave'].to(device, non_blocking=True)
        rfid = data['RFID'].to(device, non_blocking=True)
        tgt = (labels - 1).to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(wifi, mmw, rfid)
        pred = out['fused_prob'].argmax(1)
        c += pred.eq(tgt).sum().item()
        t += tgt.size(0)
    return 100.0 * c / t if t > 0 else 0.0


def _dml_loss(w_log, r_log, m_log, tgt, ce):
    """CE + pairwise KL across the three modalities (DML, Zhang et al. 2018)."""
    M = 3
    logits = [w_log, r_log, m_log]
    losses = []
    for i in range(M):
        ce_i = ce(logits[i], tgt)
        kl_i = 0.0
        for j in range(M):
            if i == j: continue
            # KL(P_i || P_j_detached) - DML: each network treats others as teachers
            kl_i = kl_i + F.kl_div(
                F.log_softmax(logits[i], dim=-1),
                F.softmax(logits[j].detach(), dim=-1),
                reduction='batchmean')
        losses.append(ce_i + kl_i / (M - 1))
    return sum(losses)


def train_and_eval(exp_name, model_cls, train_loader, eval_loaders, best_metric_fn,
                   device, epochs=200, lr=1e-3,
                   milestones=(40, 80, 120, 160), gamma=0.5,
                   log_every=10):
    """Unified single + fusion training. Last-epoch saved -> reproc/{exp_name}_last.pth"""
    set_reproducible_mode(SEED)
    model = model_cls(num_classes=55).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(milestones), gamma=gamma)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()
    save_path = os.path.join("reproc", f"{exp_name}_last.pth")

    for epoch in range(epochs):
        model.train()
        sum_loss = 0.0
        for data, labels in train_loader:
            wifi = data['WiFi'].to(device, non_blocking=True)
            mmw  = data['mmWave'].to(device, non_blocking=True)
            rfid = data['RFID'].to(device, non_blocking=True)
            tgt  = (labels - 1).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(wifi, mmw, rfid)
                loss = _dml_loss(out['wifi_logits'], out['rfid_logits'],
                                 out['mmw_logits'], tgt, criterion)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update()
            sum_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"   [Ep {epoch+1:03d}/{epochs}] Loss:{sum_loss/len(train_loader):.4f}")
        torch.cuda.empty_cache(); gc.collect()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    model.eval()
    final = {n: evaluate_model(model, ld, device) for n, ld in eval_loaders.items()}
    acc_str = " | ".join([f"{k}:{v:.2f}%" for k, v in final.items()])
    print(f"    [Final] {acc_str}")
    del model; torch.cuda.empty_cache(); gc.collect()
    return final


def _finetune_eval(model_cls, pretrain_ckpt, ft_loader, eval_loader, device,
                   epochs=100, lr=1e-5, log_every=10,
                   ft_seed=None, save_path=None):
    set_reproducible_mode(ft_seed if ft_seed is not None else SEED)
    model = model_cls(num_classes=55).to(device)
    print(f"       Loading pretrain ckpt: {pretrain_ckpt}")
    model.load_state_dict(torch.load(pretrain_ckpt, map_location=device))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        sum_loss = 0.0
        for data, labels in ft_loader:
            wifi = data['WiFi'].to(device, non_blocking=True)
            mmw  = data['mmWave'].to(device, non_blocking=True)
            rfid = data['RFID'].to(device, non_blocking=True)
            tgt  = (labels - 1).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(wifi, mmw, rfid)
                loss = _dml_loss(out['wifi_logits'], out['rfid_logits'],
                                 out['mmw_logits'], tgt, criterion)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update()
            sum_loss += loss.item()

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"      [FT Ep {epoch+1:03d}/{epochs}] Loss:{sum_loss/len(ft_loader):.4f}")

    acc = evaluate_model(model, eval_loader, device)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"       FT ckpt saved -> {save_path}")
    del model; torch.cuda.empty_cache(); gc.collect()
    return acc


def _zero_shot_eval(model_cls, pretrain_ckpt, eval_loader, device):
    model = model_cls(num_classes=55).to(device)
    print(f"       Loading pretrain ckpt (0-shot, no FT): {pretrain_ckpt}")
    model.load_state_dict(torch.load(pretrain_ckpt, map_location=device))
    acc = evaluate_model(model, eval_loader, device)
    del model; torch.cuda.empty_cache(); gc.collect()
    return acc


def print_paper_table(title, results_by_modality, shots, eval_keys=None):
    print("\n" + "="*80)
    print(f"{title}")
    print("="*80)
    eval_keys = eval_keys or list(next(iter(next(iter(results_by_modality.values())).values())).keys())
    for ek in eval_keys:
        print(f"\n[Eval target: {ek}]")
        cols = list(results_by_modality.keys())
        header = f"{'#-Shot':<8} | " + " | ".join([f"{c:>10}" for c in cols])
        print(header)
        print("-" * len(header))
        for k in shots:
            kname = ['Zero','One','Two','Three','Four','Five'][k]
            row = f"{kname:<8} | "
            row += " | ".join([
                f"{results_by_modality[c].get(k,{}).get(ek, float('nan')):>9.2f}%"
                if results_by_modality[c].get(k,{}).get(ek, float('nan')) ==
                   results_by_modality[c].get(k,{}).get(ek, float('nan'))
                else f"{'N/A':>10}"
                for c in cols])
            print(row)


def _report_param_count():
    m = XRF55_DML(num_classes=55)
    p = sum(x.numel() for x in m.parameters() if x.requires_grad)
    mem = sum(x.numel()*x.element_size() for x in m.parameters()) / (1024**2)
    print(f"   XRF55 DML baseline params: {p/1e6:.2f}M | memory: {mem:.2f}MB")
    for name, sub in [('WiFi', m.wifi), ('RFID', m.rfid), ('mmWave', m.mmw)]:
        sp = sum(x.numel() for x in sub.parameters())
        print(f"     {name:6s} ResNet18: {sp/1e6:.2f}M")

def _infer_input_shapes(root_path):
    try:
        ds = XRF55Dataset(root_path, {'Scene1': {'samples': [1], 'vols': None}})
        if len(ds) == 0: return None
        data, _ = ds[0]
        return {'wifi': tuple(data['WiFi'].shape),
                'mmw':  tuple(data['mmWave'].shape),
                'rfid': tuple(data['RFID'].shape)}
    except Exception as e:
        print(f"   WARNING: shape inference failed: {e}")
        return None

def _estimate_flops_custom(model, inputs):
    """Paper-style FLOPs estimator.
       - one multiply-add is counted as two FLOPs
       - Conv1d/Conv2d/Linear are counted by forward hooks
       - normalization/activation/pooling are not counted
    """
    model.eval()
    flops = {'total': 0}
    hooks = []

    def conv1d_hook(m, inp, out):
        B = inp[0].shape[0]
        out_len = out.shape[-1]
        kernel_ops = m.kernel_size[0] * (m.in_channels // m.groups)
        bias_ops = 1 if m.bias is not None else 0
        flops['total'] += int(B * m.out_channels * out_len * (2 * kernel_ops + bias_ops))

    def conv2d_hook(m, inp, out):
        B = inp[0].shape[0]
        out_h, out_w = out.shape[-2], out.shape[-1]
        k_h, k_w = m.kernel_size
        kernel_ops = k_h * k_w * (m.in_channels // m.groups)
        bias_ops = 1 if m.bias is not None else 0
        flops['total'] += int(B * m.out_channels * out_h * out_w * (2 * kernel_ops + bias_ops))

    def linear_hook(m, inp, out):
        num_outputs = out.numel() // m.out_features
        bias_ops = out.numel() if m.bias is not None else 0
        flops['total'] += int(num_outputs * m.in_features * m.out_features * 2 + bias_ops)

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv1d):
            hooks.append(mod.register_forward_hook(conv1d_hook))
        elif isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(conv2d_hook))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(linear_hook))

    with torch.no_grad():
        _ = model(*inputs)
    for h in hooks:
        h.remove()
    return flops['total']

def measure_model_cost(device=DEVICE, use_cache=True, save=True, verbose=True):
    cache_path = f"reproc/results/model_cost_xrf55dml_{HOSTNAME}.json"
    if use_cache and os.path.exists(cache_path):
        with open(cache_path) as f: data = json.load(f)
        if verbose:
            print(f"\nModel Cost (cached XRF55-DML, host={HOSTNAME})")
            print("-" * 64)
            print(f"{'Model':<12} | {'Params':>10} | {'Memory':>10} | {'FLOPs':>12}")
            print("-" * 64)
            for m, v in data.items():
                fl = f"{v['flops_G']:.4f}G" if v.get('flops_G') is not None else 'N/A'
                print(f"{m:<12} | {v['params_M']:>8.2f}M | {v['memory_MB']:>8.2f}MB | {fl:>12}")
            print(f"  Tip: Force re-measure: measure_model_cost(use_cache=False)")
        return data

    B = 1
    shapes = _infer_input_shapes(ROOT)
    if shapes is not None:
        if verbose:
            print(f"   [OK] Detected: WiFi={shapes['wifi']}, mmW={shapes['mmw']}, RFID={shapes['rfid']}")
        wifi = torch.randn(B, *shapes['wifi'], device=device)
        mmw  = torch.randn(B, *shapes['mmw'],  device=device)
        rfid = torch.randn(B, *shapes['rfid'], device=device)
    else:
        if verbose: print(f"   WARNING: using fallback dummy")
        wifi = torch.randn(B, 1, 270, 1000, device=device)
        mmw  = torch.randn(B, 17, 256, 128, device=device)
        rfid = torch.randn(B, 1, 23, 148, device=device)

    def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
    def occupy_mb(m):     return sum(p.numel()*p.element_size() for p in m.parameters())/(1024**2)

    if verbose:
        print(f"\nMeasuring XRF55-DML baseline cost (host={HOSTNAME})...")
        print("-" * 64)
        print(f"{'Model':<12} | {'Params':>10} | {'Memory':>10} | {'FLOPs':>12}")
        print("-" * 64)

    summary = {}

    # ----- 1) Per-backbone breakdown (params + FLOPs) -----
    sub_specs = [
        ('WiFi-R18',  WiFi_ResNet18Mutual,   (wifi,)),
        ('RFID-R18',  RFID_ResNet18Mutual,   (rfid,)),
        ('mmW-R18',   MmWave_ResNet18Mutual, (mmw,)),
    ]
    for sname, scls, sinp in sub_specs:
        sm = scls(num_classes=55).to(device).eval()
        params = count_params(sm); mem = occupy_mb(sm)
        with torch.no_grad():
            sflops = _estimate_flops_custom(sm, inputs=sinp) / 1e9
        if verbose:
            print(f"{sname:<12} | {params/1e6:>8.2f}M | {mem:>8.2f}MB | {sflops:>11.4f}G")
        summary[sname] = {'params_M': params/1e6, 'memory_MB': mem, 'flops_G': sflops,
                          'flops_note': 'custom hooks; 1 MAC = 2 FLOPs',
                          'hostname': HOSTNAME}
        del sm; torch.cuda.empty_cache(); gc.collect()

    # ----- 2) Full DML wrapper (params + FLOPs) -----
    for mname, mcls in MODEL_REGISTRY.items():
        model = mcls(num_classes=55).to(device).eval()
        params = count_params(model); mem = occupy_mb(model)
        with torch.no_grad():
            # XRF55_DML forward: (wifi, mmw, rfid, apply_drop=False)
            flops = _estimate_flops_custom(model, inputs=(wifi, mmw, rfid, False)) / 1e9

        if verbose:
            print(f"{mname:<12} | {params/1e6:>8.2f}M | {mem:>8.2f}MB | {flops:>11.4f}G")
        summary[mname] = {'params_M': params/1e6, 'memory_MB': mem, 'flops_G': flops,
                          'flops_note': 'custom hooks; 1 MAC = 2 FLOPs',
                          'hostname': HOSTNAME}
        del model; torch.cuda.empty_cache(); gc.collect()

    if save:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f: json.dump(summary, f, indent=2)
        if verbose: print(f"saved: {cache_path}")
    return summary


if os.environ.get("HYDRA_VERBOSE_IMPORT") == "1":
    print("ready (XRF55 official baseline - DML with 3 ResNet18).")
    print(f"   ROOT={ROOT}")
    print(f"   DEVICE={DEVICE}")
    print(f"   HOST={HOSTNAME}")

# Same split as the shared experiment protocol's In-Domain setting.

# ============================================================
# IN-DOMAIN Scene1 - XRF55 DML baseline (per-modality + fusion)
# ============================================================
@torch.no_grad()
def evaluate_model_all_modalities(model, loader, device):
    """Returns dict with per-modality and fusion top-1 accuracies."""
    model.eval()
    if len(loader) == 0:
        return {'WiFi': 0.0, 'RFID': 0.0, 'mmWave': 0.0, 'Fusion': 0.0}
    c_w = c_r = c_m = c_f = 0
    t = 0
    for data, labels in loader:
        wifi = data['WiFi'].to(device, non_blocking=True)
        mmw  = data['mmWave'].to(device, non_blocking=True)
        rfid = data['RFID'].to(device, non_blocking=True)
        tgt  = (labels - 1).to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(wifi, mmw, rfid)
        p_w = out['wifi_logits'].argmax(1)
        p_r = out['rfid_logits'].argmax(1)
        p_m = out['mmw_logits'].argmax(1)
        p_f = out['fused_prob'].argmax(1)
        c_w += p_w.eq(tgt).sum().item()
        c_r += p_r.eq(tgt).sum().item()
        c_m += p_m.eq(tgt).sum().item()
        c_f += p_f.eq(tgt).sum().item()
        t   += tgt.size(0)
    if t == 0:
        return {'WiFi': 0.0, 'RFID': 0.0, 'mmWave': 0.0, 'Fusion': 0.0}
    return {
        'WiFi':   100.0 * c_w / t,
        'RFID':   100.0 * c_r / t,
        'mmWave': 100.0 * c_m / t,
        'Fusion': 100.0 * c_f / t,
    }


def run_in_domain_scene1_xrf55(root_path=ROOT, device=DEVICE,
                               epochs=200, BS=64, NUM_WORKERS=4,
                               force_retrain=False):
    g = fresh_generator()
    train_samples = list(range(1, 15))
    test_samples  = list(range(15, 21))

    train_loader = make_loader(root_path,
        {'Scene1': {'samples': train_samples, 'vols': None}},
        BS, NUM_WORKERS, shuffle=True, generator=g)
    test_loader = make_loader(root_path,
        {'Scene1': {'samples': test_samples, 'vols': None}},
        BS, NUM_WORKERS, generator=g)
    eval_loaders = {'Scene1': test_loader}

    print("\n" + "="*80)
    print("[XRF55-DML | In-Domain Scene1]  Train: 1~14 | Test: 15~20")
    print("="*80)

    exp_name = "XRF55_InDomain_Scene1"
    ckpt_path = os.path.join("reproc", f"{exp_name}_last.pth")

    if (not force_retrain) and os.path.exists(ckpt_path):
        print(f"  Found checkpoint: {ckpt_path}  -> load & evaluate")
        model = XRF55_DML(num_classes=55).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        per_mod = evaluate_model_all_modalities(model, test_loader, device)
        del model; torch.cuda.empty_cache(); gc.collect()
        print(f"   [Loaded] WiFi:{per_mod['WiFi']:.2f}% | "
              f"RFID:{per_mod['RFID']:.2f}% | "
              f"mmWave:{per_mod['mmWave']:.2f}% | "
              f"Fusion:{per_mod['Fusion']:.2f}%")
    else:
        print(f"   No checkpoint -> train and save to {ckpt_path}")
        # train_and_eval is reused from the shared training routine (trains and saves on the fusion criterion)
        _ = train_and_eval(exp_name, XRF55_DML,
                           train_loader, eval_loaders, lambda r: r['Scene1'],
                           device, epochs=epochs)
        # Re-evaluate per-modality from the saved last checkpoint
        model = XRF55_DML(num_classes=55).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        per_mod = evaluate_model_all_modalities(model, test_loader, device)
        del model; torch.cuda.empty_cache(); gc.collect()

    print("\n" + "="*80)
    print("XRF55-DML In-Domain Scene1 Result")
    print("="*80)
    print(f"   WiFi   (resnet1d)      : {per_mod['WiFi']:.2f}%")
    print(f"   RFID   (resnet1d_rfid) : {per_mod['RFID']:.2f}%")
    print(f"   mmWave (resnet2d)      : {per_mod['mmWave']:.2f}%")
    print(f"   Fusion (avg softmax)   : {per_mod['Fusion']:.2f}%")

    save_result("xrf55_in_domain_scene1", {
        'WiFi':   float(per_mod['WiFi']),
        'RFID':   float(per_mod['RFID']),
        'mmWave': float(per_mod['mmWave']),
        'Fusion': float(per_mod['Fusion']),
    })
    return per_mod


# Run with scripts/run_experiment.py.


# Overlap subjects {3,4,5,6,7,13,23,24} included in source so the Cross-Scene setting can reuse this pretrain ckpt under strict subject consistency.

# ============================================================
# CROSS-SUBJECT 21-9 - XRF55 DML baseline (per-modality + fusion)
# ============================================================
OVERLAP_S2 = [5, 24]
OVERLAP_S3 = [6, 7, 23]
OVERLAP_S4 = [3, 4, 13]
OVERLAP_VOLS = sorted(OVERLAP_S2 + OVERLAP_S3 + OVERLAP_S4)


def _split_source_target(root_path, n_source=21, n_target=9):
    all_vols = get_volunteers_in_scene(root_path, 'Scene1')
    missing = [v for v in OVERLAP_VOLS if v not in all_vols]
    assert not missing, f"not in scene 1 overlap vols: {missing}"

    non_overlap = sorted([v for v in all_vols if v not in OVERLAP_VOLS])
    n_extra_src = n_source - len(OVERLAP_VOLS)
    assert n_extra_src >= 0
    assert len(non_overlap) >= n_extra_src + n_target

    src = sorted(OVERLAP_VOLS + non_overlap[:n_extra_src])
    tgt = non_overlap[n_extra_src : n_extra_src + n_target]
    assert len(src) == n_source and len(tgt) == n_target
    assert set(src).isdisjoint(tgt)
    return src, tgt


@torch.no_grad()
def evaluate_model_all_modalities(model, loader, device):
    """Per-modality + fusion top-1 accuracy in one forward pass."""
    model.eval()
    if len(loader) == 0:
        return {'WiFi': 0.0, 'RFID': 0.0, 'mmWave': 0.0, 'Fusion': 0.0}
    c_w = c_r = c_m = c_f = 0
    t = 0
    for data, labels in loader:
        wifi = data['WiFi'].to(device, non_blocking=True)
        mmw  = data['mmWave'].to(device, non_blocking=True)
        rfid = data['RFID'].to(device, non_blocking=True)
        tgt  = (labels - 1).to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(wifi, mmw, rfid)
        c_w += out['wifi_logits'].argmax(1).eq(tgt).sum().item()
        c_r += out['rfid_logits'].argmax(1).eq(tgt).sum().item()
        c_m += out['mmw_logits'].argmax(1).eq(tgt).sum().item()
        c_f += out['fused_prob'].argmax(1).eq(tgt).sum().item()
        t   += tgt.size(0)
    return {
        'WiFi':   100.0 * c_w / t,
        'RFID':   100.0 * c_r / t,
        'mmWave': 100.0 * c_m / t,
        'Fusion': 100.0 * c_f / t,
    }


def _zero_shot_eval_all(model_cls, pretrain_ckpt, eval_loader, device):
    model = model_cls(num_classes=55).to(device)
    print(f"       Loading pretrain ckpt (0-shot, no FT): {pretrain_ckpt}")
    model.load_state_dict(torch.load(pretrain_ckpt, map_location=device))
    model.eval()
    res = evaluate_model_all_modalities(model, eval_loader, device)
    del model; torch.cuda.empty_cache(); gc.collect()
    return res


def _load_and_eval_all(model_cls, ckpt_path, eval_loader, device):
    model = model_cls(num_classes=55).to(device)
    print(f"       Loading FT ckpt: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    res = evaluate_model_all_modalities(model, eval_loader, device)
    del model; torch.cuda.empty_cache(); gc.collect()
    return res


def _finetune_eval_all(model_cls, pretrain_ckpt, ft_loader, eval_loader, device,
                       epochs=100, lr=1e-5, log_every=10,
                       ft_seed=None, save_path=None):
    set_reproducible_mode(ft_seed if ft_seed is not None else SEED)
    model = model_cls(num_classes=55).to(device)
    print(f"       Loading pretrain ckpt: {pretrain_ckpt}")
    model.load_state_dict(torch.load(pretrain_ckpt, map_location=device))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        sum_loss = 0.0
        for data, labels in ft_loader:
            wifi = data['WiFi'].to(device, non_blocking=True)
            mmw  = data['mmWave'].to(device, non_blocking=True)
            rfid = data['RFID'].to(device, non_blocking=True)
            tgt  = (labels - 1).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(wifi, mmw, rfid)
                loss = _dml_loss(out['wifi_logits'], out['rfid_logits'],
                                 out['mmw_logits'], tgt, criterion)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update()
            sum_loss += loss.item()

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"      [FT Ep {epoch+1:03d}/{epochs}] Loss:{sum_loss/len(ft_loader):.4f}")

    res = evaluate_model_all_modalities(model, eval_loader, device)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"       FT ckpt saved -> {save_path}")
    del model; torch.cuda.empty_cache(); gc.collect()
    return res


def run_cross_subject_21_9_xrf55(root_path=ROOT, device=DEVICE,
                                 pretrain_epochs=200,
                                 ft_epochs=100, ft_lr=1e-5,
                                 BS=64, NUM_WORKERS=4,
                                 shots=(0, 1, 2, 3, 4, 5),
                                 force_pretrain=False):
    all_vols = get_volunteers_in_scene(root_path, 'Scene1')
    SOURCE_VOLS, TARGET_VOLS = _split_source_target(root_path, 21, 9)

    print(f"Scene1 volunteers: {len(all_vols)} -> {all_vols}")
    print(f"   Overlap subjs : {OVERLAP_VOLS}")
    print(f"   Source(21)    : {SOURCE_VOLS}")
    print(f"   Target(9)     : {TARGET_VOLS}")

    results = {'XRF55': {}}
    mod = 'XRF55'

    print(f"\n{'='*80}\n [XRF55-DML] Cross-Subject 21-9\n{'='*80}")
    pretrain_exp  = "XRF55_SCSub_21_9_pretrain"
    pretrain_ckpt = os.path.join("reproc", f"{pretrain_exp}_last.pth")

    if os.path.exists(pretrain_ckpt) and not force_pretrain:
        print(f"  [Stage 1: Pretrain] [OK] -> {pretrain_ckpt}")
    else:
        print(f"  [Stage 1: Pretrain] 21 x 55 x 20 = {21*55*20:,} samples, ep={pretrain_epochs}")
        g = fresh_generator()
        pretrain_loader = make_loader(root_path,
            {'Scene1': {'samples': list(range(1, 21)), 'vols': SOURCE_VOLS}},
            BS, NUM_WORKERS, shuffle=True, generator=g)
        sanity_loader = make_loader(root_path,
            {'Scene1': {'samples': [1], 'vols': SOURCE_VOLS}},
            BS, NUM_WORKERS, generator=g)
        _ = train_and_eval(pretrain_exp, XRF55_DML,
                           pretrain_loader, {'sanity': sanity_loader},
                           lambda r: r['sanity'], device,
                           epochs=pretrain_epochs)
        print(f"    [OK] saved -> {pretrain_ckpt}")

    for k in shots:
        print(f"\n  [Stage 2: {k}-shot] target = 9 vols")
        g = fresh_generator()

        if k == 0:
            eval_samples = list(range(1, 21))
            eval_loader = make_loader(root_path,
                {'Scene1': {'samples': eval_samples, 'vols': TARGET_VOLS}},
                BS, NUM_WORKERS, generator=g)
            print(f"    no fine-tune  (eval per-class = {len(eval_samples)})")
            acc_dict = _zero_shot_eval_all(XRF55_DML, pretrain_ckpt, eval_loader, device)
        else:
            eval_samples = list(range(k+1, 21))
            eval_loader = make_loader(root_path,
                {'Scene1': {'samples': eval_samples, 'vols': TARGET_VOLS}},
                BS, NUM_WORKERS, generator=g)

            ft_save_path = os.path.join("reproc",
                                        f"XRF55_SCSub_21_9_{k}shot_ft.pth")
            if os.path.exists(ft_save_path):
                print(f"    [{k}-shot] Found FT ckpt: {ft_save_path}  -> load & evaluate")
                acc_dict = _load_and_eval_all(XRF55_DML, ft_save_path, eval_loader, device)
            else:
                ft_loader = make_loader(root_path,
                    {'Scene1': {'samples': list(range(1, k+1)), 'vols': TARGET_VOLS}},
                    BS, NUM_WORKERS, shuffle=True, generator=g)
                print(f"    fine-tune: 9 x 55 x {k} = {9*55*k} samples"
                      f"  (ep={ft_epochs}, lr={ft_lr})  | eval per-class = {len(eval_samples)}")
                acc_dict = _finetune_eval_all(XRF55_DML, pretrain_ckpt,
                                              ft_loader, eval_loader, device,
                                              epochs=ft_epochs, lr=ft_lr,
                                              save_path=ft_save_path)

        results[mod][k] = acc_dict
        print(f"    [OK] [XRF55 {k}-shot] "
              f"WiFi:{acc_dict['WiFi']:.2f}% | "
              f"RFID:{acc_dict['RFID']:.2f}% | "
              f"mmWave:{acc_dict['mmWave']:.2f}% | "
              f"Fusion:{acc_dict['Fusion']:.2f}%")

    # Print 4 tables, one per modality
    print_paper_table("XRF55-DML | Cross-Subject 21-9", results, shots,
                      eval_keys=['WiFi', 'RFID', 'mmWave', 'Fusion'])
    save_result("xrf55_cross_subject_21_9", {
        "overlap_vols": OVERLAP_VOLS,
        "source_vols": SOURCE_VOLS,
        "target_vols": TARGET_VOLS,
        "pretrain_epochs": pretrain_epochs,
        "ft_epochs": ft_epochs, "ft_lr": ft_lr,
        "results": {m: {f"{k}shot": d for k, d in dd.items()}
                    for m, dd in results.items()}
    })
    return results


# Reuses the Cross-Subject pretrain ckpt (`XRF55_SCSub_21_9_pretrain_last.pth`) with strict subject consistency.

# ============================================================
# CROSS-SCENE - XRF55 DML baseline (per-modality + fusion)
# ============================================================
@torch.no_grad()
def evaluate_model_all_modalities(model, loader, device):
    model.eval()
    if len(loader) == 0:
        return {'WiFi': 0.0, 'RFID': 0.0, 'mmWave': 0.0, 'Fusion': 0.0}
    c_w = c_r = c_m = c_f = 0
    t = 0
    for data, labels in loader:
        wifi = data['WiFi'].to(device, non_blocking=True)
        mmw  = data['mmWave'].to(device, non_blocking=True)
        rfid = data['RFID'].to(device, non_blocking=True)
        tgt  = (labels - 1).to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(wifi, mmw, rfid)
        c_w += out['wifi_logits'].argmax(1).eq(tgt).sum().item()
        c_r += out['rfid_logits'].argmax(1).eq(tgt).sum().item()
        c_m += out['mmw_logits'].argmax(1).eq(tgt).sum().item()
        c_f += out['fused_prob'].argmax(1).eq(tgt).sum().item()
        t   += tgt.size(0)
    return {
        'WiFi':   100.0 * c_w / t,
        'RFID':   100.0 * c_r / t,
        'mmWave': 100.0 * c_m / t,
        'Fusion': 100.0 * c_f / t,
    }


def _zero_shot_eval_all(model_cls, pretrain_ckpt, eval_loader, device):
    model = model_cls(num_classes=55).to(device)
    print(f"       Loading pretrain ckpt (0-shot, no FT): {pretrain_ckpt}")
    model.load_state_dict(torch.load(pretrain_ckpt, map_location=device))
    model.eval()
    res = evaluate_model_all_modalities(model, eval_loader, device)
    del model; torch.cuda.empty_cache(); gc.collect()
    return res


def _finetune_eval_all(model_cls, pretrain_ckpt, ft_loader, eval_loader, device,
                       epochs=100, lr=1e-5, log_every=10,
                       ft_seed=None, save_path=None):
    set_reproducible_mode(ft_seed if ft_seed is not None else SEED)
    model = model_cls(num_classes=55).to(device)
    print(f"       Loading pretrain ckpt: {pretrain_ckpt}")
    model.load_state_dict(torch.load(pretrain_ckpt, map_location=device))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        sum_loss = 0.0
        for data, labels in ft_loader:
            wifi = data['WiFi'].to(device, non_blocking=True)
            mmw  = data['mmWave'].to(device, non_blocking=True)
            rfid = data['RFID'].to(device, non_blocking=True)
            tgt  = (labels - 1).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(wifi, mmw, rfid)
                loss = _dml_loss(out['wifi_logits'], out['rfid_logits'],
                                 out['mmw_logits'], tgt, criterion)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update()
            sum_loss += loss.item()

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"      [FT Ep {epoch+1:03d}/{epochs}] Loss:{sum_loss/len(ft_loader):.4f}")

    res = evaluate_model_all_modalities(model, eval_loader, device)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"       FT ckpt saved -> {save_path}")
    del model; torch.cuda.empty_cache(); gc.collect()
    return res


def run_cross_scene_xrf55(root_path=ROOT, device=DEVICE,
                          ft_epochs=100, ft_lr=1e-5,
                          BS=64, NUM_WORKERS=4,
                          shots=(0, 1, 2, 3, 4, 5),
                          strict_subject_consistency=True):
    target_scenes = ['Scene2', 'Scene3', 'Scene4']
    scene_vols_raw = {sc: get_volunteers_in_scene(root_path, sc) for sc in target_scenes}

    if strict_subject_consistency:
        SOURCE_VOLS, _ = _split_source_target(root_path, 21, 9)
        scene_vols = {sc: [v for v in vs if v in SOURCE_VOLS]
                      for sc, vs in scene_vols_raw.items()}
        for sc in target_scenes:
            removed = sorted(set(scene_vols_raw[sc]) - set(scene_vols[sc]))
            if removed:
                print(f"WARNING:  strict mode -> {sc} vols {removed} excluded")
    else:
        scene_vols = scene_vols_raw
        print("INFO:  lenient mode")

    print("\n Final target scene vols:")
    for sc, vs in scene_vols.items():
        print(f"   {sc}: {vs}")

    results = {'XRF55': {k: {} for k in shots}}
    mod = 'XRF55'

    print(f"\n{'='*80}\n [XRF55-DML] Cross-Scene\n{'='*80}")
    pretrain_ckpt = os.path.join("reproc", "XRF55_SCSub_21_9_pretrain_last.pth")
    has_pretrain = os.path.exists(pretrain_ckpt)
    if has_pretrain:
        print(f"  Reusing pretrain -> {pretrain_ckpt}")
    else:
        print(f"  No pretrain ckpt found yet: {pretrain_ckpt}")

    for target_scene in target_scenes:
        target_vols = scene_vols[target_scene]
        print(f"\n  > Target = {target_scene}  vols={target_vols}")

        for k in shots:
            g = fresh_generator()

            if k == 0:
                if not has_pretrain:
                    raise FileNotFoundError(
                        f"\n  No Pretrain ckpt: {pretrain_ckpt}"
                        f"\n  -> Run run_cross_subject_21_9_xrf55() first or omit 0-shot."
                    )
                eval_samples = list(range(1, 21))
                eval_loader = make_loader(root_path,
                    {target_scene: {'samples': eval_samples, 'vols': target_vols}},
                    BS, NUM_WORKERS, generator=g)
                print(f"    [{k}-shot] no FT  | eval per-class = {len(eval_samples)}")
                acc_dict = _zero_shot_eval_all(XRF55_DML, pretrain_ckpt,
                                               eval_loader, device)
            else:
                eval_samples = list(range(k+1, 21))
                eval_loader = make_loader(root_path,
                    {target_scene: {'samples': eval_samples, 'vols': target_vols}},
                    BS, NUM_WORKERS, generator=g)
                ft_save_path = os.path.join("reproc",
                        f"XRF55_CrossScene_{target_scene}_{k}shot_ft.pth")
                if os.path.exists(ft_save_path):
                    print(f"    [{k}-shot] Found FT ckpt: {ft_save_path}  -> load & evaluate")
                    model = XRF55_DML(num_classes=55).to(device)
                    model.load_state_dict(torch.load(ft_save_path, map_location=device))
                    acc_dict = evaluate_model_all_modalities(model, eval_loader, device)
                    del model; torch.cuda.empty_cache(); gc.collect()
                else:
                    if not has_pretrain:
                        raise FileNotFoundError(
                            f"\n  No FT ckpt: {ft_save_path}"
                            f"\n  No Pretrain ckpt: {pretrain_ckpt}"
                            f"\n  -> Provide an existing FT checkpoint or run cross-subject pretraining first."
                        )
                    ft_loader = make_loader(root_path,
                        {target_scene: {'samples': list(range(1, k+1)),
                                        'vols': target_vols}},
                        BS, NUM_WORKERS, shuffle=True, generator=g)
                    n_ft = len(target_vols) * 55 * k
                    print(f"    [{k}-shot] FT: {len(target_vols)} x 55 x {k} = {n_ft}"
                          f"  (ep={ft_epochs}, lr={ft_lr})"
                          f"  | eval per-class = {len(eval_samples)}")
                    acc_dict = _finetune_eval_all(XRF55_DML, pretrain_ckpt,
                                                  ft_loader, eval_loader, device,
                                                  epochs=ft_epochs, lr=ft_lr,
                                                  save_path=ft_save_path)

            results[mod][k][target_scene] = acc_dict
            print(f"      [OK] XRF55 {target_scene} {k}-shot: "
                  f"WiFi:{acc_dict['WiFi']:.2f}% | "
                  f"RFID:{acc_dict['RFID']:.2f}% | "
                  f"mmWave:{acc_dict['mmWave']:.2f}% | "
                  f"Fusion:{acc_dict['Fusion']:.2f}%")

    # Print 4 tables, one per modality (each with Scene2/3/4 columns)
    # The original print_paper_table builds columns from eval_keys,
    # but here results[mod][k][scene] is a dict, so we flatten per-modality and print separately.
    print("\n" + "="*80)
    print("XRF55-DML | Cross-Scene  (per-modality + fusion)")
    print("="*80)
    shot_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five']
    for modality in ['WiFi', 'RFID', 'mmWave', 'Fusion']:
        print(f"\n[Modality: {modality}]")
        header = f"{'#-Shot':<8} | " + " | ".join([f"{sc:>10}" for sc in target_scenes])
        print(header); print("-" * len(header))
        for k in shots:
            kname = shot_names[k]
            row = f"{kname:<8} | "
            row += " | ".join([
                f"{results[mod][k].get(sc, {}).get(modality, float('nan')):>9.2f}%"
                if results[mod][k].get(sc, {}).get(modality, float('nan')) ==
                   results[mod][k].get(sc, {}).get(modality, float('nan'))
                else f"{'N/A':>10}"
                for sc in target_scenes])
            print(row)

    save_result("xrf55_cross_scene", {
        "target_scenes_vols": scene_vols,
        "pretrain_source": "XRF55_SCSub_21_9",
        "ft_epochs": ft_epochs, "ft_lr": ft_lr,
        "results": {m: {f"{k}shot": d for k, d in dd.items()}
                    for m, dd in results.items()}
    })
    return results


# New pretrain on the 22 clean Scene1 subjects (excluding overlap subjs {3,4,5,6,7,13,23,24}), then adapt to S2/S3/S4.

# ============================================================
# CSC&S - XRF55 DML baseline (per-modality + fusion)
# ============================================================
@torch.no_grad()
def evaluate_model_all_modalities(model, loader, device):
    model.eval()
    if len(loader) == 0:
        return {'WiFi': 0.0, 'RFID': 0.0, 'mmWave': 0.0, 'Fusion': 0.0}
    c_w = c_r = c_m = c_f = 0
    t = 0
    for data, labels in loader:
        wifi = data['WiFi'].to(device, non_blocking=True)
        mmw  = data['mmWave'].to(device, non_blocking=True)
        rfid = data['RFID'].to(device, non_blocking=True)
        tgt  = (labels - 1).to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(wifi, mmw, rfid)
        c_w += out['wifi_logits'].argmax(1).eq(tgt).sum().item()
        c_r += out['rfid_logits'].argmax(1).eq(tgt).sum().item()
        c_m += out['mmw_logits'].argmax(1).eq(tgt).sum().item()
        c_f += out['fused_prob'].argmax(1).eq(tgt).sum().item()
        t   += tgt.size(0)
    return {
        'WiFi':   100.0 * c_w / t,
        'RFID':   100.0 * c_r / t,
        'mmWave': 100.0 * c_m / t,
        'Fusion': 100.0 * c_f / t,
    }


def _zero_shot_eval_all(model_cls, pretrain_ckpt, eval_loader, device):
    model = model_cls(num_classes=55).to(device)
    print(f"       Loading pretrain ckpt (0-shot, no FT): {pretrain_ckpt}")
    model.load_state_dict(torch.load(pretrain_ckpt, map_location=device))
    model.eval()
    res = evaluate_model_all_modalities(model, eval_loader, device)
    del model; torch.cuda.empty_cache(); gc.collect()
    return res


def _finetune_eval_all(model_cls, pretrain_ckpt, ft_loader, eval_loader, device,
                       epochs=100, lr=1e-5, log_every=10,
                       ft_seed=None, save_path=None):
    set_reproducible_mode(ft_seed if ft_seed is not None else SEED)
    model = model_cls(num_classes=55).to(device)
    print(f"       Loading pretrain ckpt: {pretrain_ckpt}")
    model.load_state_dict(torch.load(pretrain_ckpt, map_location=device))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        sum_loss = 0.0
        for data, labels in ft_loader:
            wifi = data['WiFi'].to(device, non_blocking=True)
            mmw  = data['mmWave'].to(device, non_blocking=True)
            rfid = data['RFID'].to(device, non_blocking=True)
            tgt  = (labels - 1).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(wifi, mmw, rfid)
                loss = _dml_loss(out['wifi_logits'], out['rfid_logits'],
                                 out['mmw_logits'], tgt, criterion)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update()
            sum_loss += loss.item()

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"      [FT Ep {epoch+1:03d}/{epochs}] Loss:{sum_loss/len(ft_loader):.4f}")

    res = evaluate_model_all_modalities(model, eval_loader, device)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"       FT ckpt saved -> {save_path}")
    del model; torch.cuda.empty_cache(); gc.collect()
    return res


def run_cscs_xrf55(root_path=ROOT, device=DEVICE,
                   pretrain_epochs=200,
                   ft_epochs=100, ft_lr=1e-5,
                   BS=64, NUM_WORKERS=4,
                   shots=(0, 1, 2, 3, 4, 5),
                   force_pretrain=False):
    target_scenes = ['Scene2', 'Scene3', 'Scene4']
    scene_vols = {sc: get_volunteers_in_scene(root_path, sc) for sc in target_scenes}

    all_vols = get_volunteers_in_scene(root_path, 'Scene1')
    contaminated = set()
    for sc in target_scenes:
        contaminated.update(scene_vols[sc])
    SOURCE_VOLS = sorted([v for v in all_vols if v not in contaminated])

    print("CSC&S setup:")
    print(f"   Scene1 vols total      : {len(all_vols)}")
    print(f"   Contaminated (excluded): {sorted(contaminated)}")
    print(f"   Source ({len(SOURCE_VOLS)} subjs) : {SOURCE_VOLS}")
    print(f"   Target scenes (with new subjects):")
    for sc, vs in scene_vols.items():
        print(f"     {sc}: {vs}")

    results = {'XRF55': {k: {} for k in shots}}
    mod = 'XRF55'

    print(f"\n{'='*80}\n [XRF55-DML] CSC&S\n{'='*80}")
    pretrain_exp  = "XRF55_CSCS_pretrain"
    pretrain_ckpt = os.path.join("reproc", f"{pretrain_exp}_last.pth")

    if (not os.path.exists(pretrain_ckpt)) or force_pretrain:
        n_src = len(SOURCE_VOLS)
        print(f"  [Stage 1: NEW Pretrain] {n_src} x 55 x 20 = "
              f"{n_src*55*20:,} samples,  ep={pretrain_epochs}")
        g = fresh_generator()
        pretrain_loader = make_loader(root_path,
            {'Scene1': {'samples': list(range(1, 21)), 'vols': SOURCE_VOLS}},
            BS, NUM_WORKERS, shuffle=True, generator=g)
        sanity_loader = make_loader(root_path,
            {'Scene1': {'samples': [1], 'vols': SOURCE_VOLS}},
            BS, NUM_WORKERS, generator=g)
        _ = train_and_eval(pretrain_exp, XRF55_DML,
                           pretrain_loader, {'sanity': sanity_loader},
                           lambda r: r['sanity'], device,
                           epochs=pretrain_epochs)
        print(f"    [OK] saved -> {pretrain_ckpt}")
    else:
        print(f"  [Stage 1: Pretrain] [OK] ckpt, skip -> {pretrain_ckpt}")

    for target_scene in target_scenes:
        target_vols = scene_vols[target_scene]
        print(f"\n  > Target = {target_scene}  vols={target_vols}")

        for k in shots:
            g = fresh_generator()

            if k == 0:
                eval_samples = list(range(1, 21))
                eval_loader = make_loader(root_path,
                    {target_scene: {'samples': eval_samples, 'vols': target_vols}},
                    BS, NUM_WORKERS, generator=g)
                print(f"    [{k}-shot] no FT  | eval per-class = {len(eval_samples)}")
                acc_dict = _zero_shot_eval_all(XRF55_DML, pretrain_ckpt,
                                               eval_loader, device)
            else:
                eval_samples = list(range(k+1, 21))
                eval_loader = make_loader(root_path,
                    {target_scene: {'samples': eval_samples, 'vols': target_vols}},
                    BS, NUM_WORKERS, generator=g)
                ft_save_path = os.path.join("reproc",
                        f"XRF55_CSCS_{target_scene}_{k}shot_ft.pth")
                if os.path.exists(ft_save_path):
                    print(f"    [{k}-shot] Found FT ckpt: {ft_save_path}  -> load & evaluate")
                    acc_dict = _load_and_eval_all(XRF55_DML, ft_save_path, eval_loader, device)
                else:
                    ft_loader = make_loader(root_path,
                        {target_scene: {'samples': list(range(1, k+1)),
                                        'vols': target_vols}},
                        BS, NUM_WORKERS, shuffle=True, generator=g)
                    n_ft = len(target_vols) * 55 * k
                    print(f"    [{k}-shot] FT: {len(target_vols)} x 55 x {k} = {n_ft}"
                          f"  (ep={ft_epochs}, lr={ft_lr})"
                          f"  | eval per-class = {len(eval_samples)}")
                    acc_dict = _finetune_eval_all(XRF55_DML, pretrain_ckpt,
                                                  ft_loader, eval_loader, device,
                                                  epochs=ft_epochs, lr=ft_lr,
                                                  save_path=ft_save_path)

            results[mod][k][target_scene] = acc_dict
            print(f"      [OK] XRF55 {target_scene} {k}-shot: "
                  f"WiFi:{acc_dict['WiFi']:.2f}% | "
                  f"RFID:{acc_dict['RFID']:.2f}% | "
                  f"mmWave:{acc_dict['mmWave']:.2f}% | "
                  f"Fusion:{acc_dict['Fusion']:.2f}%")

    # 4 tables, one per modality (each with Scene2/3/4 columns)
    print("\n" + "="*80)
    print("XRF55-DML | CSC&S  (per-modality + fusion)")
    print("="*80)
    shot_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five']
    for modality in ['WiFi', 'RFID', 'mmWave', 'Fusion']:
        print(f"\n[Modality: {modality}]")
        header = f"{'#-Shot':<8} | " + " | ".join([f"{sc:>10}" for sc in target_scenes])
        print(header); print("-" * len(header))
        for k in shots:
            kname = shot_names[k]
            row = f"{kname:<8} | "
            row += " | ".join([
                f"{results[mod][k].get(sc, {}).get(modality, float('nan')):>9.2f}%"
                if results[mod][k].get(sc, {}).get(modality, float('nan')) ==
                   results[mod][k].get(sc, {}).get(modality, float('nan'))
                else f"{'N/A':>10}"
                for sc in target_scenes])
            print(row)

    save_result("xrf55_cscs", {
        "source_vols": SOURCE_VOLS,
        "excluded_vols": sorted(contaminated),
        "target_scenes_vols": scene_vols,
        "pretrain_epochs": pretrain_epochs,
        "ft_epochs": ft_epochs, "ft_lr": ft_lr,
        "results": {m: {f"{k}shot": d for k, d in dd.items()}
                    for m, dd in results.items()}
    })
    return results


# This baseline ports the model configuration of [aiotgroup/XRF55-repo](https://github.com/aiotgroup/XRF55-repo) - three ResNet18 backbones trained jointly with Deep Mutual Learning (DML). What we faithfully replicate from the official code:
#
# - Three modality-specific backbones: 1D ResNet18 for WiFi, 1D ResNet18 for RFID (23 tags as input channels), 2D ResNet18 for mmWave (17 frames as input channels). All return `(logits, vec)`.
# - DML loss: per modality, `CE + (1/(M-1)) sum_{j!=i} KL(P_i || stop_grad(P_j))` with M=3 (mirrors `dml_train.py`).
# - Adam, lr=1e-3, MultiStepLR(milestones=[40,80,120,160], gamma=0.5), epochs=200, batch_size=64 (from `opts.py`).
# - Eval-time fusion = simple averaging of the three softmax outputs (mirrors `dml_eval.py`).
#
# What we deliberately leave out:
#
# - The auxiliary L1 alignment between each modality's `vec` head and a BERT-encoded class-name embedding (`bert_new_sentence_large_uncased.npy`). This requires the BERT vector file which is auxiliary and not part of the core DML idea. Our implementation still produces a `vec` head per modality for compatibility but does not regularize it.
# - The official repo trains on a fixed 70/30 split (`split_train_test.py`); we instead use the same data splits as the shared experiment protocol so all rows in the comparison table use identical train/test boundaries.
#
# Checkpoints saved with an `XRF55_` prefix to coexist with the main model's and other baseline ckpts in `reproc/`.
#
#

# ============================================================
# XRF55 Baseline (DML, 3 ResNet18) - Sensor-Failure Robustness
#   Assumption: the system can detect sensor failure and exclude
#                the dead modality's logit from the fusion (softmax average).
#   Training is already done; just load the 5-shot FT ckpts and evaluate.
#
#   ckpt path pattern: reproc/XRF55_CSCS_{Scene2,Scene3,Scene4}_5shot_ft.pth
#   (if different, pass ckpt_pattern to run_sensor_failure_xrf55)
#
#   WARNING: Eval range is range(SHOT+1, 21) - samples 1..SHOT used during FT are
#      excluded from eval (to prevent data leakage). The "all (no fail)"
#      number should therefore exactly match the 5-shot Fusion result
#      reported in the CSC&S setting.
# ============================================================
import os, glob, csv, json

SHOT = 5
CKPT_PATTERN = "reproc/XRF55_CSCS_{scene}_{shot}shot_ft.pth"

# 1) Masked forward - XRF55 DML specific
#    Original XRF55_DML fusion: (softmax(w_log) + softmax(r_log) + softmax(m_log)) / 3
#    "system knows which sensor is dead" -> drop the dead modality's softmax from the average.
#    Average over surviving modalities only.
@torch.no_grad()
def xrf55_forward_masked(model, wifi_x, mmw_x, rfid_x, keep=('W','M','R')):
    """Use only the modalities listed in `keep` for the softmax average."""
    model.eval()
    keep = set(keep)

    # All three backbones run normally (inputs are not zeroed)
    w_log, _ = model.wifi(wifi_x)
    r_log, _ = model.rfid(rfid_x)
    m_log, _ = model.mmw(mmw_x)

    probs_list = []
    if 'W' in keep: probs_list.append(F.softmax(w_log, dim=-1))
    if 'M' in keep: probs_list.append(F.softmax(m_log, dim=-1))
    if 'R' in keep: probs_list.append(F.softmax(r_log, dim=-1))

    if len(probs_list) == 0:
        # All modalities dead - should not happen (such combos are not in KEEP_COMBOS)
        return torch.zeros_like(w_log)

    fused_prob = sum(probs_list) / len(probs_list)
    return fused_prob


@torch.no_grad()
def xrf55_evaluate_masked(model, loader, device, keep):
    model.eval()
    c, t = 0, 0
    for data, labels in loader:
        wifi = data['WiFi'].to(device, non_blocking=True)
        mmw  = data['mmWave'].to(device, non_blocking=True)
        rfid = data['RFID'].to(device, non_blocking=True)
        tgt  = (labels - 1).to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            fused_prob = xrf55_forward_masked(model, wifi, mmw, rfid, keep=keep)
        pred = fused_prob.argmax(1)
        c += pred.eq(tgt).sum().item()
        t += tgt.size(0)
    return 100.0 * c / t if t > 0 else 0.0


@torch.no_grad()
def xrf55_evaluate_all_masks(model, loader, device):
    """Evaluate every keep combination with one backbone pass per batch."""
    model.eval()
    correct = {KEEP_LABEL[keep]: 0 for keep in KEEP_COMBOS}
    total = 0
    autocast_enabled = torch.device(device).type == "cuda"

    for data, labels in loader:
        wifi = data['WiFi'].to(device, non_blocking=True)
        mmw  = data['mmWave'].to(device, non_blocking=True)
        rfid = data['RFID'].to(device, non_blocking=True)
        tgt  = (labels - 1).to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=autocast_enabled):
            w_log, _ = model.wifi(wifi)
            r_log, _ = model.rfid(rfid)
            m_log, _ = model.mmw(mmw)
            probs = {
                'W': F.softmax(w_log, dim=-1),
                'M': F.softmax(m_log, dim=-1),
                'R': F.softmax(r_log, dim=-1),
            }

        for keep in KEEP_COMBOS:
            fused_prob = sum(probs[mod] for mod in keep) / len(keep)
            pred = fused_prob.argmax(1)
            correct[KEEP_LABEL[keep]] += pred.eq(tgt).sum().item()
        total += tgt.size(0)

    if total == 0:
        raise ValueError("Evaluation loader is empty.")
    return {label: 100.0 * c / total for label, c in correct.items()}


# 2) Keep combinations (same as the proposed model)
KEEP_COMBOS = [
    ('W',), ('M',), ('R',),
    ('W','M'), ('W','R'), ('M','R'),
    ('W','M','R'),
]
KEEP_LABEL = {
    ('W',):'W only', ('M',):'M only', ('R',):'R only',
    ('W','M'):'W+M (R fail)', ('W','R'):'W+R (M fail)', ('M','R'):'M+R (W fail)',
    ('W','M','R'):'all (no fail)',
}

# 3) Build eval loaders
#    WARNING: Samples 1..SHOT were used during FT, so they are excluded from eval.
#    eval set = range(SHOT+1, 21)  (identical to the CSC&S setting's 5-shot eval)
def build_cscs_eval_loaders(root_path=ROOT, BS=16, NUM_WORKERS=4, shot=SHOT):
    target_scenes = ['Scene2', 'Scene3', 'Scene4']
    scene_vols = {sc: get_volunteers_in_scene(root_path, sc) for sc in target_scenes}
    eval_samples = list(range(shot + 1, 21))
    print(f"  eval samples per class = {len(eval_samples)}  "
          f"(range({shot+1}, 21); samples 1..{shot} were used for FT)")
    loaders = {}
    for sc in target_scenes:
        g = fresh_generator()
        loaders[sc] = make_loader(root_path,
            {sc: {'samples': eval_samples, 'vols': scene_vols[sc]}},
            BS, NUM_WORKERS, generator=g)
    return loaders

# A leaky EVAL_LOADERS lingering from a previous run would silently bypass
# this fix, so always rebuild.

def run_sensor_failure_xrf55(root_path=ROOT, device=DEVICE,
                             BS=64, NUM_WORKERS=4, shot=5,
                             ckpt_pattern=None):
    """Evaluate XRF55-DML under missing modalities using CSC&S FT ckpts."""
    pattern = ckpt_pattern or CKPT_PATTERN
    eval_loaders = build_cscs_eval_loaders(root_path, BS=BS, NUM_WORKERS=NUM_WORKERS, shot=shot)
    raw = {}

    for scene in ['Scene2', 'Scene3', 'Scene4']:
        ckpt = pattern.format(scene=scene, shot=shot)
        if not os.path.exists(ckpt):
            print(f"[skip] missing checkpoint: {ckpt}")
            continue
        print(f"[sensor_failure] XRF55 {scene}: {ckpt}")
        model = XRF55_DML(num_classes=55).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        raw[scene] = xrf55_evaluate_all_masks(model, eval_loaders[scene], device)
        del model; torch.cuda.empty_cache(); gc.collect()

    keep_labels = [KEEP_LABEL[k] for k in KEEP_COMBOS]
    agg = {}
    for label in keep_labels:
        vals = [scene_d[label] for scene_d in raw.values() if label in scene_d]
        agg[label] = sum(vals) / len(vals) if vals else float("nan")
    if raw:
        full = agg.get('all (no fail)', float("nan"))
        one_fail = [agg[k] for k in ['W+M (R fail)', 'W+R (M fail)', 'M+R (W fail)']
                    if agg[k] == agg[k]]
        one_avg = sum(one_fail) / len(one_fail) if one_fail else float("nan")
        print(f"XRF55: full={full:.2f}%  one-fail-avg={one_avg:.2f}%")

    csv_path = f"reproc/results/sensor_failure_xrf55_{shot}shot.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scene", "keep", "acc"])
        for scene, keep_d in raw.items():
            for keep_label, acc in keep_d.items():
                writer.writerow([scene, keep_label, f"{acc:.4f}"])
    save_result(f"sensor_failure_xrf55_{shot}shot", raw)
    return raw
