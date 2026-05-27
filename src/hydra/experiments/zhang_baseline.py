#!/usr/bin/env python
# coding: utf-8

# Reimplementation of:
#
# > Zhang, Qin, Mu, He, Wang, Chen. *Robust Cross-Domain RF-Based Multimodal Activity Recognition With Few-Shot Adaptation*. IEEE IoT-J vol.12 no.24, Dec 2025.
#
# **Backbones** (Sec. V-A / Table II-VII): RFID = DenseNet, WiFi = ResNet18, mmWave = ResNeXt.
# **Intermodal module** (Sec. V-B, Fig. 7): 1D conv refinement -> channel-switching -> concat -> 4x (Conv1D+BN+ReLU) -> AdaptiveAvgPool. Generates 3 pairwise streams (RFID-WiFi, RFID-mmWave, WiFi-mmWave).
# **Dynamic decision** (Sec. V-C, Eqs. 7-11): 6 streams (3 uni + 3 pair) -> softmax -> confidence-weighted fusion.
#
# **Pipeline per experimental setting** (In-Domain, Cross-Subject 21-9, Cross-Scene, CSC&S):
#
# 1. Look for the matching checkpoint under `reproc/`.
# 2. If present -> load and evaluate.
# 3. If missing -> run the training pipeline (Stage A unimodal warm-up + Stage B intermodal-only for the initial training; fused-CE fine-tuning for k-shot adaptation), save the checkpoint, then evaluate.
#
# The very first end-to-end run on a fresh clone will therefore train everything from scratch and save all checkpoints; subsequent runs just load and evaluate.
#
# **Reported numbers per setting**: WiFi / RFID / mmWave / Fusion top-1 accuracy, indexed from the 6 streams `[r, w, m, rw, rm, wm]` that `ZhangBaseline.forward()` returns. The three pairwise streams are still trained (they contribute to the fused probability) but are not reported individually - for direct comparison with the XRF55-DML baseline which only reports 3 unimodals + Fusion.
#
# **Data splits are identical** to the shared experiment protocol so numbers are directly comparable: In-Domain Scene1 (samples 1-14 / 15-20), Cross-Subject 21-9, Cross-Scene Scene2/3/4, CSC&S (22 clean source subjects -> S2/3/4).
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

# Dataset
class XRF55Dataset(Dataset):
    """scene_filter: {scene: {'samples': [int], 'vols': [int] or None}}"""
    def __init__(self, root_dir, scene_filter):
        self.root_dir = Path(root_dir)
        self.scene_filter = scene_filter
        self.sensors = ['WiFi', 'mmWave', 'RFID']
        self.samples = []
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

def make_loader(root_path, scene_filter, BS=16, NUM_WORKERS=4, shuffle=False, generator=None):
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

def make_concat_loader(root_path, scene_filter_list, BS=16, NUM_WORKERS=4,
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
# Zhang et al. 2025 - Intramodal backbones
# =============================================================================

INTER_C = 128
INTER_T = 16

# ---------- DenseNet (RFID) ----------
class _DenseLayer(nn.Module):
    def __init__(self, in_ch, growth, bn_size=4):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(in_ch)
        self.conv1 = nn.Conv1d(in_ch, bn_size*growth, 1, bias=False)
        self.norm2 = nn.BatchNorm1d(bn_size*growth)
        self.conv2 = nn.Conv1d(bn_size*growth, growth, 3, padding=1, bias=False)
    def forward(self, x):
        y = self.conv1(F.relu(self.norm1(x)))
        y = self.conv2(F.relu(self.norm2(y)))
        return torch.cat([x, y], dim=1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_ch, growth, bn_size=4):
        super().__init__()
        for i in range(num_layers):
            self.add_module(f"l{i}", _DenseLayer(in_ch + i*growth, growth, bn_size))

class _Transition(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_ch)
        self.conv = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.pool = nn.AvgPool1d(2, 2)
    def forward(self, x):
        return self.pool(self.conv(F.relu(self.norm(x))))

class RFID_DenseNet(nn.Module):
    def __init__(self, num_classes=55, growth=32, block_cfg=(6, 12, 24, 16)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(23, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, 2, 1),
        )
        ch = 64
        layers = []
        for i, n in enumerate(block_cfg):
            layers.append(_DenseBlock(n, ch, growth))
            ch += n * growth
            if i != len(block_cfg)-1:
                layers.append(_Transition(ch, ch//2))
                ch //= 2
        self.blocks = nn.Sequential(*layers)
        self.norm = nn.BatchNorm1d(ch)
        self.out_dim = ch

        self.bridge = nn.Sequential(
            nn.Conv1d(ch, INTER_C, 1), nn.BatchNorm1d(INTER_C), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(INTER_T),
        )
        self.head = nn.Linear(ch, num_classes)

    def forward(self, rfid_x):
        x = rfid_x.squeeze(1)
        x = self.stem(x)
        x = F.relu(self.norm(self.blocks(x)))
        feat_seq = self.bridge(x)
        feat_vec = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        logits = self.head(feat_vec)
        return logits, feat_seq, feat_vec


# ---------- ResNet18 (WiFi) ----------
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
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)

class WiFi_ResNet18(nn.Module):
    def __init__(self, num_classes=55, base=48):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, base, 7, (2, 4), 3, bias=False),
            nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        c1, c2, c3, c4 = base, base*2, base*4, base*8
        self.layer1 = self._make_layer(c1, c1, 2, stride=1)
        self.layer2 = self._make_layer(c1, c2, 2, stride=2)
        self.layer3 = self._make_layer(c2, c3, 2, stride=2)
        self.layer4 = self._make_layer(c3, c4, 2, stride=2)
        self.out_dim = c4

        self.bridge = nn.Sequential(
            nn.Conv2d(c4, INTER_C, 1), nn.BatchNorm2d(INTER_C), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, INTER_T)),
        )
        self.head = nn.Linear(c4, num_classes)

    def _make_layer(self, in_ch, ch, n, stride):
        downsample = None
        if stride != 1 or in_ch != ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, ch, 1, stride, bias=False),
                nn.BatchNorm2d(ch)
            )
        layers = [_BasicBlock2d(in_ch, ch, stride, downsample)]
        for _ in range(1, n):
            layers.append(_BasicBlock2d(ch, ch))
        return nn.Sequential(*layers)

    def forward(self, wifi_x):
        x = self.stem(wifi_x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        feat_seq = self.bridge(x).squeeze(2)
        feat_vec = F.adaptive_avg_pool2d(x, 1).flatten(1)
        logits = self.head(feat_vec)
        return logits, feat_seq, feat_vec


# ---------- ResNeXt (mmWave) ----------
class _ResNeXtBlock(nn.Module):
    expansion = 2
    def __init__(self, in_ch, ch, stride=1, groups=32, base_width=4, downsample=None):
        super().__init__()
        D = int(ch * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(in_ch, D, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(D)
        self.conv2 = nn.Conv2d(D, D, 3, stride, 1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(D)
        self.conv3 = nn.Conv2d(D, ch * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ch * self.expansion)
        self.downsample = downsample
    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        return F.relu(out + identity, inplace=True)

class MmWave_ResNeXt(nn.Module):
    def __init__(self, num_classes=55, groups=32, base_width=4,
                 base=56, block_cfg=(2, 2, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(17, base, 7, 2, 3, bias=False),
            nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.in_ch = base
        c1, c2, c3, c4 = base, base*2, base*4, base*8
        self.layer1 = self._make_layer(c1, block_cfg[0], 1, groups, base_width)
        self.layer2 = self._make_layer(c2, block_cfg[1], 2, groups, base_width)
        self.layer3 = self._make_layer(c3, block_cfg[2], 2, groups, base_width)
        self.layer4 = self._make_layer(c4, block_cfg[3], 2, groups, base_width)
        self.out_dim = c4 * _ResNeXtBlock.expansion

        self.bridge = nn.Sequential(
            nn.Conv2d(self.out_dim, INTER_C, 1),
            nn.BatchNorm2d(INTER_C), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, INTER_T)),
        )
        self.head = nn.Linear(self.out_dim, num_classes)

    def _make_layer(self, ch, n, stride, groups, base_width):
        downsample = None
        out_ch = ch * _ResNeXtBlock.expansion
        if stride != 1 or self.in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        layers = [_ResNeXtBlock(self.in_ch, ch, stride, groups, base_width, downsample)]
        self.in_ch = out_ch
        for _ in range(1, n):
            layers.append(_ResNeXtBlock(self.in_ch, ch, 1, groups, base_width))
        return nn.Sequential(*layers)

    def forward(self, mmw_x):
        x = self.stem(mmw_x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        feat_seq = self.bridge(x).squeeze(2)
        feat_vec = F.adaptive_avg_pool2d(x, 1).flatten(1)
        logits = self.head(feat_vec)
        return logits, feat_seq, feat_vec


# =============================================================================
# Intermodal block (Sec. V-B, Fig. 7)
# =============================================================================

class IntermodalBlock(nn.Module):
    def __init__(self, c=INTER_C, t=INTER_T, num_classes=55, exchange_ratio=0.5,
                 inter_blocks=4):
        super().__init__()
        self.c = c; self.t = t
        self.k = int(c * exchange_ratio)

        self.f1 = nn.Sequential(
            nn.Conv1d(c, c, 3, padding=1, bias=False), nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
            nn.Conv1d(c, c, 3, padding=1, bias=False), nn.BatchNorm1d(c))
        self.f2 = nn.Sequential(
            nn.Conv1d(c, c, 3, padding=1, bias=False), nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
            nn.Conv1d(c, c, 3, padding=1, bias=False), nn.BatchNorm1d(c))

        blocks = []
        in_ch = 2 * c
        for _ in range(inter_blocks):
            blocks.append(nn.Sequential(
                nn.Conv1d(in_ch, c, 3, padding=1, bias=False),
                nn.BatchNorm1d(c), nn.ReLU(inplace=True)))
            in_ch = c
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(c, num_classes)
        self.out_dim = c

    def forward(self, x1, x2):
        x1p = x1 + self.f1(x1)
        x2p = x2 + self.f2(x2)
        k = self.k
        x1_sw = torch.cat([x2p[:, :k, :], x1p[:, k:, :]], dim=1)
        x2_sw = torch.cat([x1p[:, :k, :], x2p[:, k:, :]], dim=1)
        h = torch.cat([x1_sw, x2_sw], dim=1)
        h = self.blocks(h)
        v = self.pool(h).squeeze(-1)
        logits = self.head(v)
        return logits, v


# =============================================================================
# Full Zhang fusion model  (no GT leakage; top-1 confidence weighting)
# =============================================================================

class ZhangBaseline(nn.Module):
    def __init__(self, num_classes=55):
        super().__init__()
        self.rfid_enc = RFID_DenseNet(num_classes)
        self.wifi_enc = WiFi_ResNet18(num_classes)
        self.mmw_enc  = MmWave_ResNeXt(num_classes)

        self.inter_rw = IntermodalBlock(c=INTER_C, t=INTER_T, num_classes=num_classes)
        self.inter_rm = IntermodalBlock(c=INTER_C, t=INTER_T, num_classes=num_classes)
        self.inter_wm = IntermodalBlock(c=INTER_C, t=INTER_T, num_classes=num_classes)

    def forward(self, wifi_x, mmw_x, rfid_x, targets=None, apply_drop=False):
        r_logits, r_seq, _ = self.rfid_enc(rfid_x)
        w_logits, w_seq, _ = self.wifi_enc(wifi_x)
        m_logits, m_seq, _ = self.mmw_enc(mmw_x)

        rw_logits, _ = self.inter_rw(r_seq, w_seq)
        rm_logits, _ = self.inter_rm(r_seq, m_seq)
        wm_logits, _ = self.inter_wm(w_seq, m_seq)

        all_logits = torch.stack(
            [r_logits, w_logits, m_logits, rw_logits, rm_logits, wm_logits], dim=1
        )  # (B, 6, K)

        logits_f32 = all_logits.float()
        probs = F.softmax(logits_f32, dim=-1)                # (B, 6, K)

        # Top-1 confidence as fusion weight (no GT).
        conf = probs.max(-1).values                          # (B, 6)
        weights = conf / (conf.sum(-1, keepdim=True) + 1e-8) # (B, 6)
        fused_prob = (weights.unsqueeze(-1) * probs).sum(1)  # (B, K)

        return {
            'logits': all_logits,
            'fused_prob': fused_prob,
            'stream_logits': all_logits,
            'stream_probs': probs,
            'stream_weights': weights,
        }


MODEL_REGISTRY = {
    'Zhang': ZhangBaseline,
}


# =============================================================================
# Train / Eval
#
# Paper-faithful pipeline:
#   Stage A (per-modality independent training, Sec. VI-D):
#       RFID  DenseNet : 100 ep, lr 1e-4, step 30/0.5
#       WiFi  ResNet18 :  50 ep, lr 1e-3, step 10/0.5
#       mmW   ResNeXt  : 150 ep, lr 1e-4, step 35/0.5
#   Stage B (intermodal-only training, Sec. VI-D):
#       Freeze all 3 unimodal encoders, train ONLY the 3 IntermodalBlocks
#       (which produce the RW/Rm/Wm streams) for 50 epochs at lr 1e-4.
#       Loss is the paper's Eq. (11): CE on the fused probability.
#
# =============================================================================

LABEL_SMOOTHING = 0.1


@torch.no_grad()
def _quick_modality_acc(model, modality, loader, device):
    """Accuracy of a single unimodal encoder's head on `loader`."""
    if len(loader) == 0: return 0.0
    encoder = getattr(model, f"{modality}_enc")
    encoder.eval()
    correct = 0; total = 0
    for data, labels in loader:
        tgt = (labels - 1).to(device, non_blocking=True)
        if modality == 'rfid':
            x = data['RFID'].to(device, non_blocking=True)
        elif modality == 'wifi':
            x = data['WiFi'].to(device, non_blocking=True)
        else:
            x = data['mmWave'].to(device, non_blocking=True)
        logits, _, _ = encoder(x)
        pred = logits.argmax(1)
        correct += pred.eq(tgt).sum().item()
        total += tgt.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_model(model, loader, device):
    if len(loader) == 0: return 0.0
    model.eval()
    c, t = 0, 0
    for data, labels in loader:
        wifi = data['WiFi'].to(device, non_blocking=True)
        mmw  = data['mmWave'].to(device, non_blocking=True)
        rfid = data['RFID'].to(device, non_blocking=True)
        tgt = (labels - 1).to(device, non_blocking=True)
        out = model(wifi, mmw, rfid, targets=None)
        pred = out['fused_prob'].argmax(1)
        c += pred.eq(tgt).sum().item()
        t += tgt.size(0)
    return 100.0 * c / t if t > 0 else 0.0


def _stage_a_train_one_modality(model, modality, train_loader, eval_loader,
                                device, epochs, lr, wd=0.05):
    """Train a single unimodal encoder with plain CE.  All other params untouched.

    NOTE: wd argument is intentionally IGNORED here.  Stage A uses wd=0 and
    a looser clip_norm because Adam + wd=0.05 + clip=1.0 made the RFID
    DenseNet learn extremely slowly (loss stuck near ln(55)=4.0073).
    """
    encoder = getattr(model, f"{modality}_enc")
    # Override wd to 0 for Stage A; paper does not specify a non-trivial wd
    # for the per-modality stage and 0.05 was too aggressive at lr=1e-4.
    optimizer = optim.Adam(encoder.parameters(), lr=lr, weight_decay=0.0)
    if modality == 'rfid':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    elif modality == 'wifi':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=35, gamma=0.5)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    for ep in range(epochs):
        encoder.train()
        sum_loss = 0.0
        for i_batch, (data, labels) in enumerate(train_loader):
            tgt = (labels - 1).to(device, non_blocking=True)
            if modality == 'rfid':
                x = data['RFID'].to(device, non_blocking=True)
            elif modality == 'wifi':
                x = data['WiFi'].to(device, non_blocking=True)
            else:
                x = data['mmWave'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits, _, _ = encoder(x)
            loss = criterion(logits, tgt)
            loss.backward()
            # Looser clip (5.0 instead of 1.0) so early-training grads are
            # not over-suppressed.
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)
            optimizer.step()
            sum_loss += loss.item()
        scheduler.step()

        if (ep + 1) % 10 == 0 or ep == 0 or ep == epochs - 1:
            print(f"      [stageA-{modality} {ep+1:03d}/{epochs}] "
                  f"loss={sum_loss/len(train_loader):.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

def _stage_b_intermodal_only(model, train_loader, eval_loader, device,
                             epochs=50, lr=1e-4, wd=0.05):
    """Freeze all 3 unimodal encoders; train ONLY the 3 IntermodalBlocks.

    Loss = fused CE + sum of CE on the 3 intermodal stream logits.

    Why both: the paper's Eq. (11) is fused-only, but when unimodal encoders
    are already well-trained, fused_prob is dominated by them and the
    gradient through intermodal streams vanishes (loss = 0.10 from epoch 1,
    no improvement).  Adding per-intermodal-stream CE gives each intermodal
    head a direct learning signal.  We only apply CE to the 3 intermodal
    streams (indices 3,4,5); the 3 unimodal streams (0,1,2) are skipped
    because the unimodal encoders are frozen.
    """
    # Freeze encoders
    for enc in (model.rfid_enc, model.wifi_enc, model.mmw_enc):
        for p in enc.parameters(): p.requires_grad = False
        enc.eval()

    # Trainable: just the 3 intermodal blocks
    trainable = list(model.inter_rw.parameters()) \
              + list(model.inter_rm.parameters()) \
              + list(model.inter_wm.parameters())
    optimizer = optim.Adam(trainable, lr=lr, weight_decay=0.0)
    ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    for ep in range(epochs):
        model.rfid_enc.eval(); model.wifi_enc.eval(); model.mmw_enc.eval()
        model.inter_rw.train(); model.inter_rm.train(); model.inter_wm.train()

        sum_loss = 0.0
        sum_inter = 0.0
        sum_fused = 0.0
        for data, labels in train_loader:
            wifi = data['WiFi'].to(device, non_blocking=True)
            mmw  = data['mmWave'].to(device, non_blocking=True)
            rfid = data['RFID'].to(device, non_blocking=True)
            tgt  = (labels - 1).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(wifi, mmw, rfid, targets=None)

            # Per-intermodal-stream CE (indices 3=RW, 4=Rm, 5=Wm).
            # This gives each intermodal head a direct learning signal.
            all_logits = out['logits']               # (B, 6, K)
            inter_loss = (ce(all_logits[:, 3], tgt) +
                          ce(all_logits[:, 4], tgt) +
                          ce(all_logits[:, 5], tgt)) / 3.0

            # Paper Eq. (11): CE on fused probability.
            fused_logp = torch.log(out['fused_prob'].clamp_min(1e-8))
            fused_loss = F.nll_loss(fused_logp, tgt)

            loss = inter_loss + fused_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
            optimizer.step()
            sum_loss  += loss.item()
            sum_inter += inter_loss.item()
            sum_fused += fused_loss.item()

        if (ep + 1) % 10 == 0 or ep == 0 or ep == epochs - 1:
            n = len(train_loader)
            print(f"   [stageB Ep {ep+1:03d}/{epochs}] "
                  f"loss={sum_loss/n:.4f}  "
                  f"inter={sum_inter/n:.4f}  "
                  f"fused={sum_fused/n:.4f}")
        torch.cuda.empty_cache(); gc.collect()

    # Re-enable grads so downstream code (fine-tune) can update everything.
    for enc in (model.rfid_enc, model.wifi_enc, model.mmw_enc):
        for p in enc.parameters(): p.requires_grad = True

def train_and_eval(exp_name, model_cls, train_loader, eval_loaders, best_metric_fn,
                   device, epochs=100, lr_max=5e-4, lr_init=1e-4,
                   weight_decay=0.05, label_smoothing=0.1, log_every=10):
    """Paper-faithful pipeline:
         Stage A : per-modality independent training (paper hyperparams)
         Stage B : encoder freeze + intermodal-only training, fused-CE only
    `epochs` and `lr_max` are accepted for interface compatibility but the
    paper-specified hyperparameters override them."""
    set_reproducible_mode(SEED)
    model = model_cls(num_classes=55).to(device)
    save_path = os.path.join("reproc", f"{exp_name}_last.pth")

    # Use first eval loader for in-loop sanity checks
    eval_loader_for_probe = next(iter(eval_loaders.values())) if eval_loaders else None

    # ---------- Stage A ----------
    print(f"   -- Stage A: per-modality independent training (paper Sec. VI-D)")
    stage_a_cfg = {
        'rfid': dict(epochs=100, lr=1e-4),
        'wifi': dict(epochs=50,  lr=1e-3),
        'mmw':  dict(epochs=150, lr=1e-4),
    }
    for mod_key, cfg in stage_a_cfg.items():
        print(f"     - {mod_key} ({cfg['epochs']} ep, lr={cfg['lr']})")
        _stage_a_train_one_modality(
            model, mod_key, train_loader, eval_loader_for_probe, device,
            epochs=cfg['epochs'], lr=cfg['lr'], wd=weight_decay)
        torch.cuda.empty_cache(); gc.collect()

    # ---------- Sanity check between stages ----------
    if eval_loader_for_probe is not None:
        print(f"   -- Post-Stage-A unimodal sanity check:")
        for mod_key in ['rfid', 'wifi', 'mmw']:
            acc = _quick_modality_acc(model, mod_key, eval_loader_for_probe, device)
            print(f"        {mod_key:<5}: {acc:.2f}%")

    # ---------- Stage B ----------
    print(f"   -- Stage B: intermodal-only training (encoders frozen, 50 ep, lr 1e-4)")
    _stage_b_intermodal_only(
        model, train_loader, eval_loader_for_probe, device,
        epochs=50, lr=1e-4, wd=weight_decay)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    model.eval()
    final = {n: evaluate_model(model, ld, device) for n, ld in eval_loaders.items()}
    acc_str = " | ".join([f"{k}:{v:.2f}%" for k, v in final.items()])
    print(f"    [Final] {acc_str}")
    del model; torch.cuda.empty_cache(); gc.collect()
    return final


def _finetune_eval(model_cls, pretrain_ckpt, ft_loader, eval_loader, device,
                   epochs=100, lr=1e-4, weight_decay=0.05,
                   label_smoothing=0.1, log_every=10,
                   ft_seed=None, save_path=None):
    """Few-shot fine-tune.  Same recipe as Stage B: fused-CE only, all params
       trainable so the encoders can also adapt to the new domain."""
    set_reproducible_mode(ft_seed if ft_seed is not None else SEED)
    model = model_cls(num_classes=55).to(device)
    print(f"       Loading pretrain ckpt: {pretrain_ckpt}")
    model.load_state_dict(torch.load(pretrain_ckpt, map_location=device))

    # Make sure every param is trainable for fine-tune
    for p in model.parameters(): p.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(epochs):
        model.train()
        sum_loss = 0.0
        for data, labels in ft_loader:
            wifi = data['WiFi'].to(device, non_blocking=True)
            mmw  = data['mmWave'].to(device, non_blocking=True)
            rfid = data['RFID'].to(device, non_blocking=True)
            tgt  = (labels - 1).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(wifi, mmw, rfid, targets=None)
            fused_logp = torch.log(out['fused_prob'].clamp_min(1e-8))
            loss = F.nll_loss(fused_logp, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
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


# Model cost
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
    model.eval()
    flops = {'total': 0}
    hooks = []
    def conv1d_hook(m, inp, out):
        x = inp[0]; B = x.shape[0]; out_len = out.shape[-1]
        kernel_ops = m.kernel_size[0] * (m.in_channels // m.groups)
        bias_ops = 1 if m.bias is not None else 0
        flops['total'] += int(B * m.out_channels * out_len * (2 * kernel_ops + bias_ops))
    def conv2d_hook(m, inp, out):
        x = inp[0]; B = x.shape[0]
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
        if isinstance(mod, nn.Conv1d): hooks.append(mod.register_forward_hook(conv1d_hook))
        elif isinstance(mod, nn.Conv2d): hooks.append(mod.register_forward_hook(conv2d_hook))
        elif isinstance(mod, nn.Linear): hooks.append(mod.register_forward_hook(linear_hook))
    with torch.no_grad():
        _ = model(*inputs)
    for h in hooks: h.remove()
    return flops['total']

def measure_model_cost(device=DEVICE, use_cache=True, save=True, verbose=True):
    cache_path = f"reproc/results/model_cost_zhang_{HOSTNAME}.json"
    if use_cache and os.path.exists(cache_path):
        with open(cache_path) as f: data = json.load(f)
        if verbose:
            print(f"\nModel Cost (cached Zhang baseline, host={HOSTNAME})")
            print("-" * 64)
            print(f"{'Model':<10} | {'Params':>10} | {'Memory':>10} | {'FLOPs':>12}")
            print("-" * 64)
            for m, v in data.items():
                fl = f"{v['flops_G']:.4f}G" if v.get('flops_G') is not None else 'N/A'
                print(f"{m:<10} | {v['params_M']:>8.2f}M | {v['memory_MB']:>8.2f}MB | {fl:>12}")
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
        print(f"\nMeasuring Zhang baseline cost (host={HOSTNAME})...")
        print("-" * 64)
        print(f"{'Model':<10} | {'Params':>10} | {'Memory':>10} | {'FLOPs':>12}")
        print("-" * 64)

    summary = {}
    sub_specs = [
        ('WiFi',   WiFi_ResNet18,   wifi),
        ('RFID',   RFID_DenseNet,   rfid),
        ('mmWave', MmWave_ResNeXt,  mmw),
    ]
    for sname, scls, sinp in sub_specs:
        sm = scls(num_classes=55).to(device).eval()
        params = count_params(sm); mem = occupy_mb(sm)
        with torch.no_grad():
            sflops = _estimate_flops_custom(sm, inputs=(sinp,)) / 1e9
        if verbose:
            print(f"{sname:<10} | {params/1e6:>8.2f}M | {mem:>8.2f}MB | {sflops:>11.4f}G")
        summary[sname] = {'params_M': params/1e6, 'memory_MB': mem, 'flops_G': sflops,
                          'flops_note': 'custom hooks; 1 MAC = 2 FLOPs; intramodal backbone only',
                          'hostname': HOSTNAME}
        del sm; torch.cuda.empty_cache(); gc.collect()

    for mname, mcls in MODEL_REGISTRY.items():
        model = mcls(num_classes=55).to(device).eval()
        params = count_params(model); mem = occupy_mb(model)
        with torch.no_grad():
            flops = _estimate_flops_custom(model, inputs=(wifi, mmw, rfid, None, False)) / 1e9
        label = 'Fusion' if mname == 'Zhang' else mname
        if verbose:
            print(f"{label:<10} | {params/1e6:>8.2f}M | {mem:>8.2f}MB | {flops:>11.4f}G")
        summary[label] = {'params_M': params/1e6, 'memory_MB': mem, 'flops_G': flops,
                          'flops_note': 'custom hooks; 1 MAC = 2 FLOPs; full fusion (intra + inter + dynamic)',
                          'hostname': HOSTNAME}
        del model; torch.cuda.empty_cache(); gc.collect()

    if save:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f: json.dump(summary, f, indent=2)
        if verbose: print(f"saved: {cache_path}")
    return summary

if os.environ.get("HYDRA_VERBOSE_IMPORT") == "1":
    print("Zhang baseline ready (Zhang et al. 2025 baseline - paper-faithful).")
    print(f"   ROOT={ROOT}")
    print(f"   DEVICE={DEVICE}")
    print(f"   HOST={HOSTNAME}")

# Same split as the shared experiment protocol's In-Domain setting. Single model with the Zhang fusion structure.

# ============================================================
# IN-DOMAIN Scene1 - Zhang baseline
#   Train: Scene1 samples 1~14, all vols
#   Test : Scene1 samples 15~20, all vols
#
# Reports per-modality (WiFi / RFID / mmWave) + Fusion top-1 accuracy.
# Index map in ZhangBaseline.forward() -> all_logits is stacked as:
#     [r=0, w=1, m=2, rw=3, rm=4, wm=5]
# Pairwise streams (rw/rm/wm) still contribute to fused_prob but are not
# reported separately here, to keep the table comparable to the XRF55-DML
# baseline which only reports 3 unimodals + Fusion.
#
# This section defines the shared helpers used by the Cross-Subject,
# Cross-Scene, and CSC&S sections below:
#     evaluate_wmr_fusion   - per-modality + Fusion top-1 accs in one pass
#     _load_and_eval_wmr    - load ckpt -> evaluate
#     _ft_then_eval_wmr     - fine-tune from pretrain, save ckpt, evaluate
#     _print_per_modality_table  - pretty-print like the XRF55-DML baseline
# ============================================================

@torch.no_grad()
def evaluate_wmr_fusion(model, loader, device):
    """Returns dict with WiFi / mmWave / RFID / Fusion top-1 accuracy.

    Zhang forward stacks logits as [r, w, m, rw, rm, wm] along dim=1.
    We pick the three unimodal indices (0,1,2) and the model's fused_prob.
    """
    if len(loader) == 0:
        return {'WiFi': 0.0, 'mmWave': 0.0, 'RFID': 0.0, 'Fusion': 0.0}
    model.eval()
    c_w = c_m = c_r = c_f = 0
    t = 0
    for data, labels in loader:
        wifi = data['WiFi'].to(device, non_blocking=True)
        mmw  = data['mmWave'].to(device, non_blocking=True)
        rfid = data['RFID'].to(device, non_blocking=True)
        tgt  = (labels - 1).to(device, non_blocking=True)
        out = model(wifi, mmw, rfid, targets=None)
        all_logits = out['logits']                              # (B, 6, K)
        c_r += all_logits[:, 0].argmax(1).eq(tgt).sum().item()  # RFID
        c_w += all_logits[:, 1].argmax(1).eq(tgt).sum().item()  # WiFi
        c_m += all_logits[:, 2].argmax(1).eq(tgt).sum().item()  # mmWave
        c_f += out['fused_prob'].argmax(1).eq(tgt).sum().item() # Fusion
        t   += tgt.size(0)
    if t == 0:
        return {'WiFi': 0.0, 'mmWave': 0.0, 'RFID': 0.0, 'Fusion': 0.0}
    return {
        'WiFi':   100.0 * c_w / t,
        'mmWave': 100.0 * c_m / t,
        'RFID':   100.0 * c_r / t,
        'Fusion': 100.0 * c_f / t,
    }


def _load_and_eval_wmr(ckpt_path, eval_loader, device):
    """Load a ZhangBaseline ckpt and return {WiFi, mmWave, RFID, Fusion}."""
    print(f"      Loading ckpt: {ckpt_path}")
    model = ZhangBaseline(num_classes=55).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    res = evaluate_wmr_fusion(model, eval_loader, device)
    del model; torch.cuda.empty_cache(); gc.collect()
    return res


def _ft_then_eval_wmr(pretrain_ckpt, ft_loader, eval_loader, device,
                      epochs, lr, save_path):
    """Fine-tune with _finetune_eval, then
    reload the saved FT ckpt and report per-modality + Fusion accs."""
    _ = _finetune_eval(ZhangBaseline, pretrain_ckpt,
                       ft_loader, eval_loader, device,
                       epochs=epochs, lr=lr, save_path=save_path)
    return _load_and_eval_wmr(save_path, eval_loader, device)


def _print_per_modality_table(title, results, shots, eval_keys):
    """Pretty-print 4 per-modality tables in the same format as the XRF55-DML
    baseline. `results[k][eval_key]` must be a dict with keys
    'WiFi', 'mmWave', 'RFID', 'Fusion'.
    """
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)
    shot_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five']
    for modality in ['WiFi', 'RFID', 'mmWave', 'Fusion']:
        print(f"\n[Modality: {modality}]")
        header = f"{'#-Shot':<6} | " + " | ".join(eval_keys)
        print(header)
        print("-" * max(len(header), 47))
        for k in shots:
            kname = shot_names[k] if k < len(shot_names) else str(k)
            row_vals = []
            for ek in eval_keys:
                v = results.get(k, {}).get(ek, {}).get(modality, float('nan'))
                row_vals.append(f"{v:.2f}%" if v == v else "N/A")
            print(f"{kname:<6} | " + " | ".join(row_vals))


# ============================================================
# In-Domain Scene1 runner
# ============================================================
def run_in_domain_scene1_zhang(root_path=ROOT, device=DEVICE,
                                    epochs=100, BS=16, NUM_WORKERS=4,
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

    print("\n" + "=" * 80)
    print("[Zhang | In-Domain Scene1]  Train: 1~14 | Test: 15~20")
    print("=" * 80)

    exp_name  = "Zhang_InDomain_Scene1"
    ckpt_path = os.path.join("reproc", f"{exp_name}_last.pth")

    if (not force_retrain) and os.path.exists(ckpt_path):
        print(f"  Found ckpt: {ckpt_path}  -> load & evaluate")
    else:
        print(f"   No ckpt -> run 2-stage training (Stage A unimodal + Stage B intermodal)")
        print(f"            and save to {ckpt_path}")
        _ = train_and_eval(exp_name, ZhangBaseline,
                           train_loader, {'Scene1': test_loader},
                           lambda r: r['Scene1'],
                           device, epochs=epochs)

    res = _load_and_eval_wmr(ckpt_path, test_loader, device)

    # Single-column print (no shot dimension for In-Domain).
    print("\n" + "=" * 80)
    print("Zhang | In-Domain Scene1 - per-modality + Fusion")
    print("=" * 80)
    print(f"  {'Modality':<10} | {'Scene1':>8}")
    print("  " + "-" * 22)
    for m in ['WiFi', 'RFID', 'mmWave', 'Fusion']:
        print(f"  {m:<10} | {res[m]:>7.2f}%")

    save_result("zhang_in_domain_scene1", {
        'Scene1': {m: float(v) for m, v in res.items()},
    })
    return res


# Run with scripts/run_experiment.py.


# Same source/target vol split (overlap subjects {3,4,5,6,7,13,23,24} in source). 0/1/2-shot adaptation on the 9 target subjects.

# ============================================================
# CROSS-SUBJECT 21-9 - Zhang baseline
#   Same vol split as the shared experiment protocol (overlap subjects in source).
#
# Behavior per (k-shot):
#   k = 0 : load pretrain ckpt -> zero-shot eval on Target9
#   k > 0 : if FT ckpt exists  -> load and evaluate
#           else                -> fine-tune from pretrain, save FT ckpt, evaluate
# Pretrain stage trains the full 2-stage pipeline only if its ckpt is missing.
#
# Reports WiFi / RFID / mmWave / Fusion accs per k-shot, single eval key 'Target9'.
# ============================================================

OVERLAP_S2 = [5, 24]                # (+ vol 31 is Scene2-only)
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


def run_cross_subject_21_9_zhang(root_path=ROOT, device=DEVICE,
                                      pretrain_epochs=100,
                                      ft_epochs=100, ft_lr=1e-5,
                                      BS=16, NUM_WORKERS=4,
                                      shots=(0, 1, 2, 3, 4, 5),
                                      force_pretrain=False):
    all_vols = get_volunteers_in_scene(root_path, 'Scene1')
    SOURCE_VOLS, TARGET_VOLS = _split_source_target(root_path, 21, 9)

    print(f"Scene1 volunteers: {len(all_vols)} -> {all_vols}")
    print(f"   Overlap subjs : {OVERLAP_VOLS}")
    print(f"   Source(21)    : {SOURCE_VOLS}")
    print(f"   Target(9)     : {TARGET_VOLS}")

    pretrain_exp  = "Zhang_SCSub_21_9_pretrain"
    pretrain_ckpt = os.path.join("reproc", f"{pretrain_exp}_last.pth")

    print(f"\n{'='*80}\n [Zhang] Cross-Subject 21-9 (per-modality + Fusion)\n{'='*80}")

    # ========== Stage 1: Pretrain ==========
    if os.path.exists(pretrain_ckpt) and not force_pretrain:
        print(f"  [Stage 1: Pretrain] [OK] ckpt found -> {pretrain_ckpt}")
    else:
        print(f"  [Stage 1: Pretrain] 21 x 55 x 20 = {21*55*20:,} samples, ep={pretrain_epochs}")
        g = fresh_generator()
        pretrain_loader = make_loader(root_path,
            {'Scene1': {'samples': list(range(1, 21)), 'vols': SOURCE_VOLS}},
            BS, NUM_WORKERS, shuffle=True, generator=g)
        sanity_loader = make_loader(root_path,
            {'Scene1': {'samples': [1], 'vols': SOURCE_VOLS}},
            BS, NUM_WORKERS, generator=g)
        _ = train_and_eval(pretrain_exp, ZhangBaseline,
                           pretrain_loader, {'sanity': sanity_loader},
                           lambda r: r['sanity'], device,
                           epochs=pretrain_epochs)
        print(f"    [OK] saved -> {pretrain_ckpt}")

    # ========== Stage 2: K-shot adapt + per-modality eval ==========
    results = {k: {} for k in shots}
    for k in shots:
        print(f"\n  [Stage 2: {k}-shot] target = 9 vols")
        g = fresh_generator()

        if k == 0:
            eval_samples = list(range(1, 21))
            eval_loader = make_loader(root_path,
                {'Scene1': {'samples': eval_samples, 'vols': TARGET_VOLS}},
                BS, NUM_WORKERS, generator=g)
            print(f"    zero-shot (no FT)  | eval per-class = {len(eval_samples)}")
            res = _load_and_eval_wmr(pretrain_ckpt, eval_loader, device)
        else:
            eval_samples = list(range(k + 1, 21))
            eval_loader = make_loader(root_path,
                {'Scene1': {'samples': eval_samples, 'vols': TARGET_VOLS}},
                BS, NUM_WORKERS, generator=g)
            ft_save_path = os.path.join("reproc", f"Zhang_SCSub_21_9_{k}shot_ft.pth")

            if os.path.exists(ft_save_path):
                print(f"    Found FT ckpt: {ft_save_path}  -> load & evaluate")
                res = _load_and_eval_wmr(ft_save_path, eval_loader, device)
            else:
                print(f"     No FT ckpt -> fine-tune 9 x 55 x {k} = {9*55*k} samples"
                      f"  (ep={ft_epochs}, lr={ft_lr})")
                ft_loader = make_loader(root_path,
                    {'Scene1': {'samples': list(range(1, k + 1)), 'vols': TARGET_VOLS}},
                    BS, NUM_WORKERS, shuffle=True, generator=g)
                res = _ft_then_eval_wmr(pretrain_ckpt, ft_loader, eval_loader, device,
                                        epochs=ft_epochs, lr=ft_lr,
                                        save_path=ft_save_path)

        results[k]['Target9'] = res
        line = "  ".join(f"{m}={res[m]:6.2f}%"
                         for m in ['WiFi', 'RFID', 'mmWave', 'Fusion'])
        print(f"    [OK] [Zhang {k}-shot] {line}")

    _print_per_modality_table(
        "Zhang | Cross-Subject 21-9 (per-modality + Fusion)",
        results, shots, eval_keys=['Target9'],
    )
    save_result("zhang_cross_subject_21_9", {
        "overlap_vols": OVERLAP_VOLS,
        "source_vols":  SOURCE_VOLS,
        "target_vols":  TARGET_VOLS,
        "pretrain_epochs": pretrain_epochs,
        "ft_epochs": ft_epochs, "ft_lr": ft_lr,
        "results": {
            f"{k}shot": {
                'Target9': {m: float(v) for m, v in results[k]['Target9'].items()},
            } for k in shots
        },
    })
    return results


# Reuses the Cross-Subject pretrain ckpt (`Zhang_SCSub_21_9_pretrain_last.pth`) and adapts to each target scene with the overlap-subject vols. Same split as the shared experiment protocol's strict-mode Cross-Scene setting.

# ============================================================
# CROSS-SCENE - Zhang baseline
#   Reuses the Cross-Subject pretrain ckpt (Zhang_SCSub_21_9_pretrain_last.pth)
#   under strict subject consistency.
#
# Behavior per (scene, k-shot):
#   k = 0 : load pretrain ckpt -> zero-shot eval on target scene
#   k > 0 : if FT ckpt exists  -> load and evaluate
#           else                -> fine-tune from pretrain, save FT ckpt, evaluate
# ============================================================

def run_cross_scene_zhang(root_path=ROOT, device=DEVICE,
                               ft_epochs=100, ft_lr=1e-5,
                               BS=16, NUM_WORKERS=4,
                               shots=(0, 1, 2, 3, 4, 5),
                               strict_subject_consistency=True):
    target_scenes  = ['Scene2', 'Scene3', 'Scene4']
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
        print("INFO:  lenient mode: include unseen-during-pretrain users")

    print("\n Final target scene vols:")
    for sc, vs in scene_vols.items():
        print(f"   {sc}: {vs}")

    pretrain_ckpt = os.path.join("reproc", "Zhang_SCSub_21_9_pretrain_last.pth")
    print(f"\n{'='*80}\n [Zhang] Cross-Scene (per-modality + Fusion)\n{'='*80}")

    if not os.path.exists(pretrain_ckpt):
        raise FileNotFoundError(
            f"\n  Pretrain ckpt not found: {pretrain_ckpt}"
            f"\n  -> Run run_cross_subject_21_9_zhang_eval() first to create it."
        )
    print(f"  Reusing pretrain -> {pretrain_ckpt}")

    results = {k: {sc: None for sc in target_scenes} for k in shots}

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
                print(f"    [{k}-shot] zero-shot  | eval per-class = {len(eval_samples)}")
                res = _load_and_eval_wmr(pretrain_ckpt, eval_loader, device)
            else:
                eval_samples = list(range(k + 1, 21))
                eval_loader = make_loader(root_path,
                    {target_scene: {'samples': eval_samples, 'vols': target_vols}},
                    BS, NUM_WORKERS, generator=g)
                ft_save_path = os.path.join(
                    "reproc", f"Zhang_CrossScene_{target_scene}_{k}shot_ft.pth")

                if os.path.exists(ft_save_path):
                    print(f"    [{k}-shot] Found FT ckpt: {ft_save_path}  -> load & evaluate")
                    res = _load_and_eval_wmr(ft_save_path, eval_loader, device)
                else:
                    n_ft = len(target_vols) * 55 * k
                    print(f"    [{k}-shot]  No FT ckpt -> fine-tune: {len(target_vols)} x 55 x {k}"
                          f" = {n_ft} samples  (ep={ft_epochs}, lr={ft_lr})")
                    ft_loader = make_loader(root_path,
                        {target_scene: {'samples': list(range(1, k + 1)),
                                        'vols': target_vols}},
                        BS, NUM_WORKERS, shuffle=True, generator=g)
                    res = _ft_then_eval_wmr(pretrain_ckpt, ft_loader, eval_loader, device,
                                            epochs=ft_epochs, lr=ft_lr,
                                            save_path=ft_save_path)

            results[k][target_scene] = res
            line = "  ".join(f"{m}={res[m]:6.2f}%"
                             for m in ['WiFi', 'RFID', 'mmWave', 'Fusion'])
            print(f"      [OK] Zhang {target_scene} {k}-shot: {line}")

    _print_per_modality_table(
        "Zhang | Cross-Scene (per-modality + Fusion)",
        results, shots, eval_keys=target_scenes,
    )
    save_result("zhang_cross_scene", {
        "target_scenes_vols": scene_vols,
        "pretrain_source":    "Zhang_SCSub_21_9",
        "ft_epochs": ft_epochs, "ft_lr": ft_lr,
        "results": {
            f"{k}shot": {sc: {m: float(v) for m, v in results[k][sc].items()}
                         for sc in target_scenes}
            for k in shots
        },
    })
    return results


# New pretrain on the 22 clean Scene1 subjects (excluding overlap subjs {3,4,5,6,7,13,23,24}), then adapt to S2/S3/S4 with each scene's specific vols (including vol 31 in Scene2). Same split as the shared experiment protocol.

# ============================================================
# CSC&S - Zhang baseline
#   New pretrain on the 22 clean Scene1 subjects (excluding overlap subjs
#   {3,4,5,6,7,13,23,24}), then adapt to S2/S3/S4 with each scene's specific
#   vols (including vol 31 in Scene2).
#
# Behavior per (scene, k-shot):
#   k = 0 : load CSC&S pretrain ckpt -> zero-shot eval on target scene
#   k > 0 : if FT ckpt exists        -> load and evaluate
#           else                      -> fine-tune from pretrain, save FT ckpt, evaluate
# ============================================================

def run_cscs_zhang(root_path=ROOT, device=DEVICE,
                        pretrain_epochs=100,
                        ft_epochs=100, ft_lr=1e-5,
                        BS=16, NUM_WORKERS=4,
                        shots=(0, 1, 2, 3, 4, 5),
                        force_pretrain=False):
    target_scenes = ['Scene2', 'Scene3', 'Scene4']
    scene_vols    = {sc: get_volunteers_in_scene(root_path, sc) for sc in target_scenes}

    all_vols     = get_volunteers_in_scene(root_path, 'Scene1')
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

    pretrain_exp  = "Zhang_CSCS_pretrain"
    pretrain_ckpt = os.path.join("reproc", f"{pretrain_exp}_last.pth")
    print(f"\n{'='*80}\n [Zhang] CSC&S (per-modality + Fusion)\n{'='*80}")

    # ========== Stage 1: NEW Pretrain (clean source) ==========
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
        _ = train_and_eval(pretrain_exp, ZhangBaseline,
                           pretrain_loader, {'sanity': sanity_loader},
                           lambda r: r['sanity'], device,
                           epochs=pretrain_epochs)
        print(f"    [OK] saved -> {pretrain_ckpt}")
    else:
        print(f"  [Stage 1: Pretrain] [OK] ckpt found -> {pretrain_ckpt}")

    results = {k: {sc: None for sc in target_scenes} for k in shots}

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
                print(f"    [{k}-shot] zero-shot  | eval per-class = {len(eval_samples)}")
                res = _load_and_eval_wmr(pretrain_ckpt, eval_loader, device)
            else:
                eval_samples = list(range(k + 1, 21))
                eval_loader = make_loader(root_path,
                    {target_scene: {'samples': eval_samples, 'vols': target_vols}},
                    BS, NUM_WORKERS, generator=g)
                ft_save_path = os.path.join(
                    "reproc", f"Zhang_CSCS_{target_scene}_{k}shot_ft.pth")

                if os.path.exists(ft_save_path):
                    print(f"    [{k}-shot] Found FT ckpt: {ft_save_path}  -> load & evaluate")
                    res = _load_and_eval_wmr(ft_save_path, eval_loader, device)
                else:
                    n_ft = len(target_vols) * 55 * k
                    print(f"    [{k}-shot]  No FT ckpt -> fine-tune: {len(target_vols)} x 55 x {k}"
                          f" = {n_ft} samples  (ep={ft_epochs}, lr={ft_lr})")
                    ft_loader = make_loader(root_path,
                        {target_scene: {'samples': list(range(1, k + 1)),
                                        'vols': target_vols}},
                        BS, NUM_WORKERS, shuffle=True, generator=g)
                    res = _ft_then_eval_wmr(pretrain_ckpt, ft_loader, eval_loader, device,
                                            epochs=ft_epochs, lr=ft_lr,
                                            save_path=ft_save_path)

            results[k][target_scene] = res
            line = "  ".join(f"{m}={res[m]:6.2f}%"
                             for m in ['WiFi', 'RFID', 'mmWave', 'Fusion'])
            print(f"      [OK] Zhang {target_scene} {k}-shot: {line}")

    _print_per_modality_table(
        "Zhang | CSC&S (per-modality + Fusion)",
        results, shots, eval_keys=target_scenes,
    )
    save_result("zhang_cscs", {
        "source_vols":         SOURCE_VOLS,
        "excluded_vols":       sorted(contaminated),
        "target_scenes_vols":  scene_vols,
        "pretrain_epochs":     pretrain_epochs,
        "ft_epochs": ft_epochs, "ft_lr": ft_lr,
        "results": {
            f"{k}shot": {sc: {m: float(v) for m, v in results[k][sc].items()}
                         for sc in target_scenes}
            for k in shots
        },
    })
    return results


# ============================================================
# Zhang Baseline (DenseNet + ResNet18 + ResNeXt + 3 Intermodal Blocks)
# Sensor-Failure Robustness  -  5-shot FT ckpts
# ============================================================
# Assumption: the system can detect sensor failure and exclude all streams
#             that depend on the failed modality from the fusion.
#
# Zhang has 6 streams: r, w, m, rw, rm, wm
#   - W fail  -> exclude w, rw, wm  (remaining: r, m, rm)
#   - M fail  -> exclude m, rm, wm  (remaining: r, w, rw)
#   - R fail  -> exclude r, rw, rm  (remaining: w, m, wm)
#   - 2 fail (only W kept) -> only w stream usable
#     (rw/wm also need other modalities)
#
# ckpt pattern: reproc/Zhang_CSCS_{scene}_{shot}shot_ft.pth
# ============================================================
import os, glob, csv, json

SHOT = 5
CKPT_PATTERN = "reproc/Zhang_CSCS_{scene}_{shot}shot_ft.pth"

# 1) Stream index mapping
#    Zhang forward stacks: [r, w, m, rw, rm, wm] along dim=1
#    Each stream requires the following modalities:
STREAM_NEEDS = {
    0: {'R'},           # r_logits
    1: {'W'},           # w_logits
    2: {'M'},           # m_logits
    3: {'R', 'W'},      # rw_logits (RFID + WiFi)
    4: {'R', 'M'},      # rm_logits (RFID + mmWave)
    5: {'W', 'M'},      # wm_logits (WiFi + mmWave)
}
STREAM_NAMES = ['r', 'w', 'm', 'rw', 'rm', 'wm']


# 2) Masked forward - Zhang baseline specific
#    "Sensor failure" -> exclude every stream that needs that modality
@torch.no_grad()
def zhang_forward_masked(model, wifi_x, mmw_x, rfid_x, keep=('W','M','R')):
    """Use only streams whose required modalities are all in `keep`."""
    model.eval()
    keep_set = set(keep)

    # Normal forward (all backbones run; we just pick which logits to use)
    out = model(wifi_x, mmw_x, rfid_x, targets=None)
    all_logits = out['logits']  # (B, 6, K)

    # Select usable stream indices
    usable = [i for i, needs in STREAM_NEEDS.items() if needs.issubset(keep_set)]

    if not usable:
        # Should not happen (all modalities dead)
        return torch.zeros_like(all_logits[:, 0])

    # Confidence-weighted average over selected streams (Zhang's fusion rule)
    selected_logits = all_logits[:, usable, :].float()           # (B, U, K)
    probs = F.softmax(selected_logits, dim=-1)                    # (B, U, K)
    conf = probs.max(-1).values                                    # (B, U)
    weights = conf / (conf.sum(-1, keepdim=True) + 1e-8)          # (B, U)
    fused_prob = (weights.unsqueeze(-1) * probs).sum(1)            # (B, K)
    return fused_prob


@torch.no_grad()
def zhang_evaluate_masked(model, loader, device, keep):
    model.eval()
    c, t = 0, 0
    for data, labels in loader:
        wifi = data['WiFi'].to(device, non_blocking=True)
        mmw  = data['mmWave'].to(device, non_blocking=True)
        rfid = data['RFID'].to(device, non_blocking=True)
        tgt  = (labels - 1).to(device, non_blocking=True)
        fused_prob = zhang_forward_masked(model, wifi, mmw, rfid, keep=keep)
        pred = fused_prob.argmax(1)
        c += pred.eq(tgt).sum().item()
        t += tgt.size(0)
    return 100.0 * c / t if t > 0 else 0.0


# 3) Keep combinations
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


def build_cscs_eval_loaders(root_path=ROOT, BS=16, NUM_WORKERS=4, shot=SHOT):
    """Build held-out CSC&S loaders for sensor-failure evaluation."""
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


def run_sensor_failure_zhang(root_path=ROOT, device=DEVICE,
                             BS=16, NUM_WORKERS=4, shot=5,
                             ckpt_pattern=None):
    """Evaluate Zhang baseline under missing modalities using CSC&S FT ckpts."""
    pattern = ckpt_pattern or CKPT_PATTERN
    eval_loaders = build_cscs_eval_loaders(root_path, BS=BS, NUM_WORKERS=NUM_WORKERS, shot=shot)
    raw = {}

    for scene in ['Scene2', 'Scene3', 'Scene4']:
        ckpt = pattern.format(scene=scene, shot=shot)
        if not os.path.exists(ckpt):
            print(f"[skip] missing checkpoint: {ckpt}")
            continue
        print(f"[sensor_failure] Zhang {scene}: {ckpt}")
        model = ZhangBaseline(num_classes=55).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        raw[scene] = {}
        for keep in KEEP_COMBOS:
            raw[scene][KEEP_LABEL[keep]] = zhang_evaluate_masked(
                model, eval_loaders[scene], device, keep)
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
        print(f"Zhang: full={full:.2f}%  one-fail-avg={one_avg:.2f}%")

    csv_path = f"reproc/results/sensor_failure_zhang_{shot}shot.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scene", "keep", "acc"])
        for scene, keep_d in raw.items():
            for keep_label, acc in keep_d.items():
                writer.writerow([scene, keep_label, f"{acc:.4f}"])
    save_result(f"sensor_failure_zhang_{shot}shot", raw)
    return raw


# This is a faithful re-implementation from the paper text (Zhang et al., IoT-J, Dec 2025). The released code repository (`github.com/qinzuan77-beep/code.git`) was inaccessible, so the following choices were made where the paper is silent:
#
# - **Channel-exchange ratio in the intermodal block** - set to 50%, matching the half-split illustration in Fig. 7.
# - **Unified intermodal channel/time dims** - `INTER_C=128`, `INTER_T=16`. The paper says features are "compressed into a unified shape" without specifying the exact widths; the Table I total of 21.71M params is approximately matched at the modality level (DenseNet/ResNet18/ResNeXt) but the intermodal block (paper: 3.95M) is left to its natural shape here.
# - **Inference-time confidence** - Eq. 8 uses the GT one-hot during training; at inference we use `max(P_m)`, i.e., the top-1 predicted probability, as the unsupervised analogue.
# - **Optimization** - paper trains modalities with separate LR schedules (Sec. VI-D) before fusing. Here we use a single AdamW + OneCycleLR for all parameters to mirror the shared experiment protocol, since this is the head-to-head comparison configuration.
#
# Resulting checkpoints are saved with a `Zhang_` prefix to coexist with the main model's checkpoints in `reproc/`.
