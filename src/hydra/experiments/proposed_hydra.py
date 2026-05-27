#!/usr/bin/env python
# coding: utf-8


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

from mamba_ssm import Mamba

# Paths & device
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

# DataLoader / split helpers
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

# Mamba helpers (shared)
def make_mamba_layers(embed_dim, depth, use_dropout=False, dropout=0.2):
    layers = []
    for _ in range(depth):
        ld = {'norm': nn.LayerNorm(embed_dim),
              'mamba': Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2)}
        if use_dropout: ld['drop'] = nn.Dropout(dropout)
        layers.append(nn.ModuleDict(ld))
    return nn.ModuleList(layers)

def run_mamba_residual(x, layers):
    """y = mamba(norm(x)) + x  with optional dropout."""
    for l in layers:
        y = l['mamba'](l['norm'](x))
        if 'drop' in l: y = l['drop'](y)
        x = x + y
    return x

# WiFi: P4_DepthSep stem + 3-branch (antenna-preserving)
def stem_depthsep_wifi(in_chans=1, mid_dim=48, embed_dim=96, time_tokens=20):
    """Antenna-split + depthwise temporal + pointwise.
       Input per-branch: (B, 1, 90, 1000)
       Output: (B, embed_dim, 3, time_tokens) -> 60 tokens."""
    return nn.Sequential(
        # Step 1: antenna split (kernel_h=30 stride_h=30) + smooth time stride
        nn.Conv2d(in_chans, mid_dim, (30, 9), stride=(30, 5), padding=(0, 4)),  # 1000->200
        nn.BatchNorm2d(mid_dim), nn.GELU(),
        # Step 2: depthwise temporal (channels independent)
        nn.Conv2d(mid_dim, mid_dim, (1, 5), stride=(1, 2), padding=(0, 2),
                  groups=mid_dim),  # 200->100
        nn.BatchNorm2d(mid_dim), nn.GELU(),
        # Step 3: pointwise (channel mix)
        nn.Conv2d(mid_dim, embed_dim, (1, 1)),
        nn.BatchNorm2d(embed_dim), nn.GELU(),
        # Step 4: AdaptiveAvgPool to (3, time_tokens) - 100/20=5
        nn.AdaptiveAvgPool2d((3, time_tokens)),
    )

class _WiFiPhysBranch(nn.Module):
    """Stem + Mamba branch (used 3x in WiFi backbone)."""
    def __init__(self, embed_dim=96, time_tokens=20, depth=3, mid_dim=48):
        super().__init__()
        self.stem = stem_depthsep_wifi(1, mid_dim, embed_dim, time_tokens)
        num_patches = 3 * time_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.layers = make_mamba_layers(embed_dim, depth, use_dropout=False)
        self.norm_f = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.stem(x).flatten(2).transpose(1, 2)  # (B, N, D)
        x = x + self.pos_embed
        x = run_mamba_residual(x, self.layers)
        return self.norm_f(x).mean(dim=1)

# mmWave: M2_LightDim stem (24->64 channels, dim=64, depth=2)
def mmw_lightdim_stem(embed_dim=64):
    """Two-conv stem with reduced channel width.
       Input per-frame: (B*T, 1, 256, 64) -> (B*T, embed_dim, 16, 4)."""
    return nn.Sequential(
        nn.Conv2d(1, 24, 7, stride=4, padding=3), nn.BatchNorm2d(24), nn.GELU(),
        nn.Conv2d(24, embed_dim, 5, stride=4, padding=2), nn.BatchNorm2d(embed_dim), nn.GELU(),
    )

class mmw_MambaBranch(nn.Module):
    """Spatial or temporal Mamba block for mmWave."""
    def __init__(self, num_patches, embed_dim, depth):
        super().__init__()
        self.in_norm = nn.LayerNorm(embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.layers = make_mamba_layers(embed_dim, depth, use_dropout=True)
        self.norm_f = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.in_norm(x) + self.pos_embed
        x = run_mamba_residual(x, self.layers)
        return self.norm_f(x)

# RFID: R1_Original (dim=96, depth=4, conv-conv stride=2)
class rfid_MambaBranch(nn.Module):
    def __init__(self, num_patches, embed_dim, depth):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.layers = make_mamba_layers(embed_dim, depth, use_dropout=False)
        self.norm_f = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = x + self.pos_embed
        x = run_mamba_residual(x, self.layers)
        return self.norm_f(x).mean(dim=1)

# Projection helpers for branch/stream-level InfoNCE
def make_projection_head(in_dim, proj_dim=128):
    return nn.Sequential(
        nn.Linear(in_dim, 256), nn.GELU(),
        nn.Linear(256, proj_dim)
    )

# Single-modality classifiers
#   forward(wifi, mmw, rfid, apply_drop) -> dict { 'logits', optional 'nce_zs' }
#   InfoNCE policy during TRAINING only:
#     WiFi   : AP1/AP2/AP3 branch embeddings -> adjacent pairs (AP1-AP2, AP2-AP3)
#     mmWave : RD/RA stream embeddings       -> one pair (RD-RA)
#     RFID   : single branch                 -> no intra-modality InfoNCE
#
#   During EVAL/INFERENCE:
#     return only {'logits': ...}
#     -> FLOPs/latency do not include training-only projection heads.
class WiFiOnlyClassifier(nn.Module):
    """P4_DepthSep based: 3 antenna-aware branches with AP-wise InfoNCE during training."""
    def __init__(self, num_classes=55, embed_dim=96, time_tokens=20, depth=3, mid_dim=48, proj_dim=128):
        super().__init__()
        self.wifi_dim = embed_dim
        self.wifi_branches = nn.ModuleList([
            _WiFiPhysBranch(embed_dim, time_tokens, depth, mid_dim) for _ in range(3)
        ])
        self.wifi_branch_proj = make_projection_head(embed_dim, proj_dim)
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(embed_dim * 3, 256), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(256, num_classes)
        )

    def forward(self, wifi_x, mmw_x=None, rfid_x=None, apply_drop=False):
        chunks = torch.split(wifi_x, wifi_x.shape[2] // 3, dim=2)
        feats = [b(chunks[i]) for i, b in enumerate(self.wifi_branches)]  # [AP1, AP2, AP3]
        logits = self.head(torch.cat(feats, dim=1))

        out = {'logits': logits}

        # Training-only auxiliary InfoNCE path.
        # Eval/FLOPs/latency path excludes projection heads.
        if self.training:
            nce_zs = [F.normalize(self.wifi_branch_proj(f), dim=-1) for f in feats]
            out.update({'nce_zs': nce_zs, 'nce_mode': 'adjacent'})

        return out


class MmWaveOnlyClassifier(nn.Module):
    """M2_LightDim based: dim=64, depth=2, RD/RA two-stream with stream-wise InfoNCE during training."""
    def __init__(self, num_classes=55, embed_dim=64, depth=2, proj_dim=128):
        super().__init__()
        self.mmw_dim = embed_dim
        self.mmw_cnn_rd = mmw_lightdim_stem(embed_dim)
        self.mmw_s_rd = mmw_MambaBranch(64, embed_dim, depth=depth)
        self.mmw_t_rd = mmw_MambaBranch(17, embed_dim, depth=depth)
        self.mmw_cnn_ra = mmw_lightdim_stem(embed_dim)
        self.mmw_s_ra = mmw_MambaBranch(64, embed_dim, depth=depth)
        self.mmw_t_ra = mmw_MambaBranch(17, embed_dim, depth=depth)
        self.mmw_stream_proj = make_projection_head(embed_dim, proj_dim)
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(embed_dim * 2, 256), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(256, num_classes)
        )

    def forward(self, wifi_x=None, mmw_x=None, rfid_x=None, apply_drop=False):
        x_rd, x_ra = torch.split(mmw_x, 64, dim=3)

        B, T, H, W = x_rd.shape

        c_rd = self.mmw_cnn_rd(x_rd.reshape(B * T, 1, H, W)).flatten(2).transpose(1, 2)
        c_rd = self.mmw_s_rd(c_rd).max(dim=1)[0].view(B, T, -1)
        feat_rd = self.mmw_t_rd(c_rd).max(dim=1)[0]

        c_ra = self.mmw_cnn_ra(x_ra.reshape(B * T, 1, H, W)).flatten(2).transpose(1, 2)
        c_ra = self.mmw_s_ra(c_ra).max(dim=1)[0].view(B, T, -1)
        feat_ra = self.mmw_t_ra(c_ra).max(dim=1)[0]

        logits = self.head(torch.cat([feat_rd, feat_ra], dim=1))

        out = {'logits': logits}

        # Training-only auxiliary InfoNCE path.
        # With two streams, 'all' and 'adjacent' are equivalent, but adjacent is used for consistency.
        if self.training:
            nce_zs = [
                F.normalize(self.mmw_stream_proj(feat_rd), dim=-1),
                F.normalize(self.mmw_stream_proj(feat_ra), dim=-1),
            ]
            out.update({'nce_zs': nce_zs, 'nce_mode': 'adjacent'})

        return out


class RFIDOnlyClassifier(nn.Module):
    """R1_Original: dim=96, depth=4. Single RFID branch, so no intra-modality InfoNCE."""
    def __init__(self, num_classes=55, embed_dim=96, depth=4):
        super().__init__()
        self.rfid_dim = embed_dim
        self.rfid_cnn = nn.Sequential(
            nn.Conv1d(23, 64, 7, padding=3), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, embed_dim, 5, stride=2, padding=2),
            nn.BatchNorm1d(embed_dim), nn.GELU()
        )
        self.rfid_mamba = rfid_MambaBranch(
            num_patches=(148 // 2),
            embed_dim=embed_dim,
            depth=depth
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(embed_dim, 256), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(256, num_classes)
        )

    def forward(self, wifi_x=None, mmw_x=None, rfid_x=None, apply_drop=False):
        c = self.rfid_cnn(rfid_x.squeeze(1)).transpose(1, 2)
        feat = self.rfid_mamba(c)
        return {'logits': self.head(feat)}


# Fusion: best per-modality backbones + drop_only + branch/stream-level InfoNCE
#   Training-time InfoNCE uses six PRE-drop component embeddings:
#     [WiFi-AP1, WiFi-AP2, WiFi-AP3, mmWave-RD, mmWave-RA, RFID]
#
#   To avoid all-pair cost, Fusion uses adjacent pairs only:
#     AP1-AP2, AP2-AP3, AP3-RD, RD-RA, RA-RFID
#
#   During EVAL/INFERENCE:
#     return only {'logits': ...}
#     -> FLOPs/latency do not include training-only projection heads.
class FusionClassifier(nn.Module):
    """Drop_Only + adjacent branch/stream-level InfoNCE on PRE-drop features during training."""
    def __init__(self, num_classes=55, proj_dim=128,
                 wifi_E=96, wifi_T=20, wifi_D=3, wifi_mid=48,
                 mmw_E=64, mmw_D=2,
                 rfid_E=96, rfid_D=4):
        super().__init__()

        # ----- WiFi backbone (P4) -----
        self.wifi_dim = wifi_E
        self.wifi_branches = nn.ModuleList([
            _WiFiPhysBranch(wifi_E, wifi_T, wifi_D, wifi_mid) for _ in range(3)
        ])
        self.wifi_branch_proj = make_projection_head(wifi_E, proj_dim)

        # ----- mmWave backbone (M2) -----
        self.mmw_dim = mmw_E
        self.mmw_cnn_rd = mmw_lightdim_stem(mmw_E)
        self.mmw_s_rd = mmw_MambaBranch(64, mmw_E, depth=mmw_D)
        self.mmw_t_rd = mmw_MambaBranch(17, mmw_E, depth=mmw_D)

        self.mmw_cnn_ra = mmw_lightdim_stem(mmw_E)
        self.mmw_s_ra = mmw_MambaBranch(64, mmw_E, depth=mmw_D)
        self.mmw_t_ra = mmw_MambaBranch(17, mmw_E, depth=mmw_D)

        self.mmw_stream_proj = make_projection_head(mmw_E, proj_dim)

        # ----- RFID backbone (R1) -----
        self.rfid_dim = rfid_E
        self.rfid_cnn = nn.Sequential(
            nn.Conv1d(23, 64, 7, padding=3), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, rfid_E, 5, stride=2, padding=2),
            nn.BatchNorm1d(rfid_E), nn.GELU()
        )
        self.rfid_mamba = rfid_MambaBranch(
            num_patches=(148 // 2),
            embed_dim=rfid_E,
            depth=rfid_D
        )
        self.rfid_proj = make_projection_head(rfid_E, proj_dim)

        # Modality totals after concat
        wifi_total = wifi_E * 3   # 288
        mmw_total  = mmw_E  * 2   # 128
        rfid_total = rfid_E       # 96

        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(wifi_total + mmw_total + rfid_total, 512), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(512, num_classes)
        )

    def forward(self, wifi_x, mmw_x, rfid_x, apply_drop=False):
        # ----- WiFi: three AP/antenna-aware branch features -----
        wifi_chunks = torch.split(wifi_x, wifi_x.shape[2] // 3, dim=2)
        wifi_feats = [b(wifi_chunks[i]) for i, b in enumerate(self.wifi_branches)]
        wifi_full = torch.cat(wifi_feats, dim=1)

        # ----- mmWave: RD / RA stream features -----
        x_rd, x_ra = torch.split(mmw_x, 64, dim=3)
        B, T, H, W = x_rd.shape

        c_rd = self.mmw_cnn_rd(x_rd.reshape(B * T, 1, H, W)).flatten(2).transpose(1, 2)
        c_rd = self.mmw_s_rd(c_rd).max(dim=1)[0].view(B, T, -1)
        feat_rd = self.mmw_t_rd(c_rd).max(dim=1)[0]

        c_ra = self.mmw_cnn_ra(x_ra.reshape(B * T, 1, H, W)).flatten(2).transpose(1, 2)
        c_ra = self.mmw_s_ra(c_ra).max(dim=1)[0].view(B, T, -1)
        feat_ra = self.mmw_t_ra(c_ra).max(dim=1)[0]

        mmw_full = torch.cat([feat_rd, feat_ra], dim=1)

        # ----- RFID: one branch feature -----
        rfid_c = self.rfid_cnn(rfid_x.squeeze(1)).transpose(1, 2)
        rfid_full = self.rfid_mamba(rfid_c)

        # ----- Drop_only on modality-level features -----
        if apply_drop and self.training:
            Bsz = wifi_full.shape[0]
            dc = torch.randint(0, 3, (Bsz,), device=wifi_full.device)

            wifi_d = wifi_full * (dc != 0).float().unsqueeze(1)
            mmw_d  = mmw_full  * (dc != 1).float().unsqueeze(1)
            rfid_d = rfid_full * (dc != 2).float().unsqueeze(1)
        else:
            wifi_d, mmw_d, rfid_d = wifi_full, mmw_full, rfid_full

        logits = self.head(torch.cat([wifi_d, mmw_d, rfid_d], dim=1))

        out = {'logits': logits}

        # Training-only auxiliary InfoNCE path.
        # Eval/FLOPs/latency path excludes projection heads.
        if self.training:
            nce_zs = [
                F.normalize(self.wifi_branch_proj(wifi_feats[0]), dim=-1),
                F.normalize(self.wifi_branch_proj(wifi_feats[1]), dim=-1),
                F.normalize(self.wifi_branch_proj(wifi_feats[2]), dim=-1),
                F.normalize(self.mmw_stream_proj(feat_rd), dim=-1),
                F.normalize(self.mmw_stream_proj(feat_ra), dim=-1),
                F.normalize(self.rfid_proj(rfid_full), dim=-1),
            ]
            out.update({'nce_zs': nce_zs, 'nce_mode': 'adjacent'})

        return out


MODEL_REGISTRY = {
    'RFID':   RFIDOnlyClassifier,
    'WiFi':   WiFiOnlyClassifier,
    'mmWave': MmWaveOnlyClassifier,
    'Fusion': FusionClassifier,
}

# Loss
def info_nce_pair(z1, z2, temperature=0.1):
    B = z1.shape[0]
    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(B, device=z1.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) * 0.5

def component_info_nce(zs, mode='all', temperature=0.1):
    """InfoNCE over component embeddings.
       mode='all'      : all unordered pairs.
       mode='adjacent' : chain-neighbor pairs only.

       Current policy:
         WiFi-only : adjacent AP pairs = AP1-AP2, AP2-AP3
         mmWave    : adjacent/all are equivalent because there are only RD and RA
         Fusion    : adjacent six-component chain =
                     AP1-AP2, AP2-AP3, AP3-RD, RD-RA, RA-RFID
    """
    zs = [z for z in zs if z is not None]
    if len(zs) < 2:
        return None
    if mode == 'adjacent':
        pairs = [(i, i+1) for i in range(len(zs)-1)]
    else:
        pairs = [(i, j) for i in range(len(zs)) for j in range(i+1, len(zs))]
    losses = [info_nce_pair(zs[i], zs[j], temperature) for i, j in pairs]
    return sum(losses) / len(losses)

# Unified train / eval
@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    if len(loader) == 0: return 0.0
    c, t = 0, 0
    for data, labels in loader:
        wifi = data['WiFi'].to(device, non_blocking=True)
        mmw  = data['mmWave'].to(device, non_blocking=True)
        rfid = data['RFID'].to(device, non_blocking=True)
        tgt = (labels - 1).to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(wifi, mmw, rfid, apply_drop=False)
        c += out['logits'].max(1)[1].eq(tgt).sum().item()
        t += tgt.size(0)
    return 100.0 * c / t if t > 0 else 0.0

def train_and_eval(exp_name, model_cls, train_loader, eval_loaders, best_metric_fn,
                   device, epochs=100, infonce_weight=0.1, lr_max=5e-4, lr_init=1e-4,
                   weight_decay=0.05, label_smoothing=0.1, log_every=10):
    """Single + Fusion. Last-epoch saved -> reproc/{exp_name}_last.pth
       InfoNCE is applied whenever the model returns two or more component embeddings.
    """
    set_reproducible_mode(SEED)

    model = model_cls(num_classes=55).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr_init, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr_max,
        steps_per_epoch=len(train_loader), epochs=epochs)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    save_path = os.path.join("reproc", f"{exp_name}_last.pth")

    for epoch in range(epochs):
        model.train()
        sum_loss, sum_ce, sum_nce, nce_batches = 0.0, 0.0, 0.0, 0

        for data, labels in train_loader:
            wifi = data['WiFi'].to(device, non_blocking=True)
            mmw  = data['mmWave'].to(device, non_blocking=True)
            rfid = data['RFID'].to(device, non_blocking=True)
            tgt = (labels - 1).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(wifi, mmw, rfid, apply_drop=isinstance(model, FusionClassifier))
                ce_loss = criterion(out['logits'], tgt)
                nce_loss = None
                if 'nce_zs' in out and len(out['nce_zs']) >= 2:
                    nce_loss = component_info_nce(
                        out['nce_zs'], mode=out.get('nce_mode', 'all'), temperature=0.1)
                if nce_loss is not None:
                    loss = ce_loss + infonce_weight * nce_loss
                    sum_nce += nce_loss.item(); nce_batches += 1
                else:
                    loss = ce_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()

            sum_loss += loss.item(); sum_ce += ce_loss.item()
            del out, loss, ce_loss, nce_loss

        if (epoch + 1) % log_every == 0 or epoch == 0:
            n = len(train_loader)
            extra = f" NCE:{sum_nce/max(1,nce_batches):.4f}" if nce_batches > 0 else ""
            print(f"   [Ep {epoch+1:03d}/{epochs}] Train Loss:{sum_loss/n:.4f} CE:{sum_ce/n:.4f}{extra}")

        torch.cuda.empty_cache(); gc.collect()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    model.eval()
    final = {n: evaluate_model(model, ld, device) for n, ld in eval_loaders.items()}
    acc_str = " | ".join([f"{k}:{v:.2f}%" for k, v in final.items()])
    print(f"    [Final] {acc_str}")
    del model; torch.cuda.empty_cache(); gc.collect()
    return final

# Pretty printer (RFID/WiFi/mmW/Fusion)
def print_paper_table(title, results_by_modality, shots, eval_keys=None):
    """results_by_modality[mod][shot][eval_key] = acc"""
    print("\n" + "="*80)
    print(f"{title}")
    print("="*80)
    eval_keys = eval_keys or list(next(iter(next(iter(results_by_modality.values())).values())).keys())
    for ek in eval_keys:
        print(f"\n[Eval target: {ek}]")
        print(f"{'#-Shot':<8} | {'RFID':>8} | {'WiFi':>8} | {'mmWave':>8} | {'Fusion':>8}")
        print("-" * 55)
        for k in shots:
            kname = ['Zero','One','Two','Three','Four','Five'][k]
            row = f"{kname:<8} | "
            for mod in ['RFID', 'WiFi', 'mmWave', 'Fusion']:
                v = results_by_modality.get(mod, {}).get(k, {}).get(ek, float('nan'))
                row += f"{v:>7.2f}% | " if v == v else f"{'N/A':>8} | "
            print(row.rstrip(' |'))

# Model cost - auto shape detect + cache
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
       - mamba_ssm.Mamba is counted analytically and its internal modules are skipped
       - normalization/activation/pooling are not counted, matching common model-complexity tables
    """
    model.eval()
    flops = {'total': 0}
    hooks = []

    def _inside_mamba(name):
        return '.mamba.' in name or name.endswith('.mamba')

    def conv1d_hook(m, inp, out):
        x = inp[0]
        B = x.shape[0]
        out_len = out.shape[-1]
        kernel_ops = m.kernel_size[0] * (m.in_channels // m.groups)
        bias_ops = 1 if m.bias is not None else 0
        flops['total'] += int(B * m.out_channels * out_len * (2 * kernel_ops + bias_ops))

    def conv2d_hook(m, inp, out):
        x = inp[0]
        B = x.shape[0]
        out_h, out_w = out.shape[-2], out.shape[-1]
        k_h, k_w = m.kernel_size
        kernel_ops = k_h * k_w * (m.in_channels // m.groups)
        bias_ops = 1 if m.bias is not None else 0
        flops['total'] += int(B * m.out_channels * out_h * out_w * (2 * kernel_ops + bias_ops))

    def linear_hook(m, inp, out):
        # Works for both (B,D) and (...,D) inputs.
        num_outputs = out.numel() // m.out_features
        bias_ops = out.numel() if m.bias is not None else 0
        flops['total'] += int(num_outputs * m.in_features * m.out_features * 2 + bias_ops)

    def mamba_hook(m, inp, out):
        x = inp[0]
        if x.ndim != 3:
            return
        B, L, D = x.shape
        d_inner = int(getattr(m, 'd_inner', D * 2))
        d_state = int(getattr(m, 'd_state', 16))
        d_conv  = int(getattr(m, 'd_conv', 4))
        dt_rank = int(getattr(m, 'dt_rank', max(1, D // 16)))
        # Approximation for Mamba block: projections + depthwise conv + selective scan + output projection.
        flops['total'] += int(B * L * (
            D * 2 * d_inner * 2 +
            d_inner * d_conv * 2 +
            d_inner * (dt_rank + 2 * d_state) * 2 +
            dt_rank * d_inner * 2 +
            d_inner * d_state * 6 +
            d_inner * D * 2
        ))

    for name, mod in model.named_modules():
        if isinstance(mod, Mamba):
            hooks.append(mod.register_forward_hook(mamba_hook))
        elif _inside_mamba(name):
            continue
        elif isinstance(mod, nn.Conv1d):
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

def measure_model_cost(device=DEVICE, n_latency_trials=50, use_cache=True, save=True, verbose=True):
    cache_path = f"reproc/results/model_cost_custom_mamba_{HOSTNAME}.json"
    if use_cache and os.path.exists(cache_path):
        with open(cache_path) as f: data = json.load(f)
        if verbose:
            print(f"\nModel Cost (cached custom-Mamba, host={HOSTNAME})")
            print("-" * 80)
            print(f"{'Model':<10} | {'Params':>10} | {'Memory':>10} | {'FLOPs':>12} | {'Latency':>16}")
            print("-" * 80)
            for m, v in data.items():
                fl = f"{v['flops_G']:.4f}G" if v.get('flops_G') is not None else 'N/A'
                lat = f"{v['latency_ms_mean']:.2f}+/-{v['latency_ms_std']:.2f}ms"
                print(f"{m:<10} | {v['params_M']:>8.2f}M | {v['memory_MB']:>8.2f}MB | {fl:>12} | {lat:>16}")
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
        print(f"\nMeasuring model cost with custom Mamba FLOPs (host={HOSTNAME})...")
        print("-" * 80)
        print(f"{'Model':<10} | {'Params':>10} | {'Memory':>10} | {'FLOPs':>12} | {'Latency':>16}")
        print("-" * 80)

    summary = {}
    for mname, mcls in MODEL_REGISTRY.items():
        model = mcls(num_classes=55).to(device).eval()
        params = count_params(model); mem = occupy_mb(model)
        with torch.no_grad():
            flops = _estimate_flops_custom(model, inputs=(wifi, mmw, rfid, False)) / 1e9

        with torch.no_grad():
            for _ in range(10): _ = model(wifi, mmw, rfid)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            times = []
            for _ in range(n_latency_trials):
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(wifi, mmw, rfid)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                times.append((time.perf_counter() - t0)*1000)
            avg_ms, std_ms = float(np.mean(times)), float(np.std(times))

        if verbose:
            print(f"{mname:<10} | {params/1e6:>8.2f}M | {mem:>8.2f}MB | {flops:>11.4f}G | "
                  f"{avg_ms:>10.2f}+/-{std_ms:.2f}ms")
        summary[mname] = {'params_M': params/1e6, 'memory_MB': mem, 'flops_G': flops,
                          'latency_ms_mean': avg_ms, 'latency_ms_std': std_ms,
                          'flops_note': 'custom hooks; 1 MAC = 2 FLOPs; Mamba analytically approximated',
                          'hostname': HOSTNAME}
        del model; torch.cuda.empty_cache(); gc.collect()

    if save:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f: json.dump(summary, f, indent=2)
        if verbose: print(f"saved: {cache_path}")
    return summary

if os.environ.get("HYDRA_VERBOSE_IMPORT") == "1":
    print("READY")
    print(f"   ROOT={ROOT}\n   DEVICE={DEVICE}\n   HOST={HOSTNAME}")


# ============================================================
# Cell: IN-DOMAIN Scene1 (samples 1~14 train / 15~20 test, all vols)
#   - {RFID, WiFi, mmWave, Fusion} Eval.
# ============================================================
def run_in_domain_scene1(root_path=ROOT, device=DEVICE,
                         epochs=100, BS=16, NUM_WORKERS=4,
                         modalities=('RFID', 'WiFi', 'mmWave', 'Fusion'),
                         force_retrain=False):
    g = fresh_generator()
    train_samples = list(range(1, 15))   # 1~14
    test_samples  = list(range(15, 21))  # 15~20

    train_loader = make_loader(root_path,
        {'Scene1': {'samples': train_samples, 'vols': None}},
        BS, NUM_WORKERS, shuffle=True, generator=g)
    test_loader = make_loader(root_path,
        {'Scene1': {'samples': test_samples, 'vols': None}},
        BS, NUM_WORKERS, generator=g)
    eval_loaders = {'Scene1': test_loader}
    metric = lambda r: r['Scene1']

    print("\n" + "="*80)
    print("[In-Domain Scene1]  Train: Scene1 samples 1~14 | Test: Scene1 samples 15~20")
    print("="*80)

    results = {}
    for mod in modalities:
        exp_name = f"InDomain_Scene1_{mod}"
        ckpt_path = os.path.join("reproc", f"{exp_name}_last.pth")

        print(f"\n  > [{mod}] In-Domain Scene1")

        if (not force_retrain) and os.path.exists(ckpt_path):
            print(f"    Found checkpoint: {ckpt_path}  -> load & evaluate only")
            model = MODEL_REGISTRY[mod](num_classes=55).to(device)
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state)
            model.eval()
            acc = evaluate_model(model, test_loader, device)
            del model; torch.cuda.empty_cache(); gc.collect()
            res = {'Scene1': acc}
            print(f"     [Loaded] Scene1:{acc:.2f}%")
        else:
            print(f"     No checkpoint -> train and save to {ckpt_path}")
            res = train_and_eval(exp_name, MODEL_REGISTRY[mod],
                                 train_loader, eval_loaders, metric, device,
                                 epochs=epochs)

        results[mod] = float(res['Scene1'])
        print(f"    [OK] {mod}: {results[mod]:.2f}%")

    # ============== Summary ==============
    print("\n" + "="*80)
    print("In-Domain Scene1 Results")
    print("="*80)
    print("Train: Scene1 samples 1~14, all vols | Test: Scene1 samples 15~20, all vols")
    print(f"{'RFID':>10} | {'WiFi':>10} | {'mmWave':>10} | {'Fusion':>10}")
    print("-" * 50)
    print(f"{results.get('RFID', float('nan')):>9.2f}% | "
          f"{results.get('WiFi', float('nan')):>9.2f}% | "
          f"{results.get('mmWave', float('nan')):>9.2f}% | "
          f"{results.get('Fusion', float('nan')):>9.2f}%")

    save_result("in_domain_scene1", results)
    return results


# ============================================================
#  CROSS-SUBJECT 21-9  Pretrain -> Few-shot Fine-tune
#   - Source 21 vols, overlap vols {3,4,5,6,7,13,23,24} included
# ============================================================

def _finetune_eval(model_cls, pretrain_ckpt, ft_loader, eval_loader, device,
                   epochs=100, lr=1e-5, weight_decay=0.05,
                   label_smoothing=0.1, infonce_weight=0.1, log_every=10,
                   ft_seed=None,
                   save_path=None):
    set_reproducible_mode(ft_seed if ft_seed is not None else SEED)
    model = model_cls(num_classes=55).to(device)
    print(f"       Loading pretrain ckpt: {pretrain_ckpt}")
    model.load_state_dict(torch.load(pretrain_ckpt, map_location=device))

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    is_fusion = isinstance(model, FusionClassifier)

    for epoch in range(epochs):
        model.train()
        sum_loss, sum_ce, sum_nce, nce_batches = 0.0, 0.0, 0.0, 0
        for data, labels in ft_loader:
            wifi = data['WiFi'].to(device, non_blocking=True)
            mmw  = data['mmWave'].to(device, non_blocking=True)
            rfid = data['RFID'].to(device, non_blocking=True)
            tgt  = (labels - 1).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(wifi, mmw, rfid, apply_drop=is_fusion)
                ce_loss = criterion(out['logits'], tgt)
                nce_loss = None
                if 'nce_zs' in out and len(out['nce_zs']) >= 2:
                    nce_loss = component_info_nce(
                        out['nce_zs'], mode=out.get('nce_mode', 'all'), temperature=0.1)
                loss = ce_loss + (infonce_weight * nce_loss if nce_loss is not None else 0.0)
                if nce_loss is not None:
                    sum_nce += nce_loss.item(); nce_batches += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update()
            sum_loss += loss.item(); sum_ce += ce_loss.item()

        if (epoch + 1) % log_every == 0 or epoch == 0:
            n = len(ft_loader)
            extra = f" NCE:{sum_nce/max(1,nce_batches):.4f}" if nce_batches > 0 else ""
            print(f"      [FT Ep {epoch+1:03d}/{epochs}] "
                  f"Loss:{sum_loss/n:.4f} CE:{sum_ce/n:.4f}{extra}")

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


def _load_and_eval(model_cls, ckpt_path, eval_loader, device):
    model = model_cls(num_classes=55).to(device)
    print(f"       Loading FT ckpt: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    acc = evaluate_model(model, eval_loader, device)
    del model; torch.cuda.empty_cache(); gc.collect()
    return acc


# ============================================================
# Voluteer splits
# ============================================================
OVERLAP_S2 = [5, 24]              # (+ vol 31 is Scene2-only)
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


# ============================================================
# Main
# ============================================================
def run_cross_subject_21_9(root_path=ROOT, device=DEVICE,
                            pretrain_epochs=100,
                            ft_epochs=100, ft_lr=1e-5,
                            BS=16, NUM_WORKERS=4,
                            modalities=('RFID', 'WiFi', 'mmWave', 'Fusion'),
                            shots=(0, 1, 2, 3, 4, 5),
                            force_pretrain=False):
    all_vols = get_volunteers_in_scene(root_path, 'Scene1')
    SOURCE_VOLS, TARGET_VOLS = _split_source_target(root_path, 21, 9)

    print(f"Scene1 volunteers: {len(all_vols)} -> {all_vols}")
    print(f"   Overlap (S2-4, source included, 8 subjs.): {OVERLAP_VOLS}")
    print(f"     - Scene2 in S1: {OVERLAP_S2}  (+ vol 31 is S2-only)")
    print(f"     - Scene3 in S1: {OVERLAP_S3}")
    print(f"     - Scene4 in S1: {OVERLAP_S4}")
    print(f"   Source(21): {SOURCE_VOLS}")
    print(f"   Target(9) : {TARGET_VOLS}")

    results = {m: {} for m in modalities}

    for mod in modalities:
        print(f"\n{'='*80}\n [{mod}] Cross-Subject 21-9\n{'='*80}")
        pretrain_exp  = f"SCSub_21_9_{mod}_pretrain"
        pretrain_ckpt = os.path.join("reproc", f"{pretrain_exp}_last.pth")

        # ========== Stage 1: Pretrain ==========
        if os.path.exists(pretrain_ckpt) and not force_pretrain:
            print(f"  [Stage 1: Pretrain] [OK] -> {pretrain_ckpt}")
        else:
            print(f"  [Stage 1: Pretrain] 21 x 55 x 20 = {21*55*20:,} samples,  ep={pretrain_epochs}")
            g = fresh_generator()
            pretrain_loader = make_loader(root_path,
                {'Scene1': {'samples': list(range(1, 21)), 'vols': SOURCE_VOLS}},
                BS, NUM_WORKERS, shuffle=True, generator=g)
            sanity_loader = make_loader(root_path,
                {'Scene1': {'samples': [1], 'vols': SOURCE_VOLS}},
                BS, NUM_WORKERS, generator=g)
            _ = train_and_eval(pretrain_exp, MODEL_REGISTRY[mod],
                               pretrain_loader, {'sanity': sanity_loader},
                               lambda r: r['sanity'], device,
                               epochs=pretrain_epochs)
            print(f"    [OK] saved -> {pretrain_ckpt}")

        # ========== Stage 2: K-shot adapt + eval ==========
        for k in shots:
            print(f"\n  [Stage 2: {k}-shot] target = 9 vols")
            g = fresh_generator()

            if k == 0:
                eval_samples = list(range(1, 21))           # 9 x 55 x 20
                eval_loader = make_loader(root_path,
                    {'Scene1': {'samples': eval_samples, 'vols': TARGET_VOLS}},
                    BS, NUM_WORKERS, generator=g)
                print(f"    no fine-tune  (eval per-class = {len(eval_samples)})")
                acc = _zero_shot_eval(MODEL_REGISTRY[mod], pretrain_ckpt, eval_loader, device)
            else:
                eval_samples = list(range(k+1, 21))         # 9 x 55 x (20-K)
                eval_loader = make_loader(root_path,
                    {'Scene1': {'samples': eval_samples, 'vols': TARGET_VOLS}},
                    BS, NUM_WORKERS, generator=g)

                ft_save_path = os.path.join("reproc",
                                            f"SCSub_21_9_{mod}_{k}shot_ft.pth")

                if os.path.exists(ft_save_path):
                    print(f"    [{k}-shot] Found FT ckpt: {ft_save_path}  -> load & evaluate")
                    acc = _load_and_eval(MODEL_REGISTRY[mod], ft_save_path, eval_loader, device)
                else:
                    ft_loader = make_loader(root_path,
                        {'Scene1': {'samples': list(range(1, k+1)), 'vols': TARGET_VOLS}},
                        BS, NUM_WORKERS, shuffle=True, generator=g)
                    print(f"    fine-tune: 9 x 55 x {k} = {9*55*k} samples"
                          f"  (ep={ft_epochs}, lr={ft_lr})  | eval per-class = {len(eval_samples)}")
                    acc = _finetune_eval(MODEL_REGISTRY[mod], pretrain_ckpt,
                                         ft_loader, eval_loader, device,
                                         epochs=ft_epochs, lr=ft_lr,
                                         save_path=ft_save_path)

            results[mod][k] = {'Target9': acc}
            print(f"    [OK] [{mod} {k}-shot] acc = {acc:.2f}%")

    print_paper_table("Cross-Subject 21-9",
                      results, shots, eval_keys=['Target9'])
    save_result("cross_subject_21_9", {
        "overlap_vols": OVERLAP_VOLS,
        "source_vols": SOURCE_VOLS,
        "target_vols": TARGET_VOLS,
        "pretrain_samples_per_(vol,act)": 20,
        "pretrain_epochs": pretrain_epochs,
        "ft_epochs": ft_epochs, "ft_lr": ft_lr,
        "results": {m: {f"{k}shot": d for k, d in dd.items()}
                    for m, dd in results.items()}
    })
    return results


# Run


# ============================================================
# CROSS-SCENE (SCSub pretrain ckpt reuse)
#   - Pretrain ckpt:   reproc/SCSub_21_9_{mod}_pretrain_last.pth
#     - overlap {3,4,5,6,7,13,23,24} subj included
#
#   - Target scene vols (overlap subjects + Scene2 vol 31):
#       Scene2 <- {5, 24, 31}   5,24 observed during pretrain, 31 excluded
#       Scene3 <- {6, 7, 23}    observed during pretrain
#       Scene4 <- {3, 4, 13}    observed during pretrain
#   - target scene:
#       0-shot:  no fine tuning (FT)    -> 3 x 55 x 20 eval
#       1-shot:  fine-tune on 3 x 55 x 1   -> evaluate on 3 x 55 x 19
#       2-shot:  fine-tune on 3 x 55 x 2   -> evaluate on 3 x 55 x 18
# ============================================================

def run_cross_scene(root_path=ROOT, device=DEVICE,
                    ft_epochs=100, ft_lr=1e-5,
                    BS=16, NUM_WORKERS=4,
                    modalities=('RFID', 'WiFi', 'mmWave', 'Fusion'),
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
                print(f"WARNING:  strict mode -> {sc} vols {removed} excluded "
                      f"(pretrain source P_s  SCScen violation)")
    else:
        scene_vols = scene_vols_raw
        print("INFO:  lenient mode: include unseen user during pretrain")

    print("\n Final target scene vols:")
    for sc, vs in scene_vols.items():
        print(f"   {sc}: {vs}")

    results = {m: {k: {} for k in shots} for m in modalities}

    for mod in modalities:
        print(f"\n{'='*80}\n [{mod}] Cross-Scene\n{'='*80}")

        pretrain_ckpt = os.path.join("reproc", f"SCSub_21_9_{mod}_pretrain_last.pth")
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
                            f"\n  -> Run run_cross_subject_21_9() first or omit 0-shot."
                        )
                    eval_samples = list(range(1, 21))               # 3 x 55 x 20
                    eval_loader = make_loader(root_path,
                        {target_scene: {'samples': eval_samples, 'vols': target_vols}},
                        BS, NUM_WORKERS, generator=g)
                    print(f"    [{k}-shot] no FT  | eval per-class = {len(eval_samples)}")
                    acc = _zero_shot_eval(MODEL_REGISTRY[mod], pretrain_ckpt,
                                          eval_loader, device)
                else:
                    eval_samples = list(range(k+1, 21))             # 3 x 55 x (20-K)
                    eval_loader = make_loader(root_path,
                        {target_scene: {'samples': eval_samples, 'vols': target_vols}},
                        BS, NUM_WORKERS, generator=g)
                    ft_save_path = os.path.join("reproc",
                            f"CrossScene_{mod}_{target_scene}_{k}shot_ft.pth")
                    if os.path.exists(ft_save_path):
                        print(f"    [{k}-shot] Found FT ckpt: {ft_save_path}  -> load & evaluate")
                        model = MODEL_REGISTRY[mod](num_classes=55).to(device)
                        model.load_state_dict(torch.load(ft_save_path, map_location=device))
                        acc = evaluate_model(model, eval_loader, device)
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
                        print(f"    [{k}-shot] FT: {len(target_vols)} x 55 x {k} = {n_ft} samples"
                              f"  (ep={ft_epochs}, lr={ft_lr})"
                              f"  | eval per-class = {len(eval_samples)}")
                        acc = _finetune_eval(MODEL_REGISTRY[mod], pretrain_ckpt,
                                             ft_loader, eval_loader, device,
                                             epochs=ft_epochs, lr=ft_lr, save_path=ft_save_path)

                results[mod][k][target_scene] = acc
                print(f"      [OK] {mod} {target_scene} {k}-shot: acc = {acc:.2f}%")


    print_paper_table("Cross-Scene",
                      results, shots, eval_keys=target_scenes)

    save_result("cross_scene", {
        "target_scenes_vols": scene_vols,
        "pretrain_source": "SCSub_21_9 (21 vols including overlap {3,4,5,6,7,13,23,24})",
        "ft_epochs": ft_epochs, "ft_lr": ft_lr,
        "results": {m: {f"{k}shot": d for k, d in dd.items()}
                    for m, dd in results.items()}
    })
    return results


# ============================================================
# (Cross-Subject & Cross-Scene)
#   - new Pretrain: source = Scene1 \ {3,4,5,6,7,13,23,24}  (= 22 subj)
#       overlap subjects excluded in source subjs
#   - Target = Scene2/3/4
#       Scene2 <- {5, 24, 31}
#       Scene3 <- {6, 7, 23}
#       Scene4 <- {3, 4, 13}
#   - 0/1/2-shot x 20/19/18 eval per (vol, action)
# ============================================================

def run_cscs(root_path=ROOT, device=DEVICE,
             pretrain_epochs=100,
             ft_epochs=100, ft_lr=1e-5,
             BS=16, NUM_WORKERS=4,
             modalities=('RFID', 'WiFi', 'mmWave', 'Fusion'),
             shots=(0, 1, 2, 3, 4, 5),
             force_pretrain=False):
    target_scenes = ['Scene2', 'Scene3', 'Scene4']
    scene_vols = {sc: get_volunteers_in_scene(root_path, sc) for sc in target_scenes}

    # ----- Source: Scene1 \ (S2 union S3 union S4) -----
    all_vols = get_volunteers_in_scene(root_path, 'Scene1')
    contaminated = set()
    for sc in target_scenes:
        contaminated.update(scene_vols[sc])     # {3,4,5,6,7,13,23,24} (+31 is not in Scene1)
    SOURCE_VOLS = sorted([v for v in all_vols if v not in contaminated])

    print("CSC&S setup:")
    print(f"   Scene1 vols total      : {len(all_vols)}")
    print(f"   Contaminated (source exclusion): {sorted(contaminated)}")
    print(f"   Source ({len(SOURCE_VOLS)}subjs): {SOURCE_VOLS}")
    print(f"   Target scenes (with New subjects):")
    for sc, vs in scene_vols.items():
        print(f"     {sc}: {vs}")

    results = {m: {k: {} for k in shots} for m in modalities}

    for mod in modalities:
        print(f"\n{'='*80}\n [{mod}] CSC&S\n{'='*80}")
        pretrain_exp  = f"CSCS_{mod}_pretrain"
        pretrain_ckpt = os.path.join("reproc", f"{pretrain_exp}_last.pth")

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
            _ = train_and_eval(pretrain_exp, MODEL_REGISTRY[mod],
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
                    acc = _zero_shot_eval(MODEL_REGISTRY[mod], pretrain_ckpt,
                                          eval_loader, device)
                else:
                    eval_samples = list(range(k+1, 21))
                    eval_loader = make_loader(root_path,
                        {target_scene: {'samples': eval_samples, 'vols': target_vols}},
                        BS, NUM_WORKERS, generator=g)
                    ft_save_path = os.path.join("reproc",
                            f"CSCS_{mod}_{target_scene}_{k}shot_ft.pth")
                    if os.path.exists(ft_save_path):
                        print(f"    [{k}-shot] Found FT ckpt: {ft_save_path}  -> load & evaluate")
                        acc = _load_and_eval(MODEL_REGISTRY[mod], ft_save_path, eval_loader, device)
                    else:
                        ft_loader = make_loader(root_path,
                            {target_scene: {'samples': list(range(1, k+1)),
                                            'vols': target_vols}},
                            BS, NUM_WORKERS, shuffle=True, generator=g)
                        n_ft = len(target_vols) * 55 * k
                        print(f"    [{k}-shot] FT: {len(target_vols)} x 55 x {k} = {n_ft}"
                              f"  (ep={ft_epochs}, lr={ft_lr})"
                              f"  | eval per-class = {len(eval_samples)}")
                        acc = _finetune_eval(MODEL_REGISTRY[mod], pretrain_ckpt,
                                             ft_loader, eval_loader, device,
                                             epochs=ft_epochs, lr=ft_lr, save_path=ft_save_path)

                results[mod][k][target_scene] = acc
                print(f"      [OK] {mod} {target_scene} {k}-shot: acc = {acc:.2f}%")

    print_paper_table("CSC&S",
                      results, shots, eval_keys=target_scenes)
    save_result("cscs", {
        "source_vols": SOURCE_VOLS,
        "excluded_vols": sorted(contaminated),
        "target_scenes_vols": scene_vols,
        "pretrain_epochs": pretrain_epochs,
        "ft_epochs": ft_epochs, "ft_lr": ft_lr,
        "results": {m: {f"{k}shot": d for k, d in dd.items()}
                    for m, dd in results.items()}
    })
    return results


# Ablation

# ============================================================
# CSC&S Ablation Study
# ============================================================
# Reuses the CSC&S setup.
# For each ablation variant:
#   - swap in a different FusionClassifier subclass
#   - keep pretrain ckpts separate by naming them ABL_{tag}_Fusion_pretrain
#   - use the same source (Scene1 \ contaminated) and same target (S2/3/4)
#
# ============================================================

# 1) Ablation variant definitions
#    Each variant subclasses FusionClassifier and changes exactly one thing.
#    Base (current proposed) = drop ON + adjacent InfoNCE on 6 components
# ---- (A) Remove Modality Drop ----------------------------------------------
class FusionClassifier_NoDrop(FusionClassifier):
    """Drop disabled. InfoNCE unchanged."""
    def forward(self, wifi_x, mmw_x, rfid_x, apply_drop=False):
        return super().forward(wifi_x, mmw_x, rfid_x, apply_drop=False)


# ---- (B) Remove InfoNCE ----------------------------------------------------
class FusionClassifier_NoInfoNCE(FusionClassifier):
    """Drop unchanged. We drop the InfoNCE projection outputs, effectively making the NCE loss zero."""
    def forward(self, wifi_x, mmw_x, rfid_x, apply_drop=False):
        out = super().forward(wifi_x, mmw_x, rfid_x, apply_drop=apply_drop)
        # Removing nce_zs -> train_and_eval / _finetune_eval automatically skip the NCE loss
        out.pop('nce_zs', None)
        out.pop('nce_mode', None)
        return out


# ---- (C) NoDrop + NoInfoNCE (both off = pure concat baseline) --------------
class FusionClassifier_NoDropNoInfoNCE(FusionClassifier):
    def forward(self, wifi_x, mmw_x, rfid_x, apply_drop=False):
        out = super().forward(wifi_x, mmw_x, rfid_x, apply_drop=False)
        out.pop('nce_zs', None)
        out.pop('nce_mode', None)
        return out


# ---- (D) InfoNCE at the 3-modality level instead of 6-component adjacent ---
# Merge the three WiFi APs and the mmWave RD/RA into single features ->
# [WiFi, mmWave, RFID], i.e. 3 components.
# adjacent: WiFi-mmWave, mmWave-RFID
class FusionClassifier_3CompInfoNCE(FusionClassifier):
    """Apply InfoNCE only at the modality level (3 components)."""
    def __init__(self, num_classes=55, proj_dim=128, **kw):
        super().__init__(num_classes=num_classes, proj_dim=proj_dim, **kw)
        # modality-level projection heads (sized to match the concatenated full-feature dim)
        self.wifi_full_proj = make_projection_head(self.wifi_dim * 3, proj_dim)
        self.mmw_full_proj  = make_projection_head(self.mmw_dim * 2, proj_dim)
        # rfid_proj already exists in the parent (rfid_E=96)

    def forward(self, wifi_x, mmw_x, rfid_x, apply_drop=False):
        # Replicate the parent forward, but expose the modality-level concat features for projection
        wifi_chunks = torch.split(wifi_x, wifi_x.shape[2] // 3, dim=2)
        wifi_feats = [b(wifi_chunks[i]) for i, b in enumerate(self.wifi_branches)]
        wifi_full = torch.cat(wifi_feats, dim=1)

        x_rd, x_ra = torch.split(mmw_x, 64, dim=3)
        B, T, H, W = x_rd.shape

        c_rd = self.mmw_cnn_rd(x_rd.reshape(B * T, 1, H, W)).flatten(2).transpose(1, 2)
        c_rd = self.mmw_s_rd(c_rd).max(dim=1)[0].view(B, T, -1)
        feat_rd = self.mmw_t_rd(c_rd).max(dim=1)[0]

        c_ra = self.mmw_cnn_ra(x_ra.reshape(B * T, 1, H, W)).flatten(2).transpose(1, 2)
        c_ra = self.mmw_s_ra(c_ra).max(dim=1)[0].view(B, T, -1)
        feat_ra = self.mmw_t_ra(c_ra).max(dim=1)[0]

        mmw_full = torch.cat([feat_rd, feat_ra], dim=1)

        rfid_c = self.rfid_cnn(rfid_x.squeeze(1)).transpose(1, 2)
        rfid_full = self.rfid_mamba(rfid_c)

        if apply_drop and self.training:
            Bsz = wifi_full.shape[0]
            dc = torch.randint(0, 3, (Bsz,), device=wifi_full.device)
            wifi_d = wifi_full * (dc != 0).float().unsqueeze(1)
            mmw_d  = mmw_full  * (dc != 1).float().unsqueeze(1)
            rfid_d = rfid_full * (dc != 2).float().unsqueeze(1)
        else:
            wifi_d, mmw_d, rfid_d = wifi_full, mmw_full, rfid_full

        logits = self.head(torch.cat([wifi_d, mmw_d, rfid_d], dim=1))
        out = {'logits': logits}

        if self.training:
            # 3-component (modality-level) InfoNCE on PRE-drop features
            nce_zs = [
                F.normalize(self.wifi_full_proj(wifi_full), dim=-1),
                F.normalize(self.mmw_full_proj(mmw_full), dim=-1),
                F.normalize(self.rfid_proj(rfid_full), dim=-1),
            ]
            out.update({'nce_zs': nce_zs, 'nce_mode': 'adjacent'})
        return out


# ---- (E) InfoNCE: adjacent -> all-pairs (across all 6 components) ----------
class FusionClassifier_AllPairsInfoNCE(FusionClassifier):
    """6-component InfoNCE, but as all-pairs instead of adjacent."""
    def forward(self, wifi_x, mmw_x, rfid_x, apply_drop=False):
        out = super().forward(wifi_x, mmw_x, rfid_x, apply_drop=apply_drop)
        if 'nce_mode' in out:
            out['nce_mode'] = 'all'
        return out


# ---- (F) Replace Mamba with BiMamba (VMamba-style: forward + flip) ---------
# Simplest 'V/Bi-Mamba' approximation: process the input in both directions and merge.
class BiMambaLayer(nn.Module):
    """Bidirectional wrapper that merges the forward and reversed-sequence outputs.
       A 1D simplification of the cross-scan idea VMamba uses in the vision domain."""
    def __init__(self, embed_dim):
        super().__init__()
        self.fwd = Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2)
        self.bwd = Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2)
        self.proj = nn.Linear(embed_dim * 2, embed_dim)
    def forward(self, x):
        y_f = self.fwd(x)
        y_b = self.bwd(torch.flip(x, dims=[1]))
        y_b = torch.flip(y_b, dims=[1])
        return self.proj(torch.cat([y_f, y_b], dim=-1))

def make_bimamba_layers(embed_dim, depth, use_dropout=False, dropout=0.2):
    layers = []
    for _ in range(depth):
        ld = {'norm': nn.LayerNorm(embed_dim),
              'mamba': BiMambaLayer(embed_dim)}
        if use_dropout: ld['drop'] = nn.Dropout(dropout)
        layers.append(nn.ModuleDict(ld))
    return nn.ModuleList(layers)

# Branches that use the BiMamba above (same structure as the existing classes, mamba layer swapped)
class _WiFiPhysBranch_BiMamba(nn.Module):
    def __init__(self, embed_dim=96, time_tokens=20, depth=3, mid_dim=48):
        super().__init__()
        self.stem = stem_depthsep_wifi(1, mid_dim, embed_dim, time_tokens)
        num_patches = 3 * time_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.layers = make_bimamba_layers(embed_dim, depth, use_dropout=False)
        self.norm_f = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.stem(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = run_mamba_residual(x, self.layers)
        return self.norm_f(x).mean(dim=1)

class mmw_BiMambaBranch(nn.Module):
    def __init__(self, num_patches, embed_dim, depth):
        super().__init__()
        self.in_norm = nn.LayerNorm(embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.layers = make_bimamba_layers(embed_dim, depth, use_dropout=True)
        self.norm_f = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.in_norm(x) + self.pos_embed
        x = run_mamba_residual(x, self.layers)
        return self.norm_f(x)

class rfid_BiMambaBranch(nn.Module):
    def __init__(self, num_patches, embed_dim, depth):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.layers = make_bimamba_layers(embed_dim, depth, use_dropout=False)
        self.norm_f = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = x + self.pos_embed
        x = run_mamba_residual(x, self.layers)
        return self.norm_f(x).mean(dim=1)

class FusionClassifier_BiMamba(FusionClassifier):
    """Replace every Mamba layer with BiMamba (VMamba-style 1D approximation)."""
    def __init__(self, num_classes=55, proj_dim=128,
                 wifi_E=96, wifi_T=20, wifi_D=3, wifi_mid=48,
                 mmw_E=64, mmw_D=2, rfid_E=96, rfid_D=4):
        # Call the parent init first (structure is identical)
        super().__init__(num_classes=num_classes, proj_dim=proj_dim,
                         wifi_E=wifi_E, wifi_T=wifi_T, wifi_D=wifi_D, wifi_mid=wifi_mid,
                         mmw_E=mmw_E, mmw_D=mmw_D, rfid_E=rfid_E, rfid_D=rfid_D)
        # Swap in the BiMamba versions of each branch
        self.wifi_branches = nn.ModuleList([
            _WiFiPhysBranch_BiMamba(wifi_E, wifi_T, wifi_D, wifi_mid) for _ in range(3)
        ])
        self.mmw_s_rd = mmw_BiMambaBranch(64, mmw_E, depth=mmw_D)
        self.mmw_t_rd = mmw_BiMambaBranch(17, mmw_E, depth=mmw_D)
        self.mmw_s_ra = mmw_BiMambaBranch(64, mmw_E, depth=mmw_D)
        self.mmw_t_ra = mmw_BiMambaBranch(17, mmw_E, depth=mmw_D)
        self.rfid_mamba = rfid_BiMambaBranch(
            num_patches=(148 // 2), embed_dim=rfid_E, depth=rfid_D)


# ---- (G) Drop: forced-exactly-one (current) vs 0~3 independent drops -------
class FusionClassifier_DropAny(FusionClassifier):
    """Drop each modality independently with probability p=1/3 (so 0~3 modalities may be dropped)."""
    def forward(self, wifi_x, mmw_x, rfid_x, apply_drop=False):
        wifi_chunks = torch.split(wifi_x, wifi_x.shape[2] // 3, dim=2)
        wifi_feats = [b(wifi_chunks[i]) for i, b in enumerate(self.wifi_branches)]
        wifi_full = torch.cat(wifi_feats, dim=1)

        x_rd, x_ra = torch.split(mmw_x, 64, dim=3)
        B, T, H, W = x_rd.shape
        c_rd = self.mmw_cnn_rd(x_rd.reshape(B * T, 1, H, W)).flatten(2).transpose(1, 2)
        c_rd = self.mmw_s_rd(c_rd).max(dim=1)[0].view(B, T, -1)
        feat_rd = self.mmw_t_rd(c_rd).max(dim=1)[0]
        c_ra = self.mmw_cnn_ra(x_ra.reshape(B * T, 1, H, W)).flatten(2).transpose(1, 2)
        c_ra = self.mmw_s_ra(c_ra).max(dim=1)[0].view(B, T, -1)
        feat_ra = self.mmw_t_ra(c_ra).max(dim=1)[0]
        mmw_full = torch.cat([feat_rd, feat_ra], dim=1)

        rfid_c = self.rfid_cnn(rfid_x.squeeze(1)).transpose(1, 2)
        rfid_full = self.rfid_mamba(rfid_c)

        if apply_drop and self.training:
            Bsz = wifi_full.shape[0]
            m_w = (torch.rand(Bsz, device=wifi_full.device) > 1/3).float().unsqueeze(1)
            m_m = (torch.rand(Bsz, device=wifi_full.device) > 1/3).float().unsqueeze(1)
            m_r = (torch.rand(Bsz, device=wifi_full.device) > 1/3).float().unsqueeze(1)
            wifi_d = wifi_full * m_w
            mmw_d  = mmw_full  * m_m
            rfid_d = rfid_full * m_r
        else:
            wifi_d, mmw_d, rfid_d = wifi_full, mmw_full, rfid_full

        logits = self.head(torch.cat([wifi_d, mmw_d, rfid_d], dim=1))
        out = {'logits': logits}
        if self.training:
            nce_zs = [
                F.normalize(self.wifi_branch_proj(wifi_feats[0]), dim=-1),
                F.normalize(self.wifi_branch_proj(wifi_feats[1]), dim=-1),
                F.normalize(self.wifi_branch_proj(wifi_feats[2]), dim=-1),
                F.normalize(self.mmw_stream_proj(feat_rd), dim=-1),
                F.normalize(self.mmw_stream_proj(feat_ra), dim=-1),
                F.normalize(self.rfid_proj(rfid_full), dim=-1),
            ]
            out.update({'nce_zs': nce_zs, 'nce_mode': 'adjacent'})
        return out


# ---- (H) Component-level Drop (drop 1 of 6 components, not 1 of 3 modalities)
#   - WiFi : AP1, AP2, AP3   (each of dim wifi_E)
#   - mmWave: feat_rd, feat_ra (each of dim mmw_E)
#   - RFID : rfid_full       (rfid_E dim)
#   For each sample, randomly pick 1 of the 6 components and zero it out.
#   InfoNCE stays at 6-comp adjacent (same as base) on the PRE-drop features.
class FusionClassifier_CompDrop(FusionClassifier):
    """6-component drop. Only the drop location differs; InfoNCE stays 6-comp adjacent (same as parent)."""

    def forward(self, wifi_x, mmw_x, rfid_x, apply_drop=False):
        # ----- WiFi: 3 branch features -----
        wifi_chunks = torch.split(wifi_x, wifi_x.shape[2] // 3, dim=2)
        wifi_feats = [b(wifi_chunks[i]) for i, b in enumerate(self.wifi_branches)]
        # [AP1, AP2, AP3]  each (B, wifi_E)

        # ----- mmWave: RD/RA stream features -----
        x_rd, x_ra = torch.split(mmw_x, 64, dim=3)
        B, T, H, W = x_rd.shape

        c_rd = self.mmw_cnn_rd(x_rd.reshape(B * T, 1, H, W)).flatten(2).transpose(1, 2)
        c_rd = self.mmw_s_rd(c_rd).max(dim=1)[0].view(B, T, -1)
        feat_rd = self.mmw_t_rd(c_rd).max(dim=1)[0]

        c_ra = self.mmw_cnn_ra(x_ra.reshape(B * T, 1, H, W)).flatten(2).transpose(1, 2)
        c_ra = self.mmw_s_ra(c_ra).max(dim=1)[0].view(B, T, -1)
        feat_ra = self.mmw_t_ra(c_ra).max(dim=1)[0]

        # ----- RFID: one branch feature -----
        rfid_c = self.rfid_cnn(rfid_x.squeeze(1)).transpose(1, 2)
        rfid_full = self.rfid_mamba(rfid_c)

        # ----- Pre-drop copies for InfoNCE (kept as 6 components regardless of drop) -----
        ap1_pre, ap2_pre, ap3_pre = wifi_feats[0], wifi_feats[1], wifi_feats[2]
        rd_pre, ra_pre = feat_rd, feat_ra
        rfid_pre = rfid_full

        # ----- Component-level Drop_only: zero out 1 of 6 components -----
        # Index map: 0=AP1, 1=AP2, 2=AP3, 3=RD, 4=RA, 5=RFID
        if apply_drop and self.training:
            Bsz = wifi_feats[0].shape[0]
            dc = torch.randint(0, 6, (Bsz,), device=wifi_feats[0].device)

            ap1_d = wifi_feats[0] * (dc != 0).float().unsqueeze(1)
            ap2_d = wifi_feats[1] * (dc != 1).float().unsqueeze(1)
            ap3_d = wifi_feats[2] * (dc != 2).float().unsqueeze(1)
            rd_d  = feat_rd       * (dc != 3).float().unsqueeze(1)
            ra_d  = feat_ra       * (dc != 4).float().unsqueeze(1)
            rfid_d = rfid_full    * (dc != 5).float().unsqueeze(1)
        else:
            ap1_d, ap2_d, ap3_d = wifi_feats[0], wifi_feats[1], wifi_feats[2]
            rd_d,  ra_d         = feat_rd, feat_ra
            rfid_d              = rfid_full

        wifi_full = torch.cat([ap1_d, ap2_d, ap3_d], dim=1)
        mmw_full  = torch.cat([rd_d, ra_d], dim=1)
        rfid_full_d = rfid_d

        logits = self.head(torch.cat([wifi_full, mmw_full, rfid_full_d], dim=1))

        out = {'logits': logits}

        # ----- InfoNCE: pre-drop 6-component adjacent (same as base) -----
        if self.training:
            nce_zs = [
                F.normalize(self.wifi_branch_proj(ap1_pre), dim=-1),
                F.normalize(self.wifi_branch_proj(ap2_pre), dim=-1),
                F.normalize(self.wifi_branch_proj(ap3_pre), dim=-1),
                F.normalize(self.mmw_stream_proj(rd_pre),   dim=-1),
                F.normalize(self.mmw_stream_proj(ra_pre),   dim=-1),
                F.normalize(self.rfid_proj(rfid_pre),       dim=-1),
            ]
            out.update({'nce_zs': nce_zs, 'nce_mode': 'adjacent'})

        return out


# 2) Ablation runner
#    Reuse run_cscs as-is, but temporarily swap MODEL_REGISTRY['Fusion']
#    and use a different prefix so pretrain ckpt paths do not collide.
#    -> Simplest approach: vary the modality key. run_cscs forms ckpt names
#       as f"CSCS_{mod}_pretrain", so making the mod name unique naturally
#       separates them. Register a 'Fusion_<tag>' key in MODEL_REGISTRY and
#       call with modalities=(that key,).
ABLATIONS = {
    # tag                : (FusionClass,                          description)
    'base'               : (FusionClassifier,                     "Proposed (drop ON + adjacent 6-comp InfoNCE)"),
    'no_drop'            : (FusionClassifier_NoDrop,              "Drop OFF, InfoNCE ON"),
    'no_infonce'         : (FusionClassifier_NoInfoNCE,           "Drop ON, InfoNCE OFF"),
    'no_drop_no_infonce' : (FusionClassifier_NoDropNoInfoNCE,     "Drop OFF, InfoNCE OFF (pure concat)"),
    'infonce_3comp'      : (FusionClassifier_3CompInfoNCE,        "InfoNCE on 3 modality-level comps (not 6)"),
    'infonce_allpairs'   : (FusionClassifier_AllPairsInfoNCE,     "InfoNCE adjacent -> all-pairs"),
    'bimamba'            : (FusionClassifier_BiMamba,             "Mamba -> BiMamba (VMamba-style 1D)"),
    'drop_any'           : (FusionClassifier_DropAny,             "Drop: forced-one -> independent p=1/3 per modality"),
    'comp_drop'          : (FusionClassifier_CompDrop,            "6-component drop (1 of {AP1,AP2,AP3,RD,RA,RFID}) + adjacent InfoNCE"),
}


def run_cscs_ablation(tags=None, **cscs_kwargs):
    """tags=None runs every variant in ABLATIONS."""
    if tags is None:
        tags = list(ABLATIONS.keys())

    all_results = {}
    for tag in tags:
        assert tag in ABLATIONS, f"Unknown ablation tag: {tag}"
        cls, desc = ABLATIONS[tag]

        # Register a unique modality key -> ckpt names inside run_cscs are automatically separated
        mod_key = f"Fusion_{tag}"
        MODEL_REGISTRY[mod_key] = cls

        print(f"\n{'#'*80}")
        print(f"# ABLATION [{tag}]  ({desc})")
        print(f"# Pretrain ckpt -> reproc/CSCS_{mod_key}_pretrain_last.pth")
        print(f"{'#'*80}")

        try:
            results = run_cscs(modalities=(mod_key,), **cscs_kwargs)
            all_results[tag] = results
        except Exception as e:
            print(f"[ERROR] Ablation [{tag}] failed: {e}")
            import traceback; traceback.print_exc()
            all_results[tag] = {'error': str(e)}
        finally:
            # Clean up the registry so the next ablation is unaffected
            MODEL_REGISTRY.pop(mod_key, None)
            torch.cuda.empty_cache(); gc.collect()

    # ----- Summary table -----
    print("\n" + "="*80)
    print("CSC&S Ablation Summary")
    print("="*80)
    target_scenes = ['Scene2', 'Scene3', 'Scene4']
    shots = cscs_kwargs.get('shots', (0, 1, 2))

    header = f"{'Tag':<22} | {'Shot':<5} | " + " | ".join(f"{s:<8}" for s in target_scenes) + " | Avg"
    print(header)
    print("-" * len(header))
    for tag in tags:
        res = all_results.get(tag, {})
        if 'error' in res:
            print(f"{tag:<22} | ERROR: {res['error']}")
            continue
        mod_key = f"Fusion_{tag}"
        for k in shots:
            row = res.get(mod_key, {}).get(k, {})
            vals = [row.get(sc, float('nan')) for sc in target_scenes]
            avg = sum(v for v in vals if v == v) / max(1, sum(1 for v in vals if v == v))
            vals_str = " | ".join(f"{v:7.2f}%" for v in vals)
            print(f"{tag:<22} | {k}-shot| {vals_str} | {avg:6.2f}%")
        print("-" * len(header))

    save_result("cscs_ablation_summary", {
        "tags": tags,
        "descriptions": {t: ABLATIONS[t][1] for t in tags},
        "results": {t: {f"Fusion_{t}": {f"{k}shot": d for k, d in dd.items()}
                        for _, dd in (r.items() if isinstance(r, dict) and 'error' not in r else [])}
                    for t, r in all_results.items()},
    })
    return all_results


# ============================================================
# Sensor-Failure Robustness - using 5-shot FT ckpts
#   FT ckpt path pattern:
#     reproc/CSCS_{mod_key}_{scene}_{k}shot_ft.pth
#   where mod_key = "Fusion_<tag>"  (from run_cscs_ablation)
#   For tag="base", the plain CSC&S checkpoint
#     reproc/CSCS_Fusion_{scene}_{k}shot_ft.pth
#   is also accepted.
#
#   Masking is done at the modality-level concat feature stage
#   Note: evaluation uses range(SHOT+1, 21) - samples 1..SHOT
#   were used to fine-tune the FT checkpoints and must be excluded
#   from evaluation to avoid data leakage. With SHOT=5 this gives
#   15 held-out samples per (vol, action), matching the CSC&S k-shot
#   evaluation setting exactly so that "all (no fail)" equals the 5-shot Fusion.
# ============================================================
import glob, csv

# 1) Modality-masked forward
#    Mimics FusionClassifier.forward, but zeroes out the
#    modality-level concat feature of any modality not in `keep`.
@torch.no_grad()
def fusion_forward_masked(model, wifi_x, mmw_x, rfid_x,
                          keep=('W', 'M', 'R')):
    """keep: set of modalities to keep alive. e.g. ('W','R') zeros out mmWave
       at the modality-level concat feature (same as the training-time drop)."""
    model.eval()
    keep = set(keep)

    # ---- backbone forward (copied from FusionClassifier.forward) ----
    wifi_chunks = torch.split(wifi_x, wifi_x.shape[2] // 3, dim=2)
    wifi_feats = [b(wifi_chunks[i]) for i, b in enumerate(model.wifi_branches)]
    wifi_full = torch.cat(wifi_feats, dim=1)

    x_rd, x_ra = torch.split(mmw_x, 64, dim=3)
    B, T, H, W = x_rd.shape
    c_rd = model.mmw_cnn_rd(x_rd.reshape(B * T, 1, H, W)).flatten(2).transpose(1, 2)
    c_rd = model.mmw_s_rd(c_rd).max(dim=1)[0].view(B, T, -1)
    feat_rd = model.mmw_t_rd(c_rd).max(dim=1)[0]
    c_ra = model.mmw_cnn_ra(x_ra.reshape(B * T, 1, H, W)).flatten(2).transpose(1, 2)
    c_ra = model.mmw_s_ra(c_ra).max(dim=1)[0].view(B, T, -1)
    feat_ra = model.mmw_t_ra(c_ra).max(dim=1)[0]
    mmw_full = torch.cat([feat_rd, feat_ra], dim=1)

    rfid_c = model.rfid_cnn(rfid_x.squeeze(1)).transpose(1, 2)
    rfid_full = model.rfid_mamba(rfid_c)

    # Feature-level masking (the only mode now).
    if 'W' not in keep: wifi_full = torch.zeros_like(wifi_full)
    if 'M' not in keep: mmw_full  = torch.zeros_like(mmw_full)
    if 'R' not in keep: rfid_full = torch.zeros_like(rfid_full)

    logits = model.head(torch.cat([wifi_full, mmw_full, rfid_full], dim=1))
    return logits


@torch.no_grad()
def evaluate_masked(model, loader, device, keep):
    model.eval()
    c, t = 0, 0
    for data, labels in loader:
        wifi = data['WiFi'].to(device, non_blocking=True)
        mmw  = data['mmWave'].to(device, non_blocking=True)
        rfid = data['RFID'].to(device, non_blocking=True)
        tgt  = (labels - 1).to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = fusion_forward_masked(model, wifi, mmw, rfid, keep=keep)
        c += logits.max(1)[1].eq(tgt).sum().item()
        t += tgt.size(0)
    return 100.0 * c / t if t > 0 else 0.0


# 2) Keep combinations
KEEP_COMBOS = [
    ('W',),          # WiFi only
    ('M',),          # mmWave only
    ('R',),          # RFID only
    ('W', 'M'),      # RFID failed
    ('W', 'R'),      # mmWave failed
    ('M', 'R'),      # WiFi failed
    ('W', 'M', 'R'), # all normal (baseline)
]
KEEP_LABEL = {
    ('W',):         'W only',
    ('M',):         'M only',
    ('R',):         'R only',
    ('W','M'):      'W+M (R fail)',
    ('W','R'):      'W+R (M fail)',
    ('M','R'):      'M+R (W fail)',
    ('W','M','R'):  'all (no fail)',
}


def eval_one_ckpt(model_cls, ckpt_path, eval_loaders_per_scene, device):
    """eval_loaders_per_scene: {'Scene2': loader, 'Scene3': ..., 'Scene4': ...}
       Returns dict[keep_label] -> {scene: acc, ..., 'avg': acc}."""
    model = model_cls(num_classes=55).to(device)
    print(f"  loading: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    out = {}
    for keep in KEEP_COMBOS:
        per_scene = {}
        for sc, loader in eval_loaders_per_scene.items():
            per_scene[sc] = evaluate_masked(model, loader, device, keep)
        per_scene['avg'] = sum(per_scene.values()) / len(per_scene)
        out[KEEP_LABEL[keep]] = per_scene

    del model; torch.cuda.empty_cache(); gc.collect()
    return out


# 3) Eval loader builder
#    CSC&S target = S2/S3/S4, evaluated on data the FT model has NOT seen.
#    Samples 1..SHOT were used for fine-tuning, so eval = range(SHOT+1, 21).
SHOT = 5

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


# Always (re)build EVAL_LOADERS - a stale loader from a previous run could be
# using a different eval range (e.g. range(1, 21)) and would silently let FT
# samples leak into evaluation. Better to rebuild than risk it.

def _mean_sensor_failure(raw, keep_labels):
    out = {label: [] for label in keep_labels}
    for scene_d in raw.values():
        for scene_vals in scene_d.values():
            for label in keep_labels:
                val = scene_vals.get(label, float("nan"))
                if val == val:
                    out[label].append(val)
    return {
        label: (sum(vals) / len(vals) if vals else float("nan"))
        for label, vals in out.items()
    }


def run_sensor_failure(root_path=ROOT, device=DEVICE,
                       BS=16, NUM_WORKERS=4, shot=5, tags=None):
    """Evaluate missing-modality robustness using CSC&S shot-specific FT ckpts."""
    tag_to_model = {
        'base': FusionClassifier,
        'comp_drop': FusionClassifier_CompDrop,
        'no_drop': FusionClassifier_NoDrop,
        'no_drop_no_infonce': FusionClassifier_NoDropNoInfoNCE,
        'no_infonce': FusionClassifier_NoInfoNCE,
        'infonce_3comp': FusionClassifier_3CompInfoNCE,
        'infonce_allpairs': FusionClassifier_AllPairsInfoNCE,
        'drop_any': FusionClassifier_DropAny,
        'bimamba': FusionClassifier_BiMamba,
    }
    selected_tags = list(tags) if tags else list(tag_to_model)
    unknown = sorted(set(selected_tags) - set(tag_to_model))
    if unknown:
        raise ValueError(f"Unknown ablation tag(s): {unknown}")

    eval_loaders = build_cscs_eval_loaders(root_path, BS=BS, NUM_WORKERS=NUM_WORKERS, shot=shot)
    raw = {tag: {} for tag in selected_tags}

    def ckpt_candidates(tag, scene):
        if tag == "base":
            return [
                f"reproc/CSCS_Fusion_{scene}_{shot}shot_ft.pth",
                f"reproc/CSCS_Fusion_{tag}_{scene}_{shot}shot_ft.pth",
            ]
        return [f"reproc/CSCS_Fusion_{tag}_{scene}_{shot}shot_ft.pth"]

    for tag in selected_tags:
        model_cls = tag_to_model[tag]
        for scene in ['Scene2', 'Scene3', 'Scene4']:
            candidates = ckpt_candidates(tag, scene)
            ckpt = next((path for path in candidates if os.path.exists(path)), None)
            if ckpt is None:
                print(f"[skip] missing checkpoint: {candidates[0]}")
                for alt in candidates[1:]:
                    print(f"       fallback also missing: {alt}")
                continue
            print(f"[sensor_failure] {tag} {scene}: {ckpt}")
            model = model_cls(num_classes=55).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            scene_results = {}
            for keep in KEEP_COMBOS:
                scene_results[KEEP_LABEL[keep]] = evaluate_masked(
                    model, eval_loaders[scene], device, keep)
            raw[tag][scene] = scene_results
            del model; torch.cuda.empty_cache(); gc.collect()

    keep_labels = [KEEP_LABEL[k] for k in KEEP_COMBOS]
    for tag in selected_tags:
        if not raw[tag]:
            print(f"{tag}: no checkpoints found")
            continue
        agg = _mean_sensor_failure({tag: raw[tag]}, keep_labels)
        full = agg.get('all (no fail)', float("nan"))
        one_fail = [agg[k] for k in ['W+M (R fail)', 'W+R (M fail)', 'M+R (W fail)']
                    if agg[k] == agg[k]]
        one_avg = sum(one_fail) / len(one_fail) if one_fail else float("nan")
        print(f"{tag}: full={full:.2f}%  one-fail-avg={one_avg:.2f}%")

    csv_path = f"reproc/results/sensor_failure_{shot}shot.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tag", "scene", "keep", "acc"])
        for tag, scene_d in raw.items():
            for scene, keep_d in scene_d.items():
                for keep_label, acc in keep_d.items():
                    writer.writerow([tag, scene, keep_label, f"{acc:.4f}"])
    save_result(f"sensor_failure_{shot}shot", raw)
    return raw
