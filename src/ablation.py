#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import os, json, copy, warnings, gc
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import shutil
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc as pr_auc
from transformers import get_linear_schedule_with_warmup

warnings.filterwarnings("once")


# =========================
# CONFIG (更新为与文件二相同)
# =========================

@dataclass
class Config:
    # ---- optional: convert from pth to memmap ----
    do_memmap_convert: bool = False
    memmap_dtype: str = "float16"  # float16/float32

    # 8 feature pth paths (only used if do_memmap_convert=True)
    train_gpn_ref_pth: str = r"/home/yufei/20251210up/feature/GPN-MSA/train_GPN-MSA_ref.pth"
    train_gpn_alt_pth: str = r"/home/yufei/20251210up/feature/GPN-MSA/train_GPN-MSA_alt.pth"
    test_gpn_ref_pth:  str = r"/home/yufei/20251210up/feature/GPN-MSA/test_GPN-MSA_ref.pth"
    test_gpn_alt_pth:  str = r"/home/yufei/20251210up/feature/GPN-MSA/test_GPN-MSA_alt.pth"
    train_t5_wt_pth:   str = r"/home/yufei/20251210up/feature/protT5-XL/train_protT5-XL_wt.pth"
    train_t5_mut_pth:  str = r"/home/yufei/20251210up/feature/protT5-XL/train_protT5-XL_mut.pth"
    test_t5_wt_pth:    str = r"/home/yufei/20251210up/feature/protT5-XL/test_protT5-XL_wt.pth"
    test_t5_mut_pth:   str = r"/home/yufei/20251210up/feature/protT5-XL/test_protT5-XL_mut.pth"

    # memmap folder produced by file0_convert_pth_to_memmap.py
    memmap_dir: str = r"/home/yufei/20251210up/feature/merged_memmap_raw"

    # labels
    train_csv: str = r"/home/yufei/20251210up/data/train.csv"
    test_csv:  str = r"/home/yufei/20251210up/data/test.csv"
    label_col: str = "label"

    # output
    out_dir: str = r"/home/yufei/20251210up/pred/ablation"

    # device (添加与文件二相同的force_cpu选项)
    force_cpu: bool = True

    # CV
    n_folds: int = 5
    seed: int = 42

    # -------- C stage --------
    c_epochs: int = 5
    c_batch_size: int = 8
    c_lr: float = 1e-4
    c_weight_decay: float = 1e-2


    # -------- Gate (A): scalar MLP + tau schedule --------
    gate_hidden: int = 64
    gate_tau_start: float = 4.0
    gate_tau_end: float = 2.0
    gate_tau_schedule: str = "linear"

    d_model: int = 256
    branch_d: int = 128  # branch output dim (ProtT5_d = GPN-MSA_d) fixed at 128
    fused_d: int = 128   # fused representation dim fixed at 128

    n_heads: int = 8
    n_layers: int = 4
    ff_dim: int = 1024
    dropout: float = 0.1
    relpos_max: int = 64

    # pooling: (this file uses mean pooling in encoder; keep unchanged)
    pool_type: str = "mean"
    qpool_nq: int = 4

    # TCN backbone params
    tcn_layers: int = 4
    tcn_kernel: int = 5
    tcn_dropout: float = 0.1


    # CNN backbone params (for backbone="cnn" / "cnn_bilstm")
    cnn_layers: int = 2
    cnn_kernel: int = 5
    cnn_dropout: float = 0.1

    # backbone selection (used by the module sweep in main)
    t5_backbone: str = "bilstm"
    gpn_backbone: str = "gru"

    # modality switches (for ablations)
    use_t5: bool = True
    use_gpn: bool = True

    # fusion strategy sweep (avg/concat/bilinear/xattn)
    fusion_mode: str = "gate"
    bilinear_rank: int = 256
    xattn_heads: int = 8
    # BiLSTM backbone
    bilstm_layers: int = 2
    bigru_layers: int = 2

    # regularization
    moddrop_p: float = 0.0
    aux_t5_w: float = 0.0
    aux_gpn_w: float = 0.0

    # mixed precision for C (only matters on CUDA)
    use_amp: bool = True

    # -------- D stage --------
    d_epochs: int = 200
    d_batch_size: int = 64
    d_lr: float = 3e-5
    d_hidden1: int = 4096
    d_hidden2: int = 2048
    d_dropout: float = 0.3
    d_warmup_steps: int = 1000
    d_patience: int = 20
    d_early_stop_metric: str = "auc_roc"

    select_threshold: bool = True
    threshold_objective: str = "f1"  # or "mcc"


CFG = Config()


# =========================
# Utils (添加与文件二相同的转换函数)
# =========================

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_labels_csv(path: str, label_col: str) -> np.ndarray:
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise KeyError(f"{path} missing label col '{label_col}'")
    return df[label_col].values.astype(int)

def calc_metrics_binary(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = y_true.astype(int)
    prob = prob.astype(float)
    pred = (prob >= threshold).astype(int)

    out: Dict[str, float] = {}
    out["accuracy"] = metrics.accuracy_score(y_true, pred) * 100.0
    out["precision"] = metrics.precision_score(y_true, pred, zero_division=0) * 100.0
    out["recall"] = metrics.recall_score(y_true, pred, zero_division=0) * 100.0
    out["f1"] = metrics.f1_score(y_true, pred, zero_division=0) * 100.0
    out["mcc"] = metrics.matthews_corrcoef(y_true, pred) * 100.0

    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    out["specificity"] = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0

    out["auc_roc"] = roc_auc_score(y_true, prob) * 100.0
    prec, rec, _ = precision_recall_curve(y_true, prob)
    out["auc_pr"] = pr_auc(rec, prec) * 100.0
    return out

def pick_threshold(y_true: np.ndarray, prob: np.ndarray, objective: str = "f1") -> float:
    y_true = y_true.astype(int)
    prob = prob.astype(float)
    uniq = np.unique(prob)
    if len(uniq) > 500:
        uniq = np.quantile(prob, np.linspace(0.0, 1.0, 500))

    best_t, best_v = 0.5, -1e18
    for t in uniq:
        pred = (prob >= t).astype(int)
        v = metrics.matthews_corrcoef(y_true, pred) if objective == "mcc" else metrics.f1_score(y_true, pred, zero_division=0)
        if v > best_v:
            best_v, best_t = v, float(t)
    return best_t

def pseudo_pos(a_raw: torch.Tensor, b_raw: torch.Tensor) -> torch.Tensor:
    delta = b_raw - a_raw
    norm = torch.norm(delta, dim=-1)  # (B,L)
    return torch.argmax(norm, dim=1)  # (B,)

# =========================
# 添加从文件二复制的转换函数
# =========================

def _dtype_from_str(s: str):
    s = str(s).lower().strip()
    if s in ["fp16", "float16", "half"]:
        return np.float16
    if s in ["fp32", "float32", "single"]:
        return np.float32
    raise ValueError(f"Unsupported memmap_dtype={s}, use float16/float32")

def load_tensor_any(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for k in ["X", "emb", "embedding", "features", "data"]:
            if k in obj and isinstance(obj[k], torch.Tensor):
                return obj[k]
        for _, v in obj.items():
            if isinstance(v, torch.Tensor):
                return v
    raise ValueError(f"Unrecognized tensor format: {path}")

def pth_to_npy_memmap(pth_path: str, npy_path: str, dtype: np.dtype, chunk_n: int = 64):
    ensure_dir(os.path.dirname(npy_path))
    print(f"[Memmap] convert IN={pth_path}\n                 OUT={npy_path} dtype={dtype}")
    t = load_tensor_any(pth_path).contiguous()
    shape = tuple(t.shape)
    mm = np.lib.format.open_memmap(npy_path, mode="w+", dtype=dtype, shape=shape)
    n0 = shape[0]
    for i in tqdm(range(0, n0, chunk_n), desc=f"[MemmapWrite] {os.path.basename(npy_path)}", leave=False):
        j = min(i + chunk_n, n0)
        blk = t[i:j].cpu().numpy()
        if blk.dtype != dtype:
            blk = blk.astype(dtype, copy=False)
        mm[i:j] = blk
    mm.flush()
    del mm, t
    gc.collect()

def prepare_memmaps(cfg: Config) -> Dict[str, str]:
    ensure_dir(cfg.memmap_dir)
    dtype = _dtype_from_str(cfg.memmap_dtype)
    paths = {
        "train_t5_wt": os.path.join(cfg.memmap_dir, "train_t5_wt.npy"),
        "train_t5_mut": os.path.join(cfg.memmap_dir, "train_t5_mut.npy"),
        "test_t5_wt": os.path.join(cfg.memmap_dir, "test_t5_wt.npy"),
        "test_t5_mut": os.path.join(cfg.memmap_dir, "test_t5_mut.npy"),
        "train_gpn_ref": os.path.join(cfg.memmap_dir, "train_gpn_ref.npy"),
        "train_gpn_alt": os.path.join(cfg.memmap_dir, "train_gpn_alt.npy"),
        "test_gpn_ref": os.path.join(cfg.memmap_dir, "test_gpn_ref.npy"),
        "test_gpn_alt": os.path.join(cfg.memmap_dir, "test_gpn_alt.npy"),
    }
    jobs = [
        (cfg.train_t5_wt_pth, paths["train_t5_wt"]),
        (cfg.train_t5_mut_pth, paths["train_t5_mut"]),
        (cfg.test_t5_wt_pth,  paths["test_t5_wt"]),
        (cfg.test_t5_mut_pth, paths["test_t5_mut"]),
        (cfg.train_gpn_ref_pth, paths["train_gpn_ref"]),
        (cfg.train_gpn_alt_pth, paths["train_gpn_alt"]),
        (cfg.test_gpn_ref_pth,  paths["test_gpn_ref"]),
        (cfg.test_gpn_alt_pth,  paths["test_gpn_alt"]),
    ]
    for pth, npy in jobs:
        if os.path.exists(npy):
            print(f"[Memmap] exists, skip: {npy}")
            continue
        pth_to_npy_memmap(pth, npy, dtype=dtype, chunk_n=32)
    return paths


# =========================
# Memmap datasets
# =========================

class MemmapPairTokenDataset(Dataset):
    """
    Loads token embeddings from npy memmap lazily.

    Return:
      t5_wt, t5_mut, gpn_ref, gpn_alt, y (optional)
    """
    def __init__(self, t5_wt_npy: str, t5_mut_npy: str, gpn_ref_npy: str, gpn_alt_npy: str,
                 y: Optional[np.ndarray], indices: np.ndarray):
        self.t5_wt = np.load(t5_wt_npy, mmap_mode="r")
        self.t5_mut = np.load(t5_mut_npy, mmap_mode="r")
        self.gpn_ref = np.load(gpn_ref_npy, mmap_mode="r")
        self.gpn_alt = np.load(gpn_alt_npy, mmap_mode="r")
        self.y = y
        self.indices = indices.astype(int)

        assert self.t5_wt.shape == self.t5_mut.shape
        assert self.gpn_ref.shape == self.gpn_alt.shape
        assert self.t5_wt.shape[0] == self.gpn_ref.shape[0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        # 添加copy()以避免非可写numpy的警告（与文件二一致）
        t5w = torch.from_numpy(self.t5_wt[idx].copy())
        t5m = torch.from_numpy(self.t5_mut[idx].copy())
        gr  = torch.from_numpy(self.gpn_ref[idx].copy())
        ga  = torch.from_numpy(self.gpn_alt[idx].copy())
        if self.y is None:
            return t5w, t5m, gr, ga
        return t5w, t5m, gr, ga, int(self.y[idx])


# =========================
# Model blocks
# =========================

class ModalityDropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = float(p)

    def forward(self, gpn: torch.Tensor, t5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if (not self.training) or self.p <= 0:
            return gpn, t5
        B = gpn.size(0)
        u = torch.rand((B, 1), device=gpn.device, dtype=gpn.dtype)
        drop_gpn = (u < (self.p / 2)).float()
        drop_t5  = ((u >= (self.p / 2)) & (u < self.p)).float()
        return gpn * (1.0 - drop_gpn), t5 * (1.0 - drop_t5)

class TransformerEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ln = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        y = self.net(y).transpose(1, 2)
        return self.ln(x + y)

class TCNEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        blocks = []
        for i in range(int(cfg.tcn_layers)):
            blocks.append(TCNBlock(cfg.d_model, int(cfg.tcn_kernel), dilation=2**i, dropout=float(cfg.tcn_dropout)))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x

class BiLSTMEncoder(nn.Module):
    """Bidirectional LSTM backbone that preserves sequence length and outputs d_model features."""
    def __init__(self, cfg: Config):
        super().__init__()
        d = int(cfg.d_model)
        h = d // 2
        if 2 * h != d:
            raise ValueError(f"d_model must be even for BiLSTM, got d_model={d}")
        self.lstm = nn.LSTM(
            input_size=d,
            hidden_size=h,
            num_layers=int(cfg.bilstm_layers),
            batch_first=True,
            bidirectional=True,
            dropout=float(cfg.dropout) if int(cfg.bilstm_layers) > 1 else 0.0,
        )
        self.ln = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x)
        return self.ln(y)


class BiGRUEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        if int(cfg.d_model) % 2 != 0:
            raise ValueError("d_model must be even for bidirectional GRU hidden=d_model//2")
        hidden = int(cfg.d_model) // 2
        self.gru = nn.GRU(
            input_size=int(cfg.d_model),
            hidden_size=hidden,
            num_layers=int(getattr(cfg, 'bigru_layers', 2)),
            dropout=float(cfg.dropout) if int(getattr(cfg, 'bigru_layers', 2)) > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(int(cfg.d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(x)
        return self.ln(y)

class LSTMEncoder(nn.Module):
    """Unidirectional LSTM backbone that preserves sequence length and outputs d_model features."""
    def __init__(self, cfg: Config):
        super().__init__()
        d = int(cfg.d_model)
        self.lstm = nn.LSTM(
            input_size=d,
            hidden_size=d,
            num_layers=int(getattr(cfg, "bilstm_layers", 2)),
            batch_first=True,
            bidirectional=False,
            dropout=float(cfg.dropout) if int(getattr(cfg, "bilstm_layers", 2)) > 1 else 0.0,
        )
        self.ln = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x)
        return self.ln(y)


class GRUEncoder(nn.Module):
    """Unidirectional GRU backbone that preserves sequence length and outputs d_model features."""
    def __init__(self, cfg: Config):
        super().__init__()
        d = int(cfg.d_model)
        self.gru = nn.GRU(
            input_size=d,
            hidden_size=d,
            num_layers=int(getattr(cfg, "bigru_layers", 2)),
            batch_first=True,
            bidirectional=False,
            dropout=float(cfg.dropout) if int(getattr(cfg, "bigru_layers", 2)) > 1 else 0.0,
        )
        self.ln = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(x)
        return self.ln(y)


class CNNEncoder(nn.Module):
    """1D CNN backbone over sequence. Preserves length and outputs d_model features."""
    def __init__(self, cfg: Config):
        super().__init__()
        d = int(cfg.d_model)
        layers = int(getattr(cfg, "cnn_layers", 2))
        k = int(getattr(cfg, "cnn_kernel", 5))
        p = k // 2
        drop = float(getattr(cfg, "cnn_dropout", 0.1))
        self.convs = nn.ModuleList([nn.Conv1d(d, d, kernel_size=k, padding=p) for _ in range(layers)])
        self.drop = nn.Dropout(drop)
        self.ln = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        y = x.transpose(1, 2)  # (B, D, L)
        for conv in self.convs:
            z = conv(y)
            z = F.relu(z)
            z = self.drop(z)
            y = y + z  # residual
        y = y.transpose(1, 2)  # (B, L, D)
        return self.ln(y)


class CNNBiLSTMEncoder(nn.Module):
    """CNN followed by BiLSTM, outputs d_model features."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cnn = CNNEncoder(cfg)
        self.bilstm = BiLSTMEncoder(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.bilstm(x)
        return x


class TCNLSTMEncoder(nn.Module):
    """TCN followed by unidirectional LSTM, outputs d_model features."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.tcn = TCNEncoder(cfg)
        self.lstm = LSTMEncoder(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tcn(x)
        x = self.lstm(x)
        return x

class HeadProj(nn.Module):
    """Project pooled d_model features to branch_d (branch vector dim)."""
    def __init__(self, cfg: Config):
        super().__init__()
        d_in = int(cfg.d_model)
        d_out = int(getattr(cfg, 'branch_d', cfg.fused_d))
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d_in, d_out),
        )
        self.ln = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(self.mlp(x))


class Scheme2PairEncoder(nn.Module):
    """
    Scheme2: pseudo-site + relpos + site-marker, then a backbone encoder.

    - encode_tokens(a_raw, b_raw) returns token-level features after backbone: (B, 2L, d_model)
    - forward(a_raw, b_raw) returns pooled fused_d vector via global mean pooling -> HeadProj
    """
    def __init__(self, d_in: int, seq_len: int, cfg: Config, backbone: str):
        super().__init__()
        self.seq_len = seq_len
        self.cfg = cfg

        self.proj = nn.Linear(d_in, cfg.d_model)
        self.seg_emb = nn.Embedding(2, cfg.d_model)
        self.abs_pos = nn.Embedding(2 * seq_len, cfg.d_model)

        self.rel_max = int(cfg.relpos_max)
        self.rel_emb = nn.Embedding(2 * self.rel_max + 1, cfg.d_model)

        self.site_emb = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.site_emb, std=0.02)

        b = backbone.lower()
        if b == "none" or b == "identity":
            self.backbone = nn.Identity()
        elif b == "transformer":
            self.backbone = TransformerEncoder(cfg)
        elif b == "tcn":
            self.backbone = TCNEncoder(cfg)
        elif b == "cnn":
            self.backbone = CNNEncoder(cfg)
        elif b == "bilstm":
            self.backbone = BiLSTMEncoder(cfg)
        elif b == "lstm":
            self.backbone = LSTMEncoder(cfg)
        elif b == "bigru":
            self.backbone = BiGRUEncoder(cfg)
        elif b == "gru":
            self.backbone = GRUEncoder(cfg)
        elif b == "cnn_bilstm":
            self.backbone = CNNBiLSTMEncoder(cfg)
        elif b == "tcn_lstm":
            self.backbone = TCNLSTMEncoder(cfg)
        else:
            raise ValueError(
                f"Unknown backbone={backbone}, use none/transformer/tcn/cnn/bilstm/lstm/bigru/gru/cnn_bilstm/tcn_lstm"
            )

        self.head = HeadProj(cfg)

    def _build_tokens(self, a_raw: torch.Tensor, b_raw: torch.Tensor) -> torch.Tensor:
        B, L, _ = a_raw.shape
        assert L == self.seq_len

        pos_idx = pseudo_pos(a_raw, b_raw)

        a = self.proj(a_raw)
        b = self.proj(b_raw)

        base = torch.arange(L, device=a.device).unsqueeze(0).expand(B, -1)
        rel = base - pos_idx.unsqueeze(1)
        rel = torch.clamp(rel, -self.rel_max, self.rel_max) + self.rel_max
        rel_emb = self.rel_emb(rel)

        mask = torch.zeros((B, L, 1), device=a.device, dtype=a.dtype)
        mask[torch.arange(B, device=a.device), pos_idx, 0] = 1.0

        a = a + rel_emb + mask * self.site_emb
        b = b + rel_emb + mask * self.site_emb

        x = torch.cat([a, b], dim=1)  # (B,2L,d)
        seg = torch.cat([
            torch.zeros((B, L), dtype=torch.long, device=x.device),
            torch.ones((B, L), dtype=torch.long, device=x.device),
        ], dim=1)
        x = x + self.seg_emb(seg)

        pos = torch.arange(0, 2 * L, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.abs_pos(pos)

        x = self.backbone(x)  # (B,2L,d_model)
        return x

    def encode_tokens(self, a_raw: torch.Tensor, b_raw: torch.Tensor) -> torch.Tensor:
        return self._build_tokens(a_raw, b_raw)

    def forward(self, a_raw: torch.Tensor, b_raw: torch.Tensor) -> torch.Tensor:
        x = self._build_tokens(a_raw, b_raw)
        pooled = x.mean(dim=1)  # global mean pooling (keep unchanged)
        return self.head(pooled)


class DualBranchFusionModel(nn.Module):
    """Two-branch encoder + Gate(A) fusion.

    Encoders are controlled by cfg.t5_backbone / cfg.gpn_backbone.
    Each branch returns a vector of dimension cfg.branch_d (fixed at 128 in fusion_dd).
    Fusion is a scalar MLP gate with tau schedule (Gate A), then a fused classifier head.

    Returns:
      fused (B,fused_d), logits_fused (B,1), optional aux logits
    """

    def __init__(self, t5_enc: nn.Module, gpn_enc: nn.Module, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.t5_enc = t5_enc
        self.gpn_enc = gpn_enc
        self.moddrop = ModalityDropout(cfg.moddrop_p) if cfg.moddrop_p > 0 else None

        branch_d = int(getattr(cfg, "branch_d", cfg.fused_d))
        fused_d = int(cfg.fused_d)

        # ---- Gate A (scalar) computed on branch vectors ----
        self.gate_ln_g = nn.LayerNorm(branch_d)
        self.gate_ln_t = nn.LayerNorm(branch_d)

        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * branch_d, int(cfg.gate_hidden)),
            nn.GELU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(int(cfg.gate_hidden), 1),
        )

        # init last layer to 0 => alpha starts ~ 0.5
        last = self.gate_mlp[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

        self._gate_tau = float(cfg.gate_tau_start)

        # Project fused branch vector (branch_d) -> fused_d, if needed
        if fused_d == branch_d:
            self.fuse_proj = nn.Identity()
        else:
            self.fuse_proj = nn.Sequential(
                nn.Linear(branch_d, fused_d),
                nn.GELU(),
                nn.Dropout(float(cfg.dropout)),
            )

        # fused head
        self.fuse_ln = nn.LayerNorm(fused_d)
        self.head_fused = nn.Linear(fused_d, 1)

        # optional aux heads on branch vectors
        self.head_t5 = nn.Linear(branch_d, 1) if float(getattr(cfg, "aux_t5_w", 0.0)) > 0 else None
        self.head_gpn = nn.Linear(branch_d, 1) if float(getattr(cfg, "aux_gpn_w", 0.0)) > 0 else None

    def set_gate_tau(self, tau: float):
        self._gate_tau = float(tau)

    def _gate_fuse(self, g_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
        # g_in,t_in: (B,d)
        tau = max(self._gate_tau, 1e-6)
        g_gate = self.gate_ln_g(g_in)
        t_gate = self.gate_ln_t(t_in)
        xcat = torch.cat([g_gate, t_gate], dim=1)  # (B,2d)
        alpha = torch.sigmoid(self.gate_mlp(xcat) / tau)  # (B,1)
        return alpha * g_in + (1.0 - alpha) * t_in

    def forward(self, t5_wt, t5_mut, gpn_ref, gpn_alt):
        """Forward with ablation controls.

        - cfg.use_t5 / cfg.use_gpn can disable a modality (feature ablations).
        - cfg.fusion_mode supports:
            * 'gate' : Gate(A) scalar MLP gate + tau schedule (baseline)
            * 'avg'  : mean fusion (used for w/o gate fusion ablation)
        """
        use_t5 = bool(getattr(self.cfg, "use_t5", True))
        use_gpn = bool(getattr(self.cfg, "use_gpn", True))
        mode = str(getattr(self.cfg, "fusion_mode", "gate")).lower()

        t5_vec = self.t5_enc(t5_wt, t5_mut) if use_t5 else None
        gpn_vec = self.gpn_enc(gpn_ref, gpn_alt) if use_gpn else None

        # -------- fusion / bypass --------
        if use_t5 and use_gpn:
            g_in, t_in = gpn_vec, t5_vec
            if self.moddrop is not None:
                g_in, t_in = self.moddrop(g_in, t_in)

            if mode == "gate":
                fused = self._gate_fuse(g_in, t_in)
            elif mode == "avg":
                fused = 0.5 * (g_in + t_in)
            else:
                raise ValueError(f"Unsupported fusion_mode={mode} for ablation")
        elif use_t5 and (not use_gpn):
            fused = t5_vec
        elif (not use_t5) and use_gpn:
            fused = gpn_vec
        else:
            raise ValueError("Invalid ablation setting: both use_t5 and use_gpn are False")

        fused = self.fuse_proj(fused)
        fused = self.fuse_ln(fused)
        logits_fused = self.head_fused(fused)

        lt5 = self.head_t5(t5_vec) if (t5_vec is not None and self.head_t5 is not None) else None
        lgpn = self.head_gpn(gpn_vec) if (gpn_vec is not None and self.head_gpn is not None) else None
        return fused, logits_fused, lt5, lgpn

def bce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.BCEWithLogitsLoss()(logits, y.view(-1, 1))


# =========================
# C train + export to memmap (NO cat in RAM) - 修改AMP代码
# =========================

def train_c(model: nn.Module, dl: DataLoader, device: torch.device, cfg: Config, fold: int) -> Dict[str, float]:
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.c_lr, weight_decay=cfg.c_weight_decay)

    # 修改AMP代码以避免FutureWarning
    use_amp = bool(cfg.use_amp) and (device.type == "cuda")
    scaler = torch.amp.GradScaler(device_type='cuda', enabled=use_amp) if use_amp else None

    losses = []
    for ep in range(cfg.c_epochs):
        # ---- tau schedule for Gate A ----
        if hasattr(model, 'set_gate_tau'):
            tau0 = float(getattr(cfg, 'gate_tau_start', 4.0))
            tau1 = float(getattr(cfg, 'gate_tau_end', 2.0))
            if cfg.c_epochs <= 1:
                tau = tau1
            else:
                t = ep / float(cfg.c_epochs - 1)
                tau = tau0 + (tau1 - tau0) * t
            model.set_gate_tau(tau)
            if ep == 0 or ep == cfg.c_epochs - 1:
                print(f'[C][Fold {fold}] gate_tau={tau:.4f}')
        cum, n = 0.0, 0
        for bt5w, bt5m, bgref, bgalt, by in tqdm(dl, desc=f"[C][Fold {fold}] ep {ep+1}/{cfg.c_epochs}", leave=False):
            bt5w, bt5m = bt5w.to(device).float(), bt5m.to(device).float()
            bgref, bgalt = bgref.to(device).float(), bgalt.to(device).float()
            by = by.to(device).float()

            opt.zero_grad(set_to_none=True)

            if use_amp:
                # 使用新的AMP API
                with torch.amp.autocast(device_type='cuda', enabled=True):
                    _, lfused, lt5, lgpn = model(bt5w, bt5m, bgref, bgalt)
                    loss = bce(lfused, by)
                    if lt5 is not None and cfg.aux_t5_w > 0:
                        loss = loss + float(cfg.aux_t5_w) * bce(lt5, by)
                    if lgpn is not None and cfg.aux_gpn_w > 0:
                        loss = loss + float(cfg.aux_gpn_w) * bce(lgpn, by)
                
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                # 不使用AMP
                _, lfused, lt5, lgpn = model(bt5w, bt5m, bgref, bgalt)
                loss = bce(lfused, by)
                if lt5 is not None and cfg.aux_t5_w > 0:
                    loss = loss + float(cfg.aux_t5_w) * bce(lt5, by)
                if lgpn is not None and cfg.aux_gpn_w > 0:
                    loss = loss + float(cfg.aux_gpn_w) * bce(lgpn, by)
                
                loss.backward()
                opt.step()

            cum += float(loss.detach().cpu().item())
            n += 1

        avg = cum / max(n, 1)
        losses.append(avg)
        print(f"[C][Fold {fold}] ep {ep+1} loss={avg:.6f}")

    return {"c_loss_last": float(losses[-1]), "c_loss_mean": float(np.mean(losses))}


@torch.no_grad()
def export_fused_to_memmap(model: nn.Module, dl: DataLoader, device: torch.device, out_npy: str, n_rows: int, cfg: Config):
    model.eval()
    ensure_dir(os.path.dirname(out_npy))

    mm = np.lib.format.open_memmap(out_npy, mode="w+", dtype=np.float32, shape=(n_rows, int(cfg.fused_d)))

    offset = 0
    for bt5w, bt5m, bgref, bgalt in tqdm(dl, desc=f"[Export fused -> {os.path.basename(out_npy)}]", leave=False):
        bs = bt5w.size(0)
        bt5w, bt5m = bt5w.to(device).float(), bt5m.to(device).float()
        bgref, bgalt = bgref.to(device).float(), bgalt.to(device).float()
        fused, _, _, _ = model(bt5w, bt5m, bgref, bgalt)
        mm[offset:offset+bs, :] = fused.detach().cpu().numpy().astype(np.float32, copy=False)
        offset += bs

    mm.flush()
    del mm
    if offset != n_rows:
        raise RuntimeError(f"Export rows mismatch: wrote={offset}, expected={n_rows}")


# =========================
# D classifier (same)
# =========================

class Classifier2L(nn.Module):
    def __init__(self, hidden: int, hidden2: int, dropout: float, input_dim: int = 128):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.l2 = nn.Linear(hidden, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.l3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.drop(self.bn1(self.l1(x))))
        x = self.relu(self.drop(self.bn2(self.l2(x))))
        return self.l3(x)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def d_loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.BCEWithLogitsLoss()(logits, y.view(-1, 1))

def train_epoch_d(net, loader, optimizer, scheduler, device) -> Tuple[float, float]:
    net.train()
    cum, n = 0.0, 0
    ys, ps = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = net(x.float())
        loss = d_loss_fn(logits, y.float())
        loss.backward()
        optimizer.step()
        scheduler.step()
        cum += float(loss.detach().cpu().item())
        n += 1
        ys.extend(y.detach().cpu().numpy().tolist())
        ps.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten().tolist())
    loss_avg = cum / max(n, 1)
    acc = metrics.accuracy_score(np.asarray(ys).astype(int), (np.asarray(ps) >= 0.5).astype(int)) * 100.0
    return loss_avg, acc

@torch.no_grad()
def eval_epoch_d(net, loader, device) -> Tuple[float, Dict[str, float], Dict[str, np.ndarray]]:
    net.eval()
    cum, n = 0.0, 0
    ys, ps, ls = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = net(x.float())
        loss = d_loss_fn(logits, y.float())
        cum += float(loss.detach().cpu().item())
        n += 1
        ls.extend(logits.detach().cpu().numpy().flatten().tolist())
        ps.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten().tolist())
        ys.extend(y.detach().cpu().numpy().tolist())
    loss_avg = cum / max(n, 1)
    ys = np.asarray(ys).astype(int)
    ps = np.asarray(ps).astype(float)
    m = calc_metrics_binary(ys, ps, threshold=0.5)
    return loss_avg, m, {"y": ys, "prob": ps, "logit": np.asarray(ls, dtype=float)}

def train_d_complex(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray,
                    device, cfg: Config, fold: int, ckpt_dir: str):
    ensure_dir(ckpt_dir)

    tr_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                           batch_size=cfg.d_batch_size, shuffle=True, drop_last=True)
    va_loader = DataLoader(TensorDataset(torch.tensor(X_va), torch.tensor(y_va)),
                           batch_size=cfg.d_batch_size, shuffle=False, drop_last=False)

    net = Classifier2L(cfg.d_hidden1, cfg.d_hidden2, cfg.d_dropout, input_dim=int(cfg.fused_d)).to(device)
    net.apply(init_weights)

    params = list(net.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    opt_groups = [
        {"params": [p for n, p in params if not any(nd in n for nd in no_decay)], "weight_decay": 0.001},
        {"params": [p for n, p in params if any(nd in n for nd in no_decay)], "weight_decay": 0.02},
    ]

    num_steps = int(max(1, len(X_tr) // cfg.d_batch_size) * cfg.d_epochs)
    optimizer = torch.optim.AdamW(opt_groups, lr=cfg.d_lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, cfg.d_warmup_steps, max(num_steps, 1))

    best_metric, best_epoch, best_state, best_val_metrics = -1e18, 0, None, None
    patience = 0

    for ep in tqdm(range(cfg.d_epochs), desc=f"[D][Fold {fold}]", leave=False):
        tr_loss, tr_acc = train_epoch_d(net, tr_loader, optimizer, scheduler, device)
        va_loss, va_metrics, _ = eval_epoch_d(net, va_loader, device)
        cur = float(va_metrics.get(cfg.d_early_stop_metric, -1e18))

        if cur > best_metric + 1e-6:
            best_metric = cur
            best_epoch = ep
            best_state = copy.deepcopy(net.state_dict())
            best_val_metrics = va_metrics
            patience = 0
        else:
            patience += 1
            if patience >= cfg.d_patience:
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    best_thr = 0.5
    if cfg.select_threshold:
        _, _, pack = eval_epoch_d(net, va_loader, device)
        best_thr = pick_threshold(pack["y"], pack["prob"], objective=cfg.threshold_objective)

    torch.save({
        "model_state_dict": net.state_dict(),
        "best_epoch": best_epoch,
        "best_val_metric": best_metric,
        "best_val_metrics": best_val_metrics,
        "best_threshold": best_thr,
        "cfg": cfg.__dict__,
    }, os.path.join(ckpt_dir, f"d_best_model_fold{fold}.pth"))

    return net, best_val_metrics, best_epoch, float(best_thr)

@torch.no_grad()
def predict_d(net, X: np.ndarray, device, batch_size: int = 256):
    net.eval()
    dl = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                    batch_size=batch_size, shuffle=False, drop_last=False)
    logits_all, prob_all = [], []
    for (bx,) in dl:
        bx = bx.to(device)
        logits = net(bx.float())
        prob = torch.sigmoid(logits)
        logits_all.append(logits.detach().cpu().numpy().flatten())
        prob_all.append(prob.detach().cpu().numpy().flatten())
    return np.concatenate(logits_all), np.concatenate(prob_all)


# =========================
# Main (修改为与文件二相同)
# =========================

def _run_single(cfg: Config = CFG):
    set_seed(cfg.seed)
    
    # 根据force_cpu设置device（与文件二一致）
    device = torch.device("cpu") if cfg.force_cpu else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device={device}")

    # 如果设置了do_memmap_convert，则进行转换
    if cfg.do_memmap_convert:
        prepare_memmaps(cfg)
    
    ensure_dir(cfg.out_dir)
    ensure_dir(os.path.join(cfg.out_dir, "ckpt"))
    ensure_dir(os.path.join(cfg.out_dir, "pred"))
    ensure_dir(os.path.join(cfg.out_dir, "fold_artifacts"))
    ensure_dir(os.path.join(cfg.out_dir, "fused_memmap"))

    # memmap paths（与文件二相同）
    t5_tr_wt = os.path.join(cfg.memmap_dir, "train_t5_wt.npy")
    t5_tr_mut= os.path.join(cfg.memmap_dir, "train_t5_mut.npy")
    t5_te_wt = os.path.join(cfg.memmap_dir, "test_t5_wt.npy")
    t5_te_mut= os.path.join(cfg.memmap_dir, "test_t5_mut.npy")

    gpn_tr_ref = os.path.join(cfg.memmap_dir, "train_gpn_ref.npy")
    gpn_tr_alt = os.path.join(cfg.memmap_dir, "train_gpn_alt.npy")
    gpn_te_ref = os.path.join(cfg.memmap_dir, "test_gpn_ref.npy")
    gpn_te_alt = os.path.join(cfg.memmap_dir, "test_gpn_alt.npy")

    # 检查memmap文件是否存在（与文件二相同）
    for p in [t5_tr_wt, t5_tr_mut, t5_te_wt, t5_te_mut, gpn_tr_ref, gpn_tr_alt, gpn_te_ref, gpn_te_alt]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing memmap file: {p}\nSet do_memmap_convert=True or check memmap_dir.")

    # read shapes without loading fully
    _t5 = np.load(t5_tr_wt, mmap_mode="r")
    n_train, L, dp = _t5.shape
    del _t5
    _t5t = np.load(t5_te_wt, mmap_mode="r")
    n_test, L2, _ = _t5t.shape
    del _t5t
    if L != L2:
        raise ValueError("Train/Test ProtT5 L mismatch")

    _gpn = np.load(gpn_tr_ref, mmap_mode="r")
    if _gpn.shape[0] != n_train or _gpn.shape[1] != L:
        raise ValueError("GPN train shape mismatch with ProtT5")
    dg = _gpn.shape[2]
    del _gpn

    print(f"[Data] Train N={n_train}, L={L}, ProtT5_D={dp}, GPN_D={dg}")
    print(f"[Data] Test  N={n_test},  L={L2}")

    y_train = load_labels_csv(cfg.train_csv, cfg.label_col)
    y_test  = load_labels_csv(cfg.test_csv, cfg.label_col)

    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)

    fold_probs, fold_thrs, fold_summaries = [], [], []
    test_indices = np.arange(n_test)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n_train), y_train), start=1):
        print("\n" + "=" * 70)
        print(f"Fold {fold}/{cfg.n_folds}")
        print("=" * 70)

        fold_dir = os.path.join(cfg.out_dir, "fold_artifacts", f"fold{fold}")
        ensure_dir(fold_dir)

        # -------------------------
        # CHANGE ONLY HERE:
        #   ProtT5 branch: TCN
        #   GPN-MSA branch: BiLSTM
        # -------------------------
        t5_enc  = Scheme2PairEncoder(d_in=dp, seq_len=L, cfg=cfg, backbone=cfg.t5_backbone)
        gpn_enc = Scheme2PairEncoder(d_in=dg, seq_len=L, cfg=cfg, backbone=cfg.gpn_backbone)
        c_model = DualBranchFusionModel(t5_enc, gpn_enc, cfg)

        # C train loader (memmap lazy)
        ds_c_tr = MemmapPairTokenDataset(t5_tr_wt, t5_tr_mut, gpn_tr_ref, gpn_tr_alt, y_train, tr_idx)
        dl_c_tr = DataLoader(ds_c_tr, batch_size=cfg.c_batch_size, shuffle=True, drop_last=False, num_workers=0)
        c_stats = train_c(c_model, dl_c_tr, device, cfg, fold)

        torch.save({"model_state_dict": c_model.state_dict(), "cfg": cfg.__dict__, "c_stats": c_stats},
                   os.path.join(fold_dir, f"c_model_fold{fold}.pth"))

        # Export fused -> memmap files
        fused_tr_path = os.path.join(cfg.out_dir, "fused_memmap", f"fold{fold}_Xtr.npy")
        fused_va_path = os.path.join(cfg.out_dir, "fused_memmap", f"fold{fold}_Xva.npy")
        fused_te_path = os.path.join(cfg.out_dir, "fused_memmap", f"fold{fold}_Xte.npy")

        ds_tr = MemmapPairTokenDataset(t5_tr_wt, t5_tr_mut, gpn_tr_ref, gpn_tr_alt, None, tr_idx)
        ds_va = MemmapPairTokenDataset(t5_tr_wt, t5_tr_mut, gpn_tr_ref, gpn_tr_alt, None, va_idx)
        ds_te = MemmapPairTokenDataset(t5_te_wt, t5_te_mut, gpn_te_ref, gpn_te_alt, None, test_indices)

        export_fused_to_memmap(c_model, DataLoader(ds_tr, batch_size=32, shuffle=False, num_workers=0), device, fused_tr_path, n_rows=len(tr_idx), cfg=cfg)
        export_fused_to_memmap(c_model, DataLoader(ds_va, batch_size=32, shuffle=False, num_workers=0), device, fused_va_path, n_rows=len(va_idx), cfg=cfg)
        export_fused_to_memmap(c_model, DataLoader(ds_te, batch_size=32, shuffle=False, num_workers=0), device, fused_te_path, n_rows=n_test, cfg=cfg)

        # Load fused memmaps (small: N x 128)
        X_tr = np.load(fused_tr_path, mmap_mode="r")
        X_va = np.load(fused_va_path, mmap_mode="r")
        X_te = np.load(fused_te_path, mmap_mode="r")

        y_tr = y_train[tr_idx].astype(int)
        y_va = y_train[va_idx].astype(int)

        # D train
        d_model, best_val_metrics, best_epoch, best_thr = train_d_complex(
            np.asarray(X_tr), y_tr, np.asarray(X_va), y_va, device, cfg, fold, ckpt_dir=os.path.join(cfg.out_dir, "ckpt")
        )

        # ------------------------------------------------------------------
        # Explainability exports (non-intrusive):
        # Some downstream explainability scripts expect C and D checkpoints
        # to live under the same fold directory. The original code saves:
        #   - C: out_dir/fold_artifacts/fold{fold}/c_model_fold{fold}.pth
        #   - D: out_dir/ckpt/d_best_model_fold{fold}.pth
        # Here we also copy D into the fold directory and write a manifest.
        # ------------------------------------------------------------------
        try:
            d_src = os.path.join(cfg.out_dir, "ckpt", f"d_best_model_fold{fold}.pth")
            d_dst = os.path.join(fold_dir, f"d_best_model_fold{fold}.pth")
            if os.path.exists(d_src):
                shutil.copy2(d_src, d_dst)
        except Exception as e:
            print(f"[Warn] Failed to copy D checkpoint to fold_dir (fold={fold}): {e}")

        try:
            manifest = {
                "fold": int(fold),
                "paths": {
                    "fold_dir": fold_dir,
                    "c_ckpt": os.path.join(fold_dir, f"c_model_fold{fold}.pth"),
                    "d_ckpt_src": os.path.join(cfg.out_dir, "ckpt", f"d_best_model_fold{fold}.pth"),
                    "d_ckpt_fold_dir": os.path.join(fold_dir, f"d_best_model_fold{fold}.pth"),
                    "fused_memmap": {
                        "train": fused_tr_path,
                        "val": fused_va_path,
                        "test": fused_te_path,
                    },
                },
                "d_training": {
                    "best_epoch": int(best_epoch),
                    "best_threshold": float(best_thr),
                    "best_val_metrics": best_val_metrics,
                },
                "cfg": cfg.__dict__,
            }
            with open(os.path.join(fold_dir, "explain_manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Warn] Failed to write explain manifest (fold={fold}): {e}")


        te_logits, te_prob = predict_d(d_model, np.asarray(X_te), device)
        thr_use = best_thr if cfg.select_threshold else 0.5

        fold_probs.append(te_prob)
        fold_thrs.append(best_thr)

        pred_fold = pd.DataFrame({
            "sample_index": test_indices,
            "y_true": y_test.astype(int),
            "prob": te_prob.astype(float),
            "logit": te_logits.astype(float),
            "pred_label": (te_prob >= thr_use).astype(int),
            "fold": fold,
        })
        pred_fold_path = os.path.join(cfg.out_dir, "pred", f"pred_test_fold{fold}.csv")
        pred_fold.to_csv(pred_fold_path, index=False)

        test_metrics_fold = calc_metrics_binary(y_test, te_prob, threshold=thr_use)
        print(f"[Fold {fold}] TEST thr={thr_use:.4f}  "
              f"ACC={test_metrics_fold['accuracy']:.2f} F1={test_metrics_fold['f1']:.2f} "
              f"MCC={test_metrics_fold['mcc']:.2f} ROC_AUC={test_metrics_fold['auc_roc']:.2f} PR_AUC={test_metrics_fold['auc_pr']:.2f}")

        fold_summaries.append({
            "fold": fold,
            "c_stats": c_stats,
            "d_best_epoch": best_epoch,
            "d_best_val_metrics": best_val_metrics,
            "d_val_threshold": best_thr,
            "test_metrics_single_model": test_metrics_fold,
            "paths": {"pred_test_fold": pred_fold_path, "fused_tr": fused_tr_path, "fused_va": fused_va_path, "fused_te": fused_te_path},
        })

        del c_model, d_model
        torch.cuda.empty_cache()

    # Ensemble
    prob_mat = np.stack(fold_probs, axis=0)
    prob_ens = prob_mat.mean(axis=0)
    thr_ens = float(np.mean(fold_thrs)) if cfg.select_threshold else 0.5
    final_metrics = calc_metrics_binary(y_test, prob_ens, threshold=thr_ens)

    pred_ens = pd.DataFrame({
        "sample_index": test_indices,
        "y_true": y_test.astype(int),
        "prob_ensemble": prob_ens.astype(float),
        "pred_label": (prob_ens >= thr_ens).astype(int),
    })
    pred_ens_path = os.path.join(cfg.out_dir, "pred", "pred_test_ensemble.csv")
    pred_ens.to_csv(pred_ens_path, index=False)

    summary = {
        "cfg": cfg.__dict__,
        "fold_thresholds": [float(x) for x in fold_thrs],
        "ensemble_threshold": float(thr_ens),
        "final_metrics": final_metrics,
        "fold_summaries": fold_summaries,
        "outputs": {"pred_test_ensemble_csv": pred_ens_path},
    }
    summary_path = os.path.join(cfg.out_dir, "ckpt", "test_ensemble_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 90)
    print(f"[TEST Ensemble] thr={thr_ens:.4f}  "
          f"ACC={final_metrics['accuracy']:.2f} F1={final_metrics['f1']:.2f} MCC={final_metrics['mcc']:.2f} "
          f"ROC_AUC={final_metrics['auc_roc']:.2f} PR_AUC={final_metrics['auc_pr']:.2f}")
    print("=" * 90)
    print(f"[Saved] {pred_ens_path}")
    print(f"[Saved] {summary_path}")

# =========================

# =========================
# Main runner (final model)
# =========================

def main(cfg: Config = CFG):
    """
    Ablation suite (7 runs total):
      - baseline
      - feature ablations: w/o protT5, w/o GPN-MSA
      - module ablations : w/o BiLSTM, w/o GRU, w/o BiLSTM+GRU, w/o gate fusion (avg instead)

    Ablation principle:
      Change ONLY the target component; all other training/data/hyperparameters remain identical.
    """
    # Convert memmaps once (optional)
    if cfg.do_memmap_convert:
        print("[Info] do_memmap_convert=True -> converting pth to memmap once")
        prepare_memmaps(cfg)
        import copy as _copy
        cfg = _copy.deepcopy(cfg)
        cfg.do_memmap_convert = False

    # ---- fixed final dims ----
    cfg.branch_d = 128
    cfg.fused_d = 128

    # ---- fixed best encoders (baseline defaults) ----
    base = {
        "use_t5": True,
        "use_gpn": True,
        "t5_backbone": "bilstm",
        "gpn_backbone": "gru",
        "fusion_mode": "gate",
    }

    # 7 runs: baseline + 6 ablations
    experiments = [
        ("baseline",                dict(base)),
        ("w_o_protT5",              {**base, "use_t5": False}),                   # feature ablation
        ("w_o_GPN_MSA",             {**base, "use_gpn": False}),                  # feature ablation
        ("w_o_BiLSTM",              {**base, "t5_backbone": "none"}),             # module ablation
        ("w_o_GRU",                 {**base, "gpn_backbone": "none"}),            # module ablation
        ("w_o_BiLSTM_GRU",          {**base, "t5_backbone": "none", "gpn_backbone": "none"}),  # module ablation
        ("w_o_gate_fusion__avg",    {**base, "fusion_mode": "avg"}),              # module ablation
    ]

    # Root output directory
    root_out = cfg.out_dir
    ensure_dir(root_out)

    for name, override in experiments:
        import copy as _copy
        run_cfg = _copy.deepcopy(cfg)

        # apply overrides
        for k, v in override.items():
            setattr(run_cfg, k, v)

        run_cfg.out_dir = os.path.join(root_out, name)
        ensure_dir(run_cfg.out_dir)

        print("\n" + "=" * 100)
        print(f"[ABLATION] {name} | t5={getattr(run_cfg,'t5_backbone',None)} | gpn={getattr(run_cfg,'gpn_backbone',None)} | "
              f"use_t5={getattr(run_cfg,'use_t5',True)} | use_gpn={getattr(run_cfg,'use_gpn',True)} | fusion={getattr(run_cfg,'fusion_mode','gate')} | "
              f"branch_d=128 | fused_d=128")
        print(f"[OUT] {run_cfg.out_dir}")
        print("=" * 100 + "\n")

        try:
            _run_single(run_cfg)
        except Exception as e:
            print(f"[Error] {name} failed: {e}")

    print("\n[Done] All ablation runs finished.\n")


if __name__ == "__main__":
    main(CFG)
