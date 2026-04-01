#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run:
  python F:\\20251210up\\interpretability_all_in_one.py

This script is zero-argument: all paths are configured below.

Outputs:
  F:\\20251210up\\pred\\model\\explainability_6in1\\folds\\foldK\\...
  F:\\20251210up\\pred\\model\\explainability_6in1\\mean\\...   (paper-ready)
"""

from __future__ import annotations
import os, re, json, shutil, subprocess, warnings
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")


# =========================================================
# Global plotting style: consistent fonts + vector outputs
# =========================================================
# Use a single, slightly larger font size across all figures.
PLOT_FONT_SIZE: int = 9
matplotlib.rcParams.update({
    "font.size": PLOT_FONT_SIZE,
    "axes.titlesize": PLOT_FONT_SIZE,
    "axes.labelsize": PLOT_FONT_SIZE,
    "xtick.labelsize": PLOT_FONT_SIZE,
    "ytick.labelsize": PLOT_FONT_SIZE,
    "legend.fontsize": PLOT_FONT_SIZE,
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    # Default colormap for image-like plots (e.g., confusion matrix / heatmaps)
    # Keep consistent with UMAP's red-blue palette.
    "image.cmap": "RdBu_r",
})

# Force all saved figures to vector format (PDF) while keeping existing
# code logic unchanged (many places save as .png).
_ORIG_FIG_SAVEFIG = Figure.savefig

def _vectorized_savefig(self, fname, *args, **kwargs):
    # Keep original requested output (e.g., PNG) and also generate PDF + SVG.
    if not isinstance(fname, str):
        return _ORIG_FIG_SAVEFIG(self, fname, *args, **kwargs)

    root, ext = os.path.splitext(fname)
    ext_l = ext.lower()
    base = root if ext else fname

    pdf_path = base + ".pdf"
    svg_path = base + ".svg"

    out_dir = os.path.dirname(pdf_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    kw = dict(kwargs)
    kw.pop("format", None)

    result = None
    if ext_l in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        result = _ORIG_FIG_SAVEFIG(self, fname, *args, format=ext_l.lstrip('.'), **kw)

    _ORIG_FIG_SAVEFIG(self, pdf_path, *args, format="pdf", **kw)
    _ORIG_FIG_SAVEFIG(self, svg_path, *args, format="svg", **kw)
    return result

Figure.savefig = _vectorized_savefig


# =========================================================
# CONFIG (hardcoded for your exact paths)
# =========================================================
# Training outputs root
OUT_DIR = r"F:\20251210up\pred\model"

# Display name of your model used in all figures/titles/legends
MODEL_NAME = "GenProt-DSM"

# -----------------------------------------------------------------------------
# Global constants (used across plotting utilities)
# -----------------------------------------------------------------------------
FIG_DPI: int = 300
OURS_NAME: str = MODEL_NAME
LABEL_COL: str = "label"

# model.py location
MODEL_PY = r"F:\20251210up\model.py"

# IMPORTANT: override memmap_dir used by ckpt cfg (use your real test memmaps)
OVERRIDE_MEMMAP_DIR = r"F:\20251210up\feature\merged_memmap_raw"

# Optional inputs
RESULTS_DOCX = ""  # e.g. r"F:\20251210up\some_results.docx"
SEQ_CSV = r"F:\20251210up\data\seq_csv.csv"   # 改成你的真实路径

# Optional: functional annotation / tool-score table used as the 3rd input modality
# for the 6-panel UMAP (no sample_index required).
# Must contain: Chrom, Position, Reference, Alternate (case-insensitive).
TOOL_SCORE_CSV = r"F:\20251210up\data\test-score.csv"  # set "" to disable

# Subset CSVs (optional). Used for rare/gene ROC/PR overlays.
# If empty or missing, the corresponding plots will be skipped.
RARE_SUBSET_CSV = r"F:\20251210up\data\new_test\test_AF.csv"
GENE_SUBSET_CSV = r"F:\20251210up\data\new_test\test_onlygene.csv"

# Optional: for dataset plots
TRAIN_CSV = r"F:\20251210up\data\train.csv"  # set "" to disable
TEST_CSV  = r"F:\20251210up\data\test.csv"   # set "" to disable

# Optional: missing value summary for tools (tool_name,total,n_used)
MISSING_VALUE_CSV = r"F:\20251210up\metrics_result\missing_value.csv"  # set "" to disable

# Optional: ablation folder (auto-scanned)
# Expected layout (your requirement): <ABLA_ROOT>/<name>/ckpt/test_ensemble_summary.json
ABLA_ROOT = os.path.join(OUT_DIR, "ablation")

SEQ_COL_DNA_ALT = "dna_alt_128"
SEQ_COL_DNA_REF = "dna_ref_128"
SEQ_COL_PROT_MUT = "prot_mut_128"
SEQ_COL_PROT_WT  = "prot_wt_128"

# Motif tools (disabled by default)
RUN_MEME = False
RUN_TOMTOM = False
DNA_DB_MEME = ""   # e.g. r"F:\db\JASPAR_CORE_*.meme"

# Runtime
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

# Embedding plots
EMBED_METHODS = ["tsne", "umap"]  # if umap not installed, it will be skipped automatically

# ---------------------------------------------------------------------
# Figure color specs (RGB) requested for paper styling
# ---------------------------------------------------------------------
COLOR_CLASS_NEG = (222/255.0, 234/255.0, 234/255.0)  # R222 G234 B234
COLOR_CLASS_POS = (247/255.0, 228/255.0, 116/255.0)  # R247 G228 B116

DISEASE_PALETTE = [
    (170/255.0, 220/255.0, 224/255.0),  # R170 G220 B224
    (255/255.0, 208/255.0, 111/255.0),  # R255 G208 B111
    (114/255.0, 188/255.0, 213/255.0),  # R114 G188 B213
    (255/255.0, 230/255.0, 183/255.0),  # R255 G230 B183
    (82/255.0, 143/255.0, 173/255.0),   # R082 G143 B173
    (247/255.0, 170/255.0, 88/255.0),   # R247 G170 B088
    (55/255.0, 103/255.0, 149/255.0),   # R055 G103 B149
    (239/255.0, 138/255.0, 71/255.0),   # R239 G138 B071
]
# Ablation scatter colors (match the example figure style)
ABLA_COLOR_MAP = {
    "baseline": "#1f77b4",               # blue
    "without-gate_fusion": "#aec7e8",    # light blue
    "without-bilstm": "#ff7f0e",         # orange
    "without-bilstm+gru": "#9467bd",     # purple
    "without-prott5-xl_feature": "#ffbb78",  # light orange
    "without-gru": "#2ca02c",            # green
    "without-gpn-msa_feature": "#d62728" # red
}

# ---------------------------------------------------------------------
# Pink-Blue colormap (requested): #CAE5F8, #9DCBED, #FAEFF5, #F2C5DA, #E286AF
# Use ONLY for confusion matrix and substitution heatmap (do not change others).
# ---------------------------------------------------------------------
PINKBLUE_CMAP = LinearSegmentedColormap.from_list(
    "pinkblue_paper",
    ["#9DCBED", "#CAE5F8", "#FAEFF5", "#F2C5DA", "#E286AF"],
    N=256
)

def _norm_abla_name(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace(" ", "")
    # unify separators
    s = s.replace("\\", "/")
    # keep only last component if nested
    s = s.split("/")[-1]
    # unify underscore/hyphen
    s = s.replace("_", "-")
    return s

# Whether to run per-fold proxy + SHAP (can be slow)
PER_FOLD_PROXY = False

# Only generate mean outputs (paper-ready) and skip per-fold
ONLY_MEAN = False
# =========================================================


# ----------------------------
# Optional imports
# ----------------------------
def try_import_umap():
    try:
        import umap  # type: ignore
        return umap
    except Exception:
        return None

def try_import_shap():
    try:
        import shap  # type: ignore
        return shap
    except Exception:
        return None

def try_import_xgboost():
    try:
        import xgboost as xgb  # type: ignore
        return xgb
    except Exception:
        return None


# ----------------------------
# Utils
# ----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _savefig(fig, out_path: str, dpi: int = FIG_DPI):
    """Save figure with consistent settings."""
    ensure_dir(os.path.dirname(out_path) or ".")
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")

def load_json(p: str) -> Any:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def confusion_group(y_true: np.ndarray, y_prob: np.ndarray, thr=0.5) -> np.ndarray:
    y_pred = (y_prob >= thr).astype(int)
    g = np.zeros_like(y_true, dtype=int)
    g[(y_true==0)&(y_pred==0)] = 0 # TN
    g[(y_true==0)&(y_pred==1)] = 1 # FP
    g[(y_true==1)&(y_pred==0)] = 2 # FN
    g[(y_true==1)&(y_pred==1)] = 3 # TP
    return g

def hi_conf_groups(y_true: np.ndarray, y_prob: np.ndarray, thr=0.5, q=0.8) -> Dict[str, np.ndarray]:
    y_pred = (y_prob >= thr).astype(int)
    conf = np.abs(y_prob - 0.5)
    conf_thr = np.quantile(conf, q)
    hi = conf >= conf_thr
    return {
        "TP_hi": (y_true==1)&(y_pred==1)&hi,
        "TN_hi": (y_true==0)&(y_pred==0)&hi,
        "FP_hi": (y_true==0)&(y_pred==1)&hi,
        "FN_hi": (y_true==1)&(y_pred==0)&hi,
    }

def spearman_corr(x, y) -> float:
    """Spearman correlation (rank correlation) with NaN-safe handling."""
    xs = pd.to_numeric(pd.Series(x), errors="coerce")
    ys = pd.to_numeric(pd.Series(y), errors="coerce")
    m = xs.notna() & ys.notna()
    if int(m.sum()) < 2:
        return float("nan")
    return float(xs[m].corr(ys[m], method="spearman"))

def _read_csv_smart(path: str) -> pd.DataFrame:
    """Lightweight CSV reader tolerant to BOM and delimiter issues."""
    if (not path) or (not os.path.exists(path)):
        raise FileNotFoundError(path)
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
    return df


# ----------------------------
# (1) Read results docx (optional)
# ----------------------------
def read_results_docx(docx_path: str, out_csv: str) -> None:
    from docx import Document
    doc = Document(docx_path)
    rows_all = []
    for ti, table in enumerate(doc.tables):
        for ri, row in enumerate(table.rows):
            cells = [c.text.strip() for c in row.cells]
            rows_all.append([ti, ri] + cells)
    pd.DataFrame(rows_all).to_csv(out_csv, index=False, encoding="utf-8-sig")


# ----------------------------
# Model import from your model.py
# ----------------------------
def import_user_module(model_py: str):
    import importlib.util
    import sys

    module_name = "user_model_module"
    spec = importlib.util.spec_from_file_location(module_name, model_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {model_py}")

    m = importlib.util.module_from_spec(spec)

    # 关键：先注册到 sys.modules，避免 Python 3.13 dataclasses 取不到模块字典
    sys.modules[module_name] = m

    spec.loader.exec_module(m)  # type: ignore
    return m

def build_c_model_from_ckpt(user_mod, c_ckpt_path: str, device: str):
    ckpt = torch.load(c_ckpt_path, map_location="cpu")
    cfg_dict = ckpt.get("cfg", {})
    cfg_obj = user_mod.Config(**cfg_dict)

    # Force memmap_dir to your true location to avoid cfg mismatch
    if OVERRIDE_MEMMAP_DIR:
        cfg_obj.memmap_dir = OVERRIDE_MEMMAP_DIR

    # infer dp/dg/L from memmaps (lazy)
    t5_te_wt = np.load(os.path.join(cfg_obj.memmap_dir, "test_t5_wt.npy"), mmap_mode="r")
    gpn_te_ref = np.load(os.path.join(cfg_obj.memmap_dir, "test_gpn_ref.npy"), mmap_mode="r")
    _, L, dp = t5_te_wt.shape
    _, L2, dg = gpn_te_ref.shape
    del t5_te_wt, gpn_te_ref
    if L != L2:
        raise RuntimeError("ProtT5 and GPN test L mismatch in memmaps.")

    t5_enc  = user_mod.Scheme2PairEncoder(d_in=dp, seq_len=L, cfg=cfg_obj, backbone=cfg_obj.t5_backbone)
    gpn_enc = user_mod.Scheme2PairEncoder(d_in=dg, seq_len=L, cfg=cfg_obj, backbone=cfg_obj.gpn_backbone)
    c_model = user_mod.DualBranchFusionModel(t5_enc, gpn_enc, cfg_obj)

    state = ckpt.get("model_state_dict", ckpt)
    state = {re.sub(r"^module\.", "", k): v for k, v in state.items()}
    c_model.load_state_dict(state, strict=False)
    c_model.to(device).eval()

    # gate tau is NOT in state_dict; set to training end for interpretability
    if hasattr(c_model, "set_gate_tau"):
        c_model.set_gate_tau(float(getattr(cfg_obj, "gate_tau_end", 2.0)))

    return c_model, cfg_obj


# ----------------------------
# Hooks
# ----------------------------
class HookCache:
    def __init__(self):
        self.cache: Dict[str, torch.Tensor] = {}

    def hook(self, key: str):
        def fn(_m, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(t):
                self.cache[key] = t
        return fn

def register_hooks(c_model: nn.Module) -> Tuple[HookCache, List[Any]]:
    hc = HookCache()
    handles = []
    handles.append(c_model.t5_enc.register_forward_hook(hc.hook("prot_vec")))
    handles.append(c_model.gpn_enc.register_forward_hook(hc.hook("dna_vec")))
    handles.append(c_model.gate_mlp.register_forward_hook(hc.hook("gate_logit")))
    handles.append(c_model.t5_enc.backbone.register_forward_hook(hc.hook("prot_tokens")))
    handles.append(c_model.gpn_enc.backbone.register_forward_hook(hc.hook("dna_tokens")))
    return hc, handles


# ----------------------------
# Memmap dataset (test)
# ----------------------------
class TestMemmapDataset(torch.utils.data.Dataset):
    def __init__(self, memmap_dir: str):
        self.t5_wt = np.load(os.path.join(memmap_dir, "test_t5_wt.npy"), mmap_mode="r")
        self.t5_mut= np.load(os.path.join(memmap_dir, "test_t5_mut.npy"), mmap_mode="r")
        self.gpn_ref=np.load(os.path.join(memmap_dir, "test_gpn_ref.npy"), mmap_mode="r")
        self.gpn_alt=np.load(os.path.join(memmap_dir, "test_gpn_alt.npy"), mmap_mode="r")
        assert self.t5_wt.shape == self.t5_mut.shape
        assert self.gpn_ref.shape == self.gpn_alt.shape
        assert self.t5_wt.shape[0] == self.gpn_ref.shape[0]

    def __len__(self): return self.t5_wt.shape[0]

    def __getitem__(self, i: int):
        t5w = torch.from_numpy(self.t5_wt[i].copy())
        t5m = torch.from_numpy(self.t5_mut[i].copy())
        gr  = torch.from_numpy(self.gpn_ref[i].copy())
        ga  = torch.from_numpy(self.gpn_alt[i].copy())
        return t5w, t5m, gr, ga


@torch.no_grad()
def extract_branch_and_gate(c_model: nn.Module, memmap_dir: str, device: str, batch_size: int = 64) -> Dict[str, np.ndarray]:
    ds = TestMemmapDataset(memmap_dir)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    hc, handles = register_hooks(c_model)

    dna_vecs, prot_vecs, fused_vecs, gate_alphas = [], [], [], []
    prot_in_deltas, dna_in_deltas = [], []
    for bt5w, bt5m, bgref, bgalt in tqdm(dl, desc="Extract DNA/Protein/gate", leave=False):
        bt5w = bt5w.to(device).float()
        bt5m = bt5m.to(device).float()
        bgref= bgref.to(device).float()
        bgalt= bgalt.to(device).float()

        hc.cache.clear()
        fused, _logit, _lt5, _lgpn = c_model(bt5w, bt5m, bgref, bgalt)

        prot_vec = hc.cache["prot_vec"]          # [B,128]
        dna_vec  = hc.cache["dna_vec"]           # [B,128]
        gate_logit = hc.cache["gate_logit"]      # [B,1]

        tau = float(getattr(c_model, "_gate_tau", 1.0))
        alpha = torch.sigmoid(gate_logit / max(tau, 1e-6))  # [B,1]

        # Input-level summaries (3-input-panel style): mean pooled delta over sequence
        prot_in_delta = torch.nanmean(bt5m - bt5w, dim=1)
        dna_in_delta  = torch.nanmean(bgalt - bgref, dim=1)

        prot_vecs.append(prot_vec.detach().cpu().numpy())
        dna_vecs.append(dna_vec.detach().cpu().numpy())
        fused_vecs.append(fused.detach().cpu().numpy())
        gate_alphas.append(alpha.detach().cpu().numpy())
        prot_in_deltas.append(prot_in_delta.detach().cpu().numpy())
        dna_in_deltas.append(dna_in_delta.detach().cpu().numpy())

    for h in handles:
        try: h.remove()
        except Exception: pass

    return {
        "dna_vec": np.concatenate(dna_vecs, axis=0),
        "prot_vec": np.concatenate(prot_vecs, axis=0),
        "fused_vec": np.concatenate(fused_vecs, axis=0),
        "gate_alpha": np.concatenate(gate_alphas, axis=0),
        "prot_in_delta": np.concatenate(prot_in_deltas, axis=0),
        "dna_in_delta": np.concatenate(dna_in_deltas, axis=0),
    }


# ----------------------------
# (2) Score distributions
# ----------------------------
def plot_score_distributions(y_true: np.ndarray, y_prob: np.ndarray, outdir: str, thr=0.5):
    ensure_dir(outdir)

    fig = plt.figure(figsize=(7,4))
    plt.hist(y_prob[y_true==0], bins=40, alpha=0.6, label="label=0")
    plt.hist(y_prob[y_true==1], bins=40, alpha=0.6, label="label=1")
    plt.xlabel("pred_prob"); plt.ylabel("count"); plt.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "dist_label.png"), dpi=200); plt.close(fig)

    g = confusion_group(y_true, y_prob, thr)
    names = {0:"TN",1:"FP",2:"FN",3:"TP"}
    fig = plt.figure(figsize=(7,4))
    for k in [0,1,2,3]:
        plt.hist(y_prob[g==k], bins=40, alpha=0.55, label=names[k])
    plt.xlabel("pred_prob"); plt.ylabel("count"); plt.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "dist_TP_TN_FP_FN.png"), dpi=200); plt.close(fig)

def plot_confusion_matrix(y_true: np.ndarray, y_prob: np.ndarray, out_path: str, thr: float = 0.5):
    """
    Save confusion matrix (counts + normalized) with red-blue colormap consistent with UMAP.
    Output file extension can be .png; savefig hook will generate PDF+SVG.
    """
    ensure_dir(os.path.dirname(out_path) or ".")
    y_pred = (y_prob >= thr).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for ax, mat, title, fmt in [
        (axes[0], cm,  "Confusion matrix (counts)", "{:d}"),
        (axes[1], cmn, "Confusion matrix (normalized)", "{:.2f}"),
    ]:
        im = ax.imshow(
            mat,
            vmin=0.0,
            vmax=float(np.max(mat)) if title.endswith("(counts)") else 1.0,
            cmap=PINKBLUE_CMAP
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["0", "1"]); ax.set_yticklabels(["0", "1"])

        # annotate
        for i in range(2):
            for j in range(2):
                ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center", fontsize=PLOT_FONT_SIZE)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout(pad=0.4)
    _savefig(fig, out_path, dpi=FIG_DPI)
    plt.close(fig)


# ----------------------------
# (3) Embedding plots
# ----------------------------
def reduce_2d(emb: np.ndarray, method: str, random_state: int = 42) -> np.ndarray:
    emb_std = StandardScaler().fit_transform(emb)
    method = method.lower().strip()
    if method == "tsne":
        try:
            return TSNE(
                n_components=2, perplexity=30, n_iter=1500,
                random_state=random_state, init="pca"
            ).fit_transform(emb_std)
        except TypeError:
            return TSNE(
                n_components=2, perplexity=30, max_iter=1500,
                random_state=random_state, init="pca"
            ).fit_transform(emb_std)

    if method == "umap":
        umap = try_import_umap()
        if umap is None:
            raise RuntimeError("UMAP not installed. pip install umap-learn")
        return umap.UMAP(
            n_components=2, n_neighbors=30, min_dist=0.1,
            metric="euclidean", random_state=random_state
        ).fit_transform(emb_std)
    raise ValueError(method)

def scatter_save(z: np.ndarray, c: np.ndarray, title: str, outpath: str):
    fig = plt.figure(figsize=(6.2,5.3))
    plt.scatter(z[:,0], z[:,1], s=10, alpha=0.85, c=c)
    plt.title(title); plt.xlabel("dim-1"); plt.ylabel("dim-2")
    fig.tight_layout(); fig.savefig(outpath, dpi=200); plt.close(fig)

def plot_embedding(name: str, emb: np.ndarray, y_true: np.ndarray, y_prob: np.ndarray,
                   outdir: str, methods: List[str], random_state: int = 42):
    ensure_dir(outdir)
    for method in methods:
        method = method.lower().strip()
        if not method:
            continue
        try:
            z = reduce_2d(emb, method, random_state=random_state)
        except Exception as e:
            print(f"[WARN] reduce_2d failed for {name}-{method}: {e}")
            continue
        np.save(os.path.join(outdir, f"{name}_{method}_2d.npy"), z)
        scatter_save(z, y_true, f"{name}-{method.upper()} color=label",
                     os.path.join(outdir, f"{name}_{method}_label.png"))
        scatter_save(z, y_prob, f"{name}-{method.upper()} color=pred",
                     os.path.join(outdir, f"{name}_{method}_pred.png"))


# ----------------------------
# (3b) Dataset summary plots (paper style)
# ----------------------------
def plot_train_test_class_counts(train_csv: str, test_csv: str, out_path: str, label_col: str = "label"):
    """
    Paper style: horizontal 100% stacked bars (Train/Test), with in-bar counts and right-side N (INSIDE).
    """
    if not train_csv or not test_csv:
        return
    tr = _read_csv_smart(train_csv)
    te = _read_csv_smart(test_csv)
    if label_col not in tr.columns or label_col not in te.columns:
        raise KeyError(f"Missing label column '{label_col}' in train/test CSV.")

    tr_y = tr[label_col].astype(int).to_numpy()
    te_y = te[label_col].astype(int).to_numpy()

    counts = {
        "Train": {"Negative": int((tr_y == 0).sum()), "Positive": int((tr_y == 1).sum())},
        "Test":  {"Negative": int((te_y == 0).sum()), "Positive": int((te_y == 1).sum())},
    }

    groups = ["Train", "Test"]
    neg = np.array([counts[g]["Negative"] for g in groups], dtype=float)
    pos = np.array([counts[g]["Positive"] for g in groups], dtype=float)
    tot = neg + pos
    tot_safe = np.where(tot > 0, tot, 1.0)
    neg_p = neg / tot_safe
    pos_p = pos / tot_safe

    fig, ax = plt.subplots(figsize=(8.2, 2.8))
    y = np.arange(len(groups))

    ax.barh(y, neg_p, color=COLOR_CLASS_NEG, edgecolor="white", height=0.38, label="Negative")
    ax.barh(y, pos_p, left=neg_p, color=COLOR_CLASS_POS, edgecolor="white", height=0.38, label="Positive")

    for i, g in enumerate(groups):
        # segment counts
        ax.text(neg_p[i] * 0.5, y[i], f"Neg: {int(neg[i])}",
                ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(neg_p[i] + pos_p[i] * 0.5, y[i], f"Pos: {int(pos[i])}",
                ha="center", va="center", fontsize=9, fontweight="bold")

        # total N (OUTSIDE, to the right of the bar)
        ax.text(
            1.01, y[i], f"N = {int(tot[i])}",
            transform=ax.get_yaxis_transform(),  # x in axes coords, y in data coords
            ha="left", va="center", fontsize=9,
            clip_on=False
        )

    ax.set_yticks(y)
    ax.set_yticklabels(groups)
    ax.set_xlim(0, 1.0)  # keep within [0,1] so labels won't fall outside
    ax.set_xlabel("Proportion")
    ax.set_title("Dataset composition of training and test sets")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.legend(loc="lower left", frameon=True, fontsize=9)

    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path) or ".")
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

DISEASE_COUNTS_HARDCODED = [
    ("Intellectual Disability", 1470),
    ("Alzheimer", 270),
    ("Attention Deficit", 18),
    ("Language Disorder", 2),
    ("Autism", 333),
    ("Schizophrenia", 71),
    ("Tourette", 3),
    ("Obsessive-compulsive Disorder", 1),
]

def plot_disease_counts_hardcoded(out_path: str):
    """
    Paper style: donut chart with legend showing 'Category: count'.
    HARD-CODED counts; no file reading.
    Uses DISEASE_PALETTE (8 colors) with interleaved order for better contrast.
    """
    ensure_dir(os.path.dirname(out_path) or ".")

    # Keep your earlier visual order preference: 1-5-2-6-3-7-4-8 (0-based: 0,4,1,5,2,6,3,7)
    order = [0, 4, 1, 5, 2, 6, 3, 7]
    items = [DISEASE_COUNTS_HARDCODED[i] for i in order]

    labels = [x[0] for x in items]
    values = [int(x[1]) for x in items]
    colors = [DISEASE_PALETTE[i] for i in order]
    legend_labels = [f"{lab}: {cnt}" for lab, cnt in zip(labels, values)]

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    wedges, _ = ax.pie(
        values,
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.38, edgecolor="white"),
    )
    ax.set_title("Disease category counts")

    ax.legend(
        wedges,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(1.30, 1.0),
        frameon=True,
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_missing_values_bar(missing_csv: str, out_path: str):
    """
    Plot missing values for each tool based on:
      missing = total - n_used
    Required columns in CSV: tool_name, total, n_used
    """
    if (not missing_csv) or (not os.path.exists(missing_csv)):
        print(f"[WARN] Missing-value plot skipped: file not found: {missing_csv}")
        return

    df = _read_csv_smart(missing_csv)
    # normalize column names
    cols = {str(c).lower().strip(): str(c) for c in df.columns}
    for need in ["tool_name", "total", "n_used"]:
        if need not in cols:
            raise ValueError(f"{missing_csv} must contain column '{need}' (found: {list(df.columns)})")
    tool_col = cols["tool_name"]
    total_col = cols["total"]
    used_col = cols["n_used"]

    tmp = df[[tool_col, total_col, used_col]].copy()
    tmp[total_col] = pd.to_numeric(tmp[total_col], errors="coerce")
    tmp[used_col] = pd.to_numeric(tmp[used_col], errors="coerce")
    tmp = tmp.dropna(subset=[tool_col, total_col, used_col])

    tmp["missing"] = (tmp[total_col] - tmp[used_col]).astype(float)
    tmp["missing"] = tmp["missing"].clip(lower=0)

    # sort by missing descending for readability
    tmp = tmp.sort_values("missing", ascending=False)

    tools = tmp[tool_col].astype(str).tolist()
    miss = tmp["missing"].to_numpy(dtype=float)

    n = len(tools)
    if n == 0:
        print(f"[WARN] Missing-value plot skipped: empty after cleaning: {missing_csv}")
        return

    # one distinct color per bar
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(n)]

    # figure size adapts to number of tools
    fig_w = max(8.0, 0.42 * n)
    fig, ax = plt.subplots(figsize=(fig_w, 4.2))
    x = np.arange(n)
    bars = ax.bar(x, miss, color=colors, edgecolor="white", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(tools, rotation=45, ha="right")
    ax.set_ylabel("Missing count (total - n_used)")
    ax.set_title("Tool-wise missing values")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.6)

    # annotate counts on top
    for rect, val in zip(bars, miss):
        ax.text(rect.get_x() + rect.get_width() / 2.0, rect.get_height(),
                f"{int(val)}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    _savefig(fig, out_path, dpi=FIG_DPI)
    plt.close(fig)

def _detect_chr_col(df: pd.DataFrame) -> Optional[str]:
    cands = ["Chr", "CHR", "chrom", "Chrom", "chromosome", "Chromosome"]
    low = {str(c).lower(): str(c) for c in df.columns}
    for c in cands:
        if c in df.columns:
            return c
        if c.lower() in low:
            return low[c.lower()]
    return None

def _normalize_chr(x: str) -> str:
    s = str(x).strip()
    s = re.sub(r"^chr", "", s, flags=re.I)
    s = s.upper()
    if s in ["X", "Y", "MT", "M"]:
        return "MT" if s in ["MT", "M"] else s
    try:
        i = int(float(s))
        return str(i)
    except Exception:
        return s

def plot_chromosome_distribution_jitter_total(train_csv: str, test_csv: str, out_path: str):
    """
    Your requirement: do NOT split train vs test.
    Read counts from train+test, then plot total distribution (Manhattan-like jitter columns by chr).
    For chr with count n:
      x = chr_index + uniform(-0.28, 0.28)
      y = uniform(0, log10(n+1))
    """
    if (not train_csv) or (not test_csv):
        return
    if (not os.path.exists(train_csv)) or (not os.path.exists(test_csv)):
        return

    tr = _read_csv_smart(train_csv)
    te = _read_csv_smart(test_csv)

    c_tr = _detect_chr_col(tr)
    c_te = _detect_chr_col(te)
    if c_tr is None or c_te is None:
        print("[WARN] Chromosome distribution skipped: chromosome column not found.")
        return

    all_df = pd.concat(
        [tr[[c_tr]].rename(columns={c_tr: "Chr"}),
         te[[c_te]].rename(columns={c_te: "Chr"})],
        axis=0, ignore_index=True
    )
    all_df["Chr"] = all_df["Chr"].map(_normalize_chr)

    def chr_sort_key(ch: str):
        if str(ch).isdigit():
            return (0, int(ch))
        if ch == "X":
            return (1, 23)
        if ch == "Y":
            return (1, 24)
        if ch == "MT":
            return (1, 25)
        return (2, 1000, str(ch))

    vc = all_df["Chr"].value_counts()
    chrs = sorted(vc.index.tolist(), key=chr_sort_key)
    counts = [int(vc.get(c, 0)) for c in chrs]

    xs, ys, cs = [], [], []
    cmap = plt.get_cmap("tab20")
    for i, (c, n) in enumerate(zip(chrs, counts), start=1):
        if n <= 0:
            continue
        x0 = i
        ymax = float(np.log10(n + 1.0))
        xj = x0 + np.random.uniform(-0.28, 0.28, size=n)
        yj = np.random.uniform(0.0, max(ymax, 1e-6), size=n)
        xs.append(xj)
        ys.append(yj)
        cs.append(np.tile(cmap((i - 1) % 20), (n, 1)))

    if not xs:
        print("[WARN] Chromosome distribution skipped: no points.")
        return

    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    C = np.concatenate(cs, axis=0)

    fig, ax = plt.subplots(figsize=(11.8, 3.6))
    ax.scatter(X, Y, s=6, alpha=0.65, c=C, edgecolors="none")

    ax.set_xlim(0.5, len(chrs) + 0.5)
    ax.set_xticks(np.arange(1, len(chrs) + 1))
    ax.set_xticklabels(chrs, fontsize=8)
    ax.set_ylabel("log10(count+1)")
    ax.set_xlabel("Chromosome")
    ax.set_title("The distribution of samples by chromosome")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    _savefig(fig, out_path, dpi=FIG_DPI)
    plt.close(fig)


# ----------------------------
# (3c) Six-panel UMAP (3 inputs + 3 stage outputs) - colored by probability
# ----------------------------
def _make_prob_scatter(ax, z2: np.ndarray, prob: np.ndarray, title: str, vmin=0.0, vmax=1.0):
    sc = ax.scatter(
        z2[:, 0], z2[:, 1],
        s=10, alpha=0.85,
        c=prob, vmin=vmin, vmax=vmax,
        cmap="coolwarm",
        edgecolors="none"
    )
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    return sc

def _align_tool_score_table(seq_df: pd.DataFrame, tool_csv: str) -> Optional[np.ndarray]:
    """Align a tool-score (functional annotation) table to seq_df row order."""
    if not tool_csv or not os.path.exists(tool_csv):
        return None

    tool = _read_csv_smart(tool_csv)

    def resolve(df: pd.DataFrame, name: str) -> str:
        if name in df.columns:
            return name
        m = {str(c).lower(): str(c) for c in df.columns}
        if name.lower() in m:
            return m[name.lower()]
        raise KeyError(f"Column '{name}' not found in dataframe.")

    k_seq = [resolve(seq_df, "Chrom"), resolve(seq_df, "Position"), resolve(seq_df, "Reference"), resolve(seq_df, "Alternate")]
    k_tool = [resolve(tool, "Chrom"), resolve(tool, "Position"), resolve(tool, "Reference"), resolve(tool, "Alternate")]

    seq_key = seq_df[k_seq].astype(str).agg("|".join, axis=1)
    tool_key = tool[k_tool].astype(str).agg("|".join, axis=1)

    tool = tool.loc[~tool_key.duplicated()].copy()
    tool_key = tool_key.loc[~tool_key.duplicated()].copy()
    tool.index = tool_key.values

    drop_cols = set(k_tool + ["label", "Label", "y_true", "y", "Y"])
    feat_cols = [c for c in tool.columns if c not in drop_cols]
    if not feat_cols:
        return None

    sub = tool.reindex(seq_key.values)[feat_cols]
    sub = sub.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = sub.median(axis=0, skipna=True).fillna(0)
    sub = sub.fillna(med).fillna(0)
    return sub.to_numpy(dtype=np.float32)

def plot_umap_six_panels(
    reps: List[Tuple[str, np.ndarray]],
    y_prob: np.ndarray,
    out_path: str,
    random_state: int = 42,
):
    """Create a 2x3 UMAP panel. Each representation is embedded independently. Colored by predicted probability."""
    umap = try_import_umap()
    if umap is None:
        print("[WARN] UMAP not installed; six-panel UMAP skipped.")
        return

    prob = np.asarray(y_prob).reshape(-1).astype(float)
    vmin, vmax = 0.0, 1.0

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.2))
    axes = axes.reshape(-1)

    last_sc = None
    for i, (title, X) in enumerate(reps[:6]):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Rep '{title}' must be 2D, got shape={X.shape}")
        X = np.asarray(X, dtype=np.float32)
        X[~np.isfinite(X)] = np.nan

        col_med = np.nanmedian(X, axis=0)
        col_med = np.where(np.isfinite(col_med), col_med, 0.0)
        inds = np.where(np.isnan(X))
        X[inds] = col_med[inds[1]]

        Xs = StandardScaler().fit_transform(X)
        z2 = umap.UMAP(
            n_components=2, n_neighbors=30, min_dist=0.1,
            metric="euclidean", random_state=random_state,
        ).fit_transform(Xs)

        last_sc = _make_prob_scatter(axes[i], z2, prob, title, vmin=vmin, vmax=vmax)

    for j in range(len(reps), 6):
        axes[j].axis("off")

    # Leave space on the right for a dedicated colorbar axis
    fig.subplots_adjust(right=0.90, wspace=0.45, hspace=0.28)

    if last_sc is not None:
        # [left, bottom, width, height] in figure fraction coordinates
        cax = fig.add_axes([0.92, 0.14, 0.015, 0.72])
        cbar = fig.colorbar(last_sc, cax=cax)
        cbar.set_label("Predicted pathogenicity score")

    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# (4) Gate stats + proxy importance
# ----------------------------
def plot_gate_stats(alpha: np.ndarray, y_true: np.ndarray, y_prob: np.ndarray, outdir: str, thr=0.5):
    ensure_dir(outdir)
    a = alpha.reshape(-1)

    fig = plt.figure(figsize=(7,4))
    plt.hist(a[y_true==0], bins=40, alpha=0.6, label="label=0")
    plt.hist(a[y_true==1], bins=40, alpha=0.6, label="label=1")
    plt.xlabel("gate_alpha"); plt.ylabel("count"); plt.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "gate_by_label.png"), dpi=200); plt.close(fig)

    g = confusion_group(y_true, y_prob, thr)
    names = {0:"TN",1:"FP",2:"FN",3:"TP"}
    fig = plt.figure(figsize=(7,4))
    for k in [0,1,2,3]:
        plt.hist(a[g==k], bins=40, alpha=0.55, label=names[k])
    plt.xlabel("gate_alpha"); plt.ylabel("count"); plt.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "gate_by_TP_TN_FP_FN.png"), dpi=200); plt.close(fig)

def build_proxy_X(dna_vec, prot_vec, fused_vec, gate_alpha):
    parts, groups, feat_names = [], {}, []
    c0 = 0

    def add(arr: np.ndarray, gname: str):
        nonlocal c0
        parts.append(arr)
        cols = np.arange(c0, c0 + arr.shape[1])
        groups[gname] = cols
        feat_names.extend([f"{gname}_{i}" for i in range(arr.shape[1])])
        c0 += arr.shape[1]

    add(dna_vec, "DNA")
    add(prot_vec, "PROT")
    add(fused_vec, "FUSED")
    add(gate_alpha.reshape(-1,1), "GATE")

    X = np.concatenate(parts, axis=1)
    return X, groups, feat_names

def proxy_train_xgb(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    xgb = try_import_xgboost()
    if xgb is None:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.05, max_iter=400, random_state=random_state)
        m.fit(X, y)
        return m

    m = xgb.XGBClassifier(
        n_estimators=600, max_depth=5, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, objective="binary:logistic",
        eval_metric="logloss", random_state=random_state, n_jobs=8
    )
    m.fit(X, y)
    return m

def proxy_proba(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:,1]
    return model.predict(X).astype(float)

def group_permutation_importance(model, X: np.ndarray, y: np.ndarray, groups: Dict[str, np.ndarray],
                                 n_repeats: int, out_csv: str):
    base = proxy_proba(model, X)
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    base_auc = roc_auc_score(y, base)
    base_pr  = average_precision_score(y, base)
    base_f1  = f1_score(y, (base>=0.5).astype(int), zero_division=0)

    rows = []
    for gname, cols in groups.items():
        drops_auc, drops_pr, drops_f1 = [], [], []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = np.random.permutation(len(Xp))
            Xp[:, cols] = Xp[perm][:, cols]
            pp = proxy_proba(model, Xp)
            drops_auc.append(base_auc - roc_auc_score(y, pp))
            drops_pr.append(base_pr  - average_precision_score(y, pp))
            drops_f1.append(base_f1  - f1_score(y, (pp>=0.5).astype(int), zero_division=0))
        rows.append({
            "group": gname,
            "auc_drop_mean": float(np.mean(drops_auc)),
            "auc_drop_std":  float(np.std(drops_auc)),
            "aupr_drop_mean": float(np.mean(drops_pr)),
            "aupr_drop_std":  float(np.std(drops_pr)),
            "f1_drop_mean": float(np.mean(drops_f1)),
            "f1_drop_std":  float(np.std(drops_f1)),
        })

    pd.DataFrame(rows).sort_values("auc_drop_mean", ascending=False).to_csv(out_csv, index=False)

def proxy_gain_importance_if_xgb(model, feat_names: List[str], out_csv: str):
    rows = []
    try:
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
            score = booster.get_score(importance_type="gain")
            for k, v in score.items():
                fi = int(k[1:])
                rows.append({"feature": feat_names[fi], "gain": float(v)})
    except Exception:
        pass
    if rows:
        pd.DataFrame(rows).sort_values("gain", ascending=False).to_csv(out_csv, index=False)


# ----------------------------
# (5) SHAP on proxy
# ----------------------------
def run_shap(proxy_model, X: np.ndarray, y_true: np.ndarray, feat_names: List[str], outdir: str, num_local: int = 8):
    shap = try_import_shap()
    if shap is None:
        print("[WARN] shap not installed; skip SHAP (pip install shap).")
        return
    ensure_dir(outdir)

    keep_mask = np.array([not str(n).upper().startswith("GATE_") for n in feat_names], dtype=bool)
    explainer = shap.TreeExplainer(proxy_model)
    sv_full = explainer.shap_values(X)

    def _to_matrix(sv_obj):
        if isinstance(sv_obj, list):
            if len(sv_obj) >= 2:
                return sv_obj[1]
            return sv_obj[0]
        return sv_obj

    sv_full_mat = _to_matrix(sv_full)

    if keep_mask.size == sv_full_mat.shape[1] and not keep_mask.all():
        X_plot = X[:, keep_mask]
        sv_plot_mat = sv_full_mat[:, keep_mask]
        feat_names_plot = [n for n, k in zip(feat_names, keep_mask) if k]
    else:
        X_plot = X
        sv_plot_mat = sv_full_mat
        feat_names_plot = feat_names

    target_beeswarm_halfspan = 0.16
    max_abs_shap = float(np.nanmax(np.abs(sv_plot_mat))) if sv_plot_mat.size else 1.0
    display_scale = max(1.0, np.ceil(max_abs_shap / target_beeswarm_halfspan))
    sv_plot_disp = sv_plot_mat / display_scale

    def _pretty_name(n):
        s = str(n).replace("_", " ")
        while "  " in s:
            s = s.replace("  ", " ")
        return s

    feat_names_plot_disp = [_pretty_name(n) for n in feat_names_plot]

    bar_path = os.path.join(outdir, "shap_global_bar.png")
    bee_path = os.path.join(outdir, "shap_global_beeswarm.png")
    combined_path = os.path.join(outdir, "shap_global_combined.png")

    # Bar plot as a standalone wide panel.
    fig1 = plt.figure(figsize=(9.5, 12.5))
    ax1 = fig1.add_subplot(111)
    plt.sca(ax1)
    shap.summary_plot(
        sv_plot_disp, X_plot,
        feature_names=feat_names_plot_disp,
        plot_type="bar",
        max_display=40,
        show=False
    )
    ax1.set_xlim(0.0, 0.10)
    ax1.set_xticks([0.0, 0.05, 0.10])
    ax1.set_xlabel("mean(|SHAP value|) (average impact on model output magnitude)")
    fig1.tight_layout()
    fig1.savefig(bar_path, dpi=220, bbox_inches="tight", pad_inches=0.45)
    plt.close(fig1)

    # Beeswarm plot as a standalone wide panel.
    fig2 = plt.figure(figsize=(12.8, 12.5))
    ax2 = fig2.add_subplot(111)
    plt.sca(ax2)
    shap.summary_plot(
        sv_plot_disp, X_plot,
        feature_names=feat_names_plot_disp,
        max_display=40,
        show=False
    )
    beeswarm_lim = max(0.12, float(np.nanmax(np.abs(sv_plot_disp))) * 1.05)
    ax2.set_xlim(-beeswarm_lim, beeswarm_lim)
    tick_step = 0.05 if beeswarm_lim <= 0.20 else 0.10
    tick_max = np.ceil(beeswarm_lim / tick_step) * tick_step
    ticks = np.arange(-tick_max, tick_max + tick_step * 0.5, tick_step)
    ax2.set_xticks(ticks)
    ax2.tick_params(axis="x", labelsize=9)
    ax2.set_xlabel("SHAP value (impact on model output)")
    fig2.tight_layout()
    fig2.savefig(bee_path, dpi=220, bbox_inches="tight", pad_inches=0.45)
    plt.close(fig2)

    # Stitch the two standalone panels side-by-side so panel b looks like two full plots combined.
    from PIL import Image
    im1 = Image.open(bar_path).convert("RGBA")
    im2 = Image.open(bee_path).convert("RGBA")
    gap = 70
    W = im1.width + gap + im2.width
    H = max(im1.height, im2.height)
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    canvas.paste(im1, (0, (H - im1.height) // 2))
    canvas.paste(im2, (im1.width + gap, (H - im2.height) // 2))
    canvas.save(combined_path)
    canvas.convert("RGB").save(os.path.splitext(combined_path)[0] + ".pdf", "PDF", resolution=220.0)

    # Save a small set of local explanations (FULL feature space, so local arrays stay aligned).
    probs = proxy_proba(proxy_model, X)
    groups = hi_conf_groups(y_true, probs)

    picked = []
    for g in ["TP", "TN", "FP", "FN"]:
        picked.extend(groups.get(g, []))
    picked = picked[:num_local] if len(picked) > num_local else picked
    if len(picked) == 0:
        return

    local_sv = sv_full_mat[picked, :]
    local_X = X[picked, :]

    np.save(os.path.join(outdir, "local_shap_values.npy"), local_sv)
    np.save(os.path.join(outdir, "local_X.npy"), local_X)
    with open(os.path.join(outdir, "local_feat_names.txt"), "w", encoding="utf-8") as f:
        for n in feat_names:
            f.write(str(n) + "\n")

def topk_windows(seq: str, scores: np.ndarray, w: int, k: int) -> List[Tuple[int,str,float]]:
    L = min(len(seq), len(scores))
    seq = seq[:L]; scores = scores[:L]
    if L == 0:
        return []
    if L <= w:
        return [(0, seq, float(scores.mean()))]
    win = np.convolve(scores, np.ones(w, dtype=float), mode="valid")
    idx = np.argsort(win)[::-1]
    picked = []
    used = np.zeros_like(win, dtype=bool)
    for j in idx:
        if len(picked) >= k: break
        if used[max(0, j-w): min(len(used), j+w)].any(): continue
        st = int(j)
        picked.append((st, seq[st:st+w], float(win[j]/w)))
        used[st] = True
    return picked

def export_fasta(recs: List[Tuple[str,str]], outpath: str):
    with open(outpath, "w", encoding="utf-8") as f:
        for rid, s in recs:
            f.write(f">{rid}\n{s}\n")

def run_meme_cli(fasta: str, outdir: str, is_dna: bool):
    meme = shutil.which("meme")
    if meme is None:
        print("[WARN] meme not found in PATH; skip MEME.")
        return
    ensure_dir(outdir)
    cmd = [meme, fasta, "-oc", outdir, "-nostatus", "-time", "18000", "-maxsize", "10000000"]
    cmd += ["-dna"] if is_dna else ["-protein"]
    cmd += ["-nmotifs","5","-minw","6","-maxw","15"]
    subprocess.run(cmd, check=False)

def run_tomtom_cli(query_meme_txt: str, target_db_meme: str, outdir: str):
    tomtom = shutil.which("tomtom")
    if tomtom is None:
        print("[WARN] tomtom not found in PATH; skip Tomtom.")
        return
    ensure_dir(outdir)
    cmd = [tomtom, "-oc", outdir, "-no-ssc", "-min-overlap", "5", query_meme_txt, target_db_meme]
    subprocess.run(cmd, check=False)

def grad_attribution_on_tokens(c_model: nn.Module, memmap_dir: str, device: str,
                               seq_csv: Optional[str], outdir: str,
                               window_dna: int = 15, window_prot: int = 11,
                               topk: int = 2, max_samples: int = 2000,
                               run_meme: bool = False, run_tomtom: bool = False,
                               dna_db: str = ""):
    ensure_dir(outdir)

    def _pad_to_len(s: str, L: int, pad_char: str) -> str:
        s = "" if s is None else str(s)
        if len(s) >= L:
            return s[:L]
        return s + pad_char * (L - len(s))

    seq_df = None
    if seq_csv and os.path.exists(seq_csv):
        seq_df = pd.read_csv(seq_csv)
        if "sample_index" not in seq_df.columns:
            raise RuntimeError("seq_csv must have column: sample_index")
        seq_df = seq_df.set_index("sample_index")
    else:
        print("[WARN] seq_csv not provided or missing -> will export window indices only (no real FASTA).")

    ds = TestMemmapDataset(memmap_dir)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, drop_last=False, num_workers=0)

    hc, handles = register_hooks(c_model)

    n_total = len(ds)
    n_use = min(n_total, max_samples)

    dna_alt_windows, dna_ref_windows = [], []
    prot_mut_windows, prot_wt_windows = [], []

    c_model.eval()
    for p in c_model.parameters():
        p.requires_grad_(True)

    seen = 0
    for _bi, (bt5w, bt5m, bgref, bgalt) in enumerate(tqdm(dl, desc="Motif grad attribution", leave=False)):
        if seen >= n_use:
            break
        take = min(bt5w.size(0), n_use - seen)

        bt5w = bt5w[:take].to(device).float().detach()
        bgref= bgref[:take].to(device).float().detach()
        bt5m  = bt5m[:take].to(device).float().detach().requires_grad_(True)
        bgalt = bgalt[:take].to(device).float().detach().requires_grad_(True)

        hc.cache.clear()
        _fused, logits_fused, _, _ = c_model(bt5w, bt5m, bgref, bgalt)

        pos = logits_fused.view(-1).sum()
        c_model.zero_grad(set_to_none=True)
        pos.backward()

        g_imp = torch.norm(bgalt.grad, dim=-1).detach().cpu().numpy() if (bgalt.grad is not None) else None
        t_imp = torch.norm(bt5m.grad,  dim=-1).detach().cpu().numpy() if (bt5m.grad  is not None) else None

        if g_imp is None or t_imp is None:
            seen += take
            continue

        for j in range(take):
            sample_idx = seen + j

            if seq_df is not None and sample_idx in seq_df.index:
                dna_alt = _pad_to_len(seq_df.loc[sample_idx].get(SEQ_COL_DNA_ALT, ""), 128, "N")
                dna_ref = _pad_to_len(seq_df.loc[sample_idx].get(SEQ_COL_DNA_REF, ""), 128, "N")
                prot_mut = _pad_to_len(seq_df.loc[sample_idx].get(SEQ_COL_PROT_MUT, ""), 128, "X")
                prot_wt  = _pad_to_len(seq_df.loc[sample_idx].get(SEQ_COL_PROT_WT,  ""), 128, "X")
            else:
                dna_alt = "N" * len(g_imp[j])
                dna_ref = "N" * len(g_imp[j])
                prot_mut = "X" * len(t_imp[j])
                prot_wt  = "X" * len(t_imp[j])

            dna_wins = topk_windows(dna_alt, g_imp[j], window_dna, topk)
            for wi, (st, _subseq, sc) in enumerate(dna_wins):
                dna_alt_windows.append((f"s{sample_idx}_dnaALT_w{wi}_st{st}_s{sc:.4f}", dna_alt[st:st+window_dna]))
                dna_ref_windows.append((f"s{sample_idx}_dnaREF_w{wi}_st{st}_s{sc:.4f}", dna_ref[st:st+window_dna]))

            prot_wins = topk_windows(prot_mut, t_imp[j], window_prot, topk)
            for wi, (st, _subseq, sc) in enumerate(prot_wins):
                prot_mut_windows.append((f"s{sample_idx}_protMUT_w{wi}_st{st}_s{sc:.4f}", prot_mut[st:st+window_prot]))
                prot_wt_windows.append((f"s{sample_idx}_protWT_w{wi}_st{st}_s{sc:.4f}",  prot_wt[st:st+window_prot]))

        seen += take

    for h in handles:
        try: h.remove()
        except Exception: pass

    if dna_alt_windows:
        dna_alt_fa = os.path.join(outdir, "DNA_ALT_windows.fasta")
        export_fasta(dna_alt_windows, dna_alt_fa)
        if run_meme:
            meme_out = os.path.join(outdir, "DNA_ALT_meme")
            run_meme_cli(dna_alt_fa, meme_out, is_dna=True)
            if run_tomtom and dna_db:
                q = os.path.join(meme_out, "meme.txt")
                if os.path.exists(q):
                    run_tomtom_cli(q, dna_db, os.path.join(outdir, "DNA_ALT_tomtom"))

    if dna_ref_windows:
        dna_ref_fa = os.path.join(outdir, "DNA_REF_windows.fasta")
        export_fasta(dna_ref_windows, dna_ref_fa)

    if prot_mut_windows:
        prot_mut_fa = os.path.join(outdir, "PROT_MUT_windows.fasta")
        export_fasta(prot_mut_windows, prot_mut_fa)
        if run_meme:
            meme_out = os.path.join(outdir, "PROT_MUT_meme")
            run_meme_cli(prot_mut_fa, meme_out, is_dna=False)

    if prot_wt_windows:
        prot_wt_fa = os.path.join(outdir, "PROT_WT_windows.fasta")
        export_fasta(prot_wt_windows, prot_wt_fa)


# ----------------------------
# Helpers
# ----------------------------
def save_arrays(arr_root: str, dna_vec: np.ndarray, prot_vec: np.ndarray, fused_vec: np.ndarray, gate_alpha: np.ndarray):
    ensure_dir(arr_root)
    np.save(os.path.join(arr_root, "dna_vec.npy"), dna_vec)
    np.save(os.path.join(arr_root, "prot_vec.npy"), prot_vec)
    np.save(os.path.join(arr_root, "fused_vec.npy"), fused_vec)
    np.save(os.path.join(arr_root, "gate_alpha.npy"), gate_alpha)

def run_proxy_bundle(proxy_root: str, dna_vec: np.ndarray, prot_vec: np.ndarray, fused_vec: np.ndarray,
                     gate_alpha: np.ndarray, y_true: np.ndarray):
    ensure_dir(proxy_root)
    X, groups, feat_names = build_proxy_X(dna_vec, prot_vec, fused_vec, gate_alpha)
    proxy = proxy_train_xgb(X, y_true)
    group_permutation_importance(proxy, X, y_true, groups, n_repeats=10,
                                 out_csv=os.path.join(proxy_root, "group_permutation_importance.csv"))
    proxy_gain_importance_if_xgb(proxy, feat_names, os.path.join(proxy_root, "proxy_gain_importance.csv"))
    run_shap(proxy, X, y_true, feat_names, os.path.join(proxy_root, "shap"), num_local=8)

def autodetect_folds(out_dir: str) -> List[int]:
    fa = os.path.join(out_dir, "fold_artifacts")
    if not os.path.isdir(fa):
        return []
    folds = []
    for name in os.listdir(fa):
        m = re.match(r"fold(\d+)$", name)
        if m:
            folds.append(int(m.group(1)))
    return sorted(folds)

def sanity_check():
    if not os.path.isdir(OUT_DIR):
        raise FileNotFoundError(f"OUT_DIR not found: {OUT_DIR}")

    if not os.path.isfile(MODEL_PY):
        raise FileNotFoundError(f"model.py not found: {MODEL_PY}")

    pred_csv = os.path.join(OUT_DIR, "pred", "pred_test_ensemble.csv")
    if not os.path.exists(pred_csv):
        raise FileNotFoundError(f"Missing required file: {pred_csv}")

    for fn in ["test_t5_wt.npy","test_t5_mut.npy","test_gpn_ref.npy","test_gpn_alt.npy"]:
        p = os.path.join(OVERRIDE_MEMMAP_DIR, fn)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required memmap: {p}")


# ==================================================================================================
# Additional visualizations (5-8):
#   - Rare/Gene subset ROC+PR overlays
#   - Spearman correlation + prediction agreement (combined, half numeric / half color)
#   - Test score distribution + base-substitution score heatmap (left-right)
# ==================================================================================================

TOOL_COLUMN_CANDIDATES = {
    "SIFT": ["SIFT_score", "SIFT"],
    "PolyPhen2": ["Polyphen2_HVAR_score", "PolyPhen2_HVAR_score", "Polyphen2_score", "PolyPhen2_score", "PolyPhen2"],
    "FATHMM": ["FATHMM_score", "FATHMM"],
    "PROVEAN": ["PROVEAN_score", "PROVEAN"],
    "MPC": ["MPC_score", "MPC"],
    "DEOGEN2": ["DEOGEN2_score", "DEOGEN2"],
    "AlphaMissense": ["AlphaMissense_score", "alphamissense.score", "AlphaMissense"],
    "CADD v1.7": ["CADD_phred", "CADD_v1.7", "cadd", "CADD"],
    "DANN": ["DANN_score", "DANN"],
    "GenoCanyon": ["GenoCanyon_score", "GenoCanyon"],
    "PrimateAI": ["PrimateAI_score", "PrimateAI", "primateai"],
}

def _resolve_tool_col(df: pd.DataFrame, candidates: list) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _get_key_cols(df: pd.DataFrame) -> list:
    cols = ["Chrom", "Position", "Reference", "Alternate"]
    low = {c.lower(): c for c in df.columns}
    out = []
    for c in cols:
        if c in df.columns:
            out.append(c)
        elif c.lower() in low:
            out.append(low[c.lower()])
        else:
            return []
    return out

def _build_key(df: pd.DataFrame, key_cols: list) -> pd.Series:
    return df[key_cols].astype(str).agg("|".join, axis=1)

def _pick_threshold_mcc(y_true: np.ndarray, score: np.ndarray, max_grid: int = 500) -> float:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)
    uniq = np.unique(score[np.isfinite(score)])
    if uniq.size == 0:
        return 0.5
    if uniq.size > max_grid:
        uniq = np.quantile(score[np.isfinite(score)], np.linspace(0.0, 1.0, max_grid))

    best_t, best_v = 0.5, -1e18
    for t in uniq:
        pred = (score >= t).astype(int)
        v = metrics.matthews_corrcoef(y_true, pred)
        if v > best_v:
            best_v, best_t = float(v), float(t)
    return float(best_t)

def pick_threshold_mcc(y_true, scores) -> float:
    return float(_pick_threshold_mcc(np.asarray(y_true), np.asarray(scores)))

def _prepare_method_table(
    test_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    tool_df: pd.DataFrame,
    label_col: str = "label",
) -> tuple[pd.DataFrame, List[str], dict]:
    key_cols = _get_key_cols(tool_df)
    if not key_cols:
        raise ValueError("tool score table must contain Chrom/Position/Reference/Alternate (case-insensitive).")

    if "sample_index" not in test_df.columns:
        test_df = test_df.copy()
        test_df["sample_index"] = np.arange(len(test_df), dtype=int)

    if "sample_index" not in pred_df.columns:
        raise ValueError("prediction CSV must contain 'sample_index'.")
    if "prob_ensemble" not in pred_df.columns:
        raise ValueError("prediction CSV must contain 'prob_ensemble'.")

    base = test_df.copy()
    if label_col not in base.columns:
        raise ValueError(f"test CSV must contain label column '{label_col}'.")

    base = base.merge(
        pred_df[["sample_index", "prob_ensemble"]],
        on="sample_index",
        how="left",
        validate="one_to_one",
    )
    base = base.rename(columns={"prob_ensemble": "GenProt-DSM"})

    tool = tool_df.copy()
    tool_cols = {}
    for name, cands in TOOL_COLUMN_CANDIDATES.items():
        col = _resolve_tool_col(tool, cands)
        if col is not None:
            tool_cols[name] = col

    if len(tool_cols) == 0:
        raise ValueError("No tool-score columns found in tool score CSV. Please verify column names.")

    tool_keep = key_cols + list(tool_cols.values())
    tool = tool[tool_keep].copy()

    k_tool = _build_key(tool, key_cols)
    if k_tool.duplicated().any():
        tool = tool.loc[~k_tool.duplicated()].copy()

    base_key = _build_key(base, key_cols)
    tool_key = _build_key(tool, key_cols)
    tool_indexed = tool.set_index(tool_key)

    for disp, col in tool_cols.items():
        base[disp] = pd.to_numeric(tool_indexed.loc[base_key, col].to_numpy(), errors="coerce")

    flips = {}
    y = base[label_col].astype(int).to_numpy()
    for disp in tool_cols.keys():
        s = base[disp].to_numpy(dtype=float)
        mask = np.isfinite(s)
        if mask.sum() < 5 or len(np.unique(y[mask])) < 2:
            flips[disp] = False
            continue
        try:
            auc0 = roc_auc_score(y[mask], s[mask])
        except Exception:
            auc0 = np.nan
        if np.isfinite(auc0) and auc0 < 0.5:
            base[disp] = -base[disp]
            flips[disp] = True
        else:
            flips[disp] = False

    flips["GenProt-DSM"] = False

    ordered = ["GenProt-DSM"] + list(tool_cols.keys())
    keep_cols = key_cols + [label_col] + ordered
    base = base[keep_cols].copy()

    return base, ordered, flips

def _subset_mask_from_csv(base_df: pd.DataFrame, subset_csv: str) -> np.ndarray:
    if not subset_csv or not os.path.exists(subset_csv):
        return np.zeros(len(base_df), dtype=bool)

    sub = _read_csv_smart(subset_csv)
    if "sample_index" in sub.columns:
        idx = sub["sample_index"].astype(int).to_numpy()
        m = np.zeros(len(base_df), dtype=bool)
        idx = idx[(idx >= 0) & (idx < len(base_df))]
        m[idx] = True
        return m

    key_cols = _get_key_cols(base_df)
    if not key_cols:
        raise ValueError("Base table missing Chrom/Position/Reference/Alternate key columns.")

    k_base = _build_key(base_df, key_cols)
    k_sub = _build_key(sub, _get_key_cols(sub) or key_cols)
    sub_set = set(k_sub.to_list())
    return k_base.isin(sub_set).to_numpy(dtype=bool)

def _compute_curve_stats(y: np.ndarray, score: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    if len(y) < 2 or len(np.unique(y)) < 2:
        return {}

    fpr, tpr, _ = metrics.roc_curve(y, score)
    auc_roc = float(metrics.auc(fpr, tpr))

    prec, rec, _ = metrics.precision_recall_curve(y, score)
    auc_pr = float(metrics.auc(rec, prec))

    return {
        'fpr': fpr,
        'tpr': tpr,
        'prec': prec,
        'rec': rec,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
    }

def _plot_roc_pr_overlay_on_axes(
    df: pd.DataFrame,
    methods: List[str],
    label_col: str,
    ax_roc,
    ax_pr,
    title_prefix: str,
    ours_name: str = OURS_NAME,
):
    stats = {}
    for m in methods:
        if m not in df.columns:
            continue
        s = pd.to_numeric(df[m], errors='coerce').to_numpy(dtype=float)
        y = pd.to_numeric(df[label_col], errors='coerce').to_numpy(dtype=float)
        mask = np.isfinite(s) & np.isfinite(y)
        if mask.sum() < 5:
            continue
        st = _compute_curve_stats(y[mask].astype(int), s[mask].astype(float))
        if st:
            stats[m] = st

    if not stats:
        ax_roc.text(0.5, 0.5, 'No valid curves', ha='center', va='center')
        ax_pr.text(0.5, 0.5, 'No valid curves', ha='center', va='center')
        return

    base_methods = [m for m in stats.keys() if m != ours_name]
    cmap = plt.get_cmap('tab20')
    base_colors = {m: cmap(i % 20) for i, m in enumerate(sorted(base_methods))}

    roc_order = sorted(stats.keys(), key=lambda k: stats[k]['auc_roc'])
    if ours_name in roc_order:
        roc_order = [m for m in roc_order if m != ours_name] + [ours_name]

    pr_order = sorted(stats.keys(), key=lambda k: stats[k]['auc_pr'])
    if ours_name in pr_order:
        pr_order = [m for m in pr_order if m != ours_name] + [ours_name]

    ax_roc.plot([0, 1], [0, 1], linestyle='--', linewidth=1.0, color='k', alpha=0.6)
    for m in roc_order:
        st = stats[m]
        if m == ours_name:
            ax_roc.plot(st['fpr'], st['tpr'], color='red', linewidth=2.6,
                        label=f"{m} (AUROC={st['auc_roc']:.3f})")
        else:
            ax_roc.plot(st['fpr'], st['tpr'], color=base_colors.get(m, None), linewidth=1.0, alpha=0.9,
                        label=f"{m} (AUROC={st['auc_roc']:.3f})")
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title(f"{title_prefix} ROC")
    ax_roc.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)

    for m in pr_order:
        st = stats[m]
        if m == ours_name:
            ax_pr.plot(st['rec'], st['prec'], color='red', linewidth=2.6,
                       label=f"{m} (AUPR={st['auc_pr']:.3f})")
        else:
            ax_pr.plot(st['rec'], st['prec'], color=base_colors.get(m, None), linewidth=1.0, alpha=0.9,
                       label=f"{m} (AUPR={st['auc_pr']:.3f})")
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title(f"{title_prefix} PR")
    ax_pr.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)

    ax_roc.legend(loc='lower right', fontsize=7, frameon=True)
    ax_pr.legend(loc='lower left', fontsize=7, frameon=True)

def plot_alltools_rocpr_fulltest(
    df12: pd.DataFrame,
    methods12: List[str],
    out_dir: str,
    label_col: str = "label",
) -> str:
    """
    Full test set ROC+PR overlay for all methods (GenProt-DSM + 11 tools).
    Output: <out_dir>/roc_pr_all_methods_fulltest.png
    """
    ensure_dir(out_dir)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_roc, ax_pr = axes[0], axes[1]

    _plot_roc_pr_overlay_on_axes(
        df=df12,
        methods=methods12,
        label_col=label_col,
        ax_roc=ax_roc,
        ax_pr=ax_pr,
        title_prefix="Full test set",
        ours_name=OURS_NAME,
    )

    # Panel labels
    ax_roc.text(-0.18, 1.10, "A", transform=ax_roc.transAxes, fontsize=14, fontweight="bold", va="top")
    ax_pr.text(-0.18, 1.10, "B", transform=ax_pr.transAxes, fontsize=14, fontweight="bold", va="top")

    fig.tight_layout()
    out_path = os.path.join(out_dir, "roc_pr_all_methods_fulltest.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path

def plot_rare_gene_rocpr_combined(
    df12: pd.DataFrame,
    subset_rare_csv: str,
    subset_gene_csv: str,
    out_dir: str,
    methods12: List[str],
    label_col: str = "label",
) -> str:
    ensure_dir(out_dir)

    rare_mask = _subset_mask_from_csv(df12, subset_rare_csv)
    gene_mask = _subset_mask_from_csv(df12, subset_gene_csv)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axA_roc, axA_pr = axes[0, 0], axes[0, 1]
    axB_roc, axB_pr = axes[1, 0], axes[1, 1]

    df_rare = df12.loc[rare_mask].copy()
    _plot_roc_pr_overlay_on_axes(
        df=df_rare,
        methods=methods12,
        label_col=label_col,
        ax_roc=axA_roc,
        ax_pr=axA_pr,
        title_prefix='Rare subset (AF < 0.01)',
    )

    df_gene = df12.loc[gene_mask].copy()
    _plot_roc_pr_overlay_on_axes(
        df=df_gene,
        methods=methods12,
        label_col=label_col,
        ax_roc=axB_roc,
        ax_pr=axB_pr,
        title_prefix='Gene-independent subset (unseen genes)',
    )

    axA_roc.text(-0.18, 1.10, 'A', transform=axA_roc.transAxes, fontsize=14, fontweight='bold', va='top')
    axB_roc.text(-0.18, 1.10, 'B', transform=axB_roc.transAxes, fontsize=14, fontweight='bold', va='top')

    fig.tight_layout()
    out_path = os.path.join(out_dir, 'roc_pr_rare_gene_overlays.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_path

def plot_spearman_and_agreement(df12: pd.DataFrame, methods12: List[str], out_dir: str, label_col: str = 'label') -> str:
    """
    One figure with 1x2 subplots:
      Left: Spearman correlation (pairwise overlap)
      Right: Prediction agreement (pairwise overlap)
    In each subplot:
      - Upper triangle: numbers only (no color)
      - Lower triangle: colored bubbles (no numbers)
    """
    ensure_dir(out_dir)

    methods = [m for m in methods12 if m in df12.columns]
    y_all = pd.to_numeric(df12[label_col], errors='coerce').to_numpy(dtype=float)

    thresholds = {}
    for m in methods:
        s = pd.to_numeric(df12[m], errors='coerce').to_numpy(dtype=float)
        mask = np.isfinite(s) & np.isfinite(y_all)
        if mask.sum() < 10 or len(np.unique(y_all[mask].astype(int))) < 2:
            thresholds[m] = 0.5
            continue
        thresholds[m] = float(pick_threshold_mcc(y_all[mask].astype(int), s[mask].astype(float)))

    n = len(methods)
    rho = np.full((n, n), np.nan, dtype=float)
    agree = np.full((n, n), np.nan, dtype=float)

    for i in range(n):
        rho[i, i] = 1.0
        agree[i, i] = 1.0

    for i in range(n):
        si = pd.to_numeric(df12[methods[i]], errors='coerce').to_numpy(dtype=float)
        for j in range(i + 1, n):
            sj = pd.to_numeric(df12[methods[j]], errors='coerce').to_numpy(dtype=float)
            mask = np.isfinite(si) & np.isfinite(sj) & np.isfinite(y_all)
            if mask.sum() < 10:
                continue

            r = spearman_corr(si[mask], sj[mask])
            rho[i, j] = r
            rho[j, i] = r

            ti = thresholds[methods[i]]
            tj = thresholds[methods[j]]
            pi = (si[mask] >= ti).astype(int)
            pj = (sj[mask] >= tj).astype(int)
            a = float((pi == pj).mean())
            agree[i, j] = a
            agree[j, i] = a

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axL, axR = axes[0], axes[1]

    def _draw_half_numeric_half_bubble(ax, mat, title, vmin, vmax, cbar_label, is_agreement=False):
        ax.set_title(title)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_yticklabels(methods)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)

        cmap = plt.get_cmap('RdBu_r')
        norm = plt.Normalize(vmin, vmax)

        for i in range(n):
            for j in range(i + 1, n):
                v = mat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha='center', va='center', fontsize=8)

        for i in range(1, n):
            for j in range(0, i):
                v = mat[i, j]
                if not np.isfinite(v):
                    continue
                if is_agreement:
                    size = 900 * (max(v, 0.0) ** 1.2)
                    color_val = v
                else:
                    size = 900 * (abs(v) ** 1.2)
                    color_val = v
                ax.scatter(j, i, s=size, c=[cmap(norm(color_val))], edgecolors='none')

        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)

    _draw_half_numeric_half_bubble(
        ax=axL, mat=rho,
        title="Spearman correlation (pairwise overlap)",
        vmin=-1.0, vmax=1.0,
        cbar_label="Spearman ρ",
        is_agreement=False
    )
    _draw_half_numeric_half_bubble(
        ax=axR, mat=agree,
        title="Prediction agreement (pairwise overlap)",
        vmin=0.0, vmax=1.0,
        cbar_label="Agreement",
        is_agreement=True
    )

    fig.tight_layout()
    out_path = os.path.join(out_dir, "spearman_and_agreement_combined.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_path

def plot_score_distribution_and_substitution_heatmap(
    test_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    out_path: str,
    label_col: str = "label",
):
    """GenProt-DSM only: (A) score distribution (left) + (B) base-substitution score heatmap (right)."""
    if "sample_index" not in test_df.columns:
        test_df = test_df.copy()
        test_df["sample_index"] = np.arange(len(test_df), dtype=int)

    df = test_df.merge(pred_df[["sample_index", "prob_ensemble"]], on="sample_index", how="left")
    df = df.rename(columns={"prob_ensemble": "GenProt-DSM"})

    scores = pd.to_numeric(df["GenProt-DSM"], errors="coerce").to_numpy(dtype=float)
    scores = scores[np.isfinite(scores)]

    fig = plt.figure(figsize=(12.6, 4.6), dpi=FIG_DPI)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, 1, 21)
    counts, edges = np.histogram(scores, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cmap = plt.get_cmap("coolwarm")
    colors = cmap(centers)
    ax1.bar(centers, counts, width=(edges[1] - edges[0]) * 0.95, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("Prediction score")
    ax1.set_ylabel("Variant count")
    ax1.set_title("A  Prediction score distribution (GenProt-DSM)")
    ax1.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    ax2 = fig.add_subplot(gs[0, 1])
    ref_col = "Reference" if "Reference" in df.columns else ("Ref" if "Ref" in df.columns else None)
    alt_col = "Alternate" if "Alternate" in df.columns else ("Alt" if "Alt" in df.columns else None)
    if ref_col is None or alt_col is None:
        print("[WARN] substitution heatmap skipped: Reference/Alternate columns not found in test CSV.")
        _savefig(fig, out_path)
        return

    sub_df = df[[ref_col, alt_col, "GenProt-DSM"]].copy()
    sub_df[ref_col] = sub_df[ref_col].astype(str).str.upper()
    sub_df[alt_col] = sub_df[alt_col].astype(str).str.upper()
    sub_df = sub_df[np.isfinite(pd.to_numeric(sub_df["GenProt-DSM"], errors="coerce"))]
    sub_df = sub_df[(sub_df[ref_col].str.len() == 1) & (sub_df[alt_col].str.len() == 1)]
    sub_df = sub_df[sub_df[ref_col].isin(list("ACGT")) & sub_df[alt_col].isin(list("ACGT"))]
    sub_df = sub_df[sub_df[ref_col] != sub_df[alt_col]]

    types = [f"{r}→{a}" for r in "ACGT" for a in "ACGT" if r != a]

    col_scores = []
    max_len = 0
    for t in types:
        r, a = t.split("→")
        s = pd.to_numeric(
            sub_df.loc[(sub_df[ref_col] == r) & (sub_df[alt_col] == a), "GenProt-DSM"],
            errors="coerce",
        ).to_numpy(dtype=float)
        s = s[np.isfinite(s)]
        s = np.sort(s)
        col_scores.append(s)
        max_len = max(max_len, len(s))

    if max_len == 0:
        print("[WARN] substitution heatmap skipped: no SNVs found.")
        _savefig(fig, out_path)
        return

    mat = np.full((max_len, len(types)), np.nan, dtype=float)
    for j, s in enumerate(col_scores):
        if len(s) == 0:
            continue
        mat[: len(s), j] = s

    # Use the same red-blue diverging palette as UMAP-style figures.
    im = ax2.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0, cmap=plt.get_cmap("viridis"))
    ax2.set_xticks(range(len(types)))
    ax2.set_xticklabels(types, rotation=45, ha="right")
    ax2.set_yticks([])
    ax2.set_xlabel("Base substitution")
    ax2.set_title("B  Substitution-specific score heatmap (GenProt-DSM)")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Prediction score")

    _savefig(fig, out_path)

# ==================================================================================================
# (9) Hard-coded experiment summaries (NO file reading)
#   - Module experiments: (ProtT5 module) + (GPN-MSA module) + (Fusion method)
#   - Dimension experiments: (equal d) + (fixed branches=128, vary fused_d)
#   These numbers are taken from your experiment record docx.
# ==================================================================================================

HARD_MODULE_EXPERIMENTS = {
    "ProtT5-side module (GPN-MSA: none)": [
        {"name": "transformer",  "auc_roc": 0.9104, "auc_pr": 0.9201},
        {"name": "TCN",          "auc_roc": 0.9181, "auc_pr": 0.9248},
        {"name": "BiLSTM",       "auc_roc": 0.9254, "auc_pr": 0.9329},
        {"name": "CNN",          "auc_roc": 0.9208, "auc_pr": 0.9257},
        {"name": "LSTM",         "auc_roc": 0.9223, "auc_pr": 0.9298},
        {"name": "GRU",          "auc_roc": 0.9202, "auc_pr": 0.9282},
        {"name": "CNN_BiLSTM",   "auc_roc": 0.9205, "auc_pr": 0.9201},
        {"name": "TCN_LSTM",     "auc_roc": 0.9101, "auc_pr": 0.9212},
    ],
    "GPN-MSA-side module (ProtT5: BiLSTM fixed)": [
        {"name": "transformer",  "auc_roc": 0.9376, "auc_pr": 0.9432},
        {"name": "TCN",          "auc_roc": 0.9436, "auc_pr": 0.9510},
        {"name": "BiLSTM",       "auc_roc": 0.9418, "auc_pr": 0.9478},
        {"name": "CNN",          "auc_roc": 0.9323, "auc_pr": 0.9389},
        {"name": "LSTM",         "auc_roc": 0.9425, "auc_pr": 0.9476},
        {"name": "GRU",          "auc_roc": 0.9449, "auc_pr": 0.9518},
        {"name": "CNN_BiLSTM",   "auc_roc": 0.9395, "auc_pr": 0.9479},
        {"name": "TCN_LSTM",     "auc_roc": 0.9443, "auc_pr": 0.9512},
    ],
    "Fusion method (ProtT5: BiLSTM, GPN-MSA: GRU fixed)": [
        {"name": "Gate",     "auc_roc": 0.9449, "auc_pr": 0.9518},
        {"name": "Concat",   "auc_roc": 0.9436, "auc_pr": 0.9481},
        {"name": "Bilinear", "auc_roc": 0.9373, "auc_pr": 0.9410},
        {"name": "Xattn",    "auc_roc": 0.9416, "auc_pr": 0.9498},
        {"name": "Avg",      "auc_roc": 0.9401, "auc_pr": 0.9381},
    ],
}

HARD_DIMENSION_EXPERIMENTS = {
    "Equal dims: protT5_d = gpn_d = fused_d": [
        {"d": 256, "auc_roc": 0.9422, "auc_pr": 0.9487},
        {"d": 128, "auc_roc": 0.9449, "auc_pr": 0.9518},
        {"d":  64, "auc_roc": 0.9234, "auc_pr": 0.9323},
        {"d":  32, "auc_roc": 0.9301, "auc_pr": 0.9384},
    ],
    "Fixed branches=128, vary fused_d": [
        {"d": 256, "auc_roc": 0.9410, "auc_pr": 0.9484},
        {"d": 128, "auc_roc": 0.9449, "auc_pr": 0.9518},
        {"d":  64, "auc_roc": 0.9393, "auc_pr": 0.9460},
        {"d":  32, "auc_roc": 0.9397, "auc_pr": 0.9428},
    ],
}

def plot_module_experiments_summary(out_path: str):
    """
    One figure for module experiments:
      2 rows: AUROC / AUPR
      3 cols: ProtT5-module / GPN-MSA-module / Fusion
    Improvements:
      - zoom x-axis to make gaps visible (do NOT start from 0)
      - assign distinct colors for each module/method bar
    """
    from matplotlib.ticker import MaxNLocator

    ensure_dir(os.path.dirname(out_path) or ".")

    panels = list(HARD_MODULE_EXPERIMENTS.items())
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 7.8))

    cmap = plt.get_cmap("tab20")

    for col, (title, rows) in enumerate(panels):
        df = pd.DataFrame(rows).copy()

        # deterministic color mapping per panel by module name
        name_list = sorted(df["name"].astype(str).unique().tolist())
        name2color = {n: cmap(i % 20) for i, n in enumerate(name_list)}

        # Sort by AUROC for bar order
        df = df.sort_values("auc_roc", ascending=True).reset_index(drop=True)
        y = np.arange(len(df))
        bar_colors = [name2color[str(n)] for n in df["name"].astype(str).tolist()]

        # ---------------- AUROC ----------------
        ax0 = axes[0, col]
        vals0 = df["auc_roc"].to_numpy(dtype=float)
        ax0.barh(y, vals0, height=0.72, color=bar_colors, edgecolor="white", linewidth=0.8)
        ax0.set_yticks(y)
        ax0.set_yticklabels(df["name"].tolist(), fontsize=9)
        ax0.set_xlabel("AUROC")
        ax0.set_title(title)
        ax0.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.6)

        # zoom xlim (make differences visible)
        vmin0, vmax0 = float(np.min(vals0)), float(np.max(vals0))
        pad0 = max(0.002, (vmax0 - vmin0) * 0.25)
        ax0.set_xlim(max(0.0, vmin0 - pad0), min(1.0, vmax0 + pad0))
        ax0.xaxis.set_major_locator(MaxNLocator(nbins=6))

        for i, v in enumerate(vals0):
            ax0.text(v + pad0 * 0.03, i, f"{v:.4f}", va="center", fontsize=8)

        # ---------------- AUPR ----------------
        ax1 = axes[1, col]
        vals1 = df["auc_pr"].to_numpy(dtype=float)
        ax1.barh(y, vals1, height=0.72, color=bar_colors, edgecolor="white", linewidth=0.8)
        ax1.set_yticks(y)
        ax1.set_yticklabels(df["name"].tolist(), fontsize=9)
        ax1.set_xlabel("AUPR")
        ax1.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.6)

        vmin1, vmax1 = float(np.min(vals1)), float(np.max(vals1))
        pad1 = max(0.002, (vmax1 - vmin1) * 0.25)
        ax1.set_xlim(max(0.0, vmin1 - pad1), min(1.0, vmax1 + pad1))
        ax1.xaxis.set_major_locator(MaxNLocator(nbins=6))

        for i, v in enumerate(vals1):
            ax1.text(v + pad1 * 0.03, i, f"{v:.4f}", va="center", fontsize=8)

    axes[0, 0].text(-0.18, 1.10, "A", transform=axes[0, 0].transAxes, fontsize=14, fontweight="bold", va="top")
    axes[1, 0].text(-0.18, 1.10, "B", transform=axes[1, 0].transAxes, fontsize=14, fontweight="bold", va="top")

    fig.suptitle("Module experiments summary", y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_dimension_experiments_summary(out_path: str):
    """
    One figure for dimension experiments:
      2 rows: AUROC / AUPR
      2 cols: (equal d) / (fixed branches=128, vary fused_d)
    """
    ensure_dir(os.path.dirname(out_path) or ".")

    panels = list(HARD_DIMENSION_EXPERIMENTS.items())
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.2))

    for col, (title, rows) in enumerate(panels):
        df = pd.DataFrame(rows).copy()
        df = df.sort_values("d", ascending=True)
        x = df["d"].to_numpy(dtype=int)

        ax0 = axes[0, col]
        ax0.plot(x, df["auc_roc"].to_numpy(), marker="o")
        ax0.set_xlabel("d")
        ax0.set_ylabel("AUROC")
        ax0.set_title(title)
        ax0.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
        for xi, yi in zip(x, df["auc_roc"].to_numpy()):
            ax0.text(xi, yi, f"{yi:.4f}", fontsize=8, ha="left", va="bottom")

        ax1 = axes[1, col]
        ax1.plot(x, df["auc_pr"].to_numpy(), marker="o")
        ax1.set_xlabel("d")
        ax1.set_ylabel("AUPR")
        ax1.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
        for xi, yi in zip(x, df["auc_pr"].to_numpy()):
            ax1.text(xi, yi, f"{yi:.4f}", fontsize=8, ha="left", va="bottom")

    fig.suptitle("Dimension experiments summary", y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

def scan_ablation_summaries(abla_root: str) -> List[Tuple[str, float, float]]:
    """
    Scan recursively for: **/ckpt/test_ensemble_summary.json
    Name = relative path of the ablation folder (parent of 'ckpt') from abla_root.
    Returns: (name, AUROC, AUPR)
    """
    if not abla_root or (not os.path.isdir(abla_root)):
        return []

    hits = []
    for root, _dirs, files in os.walk(abla_root):
        if "test_ensemble_summary.json" not in files:
            continue
        if os.path.basename(root).lower() != "ckpt":
            continue

        p = os.path.join(root, "test_ensemble_summary.json")
        abla_dir = os.path.dirname(root)
        name = os.path.relpath(abla_dir, abla_root).replace("\\", "/")

        try:
            js = load_json(p)
            fm = js.get("final_metrics", js.get("metrics", js))
            auroc = fm.get("auc_roc", fm.get("AUROC", fm.get("auroc")))
            aupr  = fm.get("auc_pr",  fm.get("AUPR",  fm.get("aupr")))
            if auroc is None or aupr is None:
                continue
            auroc = float(auroc); aupr = float(aupr)
            if auroc > 1.0: auroc /= 100.0
            if aupr  > 1.0: aupr  /= 100.0
            hits.append((name, auroc, aupr))
        except Exception:
            continue

    return sorted(hits, key=lambda x: x[0])

def plot_ablation_scatter(points: List[Tuple[str, float, float]], out_path: str, title: str):
    if not points:
        print(f"[WARN] Ablation scatter skipped: empty points for {title}")
        return

    fig, ax = plt.subplots(figsize=(6.4, 5.6))

    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    names = [p[0] for p in points]

    # color per point (match example palette; fallback to tab20)
    cmap = plt.get_cmap("tab20")
    cols = []
    for i, nm in enumerate(names):
        key = _norm_abla_name(nm)
        cols.append(ABLA_COLOR_MAP.get(key, cmap(i % 20)))

    ax.scatter(xs, ys, s=70, alpha=0.9, c=cols, edgecolors="white", linewidths=0.7)

    # Put all text captions into a bottom-right legend (instead of annotating near points)
    from matplotlib.lines import Line2D

    handles = []
    for (name, _x, _y), c in zip(points, cols):
        handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markerfacecolor=c,
                markeredgecolor="white",
                markeredgewidth=0.7,
                markersize=8,
                label=name,
            )
        )

    ax.legend(
        handles=handles,
        loc="lower right",          # bottom-right
        frameon=True,
        fontsize=9,
    )

    ax.set_xlabel("AUROC")
    ax.set_ylabel("AUPRC")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    _savefig(fig, out_path, dpi=FIG_DPI)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------


def _panel_label(ax, label: str):
    ax.text(-0.08, 1.06, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left")

def _plot_missing_values_bar_on_ax(ax, missing_csv: str):
    df = _read_csv_smart(missing_csv)
    cols = {str(c).lower().strip(): str(c) for c in df.columns}
    for need in ["tool_name", "total", "n_used"]:
        if need not in cols:
            raise ValueError(f"{missing_csv} must contain column '{need}'")
    tmp = df[[cols["tool_name"], cols["total"], cols["n_used"]]].copy()
    tmp.columns = ["tool_name", "total", "n_used"]
    tmp[["total", "n_used"]] = tmp[["total", "n_used"]].apply(pd.to_numeric, errors="coerce")
    tmp = tmp.dropna()
    tmp["missing"] = (tmp["total"] - tmp["n_used"]).clip(lower=0)
    tmp = tmp.sort_values("missing", ascending=False)
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(tmp))]
    x = np.arange(len(tmp))
    bars = ax.bar(x, tmp["missing"].to_numpy(dtype=float), color=colors, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(tmp["tool_name"].astype(str).tolist(), rotation=45, ha="right")
    ax.set_ylabel("Missing count")
    ax.set_title("Tool-wise missing values")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    for rect, val in zip(bars, tmp["missing"].to_numpy(dtype=float)):
        ax.text(rect.get_x()+rect.get_width()/2, rect.get_height(), f"{int(val)}", ha="center", va="bottom", fontsize=7)

def _plot_train_test_class_counts_on_ax(ax, train_csv: str, test_csv: str, label_col: str = "label"):
    tr = _read_csv_smart(train_csv); te = _read_csv_smart(test_csv)
    tr_y = tr[label_col].astype(int).to_numpy(); te_y = te[label_col].astype(int).to_numpy()
    counts = {"Train": {"Negative": int((tr_y == 0).sum()), "Positive": int((tr_y == 1).sum())}, "Test": {"Negative": int((te_y == 0).sum()), "Positive": int((te_y == 1).sum())}}
    groups = ["Train", "Test"]
    neg = np.array([counts[g]["Negative"] for g in groups], dtype=float)
    pos = np.array([counts[g]["Positive"] for g in groups], dtype=float)
    tot = neg + pos
    neg_p = neg / np.where(tot > 0, tot, 1.0)
    pos_p = pos / np.where(tot > 0, tot, 1.0)
    y = np.arange(len(groups))
    ax.barh(y, neg_p, color=COLOR_CLASS_NEG, edgecolor="white", height=0.38, label="Negative")
    ax.barh(y, pos_p, left=neg_p, color=COLOR_CLASS_POS, edgecolor="white", height=0.38, label="Positive")
    for i, g in enumerate(groups):
        ax.text(neg_p[i] * 0.5, y[i], f"Neg: {int(neg[i])}", ha="center", va="center", fontsize=7, fontweight="bold")
        ax.text(neg_p[i] + pos_p[i] * 0.5, y[i], f"Pos: {int(pos[i])}", ha="center", va="center", fontsize=7, fontweight="bold")
        ax.text(1.01, y[i], f"N = {int(tot[i])}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7, clip_on=False)
    ax.set_yticks(y); ax.set_yticklabels(groups); ax.set_xlim(0, 1.0)
    ax.set_xlabel("Proportion"); ax.set_title("Dataset composition of training and test sets")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.6); ax.legend(loc="lower left", frameon=True, fontsize=7)

def _plot_disease_counts_on_ax(ax):
    order = [0, 4, 1, 5, 2, 6, 3, 7]
    items = [DISEASE_COUNTS_HARDCODED[i] for i in order]
    labels = [x[0] for x in items]; values = [int(x[1]) for x in items]; colors = [DISEASE_PALETTE[i] for i in order]
    legend_labels = [f"{lab}: {cnt}" for lab, cnt in zip(labels, values)]
    wedges, _ = ax.pie(values, colors=colors, startangle=90, wedgeprops=dict(width=0.38, edgecolor="white"))
    ax.set_title("Disease category counts")
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=True, fontsize=7)

def _plot_chromosome_distribution_on_ax(ax, train_csv: str, test_csv: str):
    tr = _read_csv_smart(train_csv); te = _read_csv_smart(test_csv)
    c_tr = _detect_chr_col(tr); c_te = _detect_chr_col(te)
    all_df = pd.concat([tr[[c_tr]].rename(columns={c_tr: "Chr"}), te[[c_te]].rename(columns={c_te: "Chr"})], axis=0, ignore_index=True)
    all_df["Chr"] = all_df["Chr"].map(_normalize_chr)
    def chr_sort_key(ch: str):
        if str(ch).isdigit(): return (0, int(ch))
        if ch == "X": return (1, 23)
        if ch == "Y": return (1, 24)
        if ch == "MT": return (1, 25)
        return (2, 1000, str(ch))
    vc = all_df["Chr"].value_counts(); chrs = sorted(vc.index.tolist(), key=chr_sort_key); counts = [int(vc.get(c, 0)) for c in chrs]
    xs, ys, cs = [], [], []; cmap = plt.get_cmap("tab20")
    for i, (c, n) in enumerate(zip(chrs, counts), start=1):
        if n <= 0: continue
        ymax = float(np.log10(n + 1.0)); xs.append(i + np.random.uniform(-0.28, 0.28, size=n)); ys.append(np.random.uniform(0.0, max(ymax, 1e-6), size=n)); cs.append(np.tile(cmap((i - 1) % 20), (n, 1)))
    X = np.concatenate(xs); Y = np.concatenate(ys); C = np.concatenate(cs, axis=0)
    ax.scatter(X, Y, s=6, alpha=0.65, c=C, edgecolors="none")
    ax.set_xlim(0.5, len(chrs)+0.5); ax.set_xticks(np.arange(1, len(chrs)+1)); ax.set_xticklabels(chrs)
    ax.set_ylabel("log10(count+1)"); ax.set_xlabel("Chromosome"); ax.set_title("The distribution of samples by chromosome")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

def create_combined_fig1(train_csv: str, test_csv: str, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.2), gridspec_kw={"width_ratios": [1.2, 1.0]})
    _plot_chromosome_distribution_on_ax(axes[0], train_csv, test_csv); _panel_label(axes[0], "a")
    _plot_disease_counts_on_ax(axes[1]); _panel_label(axes[1], "b")
    fig.tight_layout()
    _savefig(fig, out_path, dpi=FIG_DPI)
    plt.close(fig)

def create_combined_fig3(out_path: str):
    fig = plt.figure(figsize=(19.0, 8.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.08, 0.92], hspace=0.42, wspace=0.32)

    def _module_panel(axspec, title, rows, label):
        sub = axspec.subgridspec(2, 1, hspace=0.36)
        ax_top = fig.add_subplot(sub[0, 0])
        ax_bot = fig.add_subplot(sub[1, 0])
        df = pd.DataFrame(rows).copy()
        order = df["auc_roc"].astype(float).sort_values(ascending=True).index
        df = df.loc[order].reset_index(drop=True)
        y = np.arange(len(df))
        names = df["name"].astype(str).tolist()
        name_list = sorted(set(names))
        cmap = plt.get_cmap("tab20")
        name2color = {n: cmap(i % 20) for i, n in enumerate(name_list)}
        colors = [name2color[n] for n in names]

        def _draw(ax, vals, xlabel):
            ax.barh(y, vals, height=0.62, color=colors, edgecolor="white", linewidth=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels(names, fontsize=7.5)
            ax.invert_yaxis()
            v = np.asarray(vals, dtype=float)
            vmin, vmax = float(np.min(v)), float(np.max(v))
            pad = max(0.0015, (vmax - vmin) * 0.18)
            ax.set_xlim(max(0.0, vmin - pad), min(1.0, vmax + pad))
            ax.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
            ax.set_xlabel(xlabel)
            for yi, vv in zip(y, v):
                ax.text(vv + pad * 0.08, yi, f"{vv:.4f}", va="center", ha="left", fontsize=7, color="black")

        _draw(ax_top, df["auc_roc"].to_numpy(dtype=float), "AUROC")
        _draw(ax_bot, df["auc_pr"].to_numpy(dtype=float), "AUPR")
        ax_top.set_title(title)
        _panel_label(ax_top, label)

    panels = list(HARD_MODULE_EXPERIMENTS.items())
    for col, (title, rows) in enumerate(panels):
        _module_panel(gs[0, col], title, rows, chr(ord('a') + col))

    subgs = gs[1, :].subgridspec(1, 2, wspace=0.24)
    axes = [fig.add_subplot(subgs[0, 0]), fig.add_subplot(subgs[0, 1])]
    for ax, (title, rows) in zip(axes, HARD_DIMENSION_EXPERIMENTS.items()):
        df = pd.DataFrame(rows).copy().sort_values("d", ascending=True)
        x = df["d"].to_numpy(dtype=int)
        ax.plot(x, df["auc_roc"].to_numpy(dtype=float), marker="o", label="AUROC")
        ax.plot(x, df["auc_pr"].to_numpy(dtype=float), marker="s", label="AUPR")
        ax.set_xticks([32, 64, 128, 256])
        ax.set_xticklabels(["32", "64", "128", "256"])
        for xi, yi in zip(x, df["auc_roc"].to_numpy(dtype=float)):
            ax.text(xi, yi, f"{yi:.4f}", fontsize=7, ha="left", va="bottom")
        for xi, yi in zip(x, df["auc_pr"].to_numpy(dtype=float)):
            ax.text(xi, yi, f"{yi:.4f}", fontsize=7, ha="left", va="top")
        ax.set_xlabel("d")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
        ax.legend(loc="lower right", fontsize=7)
    _panel_label(axes[0], "d")
    fig.tight_layout()
    _savefig(fig, out_path, dpi=FIG_DPI)
    plt.close(fig)

def _create_dual_curve_panel(fig, subspec, df, methods12, label_col, title_prefix, label):
    container = fig.add_subplot(subspec)
    container.axis("off")
    _panel_label(container, label)
    inner = subspec.subgridspec(1, 2, wspace=0.22)
    ax1 = fig.add_subplot(inner[0,0]); ax2 = fig.add_subplot(inner[0,1])
    _plot_roc_pr_overlay_on_axes(df=df, methods=methods12, label_col=label_col, ax_roc=ax1, ax_pr=ax2, title_prefix=title_prefix, ours_name=OURS_NAME)

def _maximal_shared_subset_mask(df: pd.DataFrame, methods: List[str], label_col: str = "label") -> np.ndarray:
    mask = np.isfinite(pd.to_numeric(df[label_col], errors="coerce").to_numpy(dtype=float))
    for m in methods:
        if m in df.columns:
            mask &= np.isfinite(pd.to_numeric(df[m], errors="coerce").to_numpy(dtype=float))
    return mask


def create_combined_fig4(df12: pd.DataFrame, methods12: List[str], missing_csv: str, subset_rare_csv: str, subset_gene_csv: str, out_path: str, label_col: str="label"):
    fig = plt.figure(figsize=(16.0, 13.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 1.0], width_ratios=[0.95, 1.35], hspace=0.34, wspace=0.22)

    # (a) overall VarCloseTest comparison as a dual ROC/PR panel
    _create_dual_curve_panel(fig, gs[0, :], df12.copy(), methods12, label_col, "VarCloseTest", "a")

    # (b) tool-specific missingness
    axb = fig.add_subplot(gs[1, 0])
    _plot_missing_values_bar_on_ax(axb, missing_csv)
    _panel_label(axb, "b")

    # (c) maximal shared subset comparison
    shared_mask = _maximal_shared_subset_mask(df12, methods12, label_col=label_col)
    _create_dual_curve_panel(fig, gs[1, 1], df12.loc[shared_mask].copy(), methods12, label_col, "Shared subset", "c")

    # (d) rare-variant comparison
    rare_mask = _subset_mask_from_csv(df12, subset_rare_csv)
    _create_dual_curve_panel(fig, gs[2, 0], df12.loc[rare_mask].copy(), methods12, label_col, "VarRareTest", "d")

    # (e) unseen-gene comparison
    gene_mask = _subset_mask_from_csv(df12, subset_gene_csv)
    _create_dual_curve_panel(fig, gs[2, 1], df12.loc[gene_mask].copy(), methods12, label_col, "VarGeneOutTest", "e")

    fig.tight_layout()
    _savefig(fig, out_path, dpi=FIG_DPI)
    plt.close(fig)



def _crop_whitespace(img):
    arr = np.asarray(img)
    if arr.ndim == 3:
        bg = np.array([255,255,255,255]) if arr.shape[2] == 4 else np.array([255,255,255])
        mask = np.any(arr[:, :, :len(bg)] < 250, axis=2)
    else:
        mask = arr < 250
    if not np.any(mask):
        return img
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()+1
    x0, x1 = xs.min(), xs.max()+1
    if arr.ndim == 3:
        return arr[y0:y1, x0:x1, :]
    return arr[y0:y1, x0:x1]

def _show_image_on_ax(ax, img_path: str, label: str):
    img = plt.imread(img_path)
    ax.imshow(img)
    ax.axis("off")
    _panel_label(ax, label)

def create_combined_fig5(mean_root: str, out_path: str):
    from PIL import Image, ImageDraw, ImageFont

    def _np_to_pil(arr):
        arr = np.asarray(arr)
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr, mode='L').convert('RGB')
        if arr.shape[-1] == 4:
            return Image.fromarray(arr, mode='RGBA').convert('RGB')
        return Image.fromarray(arr, mode='RGB')

    def _load_crop_pil(path):
        return _np_to_pil(_crop_whitespace(plt.imread(path)))

    img_a = _load_crop_pil(os.path.join(mean_root, "plots", "performance", "ablation_all_auroc_auprc_scatter.png"))
    img_b = _load_crop_pil(os.path.join(mean_root, "proxy", "shap", "shap_global_combined.png"))
    img_c = _load_crop_pil(os.path.join(mean_root, "plots", "embeddings", "UMAP_six_panel.png"))

    # Solve widths so that the stacked left column roughly matches the height of the long SHAP panel,
    # while keeping the whole figure compact and avoiding the large blank gap seen before.
    total_w = 3600
    margin = 24
    gap_x = 26
    gap_y = 18
    usable_w = total_w - 2 * margin - gap_x

    ra = img_a.height / float(img_a.width)
    rb = img_b.height / float(img_b.width)
    rc = img_c.height / float(img_c.width)

    left_w = int(max(900, min(1500, (rb * usable_w - gap_y) / (ra + rc + rb))))
    right_w = usable_w - left_w
    if right_w < 1400:
        right_w = 1400
        left_w = usable_w - right_w

    h_a = int(round(left_w * ra))
    h_c = int(round(left_w * rc))
    h_b = int(round(right_w * rb))
    left_h = h_a + gap_y + h_c
    H = max(left_h, h_b) + 2 * margin + 40

    canvas = Image.new('RGB', (total_w, H), (255, 255, 255))

    img_a_r = img_a.resize((left_w, h_a), Image.LANCZOS)
    img_c_r = img_c.resize((left_w, h_c), Image.LANCZOS)
    img_b_r = img_b.resize((right_w, h_b), Image.LANCZOS)

    xa = margin
    ya = margin + 22
    xc = margin
    yc = ya + h_a + gap_y
    xb = margin + left_w + gap_x
    yb = margin + 22 + max(0, (left_h - h_b) // 2)

    canvas.paste(img_a_r, (xa, ya))
    canvas.paste(img_c_r, (xc, yc))
    canvas.paste(img_b_r, (xb, yb))

    try:
        font = ImageFont.truetype("arial.ttf", 34)
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 34)
        except Exception:
            font = ImageFont.load_default()

    draw = ImageDraw.Draw(canvas)
    draw.text((xa - 16, ya - 36), "a", fill=(0, 0, 0), font=font)
    draw.text((xb - 16, yb - 36), "b", fill=(0, 0, 0), font=font)
    draw.text((xc - 16, yc - 36), "c", fill=(0, 0, 0), font=font)

    ensure_dir(os.path.dirname(out_path) or ".")
    canvas.save(out_path)
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    canvas.save(pdf_path, "PDF", resolution=FIG_DPI)

def create_combined_fig7(mean_root: str, out_path: str):
    fig = plt.figure(figsize=(14.0, 10.5))
    gs = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.15)
    axa = fig.add_subplot(gs[0,0]); axb = fig.add_subplot(gs[0,1]); axc = fig.add_subplot(gs[1,0]); axd = fig.add_subplot(gs[1,1])
    p1 = os.path.join(mean_root, "plots", "performance", "spearman_and_agreement_combined.png")
    p2 = os.path.join(mean_root, "plots", "performance", "genprot_score_distribution_and_substitution.png")
    # crop left/right halves
    img1 = plt.imread(p1); img2 = plt.imread(p2)
    mid1 = img1.shape[1] // 2; mid2 = img2.shape[1] // 2
    for ax, img, slc, label in [(axa, img1, slice(0, mid1), "a"), (axb, img1, slice(mid1, None), "b"), (axc, img2, slice(0, mid2), "c"), (axd, img2, slice(mid2, None), "d")]:
        ax.imshow(img[:, slc, ...]); ax.axis("off"); _panel_label(ax, label)
    fig.tight_layout()
    _savefig(fig, out_path, dpi=FIG_DPI)
    plt.close(fig)

def main():
    sanity_check()

    folds = autodetect_folds(OUT_DIR)
    if not folds:
        raise FileNotFoundError(f"No folds detected under: {os.path.join(OUT_DIR,'fold_artifacts')}")

    work = os.path.join(OUT_DIR, "explainability_6in1")
    folds_root = os.path.join(work, "folds")
    mean_root  = os.path.join(work, "mean")
    ensure_dir(folds_root)
    ensure_dir(mean_root)

    if RESULTS_DOCX and os.path.exists(RESULTS_DOCX):
        read_results_docx(RESULTS_DOCX, os.path.join(work, "results_docx_dump.csv"))

    pred_ens_path = os.path.join(OUT_DIR, "pred", "pred_test_ensemble.csv")
    pred_ens = pd.read_csv(pred_ens_path)

    test_df = pd.read_csv(TEST_CSV) if (TEST_CSV and os.path.exists(TEST_CSV)) else None

    if "sample_index" not in pred_ens.columns:
        pred_ens = pred_ens.copy()
        pred_ens["sample_index"] = np.arange(len(pred_ens), dtype=int)
    pred_df = pred_ens[["sample_index", "prob_ensemble"]].copy()

    if "y_true" not in pred_ens.columns or "prob_ensemble" not in pred_ens.columns:
        raise RuntimeError("pred_test_ensemble.csv must include columns: y_true, prob_ensemble")
    y_true = pred_ens["y_true"].values.astype(int)
    y_prob = pred_ens["prob_ensemble"].values.astype(float)

    thr = 0.5
    summ_path = os.path.join(OUT_DIR, "ckpt", "test_ensemble_summary.json")
    if os.path.exists(summ_path):
        summ = load_json(summ_path)
        thr = float(summ.get("ensemble_threshold", 0.5))

    plot_score_distributions(y_true, y_prob, os.path.join(mean_root, "plots", "score_distributions"), thr=thr)
    plot_confusion_matrix(y_true, y_prob, os.path.join(mean_root, "plots", "performance", "confusion_matrix.png"), thr=thr)

    # Dataset summary plots (paper style)
    try:
        ds_dir = os.path.join(mean_root, "plots", "dataset")
        os.makedirs(ds_dir, exist_ok=True)

        if TRAIN_CSV and TEST_CSV and os.path.exists(TRAIN_CSV) and os.path.exists(TEST_CSV):
            plot_train_test_class_counts(
                TRAIN_CSV, TEST_CSV,
                os.path.join(ds_dir, "dataset_composition_train_test.png"),
                label_col=LABEL_COL
            )
            plot_chromosome_distribution_jitter_total(
                TRAIN_CSV, TEST_CSV,
                os.path.join(ds_dir, "chromosome_distribution_total_jitter.png")
            )

        plot_disease_counts_hardcoded(os.path.join(ds_dir, "disease_counts_donut.png"))
        create_combined_fig1(TRAIN_CSV, TEST_CSV, os.path.join(ds_dir, "Fig1_combined.png"))
        # Tool-wise missing value visualization
        if MISSING_VALUE_CSV:
            plot_missing_values_bar(
                MISSING_VALUE_CSV,
                os.path.join(ds_dir, "tool_missing_values_bar.png")
            )
    except Exception as e:
        print(f"[WARN] Skipped dataset summary plots: {e}")

    user_mod = import_user_module(MODEL_PY)

    dna_all, prot_all, gate_all, fused_all = [], [], [], []
    prot_in_all, dna_in_all = [], []
    methods = EMBED_METHODS[:]

    for fold in folds:
        c_ckpt = os.path.join(OUT_DIR, "fold_artifacts", f"fold{fold}", f"c_model_fold{fold}.pth")
        if not os.path.exists(c_ckpt):
            raise FileNotFoundError(f"Missing C ckpt: {c_ckpt}")

        c_model, cfg_obj = build_c_model_from_ckpt(user_mod, c_ckpt, device=DEVICE)

        pack = extract_branch_and_gate(c_model, cfg_obj.memmap_dir, device=DEVICE, batch_size=BATCH_SIZE)
        dna_vec_fold   = pack["dna_vec"]
        prot_vec_fold  = pack["prot_vec"]
        gate_alpha_fold= pack["gate_alpha"]
        prot_in_fold   = pack.get("prot_in_delta", None)
        dna_in_fold    = pack.get("dna_in_delta", None)

        fused_te = os.path.join(OUT_DIR, "fused_memmap", f"fold{fold}_Xte.npy")
        if not os.path.exists(fused_te):
            raise FileNotFoundError(f"Missing fused memmap: {fused_te}")
        fused_vec_fold = np.array(np.load(fused_te, mmap_mode="r"))

        dna_all.append(dna_vec_fold)
        prot_all.append(prot_vec_fold)
        gate_all.append(gate_alpha_fold)
        fused_all.append(fused_vec_fold)
        if prot_in_fold is not None:
            prot_in_all.append(prot_in_fold)
        if dna_in_fold is not None:
            dna_in_all.append(dna_in_fold)

        if not ONLY_MEAN:
            fold_root = os.path.join(folds_root, f"fold{fold}")
            save_arrays(os.path.join(fold_root, "arrays"), dna_vec_fold, prot_vec_fold, fused_vec_fold, gate_alpha_fold)

            emb_dir = os.path.join(fold_root, "plots", "embeddings")
            plot_embedding("DNA",     dna_vec_fold,  y_true, y_prob, os.path.join(emb_dir, "DNA"),     methods, random_state=42+fold)
            plot_embedding("Protein", prot_vec_fold, y_true, y_prob, os.path.join(emb_dir, "Protein"), methods, random_state=42+fold)
            plot_embedding("Fused",   fused_vec_fold,y_true, y_prob, os.path.join(emb_dir, "Fused"),   methods, random_state=42+fold)

            plot_gate_stats(gate_alpha_fold, y_true, y_prob, os.path.join(fold_root, "plots", "gate"), thr=thr)

            if PER_FOLD_PROXY:
                run_proxy_bundle(os.path.join(fold_root, "proxy"),
                                 dna_vec_fold, prot_vec_fold, fused_vec_fold, gate_alpha_fold, y_true)

        del c_model

    dna_vec    = np.mean(np.stack(dna_all,  axis=0), axis=0)
    prot_vec   = np.mean(np.stack(prot_all, axis=0), axis=0)
    gate_alpha = np.mean(np.stack(gate_all, axis=0), axis=0)
    fused_vec  = np.mean(np.stack(fused_all,axis=0), axis=0)

    prot_input_delta_mean = None
    dna_input_delta_mean = None
    if len(prot_in_all) > 0:
        prot_input_delta_mean = np.mean(np.stack(prot_in_all, axis=0), axis=0)
    if len(dna_in_all) > 0:
        dna_input_delta_mean = np.mean(np.stack(dna_in_all, axis=0), axis=0)

    save_arrays(os.path.join(mean_root, "arrays"), dna_vec, prot_vec, fused_vec, gate_alpha)

    emb_dir = os.path.join(mean_root, "plots", "embeddings")
    plot_embedding("DNA",     dna_vec,  y_true, y_prob, os.path.join(emb_dir, "DNA"),     methods, random_state=42)
    plot_embedding("Protein", prot_vec, y_true, y_prob, os.path.join(emb_dir, "Protein"), methods, random_state=42)
    plot_embedding("Fused",   fused_vec,y_true, y_prob, os.path.join(emb_dir, "Fused"),   methods, random_state=42)

    plot_gate_stats(gate_alpha, y_true, y_prob, os.path.join(mean_root, "plots", "gate"), thr=thr)
    run_proxy_bundle(os.path.join(mean_root, "proxy"),
                     dna_vec, prot_vec, fused_vec, gate_alpha, y_true)

    # Six-panel UMAP (3 inputs + 3 stage outputs), colored by probability
    try:
        reps: List[Tuple[str, np.ndarray]] = []
        if prot_input_delta_mean is not None:
            reps.append(("ProtT5 Input", prot_input_delta_mean))
        if dna_input_delta_mean is not None:
            reps.append(("GPN-MSA Input", dna_input_delta_mean))

        if TOOL_SCORE_CSV and os.path.exists(TOOL_SCORE_CSV):
            meta_csv = None
            if TEST_CSV and os.path.exists(TEST_CSV):
                meta_csv = TEST_CSV
            elif SEQ_CSV and os.path.exists(SEQ_CSV):
                meta_csv = SEQ_CSV
            if meta_csv is not None:
                meta_df = pd.read_csv(meta_csv)
                func_X = _align_tool_score_table(meta_df, TOOL_SCORE_CSV)
                if func_X is not None:
                    reps.append(("Functional Annotation", func_X))

        if len(reps) == 2:
            reps.append(("Combined Input", np.concatenate([reps[0][1], reps[1][1]], axis=1)))

        reps.extend([
            ("Protein Encoder Output", prot_vec),
            ("DNA Encoder Output", dna_vec),
            ("Gate Fusion Output", fused_vec),
        ])

        umap_dir = os.path.join(mean_root, "plots", "embeddings")
        os.makedirs(umap_dir, exist_ok=True)
        plot_umap_six_panels(reps, y_prob, os.path.join(umap_dir, "UMAP_six_panel.png"), random_state=42)
    except Exception as e:
        print(f"[WARN] Skipped six-panel UMAP: {e}")

    # Performance/agreement visualizations (tools + GenProt-DSM)
    try:
        if not TOOL_SCORE_CSV or (not os.path.exists(TOOL_SCORE_CSV)):
            raise FileNotFoundError(f"TOOL_SCORE_CSV not found: {TOOL_SCORE_CSV}")
        if test_df is None:
            raise FileNotFoundError("TEST_CSV not found or not loaded; required for performance visualizations.")

        tool_df = _read_csv_smart(TOOL_SCORE_CSV)
        df12, methods12, _flipped_map = _prepare_method_table(
            test_df=test_df,
            pred_df=pred_df,
            tool_df=tool_df,
            label_col=LABEL_COL,
        )

        perf_dir = os.path.join(mean_root, "plots", "performance")
        os.makedirs(perf_dir, exist_ok=True)
                # ✅ Full test set ROC/PR overlay (12 tools + GenProt-DSM)
        plot_alltools_rocpr_fulltest(
            df12=df12,
            methods12=methods12,
            out_dir=perf_dir,
            label_col=LABEL_COL,
        )

        # Rare/Gene overlay (if subset CSVs exist)
        if os.path.isfile(RARE_SUBSET_CSV) and os.path.isfile(GENE_SUBSET_CSV):
            plot_rare_gene_rocpr_combined(
                df12=df12,
                subset_rare_csv=RARE_SUBSET_CSV,
                subset_gene_csv=GENE_SUBSET_CSV,
                out_dir=perf_dir,
                methods12=methods12,
                label_col=LABEL_COL,
            )
        else:
            print(f"[WARN] Subset CSVs not found; skip subset ROC/PR overlays. rare={RARE_SUBSET_CSV} gene={GENE_SUBSET_CSV}")

        # Spearman + Agreement (combined single figure)
        plot_spearman_and_agreement(
            df12=df12,
            methods12=methods12,
            out_dir=perf_dir,
            label_col=LABEL_COL,
        )

        create_combined_fig4(
            df12=df12,
            methods12=methods12,
            missing_csv=MISSING_VALUE_CSV,
            subset_rare_csv=RARE_SUBSET_CSV,
            subset_gene_csv=GENE_SUBSET_CSV,
            out_path=os.path.join(perf_dir, "Fig4_combined.png"),
            label_col=LABEL_COL,
        )

        # GenProt-DSM score distribution + base-substitution heatmap (left-right)
        plot_score_distribution_and_substitution_heatmap(
            test_df=test_df,
            pred_df=pred_df,
            out_path=os.path.join(perf_dir, "genprot_score_distribution_and_substitution.png"),
            label_col=LABEL_COL,
        )

        # Ablation (single scatter, your new path style)
        try:
            pts = scan_ablation_summaries(ABLA_ROOT)
            plot_ablation_scatter(
                points=pts,
                out_path=os.path.join(perf_dir, "ablation_all_auroc_auprc_scatter.png"),
                title="Ablation study (AUROC vs AUPRC)",
            )
        except Exception as _e:
            print(f"[WARN] Skipped ablation scatter figure: {_e}")

    except Exception as e:
        print(f"[WARN] Skipped performance/agreement visualizations (tools): {e}")

        # (9) Hard-coded experiment summary figures (no file reading)
    try:
        exp_dir = os.path.join(mean_root, "plots", "experiments")
        os.makedirs(exp_dir, exist_ok=True)
        plot_module_experiments_summary(os.path.join(exp_dir, "module_experiments_summary.png"))
        plot_dimension_experiments_summary(os.path.join(exp_dir, "dimension_experiments_summary.png"))
        create_combined_fig3(os.path.join(exp_dir, "Fig3_combined.png"))
    except Exception as e:
        print(f"[WARN] Skipped hard-coded experiment summary figures: {e}")

    try:
        create_combined_fig5(mean_root, os.path.join(mean_root, "plots", "Fig5_combined.png"))
    except Exception as e:
        print(f"[WARN] Skipped Fig. 5 combination: {e}")
    try:
        create_combined_fig7(mean_root, os.path.join(mean_root, "plots", "performance", "Fig7_combined.png"))
    except Exception as e:
        print(f"[WARN] Skipped Fig. 7 combination: {e}")

    # motif (optional; use fold1 to keep bounded runtime)
    fold_for_motif = folds[0]
    fold_ckpt = os.path.join(OUT_DIR, "fold_artifacts", f"fold{fold_for_motif}", f"c_model_fold{fold_for_motif}.pth")
    c_model, cfg_obj = build_c_model_from_ckpt(user_mod, fold_ckpt, device=DEVICE)
    grad_attribution_on_tokens(
        c_model=c_model,
        memmap_dir=cfg_obj.memmap_dir,
        device=DEVICE,
        seq_csv=(SEQ_CSV if SEQ_CSV else None),
        outdir=os.path.join(mean_root, "motif"),
        run_meme=RUN_MEME,
        run_tomtom=RUN_TOMTOM,
        dna_db=DNA_DB_MEME
    )

    print("[DONE] per-fold saved to:", folds_root if not ONLY_MEAN else "(skipped)")
    print("[DONE] mean (paper-ready) saved to:", mean_root)
    print("[INFO] If you only want paper plots, set ONLY_MEAN=True in CONFIG and re-run.")


if __name__ == "__main__":
    main()
