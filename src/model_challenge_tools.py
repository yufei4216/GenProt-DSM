#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import os, json, copy, warnings, gc, shutil
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
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
# CSV/TSV robust reader + preflight
# =========================

def read_table_auto(path: str, **kwargs) -> pd.DataFrame:
    """
    Thin wrapper over pandas.read_csv that also normalizes column names.
    Accepts **kwargs so other helpers can pass sep/engine/etc.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, **kwargs)
    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
    return df


def _read_csv_smart(path: str) -> pd.DataFrame:
    """Read CSV/TSV robustly (handles UTF-8 BOM, delimiter issues, and stray whitespace in headers).

    Supports:
      - comma / tab / semicolon separated files
      - files mislabeled as .csv but actually TSV
      - UTF-8 BOM on the first column (\ufeff)
      - trailing/leading whitespace in column names
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # Try pandas' delimiter sniffing first (python engine).
    try:
        df = read_table_auto(path, sep=None, engine="python")
    except Exception:
        # Safe fallbacks (NO recursion)
        tried = []
        last_e = None
        for sep in [",", "\t", ";"]:
            try:
                df = read_table_auto(path, sep=sep, engine="python")
                break
            except Exception as e:
                tried.append(sep)
                last_e = e
                df = None
        if df is None:
            raise RuntimeError(f"Failed to read table: {path}. Tried seps={tried}. Last error={last_e}") from last_e

    # If delimiter was mis-detected and everything landed in one column, retry with tab / semicolon.
    if df.shape[1] == 1:
        col0 = str(df.columns[0]) if len(df.columns) else ""
        if "\t" in col0:
            df = read_table_auto(path, sep="\t", engine="python")
        elif ";" in col0:
            df = read_table_auto(path, sep=";", engine="python")

    # Normalize column names (strip whitespace & UTF-8 BOM; remove common invisible characters)
    def _norm(c: object) -> str:
        s = str(c)
        s = s.replace("\u200b", "")  # zero-width space
        s = s.strip().lstrip("\ufeff")
        return s

    df.columns = [_norm(c) for c in df.columns]
    return df


def preflight_check(cfg: "Config") -> None:
    """
    Fast checks to fail early (before long training):
      - memmap presence & shapes
      - train/test CSV label column and row counts
      - challenge subset mapping sanity
      - tool-score CSV column presence + duplicate key detection
      - tool-score table can map back to test.csv for common-subset eval
    """
    print("\n" + "-" * 100)
    print("[Preflight] Start")

    # ---- memmap files ----
    memmap_paths = [
        os.path.join(cfg.memmap_dir, "train_t5_wt.npy"),
        os.path.join(cfg.memmap_dir, "train_t5_mut.npy"),
        os.path.join(cfg.memmap_dir, "test_t5_wt.npy"),
        os.path.join(cfg.memmap_dir, "test_t5_mut.npy"),
        os.path.join(cfg.memmap_dir, "train_gpn_ref.npy"),
        os.path.join(cfg.memmap_dir, "train_gpn_alt.npy"),
        os.path.join(cfg.memmap_dir, "test_gpn_ref.npy"),
        os.path.join(cfg.memmap_dir, "test_gpn_alt.npy"),
    ]
    missing = [p for p in memmap_paths if not os.path.exists(p)]
    if missing and (not getattr(cfg, "do_memmap_convert", False)):
        msg = (
            "[Preflight] Missing memmap files:\n  - "
            + "\n  - ".join(missing)
            + "\nSet Config.do_memmap_convert=True (first run) OR fix Config.memmap_dir."
        )
        raise FileNotFoundError(msg)

    # Read memmap shapes (fast)
    try:
        _t5 = np.load(memmap_paths[0], mmap_mode="r")
        n_train, L, dp = _t5.shape
        del _t5
        _t5t = np.load(memmap_paths[2], mmap_mode="r")
        n_test, L2, _ = _t5t.shape
        del _t5t
        _gpn = np.load(memmap_paths[4], mmap_mode="r")
        dg = int(_gpn.shape[2])
        del _gpn
        if L != L2:
            raise ValueError(f"ProtT5 train/test L mismatch: train={L}, test={L2}")
        print(f"[Preflight] Memmap OK: train N={n_train}, test N={n_test}, L={L}, ProtT5_D={dp}, GPN_D={dg}")
    except Exception as e:
        raise RuntimeError(f"[Preflight] Failed to read memmap shapes: {e}") from e

    # ---- train/test CSVs ----
    tr_df = _read_csv_smart(cfg.train_csv)
    te_df = _read_csv_smart(cfg.test_csv)
    if cfg.label_col not in tr_df.columns:
        raise KeyError(f"[Preflight] train_csv missing label col '{cfg.label_col}'. Columns={list(tr_df.columns)[:20]}")
    if cfg.label_col not in te_df.columns:
        raise KeyError(f"[Preflight] test_csv missing label col '{cfg.label_col}'. Columns={list(te_df.columns)[:20]}")
    if len(tr_df) != n_train:
        raise ValueError(f"[Preflight] train_csv rows != train memmap rows: csv={len(tr_df)} vs memmap={n_train}")
    if len(te_df) != n_test:
        raise ValueError(f"[Preflight] test_csv rows != test memmap rows: csv={len(te_df)} vs memmap={n_test}")
    print(f"[Preflight] train/test CSV OK: train rows={len(tr_df)}, test rows={len(te_df)}")

    # ---- challenge mapping sanity (against test_csv) ----
    for name, path in [("rare", getattr(cfg, "rare_csv", "")), ("gene_independent", getattr(cfg, "gene_csv", ""))]:
        path = str(path or "").strip()
        if not path:
            continue
        if not os.path.exists(path):
            raise FileNotFoundError(f"[Preflight] {name} CSV not found: {path}")
        try:
            idx = subset_indices_from_csv(path, te_df, cfg, subset_name=name)
            print(f"[Preflight] {name} subset mapping OK: n={len(idx)}")
        except Exception as e:
            raise RuntimeError(f"[Preflight] {name} subset mapping failed: {e}") from e

    # ---- tool-score CSV sanity (if provided) ----
    tool_path = str(getattr(cfg, "tool_test_score_csv", "")).strip()
    if tool_path:
        tool_df = _read_csv_smart(tool_path)
        if "label" not in tool_df.columns:
            raise KeyError(f"[Preflight] tool_test_score_csv missing 'label' col. Columns={list(tool_df.columns)[:30]}")

        # match cols presence
        match_cols = _choose_tool_match_cols(cfg, tool_df)
        missing_cols = [c for c in match_cols if c not in tool_df.columns and c.lower() not in [x.lower() for x in tool_df.columns]]
        if missing_cols:
            raise KeyError(
                f"[Preflight] tool_test_score_csv missing match cols {missing_cols}. "
                f"Configured tool_match_cols/match_cols={match_cols}. Available columns head={list(tool_df.columns)[:30]}"
            )

        # duplicate key check
        key = _df_key_series(tool_df, match_cols)
        dup = int(key.duplicated().sum())
        if dup > 0:
            raise ValueError(
                f"[Preflight] tool_test_score_csv has duplicate keys for match_cols={match_cols} (duplicates={dup}). "
                f"Extend Config.tool_match_cols to make keys unique."
            )

        # at least one tool column resolvable
        resolved = 0
        for _, cands in TOOL_COLUMN_CANDIDATES.items():
            if _resolve_tool_column(tool_df, cands) is not None:
                resolved += 1
        if resolved == 0:
            raise ValueError(
                "[Preflight] No tool score columns were recognized in tool_test_score_csv. "
                f"Columns head={list(tool_df.columns)[:40]}"
            )

        # mapping tool table back to test_df (for common subset)
        tool_to_test = map_tool_rows_to_test_indices(tool_df, te_df, match_cols)
        mapped = int((tool_to_test >= 0).sum())
        if mapped == 0:
            raise ValueError(
                "[Preflight] tool_test_score_csv cannot map back to test_csv with the current match cols. "
                f"match_cols={match_cols}. Please set Config.tool_match_cols / Config.match_cols correctly."
            )
        # if too low mapping ratio, error early
        ratio = mapped / max(1, len(tool_df))
        if ratio < 0.8:
            raise ValueError(
                f"[Preflight] tool_test_score_csv mapping back to test_csv is too low: {mapped}/{len(tool_df)} ({ratio:.2%}). "
                f"match_cols={match_cols}. Please fix key columns."
            )
        print(f"[Preflight] tool->test mapping OK: mapped {mapped}/{len(tool_df)} ({ratio:.2%})")

        # mapping challenge subsets onto tool table too
        tmp_cfg = copy.copy(cfg)
        tmp_cfg.match_cols = ",".join(match_cols)
        for name, path in [("rare", getattr(cfg, "rare_csv", "")), ("gene_independent", getattr(cfg, "gene_csv", ""))]:
            path = str(path or "").strip()
            if not path:
                continue
            idx = subset_indices_from_csv(path, tool_df, tmp_cfg, subset_name=name)
            print(f"[Preflight] {name} subset mapping onto tool table OK: n={len(idx)}")

        print(f"[Preflight] tool_test_score_csv OK: rows={len(tool_df)}, match_cols={match_cols}, resolved_tools={resolved}")
    else:
        print("[Preflight] tool_test_score_csv empty -> skip tool-score checks.")

    print("[Preflight] PASS")
    print("-" * 100 + "\n")


# =========================
# CONFIG
# =========================

@dataclass
class Config:
    # ---- optional: convert from pth to memmap ----
    do_memmap_convert: bool = False
    memmap_dtype: str = "float16"  # float16/float32

    # 8 feature pth paths (only used if do_memmap_convert=True)
    train_gpn_ref_pth: str = r"F:\20251210up\feature\GPN-MSA\train_GPN-MSA_ref.pth"
    train_gpn_alt_pth: str = r"F:\20251210up\feature\GPN-MSA\train_GPN-MSA_alt.pth"
    test_gpn_ref_pth:  str = r"F:\20251210up\feature\GPN-MSA\test_GPN-MSA_ref.pth"
    test_gpn_alt_pth:  str = r"F:\20251210up\feature\GPN-MSA\test_GPN-MSA_alt.pth"
    train_t5_wt_pth:   str = r"F:\20251210up\feature\protT5-XL\train_protT5-XL_wt.pth"
    train_t5_mut_pth:  str = r"F:\20251210up\feature\protT5-XL\train_protT5-XL_mut.pth"
    test_t5_wt_pth:    str = r"F:\20251210up\feature\protT5-XL\test_protT5-XL_wt.pth"
    test_t5_mut_pth:   str = r"F:\20251210up\feature\protT5-XL\test_protT5-XL_mut.pth"

    # memmap folder produced by file0_convert_pth_to_memmap.py
    memmap_dir: str = r"F:\20251210up\feature\merged_memmap_raw"

    # labels
    train_csv: str = r"F:\20251210up\data\train.csv"
    test_csv:  str = r"F:\20251210up\data\test.csv"
    label_col: str = "label"

    # challenge set CSVs (optional). Leave empty ("") to skip.
    rare_csv: str = r"F:\20251210up\data\new_test\test_AF.csv"
    gene_csv: str = r"F:\20251210up\data\new_test\test_onlygene.csv"

    # how to match rows between test_csv and challenge CSVs
    match_cols: str = ""  # e.g. "Chr,Start,End,Ref,Alt"

    # bootstrap CI settings
    bootstrap_n: int = 1000
    bootstrap_seed: int = 123
    bootstrap_alpha: float = 0.95

    # ---- tool score evaluation ----
    tool_test_score_csv: str = r"F:\20251210up\data\test-score.csv"
    tool_train_score_csv: str = ""
    tool_match_cols: str = ""
    tools_bootstrap_n: int = 1000

    # NEW: how to name your model row inside tool_metrics_challenge.csv
    model_name: str = "GeneProt-DSM"

    # output
    out_dir: str = r"F:\20251210up\pred\model_challenge_tools"

    # runtime toggles
    preflight_only: bool = False
    tools_only: bool = False
    skip_tools: bool = False

    # device
    force_cpu: bool = True

    # CV
    n_folds: int = 5
    seed: int = 42

    # -------- C stage --------
    c_epochs: int = 5
    c_batch_size: int = 8
    c_lr: float = 1e-4
    c_weight_decay: float = 1e-2

    # -------- Gate (A) --------
    gate_hidden: int = 64
    gate_tau_start: float = 4.0
    gate_tau_end: float = 2.0
    gate_tau_schedule: str = "linear"

    d_model: int = 256
    branch_d: int = 128
    fused_d: int = 128

    n_heads: int = 8
    n_layers: int = 4
    ff_dim: int = 1024
    dropout: float = 0.1
    relpos_max: int = 64

    pool_type: str = "mean"
    qpool_nq: int = 4

    # TCN params
    tcn_layers: int = 4
    tcn_kernel: int = 5
    tcn_dropout: float = 0.1

    # CNN params
    cnn_layers: int = 2
    cnn_kernel: int = 5
    cnn_dropout: float = 0.1

    # backbone selection
    t5_backbone: str = "bilstm"
    gpn_backbone: str = "gru"

    fusion_mode: str = "gate"
    bilinear_rank: int = 256
    xattn_heads: int = 8
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
# Utils
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
    df = _read_csv_smart(path)
    if label_col not in df.columns:
        raise KeyError(f"{path} missing label col '{label_col}'")
    return df[label_col].values.astype(int)


# =========================
# Challenge-set helpers
# =========================

def _parse_match_cols(match_cols: str) -> List[str]:
    s = str(match_cols or "").strip()
    if not s:
        return []
    return [c.strip() for c in s.split(",") if c.strip()]

def _auto_match_cols(df: pd.DataFrame) -> List[str]:
    cols = set(df.columns)
    candidates = [
        ["Chrom", "Position", "Reference", "Alternate"],
        ["Chr", "Start", "End", "Ref", "Alt"],
        ["CHROM", "POS", "REF", "ALT"],
        ["chrom", "pos", "ref", "alt"],
        ["Chrom", "Pos", "Ref", "Alt"],
        ["Location", "Allele"],
        ["chr", "pos", "ref", "alt"],
    ]
    for cand in candidates:
        if all(c in cols for c in cand):
            return cand
    fallback = ["Gene", "Chr", "Start", "Ref", "Alt"]
    if all(c in cols for c in fallback):
        return fallback
    common = [c for c in ["Chr", "CHROM", "chrom", "Start", "POS", "pos", "Ref", "REF", "ref", "Alt", "ALT", "alt"] if c in cols]
    if len(common) >= 3:
        uniq = []
        for c in common:
            if c not in uniq:
                uniq.append(c)
        return uniq[:5]
    raise ValueError(
        "Unable to auto-detect match columns. Please set Config.match_cols explicitly, e.g. 'Chr,Start,End,Ref,Alt'."
    )

def _resolve_cols_case_insensitive(df: pd.DataFrame, cols: List[str]) -> List[str]:
    actual: List[str] = []
    lower_map = {str(c).lower(): str(c) for c in df.columns}
    for c in cols:
        if c in df.columns:
            actual.append(c)
            continue
        lc = str(c).lower()
        if lc in lower_map:
            actual.append(lower_map[lc])
            continue
        raise KeyError(f"Column '{c}' not found in dataframe (first columns={list(df.columns)[:20]}..., total={len(df.columns)})")
    return actual

def _build_key(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    cols = _resolve_cols_case_insensitive(df, cols)
    parts = []
    for c in cols:
        parts.append(df[c].astype(str).fillna(""))
    key = parts[0]
    for p in parts[1:]:
        key = key + "|" + p
    return key

def subset_indices_from_csv(sub_csv: str, test_df: pd.DataFrame, cfg: Config, subset_name: str) -> np.ndarray:
    if (sub_csv is None) or (str(sub_csv).strip() == ""):
        return np.asarray([], dtype=int)
    if not os.path.exists(sub_csv):
        raise FileNotFoundError(f"{subset_name} CSV not found: {sub_csv}")

    sub_df = _read_csv_smart(sub_csv)
    if "sample_index" in sub_df.columns:
        idx = sub_df["sample_index"].values.astype(int)
        if idx.min() < 0 or idx.max() >= len(test_df):
            raise ValueError(
                f"{subset_name} has 'sample_index' out of range (0..{len(test_df)-1}). "
                f"min={idx.min()}, max={idx.max()}"
            )
        return idx

    cols = _parse_match_cols(getattr(cfg, "match_cols", ""))
    if not cols:
        cols = _auto_match_cols(test_df)

    test_key = _build_key(test_df, cols)
    sub_key  = _build_key(sub_df, cols)

    if test_key.duplicated().any():
        dup_n = int(test_key.duplicated().sum())
        raise ValueError(
            f"Chosen match_cols={cols} are not unique in test_csv (duplicates={dup_n}). "
            f"Please set Config.match_cols to a set of columns that uniquely identify a variant."
        )

    mapper = pd.Series(np.arange(len(test_df), dtype=int), index=test_key.values)
    try:
        idx = mapper.loc[sub_key.values].values.astype(int)
    except KeyError as e:
        missing = set(sub_key.values) - set(mapper.index.values)
        raise KeyError(
            f"{subset_name}: failed to map {len(missing)} rows back to test_csv using match_cols={cols}. "
            f"Example missing key: {next(iter(missing)) if missing else 'N/A'}"
        ) from e

    return idx

def _pr_auc_score(y_true: np.ndarray, prob: np.ndarray) -> float:
    prec, rec, _ = precision_recall_curve(y_true.astype(int), prob.astype(float))
    return float(pr_auc(rec, prec))

def bootstrap_ci(y_true: np.ndarray, prob: np.ndarray, metric: str, threshold: float,
                 n_boot: int, seed: int, alpha: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    rng = np.random.default_rng(int(seed))
    n = len(y_true)
    vals = []
    max_draws = int(n_boot * 1.2) + 50

    def _metric(y, p):
        if metric == "auc_roc":
            return float(roc_auc_score(y, p))
        if metric == "auc_pr":
            return float(_pr_auc_score(y, p))
        if metric == "mcc":
            pred = (p >= float(threshold)).astype(int)
            return float(metrics.matthews_corrcoef(y, pred))
        raise ValueError(f"Unknown metric for bootstrap_ci: {metric}")

    point = _metric(y_true, prob)

    draws = 0
    while (len(vals) < n_boot) and (draws < max_draws):
        draws += 1
        idx = rng.integers(0, n, size=n, endpoint=False)
        yb = y_true[idx]
        pb = prob[idx]
        if metric in ("auc_roc", "auc_pr"):
            if len(np.unique(yb)) < 2:
                continue
        try:
            v = _metric(yb, pb)
        except Exception:
            continue
        if np.isfinite(v):
            vals.append(v)

    if len(vals) < max(50, int(0.2 * n_boot)):
        warnings.warn(
            f"bootstrap_ci({metric}) collected only {len(vals)}/{n_boot} valid samples; CI may be unstable.",
            RuntimeWarning,
        )

    vals = np.asarray(vals, dtype=float)
    lo_q = (1.0 - float(alpha)) / 2.0
    hi_q = 1.0 - lo_q
    lo = float(np.quantile(vals, lo_q))
    hi = float(np.quantile(vals, hi_q))
    return {"point": point, "ci_lo": lo, "ci_hi": hi, "n_eff": int(len(vals))}

def eval_challenge_set(name: str, idx: np.ndarray, y_test: np.ndarray, prob_test: np.ndarray,
                       threshold: float, cfg: Config, primary_auc: str) -> Dict[str, Any]:
    idx = np.asarray(idx).astype(int)
    y = y_test[idx].astype(int)
    p = prob_test[idx].astype(float)

    out = {}
    out["n"] = int(len(idx))
    out["n_pos"] = int(y.sum())
    out["n_neg"] = int(len(y) - y.sum())
    out["prevalence"] = float(y.mean()) if len(y) else 0.0
    out["threshold"] = float(threshold)

    out["auc_roc"] = float(roc_auc_score(y, p)) * 100.0 if len(np.unique(y)) == 2 else float("nan")
    out["auc_pr"] = float(_pr_auc_score(y, p)) * 100.0 if len(np.unique(y)) == 2 else float("nan")

    pred = (p >= float(threshold)).astype(int)
    out["mcc"] = float(metrics.matthews_corrcoef(y, pred)) * 100.0
    out["f1"] = float(metrics.f1_score(y, pred, zero_division=0)) * 100.0

    n_boot = int(getattr(cfg, "bootstrap_n", 1000))
    seed = int(getattr(cfg, "bootstrap_seed", 123))
    alpha = float(getattr(cfg, "bootstrap_alpha", 0.95))

    if primary_auc not in ("auc_roc", "auc_pr"):
        raise ValueError("primary_auc must be 'auc_roc' or 'auc_pr'")

    if len(np.unique(y)) == 2:
        auc_ci = bootstrap_ci(y, p, metric=primary_auc, threshold=threshold, n_boot=n_boot, seed=seed, alpha=alpha)
        out[primary_auc + "_ci"] = {k: (float(v) * 100.0 if k in ["point", "ci_lo", "ci_hi"] else v) for k, v in auc_ci.items()}
    else:
        out[primary_auc + "_ci"] = None

    mcc_ci = bootstrap_ci(y, p, metric="mcc", threshold=threshold, n_boot=n_boot, seed=seed + 17, alpha=alpha)
    out["mcc_ci"] = {k: (float(v) * 100.0 if k in ["point", "ci_lo", "ci_hi"] else v) for k, v in mcc_ci.items()}

    return out

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
# Tool-score evaluation
# =========================

TOOL_COLUMN_CANDIDATES: Dict[str, List[str]] = {
    "SIFT": ["SIFT_score", "SIFT"],
    "PolyPhen2": ["Polyphen2_HVAR_score", "PolyPhen2_HVAR_score", "Polyphen2_score", "PolyPhen2_score", "PolyPhen2"],
    "FATHMM": ["FATHMM_score", "FATHMM"],
    "PROVEAN": ["PROVEAN_score", "PROVEAN"],
    "MPC": ["MPC_score", "MPC"],
    "DEOGEN2": ["DEOGEN2_score", "DEOGEN2"],
    "AlphaMissense": ["AlphaMissense_score", "alphamissense.score", "AlphaMissense"],
    "CADD_v1.7": ["CADD_phred", "CADD_v1.7", "cadd", "CADD"],
    "DANN": ["DANN_score", "DANN"],
    "GenoCanyon": ["GenoCanyon_score", "GenoCanyon"],
    "PrimateAI": ["PrimateAI_score", "PrimateAI", "primateai"],
}

def _resolve_tool_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive fallback
    lower_map = {str(x).lower(): str(x) for x in df.columns}
    for c in candidates:
        lc = str(c).lower()
        if lc in lower_map:
            return lower_map[lc]
    return None

def _df_key_series(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    cols = _resolve_cols_case_insensitive(df, cols)
    key = df[cols].astype(str).agg("|".join, axis=1)
    return key

def _choose_tool_match_cols(cfg: Config, df: pd.DataFrame) -> List[str]:
    cols = _parse_match_cols(getattr(cfg, "tool_match_cols", ""))
    if not cols:
        cols = _parse_match_cols(getattr(cfg, "match_cols", ""))
    if not cols:
        cols = _auto_match_cols(df)
    return cols

def map_tool_rows_to_test_indices(tool_df: pd.DataFrame, test_df: pd.DataFrame, match_cols: List[str]) -> np.ndarray:
    """
    Map each row of tool_df to an index in test_df via key(match_cols).
    Returns array len(tool_df) with test index, or -1 if missing.
    Requires uniqueness in test_df for the chosen key. If tool_df duplicates key, should be handled earlier.
    """
    cols_t = _resolve_cols_case_insensitive(tool_df, match_cols)
    cols_s = _resolve_cols_case_insensitive(test_df, match_cols)

    tool_key = tool_df[cols_t].astype(str).agg("|".join, axis=1)
    test_key = test_df[cols_s].astype(str).agg("|".join, axis=1)

    if test_key.duplicated().any():
        dup_n = int(test_key.duplicated().sum())
        raise ValueError(f"test_csv key is not unique for match_cols={match_cols} (duplicates={dup_n}).")

    mapper = pd.Series(np.arange(len(test_df), dtype=int), index=test_key.values)
    out = np.full((len(tool_df),), -1, dtype=int)

    # vectorized mapping
    common_mask = tool_key.isin(mapper.index)
    if common_mask.any():
        out[common_mask.values] = mapper.loc[tool_key[common_mask].values].values.astype(int)
    return out

def _compute_tool_orientation_and_threshold(base_df: pd.DataFrame, y_col: str, score_col: str, seed: int) -> Dict[str, Any]:
    y = base_df[y_col].astype(int).to_numpy()
    s = pd.to_numeric(base_df[score_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(s) & np.isfinite(y)
    y = y[mask]
    s = s[mask]
    out: Dict[str, Any] = {"flipped": False, "threshold": float("nan"), "n_used": int(len(y))}
    if len(y) < 2 or len(np.unique(y)) < 2:
        return out

    try:
        auc0 = float(roc_auc_score(y, s))
    except Exception:
        auc0 = float("nan")
    flipped = False
    if np.isfinite(auc0) and auc0 < 0.5:
        s = -s
        flipped = True

    thr = float(pick_threshold(y, s, objective="mcc"))
    out.update({"flipped": flipped, "threshold": thr})
    return out

def _eval_tool_on_index(df: pd.DataFrame, idx: np.ndarray, y_col: str, score_col: str,
                        flipped: bool, threshold: float, primary_auc: str,
                        n_boot: int, seed: int, alpha: float) -> Dict[str, Any]:
    idx = np.asarray(idx).astype(int)
    sub = df.iloc[idx].copy()

    y = sub[y_col].astype(int).to_numpy()
    s = pd.to_numeric(sub[score_col], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    if flipped:
        s = -s

    out: Dict[str, Any] = {
        "n_used": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int(len(y) - y.sum()),
        "prevalence": float(y.mean()) if len(y) else 0.0,
        "threshold": float(threshold),
        "flipped": bool(flipped),
        "auc_roc": float(roc_auc_score(y, s)) * 100.0 if len(np.unique(y)) == 2 else float("nan"),
        "auc_pr": float(_pr_auc_score(y, s)) * 100.0 if len(np.unique(y)) == 2 else float("nan"),
        "mcc": float(metrics.matthews_corrcoef(y, (s >= float(threshold)).astype(int))) * 100.0 if len(np.unique(y)) == 2 else float("nan"),
    }

    if len(np.unique(y)) == 2 and len(y) >= 5:
        try:
            ci_auc = bootstrap_ci(y, s, metric=primary_auc, threshold=float(threshold),
                                  n_boot=int(n_boot), seed=int(seed), alpha=float(alpha))
            out[primary_auc + "_ci"] = {k: float(v) * (100.0 if k in ("point","ci_lo","ci_hi") else 1.0) if k!="n_eff" else int(v) for k,v in ci_auc.items()}
        except Exception:
            out[primary_auc + "_ci"] = {}
        try:
            ci_mcc = bootstrap_ci(y, s, metric="mcc", threshold=float(threshold),
                                  n_boot=int(n_boot), seed=int(seed)+7, alpha=float(alpha))
            out["mcc_ci"] = {k: float(v) * (100.0 if k in ("point","ci_lo","ci_hi") else 1.0) if k!="n_eff" else int(v) for k,v in ci_mcc.items()}
        except Exception:
            out["mcc_ci"] = {}
    else:
        out[primary_auc + "_ci"] = {}
        out["mcc_ci"] = {}
    return out

def _eval_model_on_tool_index_common(
    y_test: np.ndarray,
    prob_test: np.ndarray,
    tool_to_test_idx: np.ndarray,
    idx_tool: np.ndarray,
    threshold: float,
    primary_auc: str,
    n_boot: int,
    seed: int,
    alpha: float,
) -> Dict[str, Any]:
    """
    Evaluate model on a subset defined on tool_df row indices (idx_tool),
    via tool_to_test_idx mapping back to test indices.
    """
    idx_tool = np.asarray(idx_tool).astype(int)
    test_idx = tool_to_test_idx[idx_tool]
    test_idx = test_idx[test_idx >= 0]
    test_idx = np.asarray(test_idx, dtype=int)
    y = y_test[test_idx].astype(int)
    p = prob_test[test_idx].astype(float)

    out: Dict[str, Any] = {
        "n_used": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int(len(y) - y.sum()),
        "prevalence": float(y.mean()) if len(y) else 0.0,
        "threshold": float(threshold),
        "flipped": False,
        "auc_roc": float(roc_auc_score(y, p)) * 100.0 if len(np.unique(y)) == 2 else float("nan"),
        "auc_pr": float(_pr_auc_score(y, p)) * 100.0 if len(np.unique(y)) == 2 else float("nan"),
        "mcc": float(metrics.matthews_corrcoef(y, (p >= float(threshold)).astype(int))) * 100.0 if len(np.unique(y)) == 2 else float("nan"),
    }

    if len(np.unique(y)) == 2 and len(y) >= 5:
        try:
            ci_auc = bootstrap_ci(y, p, metric=primary_auc, threshold=float(threshold),
                                  n_boot=int(n_boot), seed=int(seed), alpha=float(alpha))
            out[primary_auc + "_ci"] = {k: float(v) * (100.0 if k in ("point","ci_lo","ci_hi") else 1.0) if k!="n_eff" else int(v) for k,v in ci_auc.items()}
        except Exception:
            out[primary_auc + "_ci"] = {}
        try:
            ci_mcc = bootstrap_ci(y, p, metric="mcc", threshold=float(threshold),
                                  n_boot=int(n_boot), seed=int(seed)+7, alpha=float(alpha))
            out["mcc_ci"] = {k: float(v) * (100.0 if k in ("point","ci_lo","ci_hi") else 1.0) if k!="n_eff" else int(v) for k,v in ci_mcc.items()}
        except Exception:
            out["mcc_ci"] = {}
    else:
        out[primary_auc + "_ci"] = {}
        out["mcc_ci"] = {}
    return out

def eval_tools_on_challenge_sets(
    cfg: Config,
    test_df: pd.DataFrame,
    y_test: Optional[np.ndarray] = None,
    model_prob_test: Optional[np.ndarray] = None,
    model_threshold: float = 0.5,
) -> None:
    """
    Evaluate classic tools on:
      - test / rare / gene_independent (as before)
      - PLUS common subsets: test_common / rare_common / gene_independent_common
        where "common" means:
          - tool row can map back to test.csv
          - and ALL included tool columns have finite scores on that row

    Also writes your model (if provided) into tool_metrics_challenge.csv on the *_common subsets.
    """
    tool_test_path = str(getattr(cfg, "tool_test_score_csv", "")).strip()
    if not tool_test_path:
        print("[Tools] Skip (Config.tool_test_score_csv is empty).")
        return

    os.makedirs(os.path.join(cfg.out_dir, "metrics"), exist_ok=True)

    tool_test_df = _read_csv_smart(tool_test_path)
    if "label" not in tool_test_df.columns:
        raise ValueError("tool_test_score_csv must contain a 'label' column (0/1).")

    # choose match cols for mapping rare/gene onto the tool-score table
    match_cols = _choose_tool_match_cols(cfg, tool_test_df)

    # validate uniqueness in tool_test_df
    key = _df_key_series(tool_test_df, match_cols)
    dup = int(key.duplicated().sum())
    if dup > 0:
        raise ValueError(
            f"tool_test_score_csv has duplicate keys for match_cols={match_cols} (duplicates={dup}). "
            f"Please extend Config.tool_match_cols to make keys unique."
        )

    # thresholds/orientation: prefer tool_train_score_csv if provided (no peeking)
    base_path = str(getattr(cfg, "tool_train_score_csv", "")).strip()
    base_df = None
    base_tag = "test"
    if base_path:
        base_df = _read_csv_smart(base_path)
        if "label" not in base_df.columns:
            raise ValueError("tool_train_score_csv must contain a 'label' column (0/1).")
        base_tag = "train"
    else:
        base_df = tool_test_df

    # resolve columns and prepare per-tool settings
    tool_settings: Dict[str, Dict[str, Any]] = {}
    for tname, cands in TOOL_COLUMN_CANDIDATES.items():
        col = _resolve_tool_column(tool_test_df, cands)
        if col is None:
            print(f"[Tools] {tname}: skipped (no matching column found). Candidates={cands}")
            continue
        if col not in base_df.columns:
            print(f"[Tools] {tname}: skipped (threshold base='{base_tag}' missing column '{col}').")
            continue
        st = _compute_tool_orientation_and_threshold(base_df, "label", col, seed=int(cfg.bootstrap_seed))
        st.update({"tool": tname, "column": col, "threshold_base": base_tag})
        tool_settings[tname] = st

    settings_path = os.path.join(cfg.out_dir, "metrics", "tool_thresholds.json")
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump({"match_cols": match_cols, "settings": tool_settings}, f, indent=2, ensure_ascii=False)
    print(f"[Tools] Saved threshold/orientation: {settings_path}")

    # Build subset indices on the tool-score table
    tmp_cfg = copy.copy(cfg)
    tmp_cfg.match_cols = ",".join(match_cols)

    subsets_base: List[Tuple[str, Optional[str], str]] = [
        ("test", None, "auc_roc"),
        ("rare", str(getattr(cfg, "rare_csv", "")).strip() or None, "auc_roc"),
        ("gene_independent", str(getattr(cfg, "gene_csv", "")).strip() or None, "auc_pr"),
    ]

    # ---- common subset mask construction ----
    # mapping tool rows -> test indices
    tool_to_test_idx = map_tool_rows_to_test_indices(tool_test_df, test_df, match_cols)
    mask_in_test = tool_to_test_idx >= 0

    # "all tools have finite scores" across INCLUDED tools only
    mask_all_scores = mask_in_test.copy()
    included_cols = []
    for tname, st in tool_settings.items():
        col = st["column"]
        included_cols.append(col)
        s = pd.to_numeric(tool_test_df[col], errors="coerce").to_numpy(dtype=float)
        mask_all_scores &= np.isfinite(s)

    # also require label to be finite/int (should already be)
    ycol = tool_test_df["label"].to_numpy()
    mask_all_scores &= np.isfinite(ycol.astype(float))

    n_common = int(mask_all_scores.sum())
    print(f"[Tools][Common] included_tools={len(tool_settings)} | common_rows={n_common}/{len(tool_test_df)}")

    # output rows
    rows = []
    n_boot = int(getattr(cfg, "tools_bootstrap_n", cfg.bootstrap_n))
    seed0 = int(cfg.bootstrap_seed)
    alpha0 = float(cfg.bootstrap_alpha)

    for subset_name, subset_csv, primary_auc in subsets_base:
        # indices on tool_test_df
        if subset_name == "test":
            idx = np.arange(len(tool_test_df), dtype=int)
        else:
            if not subset_csv:
                print(f"[Tools] {subset_name}: skip (csv path empty)")
                continue
            idx = subset_indices_from_csv(subset_csv, tool_test_df, tmp_cfg, subset_name=subset_name)

        if idx.size == 0:
            print(f"[Tools] {subset_name}: skip (empty subset)")
            continue

        # 1) original subset metrics for tools (legacy behavior)
        for tname, st in tool_settings.items():
            col = st["column"]
            res = _eval_tool_on_index(
                tool_test_df, idx, "label", col,
                flipped=bool(st.get("flipped", False)),
                threshold=float(st.get("threshold", 0.0)),
                primary_auc=primary_auc,
                n_boot=n_boot,
                seed=seed0 + 31,
                alpha=alpha0,
            )
            row = {
                "subset": subset_name,
                "tool": tname,
                "column": col,
                "threshold_base": st.get("threshold_base", "test"),
                "flipped": bool(st.get("flipped", False)),
                "threshold": float(st.get("threshold", float("nan"))),
                "n_used": res.get("n_used", 0),
                "n_pos": res.get("n_pos", 0),
                "n_neg": res.get("n_neg", 0),
                "prevalence": res.get("prevalence", float("nan")),
                "auc_roc": res.get("auc_roc", float("nan")),
                "auc_pr": res.get("auc_pr", float("nan")),
                "mcc": res.get("mcc", float("nan")),
            }
            auc_ci = res.get(primary_auc + "_ci") or {}
            mcc_ci = res.get("mcc_ci") or {}
            row.update({
                f"{primary_auc}_ci_lo": auc_ci.get("ci_lo", float("nan")),
                f"{primary_auc}_ci_hi": auc_ci.get("ci_hi", float("nan")),
                "mcc_ci_lo": mcc_ci.get("ci_lo", float("nan")),
                "mcc_ci_hi": mcc_ci.get("ci_hi", float("nan")),
                "ci_n_eff": int(auc_ci.get("n_eff", mcc_ci.get("n_eff", 0)) or 0),
            })
            rows.append(row)

        # 2) common subset metrics (new)
        idx_common = idx[mask_all_scores[idx]]
        if idx_common.size == 0:
            print(f"[Tools][Common] {subset_name}_common: skip (empty)")
            continue

        subset_common_name = f"{subset_name}_common"
        for tname, st in tool_settings.items():
            col = st["column"]
            res = _eval_tool_on_index(
                tool_test_df, idx_common, "label", col,
                flipped=bool(st.get("flipped", False)),
                threshold=float(st.get("threshold", 0.0)),
                primary_auc=primary_auc,
                n_boot=n_boot,
                seed=seed0 + 131,
                alpha=alpha0,
            )
            row = {
                "subset": subset_common_name,
                "tool": tname,
                "column": col,
                "threshold_base": st.get("threshold_base", "test"),
                "flipped": bool(st.get("flipped", False)),
                "threshold": float(st.get("threshold", float("nan"))),
                "n_used": res.get("n_used", 0),
                "n_pos": res.get("n_pos", 0),
                "n_neg": res.get("n_neg", 0),
                "prevalence": res.get("prevalence", float("nan")),
                "auc_roc": res.get("auc_roc", float("nan")),
                "auc_pr": res.get("auc_pr", float("nan")),
                "mcc": res.get("mcc", float("nan")),
            }
            auc_ci = res.get(primary_auc + "_ci") or {}
            mcc_ci = res.get("mcc_ci") or {}
            row.update({
                f"{primary_auc}_ci_lo": auc_ci.get("ci_lo", float("nan")),
                f"{primary_auc}_ci_hi": auc_ci.get("ci_hi", float("nan")),
                "mcc_ci_lo": mcc_ci.get("ci_lo", float("nan")),
                "mcc_ci_hi": mcc_ci.get("ci_hi", float("nan")),
                "ci_n_eff": int(auc_ci.get("n_eff", mcc_ci.get("n_eff", 0)) or 0),
            })
            rows.append(row)

        # 3) add your model row on the same common subset
        if (y_test is not None) and (model_prob_test is not None):
            res_m = _eval_model_on_tool_index_common(
                y_test=y_test,
                prob_test=model_prob_test,
                tool_to_test_idx=tool_to_test_idx,
                idx_tool=idx_common,
                threshold=float(model_threshold),
                primary_auc=primary_auc,
                n_boot=n_boot,
                seed=seed0 + 231,
                alpha=alpha0,
            )
            row = {
                "subset": subset_common_name,
                "tool": str(getattr(cfg, "model_name", "Ours")),
                "column": "prob_ensemble",
                "threshold_base": "model",
                "flipped": False,
                "threshold": float(model_threshold),
                "n_used": res_m.get("n_used", 0),
                "n_pos": res_m.get("n_pos", 0),
                "n_neg": res_m.get("n_neg", 0),
                "prevalence": res_m.get("prevalence", float("nan")),
                "auc_roc": res_m.get("auc_roc", float("nan")),
                "auc_pr": res_m.get("auc_pr", float("nan")),
                "mcc": res_m.get("mcc", float("nan")),
            }
            auc_ci = res_m.get(primary_auc + "_ci") or {}
            mcc_ci = res_m.get("mcc_ci") or {}
            row.update({
                f"{primary_auc}_ci_lo": auc_ci.get("ci_lo", float("nan")),
                f"{primary_auc}_ci_hi": auc_ci.get("ci_hi", float("nan")),
                "mcc_ci_lo": mcc_ci.get("ci_lo", float("nan")),
                "mcc_ci_hi": mcc_ci.get("ci_hi", float("nan")),
                "ci_n_eff": int(auc_ci.get("n_eff", mcc_ci.get("n_eff", 0)) or 0),
            })
            rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(cfg.out_dir, "metrics", "tool_metrics_challenge.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[Tools] Saved tool+model metrics: {out_path}")


# =========================
# Memmap conversion helpers
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
        y = x.transpose(1, 2)
        for conv in self.convs:
            z = conv(y)
            z = F.relu(z)
            z = self.drop(z)
            y = y + z
        y = y.transpose(1, 2)
        return self.ln(y)

class CNNBiLSTMEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cnn = CNNEncoder(cfg)
        self.bilstm = BiLSTMEncoder(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.bilstm(x)
        return x

class TCNLSTMEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.tcn = TCNEncoder(cfg)
        self.lstm = LSTMEncoder(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tcn(x)
        x = self.lstm(x)
        return x

class HeadProj(nn.Module):
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

        x = torch.cat([a, b], dim=1)
        seg = torch.cat([
            torch.zeros((B, L), dtype=torch.long, device=x.device),
            torch.ones((B, L), dtype=torch.long, device=x.device),
        ], dim=1)
        x = x + self.seg_emb(seg)

        pos = torch.arange(0, 2 * L, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.abs_pos(pos)

        x = self.backbone(x)
        return x

    def encode_tokens(self, a_raw: torch.Tensor, b_raw: torch.Tensor) -> torch.Tensor:
        return self._build_tokens(a_raw, b_raw)

    def forward(self, a_raw: torch.Tensor, b_raw: torch.Tensor) -> torch.Tensor:
        x = self._build_tokens(a_raw, b_raw)
        pooled = x.mean(dim=1)
        return self.head(pooled)

class DualBranchFusionModel(nn.Module):
    def __init__(self, t5_enc: nn.Module, gpn_enc: nn.Module, cfg: Config):
        super().__init__()
        self.t5_enc = t5_enc
        self.gpn_enc = gpn_enc
        self.moddrop = ModalityDropout(cfg.moddrop_p) if cfg.moddrop_p > 0 else None

        branch_d = int(getattr(cfg, "branch_d", cfg.fused_d))
        fused_d = int(cfg.fused_d)

        self.gate_ln_g = nn.LayerNorm(branch_d)
        self.gate_ln_t = nn.LayerNorm(branch_d)

        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * branch_d, int(cfg.gate_hidden)),
            nn.GELU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(int(cfg.gate_hidden), 1),
        )

        last = self.gate_mlp[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

        self._gate_tau = float(cfg.gate_tau_start)

        if fused_d == branch_d:
            self.fuse_proj = nn.Identity()
        else:
            self.fuse_proj = nn.Sequential(
                nn.Linear(branch_d, fused_d),
                nn.GELU(),
                nn.Dropout(float(cfg.dropout)),
            )

        self.fuse_ln = nn.LayerNorm(fused_d)
        self.head_fused = nn.Linear(fused_d, 1)

        self.head_t5 = nn.Linear(branch_d, 1) if float(getattr(cfg, "aux_t5_w", 0.0)) > 0 else None
        self.head_gpn = nn.Linear(branch_d, 1) if float(getattr(cfg, "aux_gpn_w", 0.0)) > 0 else None

    def set_gate_tau(self, tau: float):
        self._gate_tau = float(tau)

    def _gate_fuse(self, g_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
        tau = max(self._gate_tau, 1e-6)
        g_gate = self.gate_ln_g(g_in)
        t_gate = self.gate_ln_t(t_in)
        xcat = torch.cat([g_gate, t_gate], dim=1)
        alpha = torch.sigmoid(self.gate_mlp(xcat) / tau)
        return alpha * g_in + (1.0 - alpha) * t_in

    def forward(self, t5_wt, t5_mut, gpn_ref, gpn_alt):
        t5_vec = self.t5_enc(t5_wt, t5_mut)
        gpn_vec = self.gpn_enc(gpn_ref, gpn_alt)

        g_in, t_in = gpn_vec, t5_vec
        if self.moddrop is not None:
            g_in, t_in = self.moddrop(g_in, t_in)

        fused = self._gate_fuse(g_in, t_in)
        fused = self.fuse_proj(fused)
        fused = self.fuse_ln(fused)
        logits_fused = self.head_fused(fused)
        lt5 = self.head_t5(t5_vec) if self.head_t5 is not None else None
        lgpn = self.head_gpn(gpn_vec) if self.head_gpn is not None else None
        return fused, logits_fused, lt5, lgpn

def bce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.BCEWithLogitsLoss()(logits, y.view(-1, 1))


# =========================
# C train + export to memmap
# =========================

def train_c(model: nn.Module, dl: DataLoader, device: torch.device, cfg: Config, fold: int) -> Dict[str, float]:
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.c_lr, weight_decay=cfg.c_weight_decay)

    use_amp = bool(cfg.use_amp) and (device.type == "cuda")
    scaler = torch.amp.GradScaler(device_type='cuda', enabled=use_amp) if use_amp else None

    losses = []
    for ep in range(cfg.c_epochs):
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
# D classifier
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
# Main
# =========================

def _run_single(cfg: Config = CFG):
    set_seed(cfg.seed)

    device = torch.device("cpu") if cfg.force_cpu else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device={device}")

    if cfg.do_memmap_convert:
        prepare_memmaps(cfg)

    ensure_dir(cfg.out_dir)
    ensure_dir(os.path.join(cfg.out_dir, "ckpt"))
    ensure_dir(os.path.join(cfg.out_dir, "pred"))
    ensure_dir(os.path.join(cfg.out_dir, "fold_artifacts"))
    ensure_dir(os.path.join(cfg.out_dir, "fused_memmap"))

    t5_tr_wt = os.path.join(cfg.memmap_dir, "train_t5_wt.npy")
    t5_tr_mut= os.path.join(cfg.memmap_dir, "train_t5_mut.npy")
    t5_te_wt = os.path.join(cfg.memmap_dir, "test_t5_wt.npy")
    t5_te_mut= os.path.join(cfg.memmap_dir, "test_t5_mut.npy")

    gpn_tr_ref = os.path.join(cfg.memmap_dir, "train_gpn_ref.npy")
    gpn_tr_alt = os.path.join(cfg.memmap_dir, "train_gpn_alt.npy")
    gpn_te_ref = os.path.join(cfg.memmap_dir, "test_gpn_ref.npy")
    gpn_te_alt = os.path.join(cfg.memmap_dir, "test_gpn_alt.npy")

    for p in [t5_tr_wt, t5_tr_mut, t5_te_wt, t5_te_mut, gpn_tr_ref, gpn_tr_alt, gpn_te_ref, gpn_te_alt]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing memmap file: {p}\nSet do_memmap_convert=True or check memmap_dir.")

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
    test_df = _read_csv_smart(cfg.test_csv)

    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)

    fold_probs, fold_thrs, fold_summaries = [], [], []
    test_indices = np.arange(n_test)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n_train), y_train), start=1):
        print("\n" + "=" * 70)
        print(f"Fold {fold}/{cfg.n_folds}")
        print("=" * 70)

        fold_dir = os.path.join(cfg.out_dir, "fold_artifacts", f"fold{fold}")
        ensure_dir(fold_dir)

        t5_enc  = Scheme2PairEncoder(d_in=dp, seq_len=L, cfg=cfg, backbone=cfg.t5_backbone)
        gpn_enc = Scheme2PairEncoder(d_in=dg, seq_len=L, cfg=cfg, backbone=cfg.gpn_backbone)
        c_model = DualBranchFusionModel(t5_enc, gpn_enc, cfg)

        ds_c_tr = MemmapPairTokenDataset(t5_tr_wt, t5_tr_mut, gpn_tr_ref, gpn_tr_alt, y_train, tr_idx)
        dl_c_tr = DataLoader(ds_c_tr, batch_size=cfg.c_batch_size, shuffle=True, drop_last=False, num_workers=0)
        c_stats = train_c(c_model, dl_c_tr, device, cfg, fold)

        torch.save({"model_state_dict": c_model.state_dict(), "cfg": cfg.__dict__, "c_stats": c_stats},
                   os.path.join(fold_dir, f"c_model_fold{fold}.pth"))

        fused_tr_path = os.path.join(cfg.out_dir, "fused_memmap", f"fold{fold}_Xtr.npy")
        fused_va_path = os.path.join(cfg.out_dir, "fused_memmap", f"fold{fold}_Xva.npy")
        fused_te_path = os.path.join(cfg.out_dir, "fused_memmap", f"fold{fold}_Xte.npy")

        ds_tr = MemmapPairTokenDataset(t5_tr_wt, t5_tr_mut, gpn_tr_ref, gpn_tr_alt, None, tr_idx)
        ds_va = MemmapPairTokenDataset(t5_tr_wt, t5_tr_mut, gpn_tr_ref, gpn_tr_alt, None, va_idx)
        ds_te = MemmapPairTokenDataset(t5_te_wt, t5_te_mut, gpn_te_ref, gpn_te_alt, None, test_indices)

        export_fused_to_memmap(c_model, DataLoader(ds_tr, batch_size=32, shuffle=False, num_workers=0), device, fused_tr_path, n_rows=len(tr_idx), cfg=cfg)
        export_fused_to_memmap(c_model, DataLoader(ds_va, batch_size=32, shuffle=False, num_workers=0), device, fused_va_path, n_rows=len(va_idx), cfg=cfg)
        export_fused_to_memmap(c_model, DataLoader(ds_te, batch_size=32, shuffle=False, num_workers=0), device, fused_te_path, n_rows=n_test, cfg=cfg)

        X_tr = np.load(fused_tr_path, mmap_mode="r")
        X_va = np.load(fused_va_path, mmap_mode="r")
        X_te = np.load(fused_te_path, mmap_mode="r")

        y_tr = y_train[tr_idx].astype(int)
        y_va = y_train[va_idx].astype(int)

        d_model, best_val_metrics, best_epoch, best_thr = train_d_complex(
            np.asarray(X_tr), y_tr, np.asarray(X_va), y_va, device, cfg, fold, ckpt_dir=os.path.join(cfg.out_dir, "ckpt")
        )

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

    # Challenge-set evaluation (optional) on your model predictions (original subset, not necessarily common)
    challenge_results: Dict[str, Any] = {}
    challenge_specs = [
        ("rare", str(getattr(cfg, "rare_csv", "")).strip(), "auc_roc"),
        ("gene_independent", str(getattr(cfg, "gene_csv", "")).strip(), "auc_pr"),
    ]

    for cname, cpath, primary_auc in challenge_specs:
        if not cpath:
            print(f"[Challenge] {cname}: skip (cfg path empty)")
            continue
        if not os.path.exists(cpath):
            print(f"[Challenge] {cname}: skip (file not found: {cpath})")
            continue
        try:
            idx = subset_indices_from_csv(cpath, test_df, cfg, subset_name=cname)
            if idx.size == 0:
                print(f"[Challenge] {cname}: skip (empty subset)")
                continue
            res = eval_challenge_set(cname, idx, y_test, prob_ens, threshold=thr_ens, cfg=cfg, primary_auc=primary_auc)
            challenge_results[cname] = res

            sub_pred = pd.DataFrame({
                "sample_index": idx.astype(int),
                "y_true": y_test[idx].astype(int),
                "prob_ensemble": prob_ens[idx].astype(float),
                "pred_label": (prob_ens[idx] >= float(thr_ens)).astype(int),
            })
            sub_csv_path = os.path.join(cfg.out_dir, "pred", f"pred_{cname}_ensemble.csv")
            sub_pred.to_csv(sub_csv_path, index=False)

            auc_key = "auc_roc" if primary_auc == "auc_roc" else "auc_pr"
            auc_ci = res.get(primary_auc + "_ci") or {}
            mcc_ci = res.get("mcc_ci") or {}
            print(f"[Challenge:{cname}] N={res['n']} pos={res['n_pos']} neg={res['n_neg']} prev={res['prevalence']:.3f}  "
                  f"{auc_key}={res[auc_key]:.2f} (CI {auc_ci.get('ci_lo', float('nan')):.2f}-{auc_ci.get('ci_hi', float('nan')):.2f})  "
                  f"MCC={res['mcc']:.2f} (CI {mcc_ci.get('ci_lo', float('nan')):.2f}-{mcc_ci.get('ci_hi', float('nan')):.2f})  "
                  f"[Saved] {sub_csv_path}")
        except Exception as e:
            print(f"[Challenge] {cname}: failed: {e}")

    if len(challenge_results) > 0:
        challenge_path = os.path.join(cfg.out_dir, "ckpt", "challenge_sets_summary.json")
        payload = {
            "cfg": cfg.__dict__,
            "ensemble_threshold": float(thr_ens),
            "challenge_sets": challenge_results,
        }
        with open(challenge_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[Saved] {challenge_path}")

    # Tool + model comparison on COMMON subsets (new)
    if not getattr(cfg, "skip_tools", False):
        try:
            eval_tools_on_challenge_sets(
                cfg,
                test_df=test_df,
                y_test=y_test,
                model_prob_test=prob_ens,
                model_threshold=thr_ens,
            )
        except Exception as e:
            print(f"[Tools] failed: {e}")
    else:
        print("[Tools] skipped (cfg.skip_tools=True)")


def main(cfg: Config = CFG):
    """
    Final model configuration:
      - protT5 encoder: BiLSTM
      - GPN-MSA encoder: GRU
      - fusion: Gate(A) scalar MLP gate + tau schedule
      - dims: protT5_d = 128, gpn_d = 128, fused_d = 128
    """
    preflight_check(cfg)
    if getattr(cfg, "preflight_only", False):
        print("[Exit] preflight_only=True")
        return
    if getattr(cfg, "tools_only", False):
        print("[Mode] tools_only=True -> evaluating classical tools only (no model training).")
        test_df = _read_csv_smart(cfg.test_csv)
        # tools-only mode has no model predictions; will output tool rows + common tool rows
        eval_tools_on_challenge_sets(cfg, test_df=test_df, y_test=None, model_prob_test=None, model_threshold=0.5)
        return

    if cfg.do_memmap_convert:
        print("[Info] do_memmap_convert=True -> converting pth to memmap once")
        prepare_memmaps(cfg)
        import copy as _copy
        cfg = _copy.deepcopy(cfg)
        cfg.do_memmap_convert = False

    cfg.t5_backbone = "bilstm"
    cfg.gpn_backbone = "gru"
    cfg.fusion_mode = "gate"
    cfg.branch_d = 128
    cfg.fused_d = 128

    ensure_dir(cfg.out_dir)

    print("\n" + "=" * 100)
    print("[FINAL MODEL] protT5=bilstm | GPN-MSA=gru | fusion=gate | prot_d=128 | gpn_d=128 | fused_d=128")
    print(f"[OUT] {cfg.out_dir}")
    print("=" * 100 + "\n")

    _run_single(cfg)


if __name__ == "__main__":
    import argparse as _argparse
    ap = _argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true", help="Only run preflight checks and exit.")
    ap.add_argument("--tools-only", action="store_true", help="Only evaluate classical tools (no model training).")
    ap.add_argument("--skip-tools", action="store_true", help="Skip tool evaluation at the end of training.")
    args = ap.parse_args()

    cfg = CFG
    if args.preflight:
        cfg.preflight_only = True
    if args.tools_only:
        cfg.tools_only = True
    if args.skip_tools:
        cfg.skip_tools = True

    main(cfg)