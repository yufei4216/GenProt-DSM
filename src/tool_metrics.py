import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

file_path = r"F:\20251210up\metrics_result\test-score.csv"
df = pd.read_csv(file_path)

# label 必须是 0/1
y_all = pd.to_numeric(df["label"], errors="coerce").astype("Int64")

# 简名 -> 列名
tool_map = {
    "SIFT": "SIFT_score",
    "Polyphen2": "Polyphen2_HVAR_score",
    "FATHMM": "FATHMM_score",
    "PROVEAN": "PROVEAN_score",
    "MPC": "MPC_score",
    "PrimateAI": "PrimateAI_score",
    "DEOGEN2": "DEOGEN2_score",
    "AlphaMissense": "AlphaMissense_score",
    "CADD": "CADD_phred",
    "DANN": "DANN_score",
    "GenoCanyon": "GenoCanyon_score",
    "Ours": "Ours_score"
}

eps = 1e-15
rows_original = []
rows_common_subset = []

# 第一步：计算公共子集（所有工具列和label都不为空的行）
common_columns = list(tool_map.values())
common_mask = y_all.notna()
for col in common_columns:
    if col in df.columns:
        common_mask = common_mask & pd.to_numeric(df[col], errors="coerce").notna()
    else:
        common_mask = pd.Series(False, index=df.index)

y_common = y_all[common_mask].astype(int).to_numpy()
common_n_used = int(common_mask.sum())

print(f"公共子集大小: {common_n_used}")

# 计算两种情况的性能
def calculate_metrics(y_true, y_score, tool_name, tool_col, subset_type):
    """计算性能指标的通用函数"""
    n_used = len(y_true)
    
    if n_used == 0 or len(np.unique(y_true)) < 2:
        return {
            "tool_name": tool_name,
            "tool_column": tool_col,
            "subset_type": subset_type,
            "n_used": n_used,
            "flipped_direction": np.nan,
            "AUC_ROC": np.nan,
            "AUC_PR": np.nan,
            "ACC": np.nan,
            "F1": np.nan,
            "PREC": np.nan,
            "REC": np.nan,
            "Best_Threshold_oriented": np.nan
        }
    
    # 自动纠正方向：如果 AUC < 0.5，翻转分数方向
    auc_tmp = roc_auc_score(y_true, y_score)
    flipped = False
    if auc_tmp < 0.5:
        y_score = -y_score
        flipped = True
    
    # 连续分数指标
    auc_roc = roc_auc_score(y_true, y_score)
    auc_pr = average_precision_score(y_true, y_score)  # AUC-PR(AP)
    
    # 阈值：用 PR 曲线阈值集合找 F1 最大
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + eps)
    best_idx = int(np.nanargmax(f1s))
    best_thr = float(thr[best_idx]) if len(thr) > 0 else np.nan
    
    # 二值化后指标
    y_pred = (y_score >= best_thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    
    return {
        "tool_name": tool_name,
        "tool_column": tool_col,
        "subset_type": subset_type,
        "n_used": n_used,
        "flipped_direction": flipped,
        "AUC_ROC": float(auc_roc),
        "AUC_PR": float(auc_pr),
        "ACC": float(acc),
        "F1": float(f1),
        "PREC": float(p),
        "REC": float(r),
        "Best_Threshold_oriented": float(best_thr)
    }

# 循环计算每个工具的性能
for tool_name, tool_col in tool_map.items():
    if tool_col not in df.columns:
        # 原始性能
        rows_original.append({
            "tool_name": tool_name,
            "tool_column": tool_col,
            "subset_type": "original",
            "n_used": 0,
            "flipped_direction": np.nan,
            "AUC_ROC": np.nan,
            "AUC_PR": np.nan,
            "ACC": np.nan,
            "F1": np.nan,
            "PREC": np.nan,
            "REC": np.nan,
            "Best_Threshold_oriented": np.nan
        })
        # 公共子集性能
        rows_common_subset.append({
            "tool_name": tool_name,
            "tool_column": tool_col,
            "subset_type": "common_subset",
            "n_used": 0,
            "flipped_direction": np.nan,
            "AUC_ROC": np.nan,
            "AUC_PR": np.nan,
            "ACC": np.nan,
            "F1": np.nan,
            "PREC": np.nan,
            "REC": np.nan,
            "Best_Threshold_oriented": np.nan
        })
        print(f"{tool_name}: column not found -> {tool_col}")
        continue
    
    # 1. 原始性能（各自非空子集）
    s_raw = pd.to_numeric(df[tool_col], errors="coerce")
    mask = y_all.notna() & s_raw.notna()
    y_true = y_all[mask].astype(int).to_numpy()
    y_score = s_raw[mask].astype(float).to_numpy()
    
    original_metrics = calculate_metrics(y_true, y_score, tool_name, tool_col, "original")
    rows_original.append(original_metrics)
    
    print(f"\n{tool_name} [{tool_col}] - 原始性能 (n={original_metrics['n_used']})")
    print(f"  AUC-ROC: {original_metrics['AUC_ROC']:.4f}")
    print(f"  AUC-PR : {original_metrics['AUC_PR']:.4f}")
    print(f"  ACC    : {original_metrics['ACC']:.4f}")
    
    # 2. 公共子集性能
    if common_n_used > 0:
        y_true_common = y_common
        y_score_common = pd.to_numeric(df.loc[common_mask, tool_col], errors="coerce").astype(float).to_numpy()
        common_metrics = calculate_metrics(y_true_common, y_score_common, tool_name, tool_col, "common_subset")
        rows_common_subset.append(common_metrics)
        
        print(f"{tool_name} [{tool_col}] - 公共子集性能 (n={common_metrics['n_used']})")
        print(f"  AUC-ROC: {common_metrics['AUC_ROC']:.4f}")
        print(f"  AUC-PR : {common_metrics['AUC_PR']:.4f}")
        print(f"  ACC    : {common_metrics['ACC']:.4f}")
    else:
        rows_common_subset.append({
            "tool_name": tool_name,
            "tool_column": tool_col,
            "subset_type": "common_subset",
            "n_used": 0,
            "flipped_direction": np.nan,
            "AUC_ROC": np.nan,
            "AUC_PR": np.nan,
            "ACC": np.nan,
            "F1": np.nan,
            "PREC": np.nan,
            "REC": np.nan,
            "Best_Threshold_oriented": np.nan
        })

# 合并两种性能结果
all_rows = rows_original + rows_common_subset
out_df = pd.DataFrame(all_rows)

# 重新排序列顺序
column_order = [
    "tool_name", "tool_column", "subset_type", "n_used", "flipped_direction",
    "AUC_ROC", "AUC_PR", "ACC", "F1", "PREC", "REC", "Best_Threshold_oriented"
]
out_df = out_df[column_order]

# 保存结果
output_path = r"F:\20251210up\metrics_result\metrics_tools_with_common_subset.csv"
out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\nSaved metrics to: {output_path}")
print(f"包含 {len(rows_original)} 个原始性能结果和 {len(rows_common_subset)} 个公共子集性能结果")