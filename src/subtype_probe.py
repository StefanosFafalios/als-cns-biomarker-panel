"""Quick probe: split FTLD/PSP scores by institution to detect FTLD-TDP vs PSP."""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ALS_DIR = Path(__file__).parents[1]
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(ALS_DIR.parent))
sys.path.insert(0, str(ALS_DIR))

from utils import _load_suppl_expression, _parse_series_matrix
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

matrix_path = (
    ALS_DIR / "resources/GSE153960/matrix/GSE153960-GPL24676_series_matrix.txt.gz"
)
meta_df, _ = _parse_series_matrix(matrix_path)
expr_df = _load_suppl_expression("GSE153960", meta_df, ALS_DIR / "resources")
common = meta_df.index.intersection(expr_df.columns)
meta_df = meta_df.loc[common]
expr_df = expr_df[common]

panel = pd.read_csv(SCRIPT_DIR / "lgbm_core25_panel.csv")
feat_names = list(expr_df.index)
feat_base = {n.split(".")[0]: i for i, n in enumerate(feat_names)}
panel_cols = [
    feat_base[f.split(".")[0]]
    for f in panel["feature"]
    if f.split(".")[0] in feat_base
]

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    X_log = np.log1p(expr_df.values.T.astype(np.float32))
X_panel = X_log[:, panel_cols]

grp = meta_df["group"].fillna("").astype(str)
als_mask = grp.str.match(r"^ALS Spectrum MND", na=False)
ctrl_mask = grp.str.match(r"^Non-Neurological Control", na=False)
ftld_mask = grp.str.match(r"^Other Neurological Dis", na=False) & ~als_mask

tr_idx = [i for i, m in enumerate(als_mask | ctrl_mask) if m]
y_train = np.array([1 if als_mask.iloc[i] else 0 for i in tr_idx])

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_panel[tr_idx]).astype(np.float32)
X_all_sc = scaler.transform(X_panel).astype(np.float32)

params = json.loads((SCRIPT_DIR / "lgbm_top500_best_params.json").read_text())
params["colsample_bytree"] = 1.0
clf = LGBMClassifier(**params)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    clf.fit(X_tr_sc, y_train)

all_scores = clf.predict_proba(X_all_sc)[:, 1]

ftld_rows = [i for i, m in enumerate(ftld_mask) if m]
ftld_meta = meta_df.iloc[ftld_rows].copy()
ftld_meta["score"] = all_scores[ftld_rows]
ftld_meta["prefix"] = ftld_meta["subject id"].str.split("-").str[:2].str.join("-")

print("Scores by institution (FTLD/PSP group):")
for prefix, sub in ftld_meta.groupby("prefix"):
    s = sub["score"].values
    print(
        f"  {prefix:<12}  n={len(s):3d}  median={np.median(s):.4f}"
        f"  mean={np.mean(s):.4f}"
        f"  IQR=[{np.percentile(s, 25):.4f}, {np.percentile(s, 75):.4f}]"
    )

print("\nTissue distribution by institution:")
for prefix, sub in ftld_meta.groupby("prefix"):
    print(f"\n  {prefix}:")
    for tissue, cnt in sub["tissue"].value_counts().items():
        print(f"    {tissue:<30} {cnt}")

# Score threshold split: high (≥0.5) vs low (<0.5)
high = ftld_meta[ftld_meta["score"] >= 0.5]
low = ftld_meta[ftld_meta["score"] < 0.5]
print(f"\nScore ≥ 0.5 (ALS-like):  n={len(high)}")
print(high["prefix"].value_counts().to_string())
print(f"\nScore < 0.5 (ctrl-like): n={len(low)}")
print(low["prefix"].value_counts().to_string())

# Scores by tissue
from scipy.stats import mannwhitneyu

print("\nScores by tissue within FTLD/PSP group:")
for tissue, sub in ftld_meta.groupby("tissue"):
    s = sub["score"].values
    print(
        f"  {tissue:<30} n={len(s):3d}  median={np.median(s):.4f}"
        f"  IQR=[{np.percentile(s, 25):.3f}, {np.percentile(s, 75):.3f}]"
    )

cortex = ftld_meta[ftld_meta["tissue"].str.contains("Cortex Frontal|Cortex Temporal", na=False)]["score"].values
cerebellum = ftld_meta[ftld_meta["tissue"] == "Cerebellum"]["score"].values
_, p = mannwhitneyu(cortex, cerebellum, alternative="two-sided")
print(f"\nFrontal+Temporal cortex (n={len(cortex)}) vs Cerebellum (n={len(cerebellum)}):")
print(f"  Cortex     median={np.median(cortex):.4f}")
print(f"  Cerebellum median={np.median(cerebellum):.4f}")
print(f"  Mann-Whitney U p={p:.3e}")

# Per-subject median — rank subjects high to low
print("\nPer-subject median score:")
subj_med = ftld_meta.groupby("subject id")["score"].median().sort_values(ascending=False)
print("  HIGH scorers (FTLD-TDP candidates):")
print(subj_med.head(8).to_string())
print("  LOW scorers (PSP candidates):")
print(subj_med.tail(8).to_string())
