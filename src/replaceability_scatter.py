"""Replaceability vs LOO Δ scatter plot for the core 25-gene ALS panel.

x-axis : LOO Δ AUC (within 25-gene panel context)
y-axis : Replaceability % (fraction of candidates that are valid replacements)
Size   : mean |SHAP|  (scaled for visibility)
Colour : biological theme
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

BASE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Biological theme assignments
# ---------------------------------------------------------------------------
_PC15 = {
    "MECOM", "SERTAD1", "FCN3", "PROS1", "ANGPT2", "EMP1", "TINAGL1",
    "CKMT2", "VWF", "CLDN5", "NR4A1", "SOHLH2", "HEXB", "MCEE", "SLC37A2",
}

THEME = {
    "MECOM":    "Cell cycle / TF",
    "SERTAD1":  "Cell cycle / TF",
    "FCN3":     "Immune / complement",
    "PROS1":    "Vascular / BBB",
    "ANGPT2":   "Vascular / BBB",
    "EMP1":     "Vascular / BBB",
    "TINAGL1":  "Vascular / BBB",
    "CKMT2":    "Mitochondria / metabolism",
    "VWF":      "Vascular / BBB",
    "CLDN5":    "Vascular / BBB",
    "NR4A1":    "Cell cycle / TF",
    "SOHLH2":   "Cell cycle / TF",
    "HEXB":     "Lysosomal",
    "MCEE":     "Mitochondria / metabolism",
    "SLC37A2":  "Mitochondria / metabolism",
}

PALETTE = {
    "Vascular / BBB":            "#E05C5C",   # red
    "Mitochondria / metabolism": "#F5A623",   # amber
    "Cell cycle / TF":           "#4A90D9",   # blue
    "Immune / complement":       "#27AE60",   # green
    "Lysosomal":                 "#8E44AD",   # purple
    "Non-coding / pseudogene":   "#95A5A6",   # grey
}

# ---------------------------------------------------------------------------
# Load LOO Δ + SHAP from lgbm_core25_panel.csv
# ---------------------------------------------------------------------------
panel = pd.read_csv(BASE / "lgbm_core25_panel.csv")
# columns: rank, feature, symbol, mean_abs_shap, loo_delta_auc

# ---------------------------------------------------------------------------
# Parse gene_replacement_statistics.txt
# ---------------------------------------------------------------------------
repl_path = BASE / "gene_replacement_statistics.txt"
repl_rows = []
in_table = False
for line in repl_path.read_text().splitlines():
    if line.startswith("Gene ") and "n_corr" in line:
        in_table = True
        continue
    if in_table and line.startswith("-"):
        continue
    if in_table and line.strip() == "":
        break
    if in_table:
        parts = line.split()
        if len(parts) >= 7:
            gene = parts[0]
            repl_pct_str = parts[5].rstrip("%")
            try:
                repl_pct = float(repl_pct_str)
            except ValueError:
                continue
            repl_rows.append({"symbol": gene, "repl_pct": repl_pct})

repl_df = pd.DataFrame(repl_rows)

# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
df = panel.merge(repl_df, on="symbol", how="inner")
df = df[df["symbol"].isin(_PC15)].copy()
df["theme"] = df["symbol"].map(THEME).fillna("Other")

# Scale dot size: map mean_abs_shap to reasonable marker area
shap_vals = df["mean_abs_shap"].values
size_min, size_max = 60, 600
sizes = size_min + (shap_vals - shap_vals.min()) / (shap_vals.max() - shap_vals.min() + 1e-9) * (size_max - size_min)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 8))

for _, row in df.iterrows():
    color = PALETTE.get(row["theme"], "#333333")
    s = size_min + (row["mean_abs_shap"] - shap_vals.min()) / (shap_vals.max() - shap_vals.min() + 1e-9) * (size_max - size_min)
    ax.scatter(row["loo_delta_auc"], row["repl_pct"], s=s, color=color,
               alpha=0.80, edgecolors="white", linewidths=0.8, zorder=3)

# Gene labels — offset to reduce overlap
label_offsets: dict[str, tuple[float, float]] = {
    "CKMT2":   (-0.00007, -3),
    "NR4A1":   (+0.00004, -3),
    "TINAGL1": (-0.00012, +2),
    "SERTAD1": (+0.00004, +2),
    "EMP1":    (-0.00014, +2),
    "MECOM":   (+0.00004, +2),
    "HEXB":    (+0.00004, +2),
    "PROS1":   (-0.00007, +2),
    "VWF":     (+0.00004, -3),
    "CLDN5":   (-0.00014, +2),
    "FCN3":    (+0.00004, +2),
    "ANGPT2":  (+0.00004, +2),
    "MCEE":    (+0.00004, -3),
    "SLC37A2": (-0.00014, +2),
    "SOHLH2":  (+0.00004, +2),
}

for _, row in df.iterrows():
    ox, oy = label_offsets.get(row["symbol"], (0.00005, 3))
    ax.annotate(
        row["symbol"],
        xy=(row["loo_delta_auc"], row["repl_pct"]),
        xytext=(row["loo_delta_auc"] + ox, row["repl_pct"] + oy),
        fontsize=7.5,
        ha="left" if ox >= 0 else "right",
        va="bottom" if oy >= 0 else "top",
        color="#222222",
    )

# Reference lines
ax.axhline(50, color="#AAAAAA", lw=0.8, ls="--", zorder=1)
ax.axvline(0, color="#AAAAAA", lw=0.8, ls="--", zorder=1)

# Quadrant annotations
ax.text(0.16, 0.56, "Low LOO Δ\nHigh repl%\n(redundant, widely substitutable)",
        transform=ax.transAxes, fontsize=7, color="#888888",
        ha="center", va="center", style="italic")
ax.text(0.15, 0.03, "Low LOO Δ\nLow repl%\n(redundant but unique signal)",
        transform=ax.transAxes, fontsize=7, color="#888888",
        ha="center", va="bottom", style="italic")
ax.text(0.85, 0.03, "High LOO Δ\nLow repl%\n(irreplaceable, unique signal)",
        transform=ax.transAxes, fontsize=7, color="#E05C5C",
        ha="center", va="bottom", style="italic", weight="bold")
ax.text(0.85, 0.97, "High LOO Δ\nHigh repl%\n(important but substitutable)",
        transform=ax.transAxes, fontsize=7, color="#4A90D9",
        ha="center", va="top", style="italic")

# Legend: theme colours
theme_patches = [
    mpatches.Patch(color=c, label=t)
    for t, c in PALETTE.items()
]
# Size legend (SHAP)
size_vals = [shap_vals.min(), np.median(shap_vals), shap_vals.max()]
size_labels = [f"|SHAP| = {v:.3f}" for v in size_vals]
size_handles = [
    plt.scatter([], [], s=size_min + (v - shap_vals.min()) / (shap_vals.max() - shap_vals.min() + 1e-9) * (size_max - size_min),
                color="#888888", alpha=0.7, edgecolors="white")
    for v in size_vals
]

leg1 = ax.legend(handles=theme_patches, title="Biological theme",
                  loc="lower left", bbox_to_anchor=(0.0, 0.14),
                  fontsize=8, title_fontsize=8, framealpha=0.9)
ax.add_artist(leg1)
ax.legend(handles=size_handles, labels=size_labels, title="Dot size ∝ mean |SHAP|",
          loc="lower right", bbox_to_anchor=(1.0, 0.14), fontsize=7.5,
          title_fontsize=7.5, framealpha=0.9)

ax.set_xlabel("LOO Δ AUC  (within 25-gene panel context)", fontsize=11)
ax.set_ylabel("Replaceability  (%)", fontsize=11)
ax.set_title("Replaceability vs LOO Δ — 15 Protein-Coding Panel Members\n"
             "(dot size ∝ mean |SHAP|; LOO Δ from full-model 5-fold CV, replaceability from fast-model screen)",
             fontsize=10)
ax.set_ylim(-5, 110)
ax.set_xlim(df["loo_delta_auc"].min() - 0.0005, df["loo_delta_auc"].max() + 0.0010)
ax.grid(axis="both", color="#EEEEEE", lw=0.6, zorder=0)

plt.tight_layout()
out = BASE / "replaceability_scatter.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# ---------------------------------------------------------------------------
# Print data table for verification
# ---------------------------------------------------------------------------
cols = ["symbol", "loo_delta_auc", "repl_pct", "mean_abs_shap", "theme"]
print(df[cols].sort_values("repl_pct").to_string(index=False))
