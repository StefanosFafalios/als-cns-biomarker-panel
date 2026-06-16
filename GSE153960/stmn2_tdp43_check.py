"""STMN2 expression in panel-positive vs panel-negative samples (TDP-43 bridge check).

The Discussion claims the panel captures a TDP-43 proteinopathy signature.
A direct test is whether STMN2 — a canonical TDP-43 loss-of-function readout
(Klim 2019) — is depleted in samples that the panel scores as TDP-43-positive.

Compares STMN2 (ENSG00000104435) in:
  (a) ALS samples scoring P(ALS=1) >= 0.5 vs Control samples scoring < 0.5
  (b) Held-out FTLD/PSP samples by panel score (high vs low)

Outputs: stmn2_tdp43_check_statistics.txt, stmn2_tdp43_check.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

matplotlib.use("Agg")

ALS_DIR = Path(__file__).parents[1]
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset  # noqa: E402

STMN2_BASE = "ENSG00000104435"


def _rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    u, _ = mannwhitneyu(x, y, alternative="two-sided")
    return float(2 * u / (len(x) * len(y)) - 1)


def main() -> None:
    print("Loading GPL24676 ...")
    (ds,) = load_dataset(
        "GSE153960",
        platform="GPL24676",
        resources_dir=ALS_DIR / "resources",
    )
    print(f"  shape: {ds.X.shape}")
    print(f"  unique groups: {ds.meta['group'].unique().tolist()}")

    # Locate STMN2
    stmn2_col = None
    for c in ds.X.columns:
        if c.startswith(STMN2_BASE):
            stmn2_col = c
            break
    if stmn2_col is None:
        print("  STMN2 not found")
        return
    print(f"  STMN2 column: {stmn2_col}")

    stmn2_expr = np.log1p(ds.X[stmn2_col].values.astype(np.float32))

    # Group by diagnosis (using ds.meta['group'])
    diag = ds.meta["group"].astype(str).str.upper()
    g_als = diag.str.contains("ALS|AMYOTROPHIC", regex=True)
    g_ctrl = diag.str.contains("NON-NEUROLOGICAL|CONTROL", regex=True)
    g_other = diag.str.contains("OTHER NEUROLOGICAL|FTLD|PSP|CBS", regex=True)

    als_expr = stmn2_expr[g_als.values]
    ctrl_expr = stmn2_expr[g_ctrl.values]
    other_expr = stmn2_expr[g_other.values]

    print(f"\nSTMN2 (log1p RSEM) by diagnosis:")
    print(f"  ALS    : n={len(als_expr):4d}  mean={als_expr.mean():.3f}  median={np.median(als_expr):.3f}")
    print(f"  Control: n={len(ctrl_expr):4d}  mean={ctrl_expr.mean():.3f}  median={np.median(ctrl_expr):.3f}")
    print(f"  Other  : n={len(other_expr):4d}  mean={other_expr.mean():.3f}  median={np.median(other_expr):.3f}")

    u_als_ctrl, p_als_ctrl = mannwhitneyu(als_expr, ctrl_expr, alternative="two-sided")
    rb_als_ctrl = _rank_biserial(als_expr, ctrl_expr)
    lfc_als_ctrl = np.log2((np.expm1(als_expr).mean() + 1) / (np.expm1(ctrl_expr).mean() + 1))
    print(f"\nALS vs Control: rank-biserial = {rb_als_ctrl:+.3f}, p = {p_als_ctrl:.3e}")
    print(f"  log2 FC (ALS/Control) on raw RSEM counts: {lfc_als_ctrl:+.3f}")

    # Other-neurological subjects split by panel-positive/negative
    # No saved panel scores available, so we use a panel-membership proxy
    # via the up-down composite z-score using the protein-coding critical
    # panel
    UP_GENES = {
        "ENSG00000142748": "FCN3",
        "ENSG00000184500": "PROS1",
        "ENSG00000134531": "EMP1",
        "ENSG00000120669": "SOHLH2",
        "ENSG00000049860": "HEXB",
        "ENSG00000124370": "MCEE",
        "ENSG00000134955": "SLC37A2",
    }
    DOWN_GENES = {
        "ENSG00000091879": "ANGPT2",
        "ENSG00000131730": "CKMT2",
        "ENSG00000184113": "CLDN5",
        "ENSG00000142910": "TINAGL1",
    }

    def _find(base: str) -> str | None:
        for c in ds.X.columns:
            if c.startswith(base):
                return c
        return None

    up_cols = [_find(b) for b in UP_GENES]
    dn_cols = [_find(b) for b in DOWN_GENES]
    up_cols = [c for c in up_cols if c]
    dn_cols = [c for c in dn_cols if c]
    print(f"\nPanel composite — up={len(up_cols)}/{len(UP_GENES)}, down={len(dn_cols)}/{len(DOWN_GENES)}")

    X_up = np.log1p(ds.X[up_cols].values.astype(np.float32))
    X_dn = np.log1p(ds.X[dn_cols].values.astype(np.float32))
    # z-score per gene across all 874
    z_up = (X_up - X_up.mean(axis=0)) / (X_up.std(axis=0) + 1e-9)
    z_dn = (X_dn - X_dn.mean(axis=0)) / (X_dn.std(axis=0) + 1e-9)
    panel_score = z_up.mean(axis=1) - z_dn.mean(axis=1)

    # Split Other-neuro by panel score (median split: panel-high = TDP-43-like)
    other_idx = np.where(g_other.values)[0]
    if len(other_idx) >= 10:
        scores = panel_score[other_idx]
        thresh = np.median(scores)
        hi = other_idx[scores >= thresh]
        lo = other_idx[scores < thresh]
        stmn2_hi = stmn2_expr[hi]
        stmn2_lo = stmn2_expr[lo]
        u_hl, p_hl = mannwhitneyu(stmn2_hi, stmn2_lo, alternative="two-sided")
        rb_hl = _rank_biserial(stmn2_hi, stmn2_lo)
        print(f"\nOther-neuro split by panel composite (median; n={len(hi)} hi, {len(lo)} lo):")
        print(f"  STMN2 hi-panel: mean={stmn2_hi.mean():.3f}")
        print(f"  STMN2 lo-panel: mean={stmn2_lo.mean():.3f}")
        print(f"  rank-biserial  = {rb_hl:+.3f}, p = {p_hl:.3e}")
        print(f"  expectation: TDP-43-positive (hi panel) should show LOWER STMN2")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    parts = ax.violinplot([ctrl_expr, als_expr, other_expr], positions=[0, 1, 2],
                          showmeans=False, showmedians=True, widths=0.7)
    for body, color in zip(parts["bodies"], ["#1f77b4", "#d62728", "#7f7f7f"]):
        body.set_facecolor(color)
        body.set_alpha(0.7)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([f"Control\n(n={len(ctrl_expr)})",
                         f"ALS Spectrum MND\n(n={len(als_expr)})",
                         f"Other Neuro\n(n={len(other_expr)})"])
    ax.set_ylabel(r"STMN2 expression (log$_1$p RSEM)")
    ax.set_title(f"STMN2 (TDP-43 loss-of-function readout) by diagnosis\nALS vs Ctrl: $r_b$={rb_als_ctrl:+.3f}, $p$={p_als_ctrl:.2e}", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    if len(other_idx) >= 10:
        ax2.scatter(scores, stmn2_expr[other_idx], s=30, alpha=0.7, color="#7f7f7f")
        ax2.axvline(thresh, ls="--", color="black", lw=0.6)
        ax2.set_xlabel(r"Panel composite z-score (up$-$down)")
        ax2.set_ylabel(r"STMN2 expression (log$_1$p RSEM)")
        ax2.set_title(f"Other-Neurological samples\nrank-biserial (hi vs lo panel) = {rb_hl:+.3f}, $p$={p_hl:.3e}", fontsize=11)
        ax2.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(SCRIPT_DIR / "stmn2_tdp43_check.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # Statistics
    lines = [
        "STMN2 TDP-43 bridge check",
        "=" * 60,
        f"STMN2 column: {stmn2_col}",
        "",
        f"ALS Spectrum MND (n={len(als_expr)})  mean log1p = {als_expr.mean():.3f}",
        f"Non-Neuro Control (n={len(ctrl_expr)}) mean log1p = {ctrl_expr.mean():.3f}",
        f"Other Neurological (n={len(other_expr)}) mean log1p = {other_expr.mean():.3f}",
        "",
        f"ALS vs Control:",
        f"  rank-biserial = {rb_als_ctrl:+.3f}",
        f"  Mann-Whitney p = {p_als_ctrl:.3e}",
        f"  log2 FC (ALS/Control) raw RSEM = {lfc_als_ctrl:+.3f}",
        f"  direction: {'DOWN in ALS' if lfc_als_ctrl < 0 else 'UP in ALS'}",
        f"  TDP-43 expectation: STMN2 should be DOWN in ALS (TDP-43 LOF)",
        "",
    ]
    if len(other_idx) >= 10:
        lines += [
            f"Other-Neurological cohort split by panel composite (median):",
            f"  hi-panel STMN2 mean = {stmn2_hi.mean():.3f}  (n={len(hi)})",
            f"  lo-panel STMN2 mean = {stmn2_lo.mean():.3f}  (n={len(lo)})",
            f"  rank-biserial (hi vs lo) = {rb_hl:+.3f}",
            f"  Mann-Whitney p = {p_hl:.3e}",
            f"  TDP-43 expectation: hi-panel (FTLD-TDP-like) should show LOWER STMN2",
        ]
    (SCRIPT_DIR / "stmn2_tdp43_check_statistics.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
