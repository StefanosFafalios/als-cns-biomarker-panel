# ruff: noqa: E402
"""ALS progression proxy analysis — core 25-gene panel.

Two complementary analyses that use available GSE153960 metadata to assess
which panel genes track disease severity rather than merely disease presence:

1. Tissue vulnerability gradient
   ─────────────────────────────
   ALS preferentially destroys motor neurons before spreading to other regions.
   Tissues are ranked by established ALS vulnerability (spinal cord > motor
   cortex > associative cortex > cerebellum / occipital). For each panel gene,
   the per-tissue ALS effect size (rank-biserial r) is correlated (Spearman)
   with the tissue vulnerability rank. A high "progression ρ" means the gene's
   ALS signal is strongest precisely where ALS pathology is worst → the gene
   tracks neurodegeneration, not just disease presence.

2. ALS vs ALS-FTD subgroup comparison
   ─────────────────────────────────────
   GSE153960 contains 172 samples labelled "ALS Spectrum MND, Other
   Neurological Disorders" — these are ALS-FTD subjects. Comparing pure ALS
   vs ALS-FTD isolates FTD-specific transcriptomic signal from pure motor-
   neuron-degeneration signal. Panel genes that differ between these two groups
   are sensitive to the FTD component.

3. Pre-fALS descriptive comparison (n=3)
   ─────────────────────────────────────
   Three "Pre-fALS" samples (symptomatic-free familial ALS carriers) provide
   a qualitative glimpse at pre-symptomatic gene expression.

Outputs
-------
  als_progression_proxies.png      — heatmap + bar chart
  als_progression_proxies.txt      — statistics
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ALS_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ALS_DIR))

SCRIPT_DIR = Path(__file__).parent
RESOURCES_DIR = ALS_DIR / "resources"

_MATRIX_PATH = (
    RESOURCES_DIR / "GSE153960" / "matrix"
    / "GSE153960-GPL24676_series_matrix.txt.gz"
)
_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Tissue vulnerability ranks (ALS literature)
# Higher = more severely affected in ALS pathology
# ---------------------------------------------------------------------------

_VULN: dict[str, int] = {
    "Spinal Cord Cervical":        5,
    "Spinal Cord Lumbar":          5,
    "Spinal Cord Thoracic":        5,
    "Cortex Motor Lateral":        4,
    "Cortex Motor Medial":         4,
    "Cortex Motor Unspecified":    4,
    "Cortex Frontal":              3,
    "Cortex Temporal":             2,
    "Hippocampus":                 2,
    "Cortex Sensory":              2,
    "Cortex Occipital":            1,
    "Cerebellum":                  1,
}

# ---------------------------------------------------------------------------
# 15 protein-coding panel genes (Ensembl base IDs, no version suffix)
# ---------------------------------------------------------------------------

_PANEL_GENES: list[dict[str, str]] = [
    {"symbol": "MECOM",   "ensg": "ENSG00000085276"},
    {"symbol": "SERTAD1", "ensg": "ENSG00000197019"},
    {"symbol": "FCN3",    "ensg": "ENSG00000142748"},
    {"symbol": "PROS1",   "ensg": "ENSG00000184500"},
    {"symbol": "ANGPT2",  "ensg": "ENSG00000091879"},
    {"symbol": "EMP1",    "ensg": "ENSG00000134531"},
    {"symbol": "TINAGL1", "ensg": "ENSG00000142910"},
    {"symbol": "CKMT2",   "ensg": "ENSG00000131730"},
    {"symbol": "VWF",     "ensg": "ENSG00000110799"},
    {"symbol": "CLDN5",   "ensg": "ENSG00000184113"},
    {"symbol": "NR4A1",   "ensg": "ENSG00000123358"},
    {"symbol": "SOHLH2",  "ensg": "ENSG00000120669"},
    {"symbol": "HEXB",    "ensg": "ENSG00000049860"},
    {"symbol": "MCEE",    "ensg": "ENSG00000124370"},
    {"symbol": "SLC37A2", "ensg": "ENSG00000134955"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from utils import _load_suppl_expression, _parse_series_matrix  # noqa: E402


def _rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-biserial correlation (effect size for Mann-Whitney U).

    Equivalent to 2*(U / (n1*n2)) - 1, ranges −1 to +1.
    Positive = x values tend to be larger.
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    u_stat, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    return float(2 * u_stat / (n1 * n2) - 1)


def _match_feature(ensg_base: str, feature_names: list[str]) -> str | None:
    """Return the versioned feature name matching the base Ensembl ID."""
    for fn in feature_names:
        if fn.split(".")[0] == ensg_base:
            return fn
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("ALS progression proxy analysis — core 25-gene panel")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load expression + metadata
    # ------------------------------------------------------------------
    print("\n[1] Loading expression data ...")
    meta_df, _ = _parse_series_matrix(_MATRIX_PATH)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        expr_df = _load_suppl_expression("GSE153960", meta_df, RESOURCES_DIR)

    common = meta_df.index.intersection(expr_df.columns)
    meta_df = meta_df.loc[common]
    expr_df = expr_df[common]
    feature_names: list[str] = list(expr_df.index)
    print(f"  Samples: {len(common)}, Genes: {len(feature_names)}")

    # log1p transform
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        X_log = np.log1p(expr_df.values.astype(np.float32))

    grp = meta_df["group"].fillna("").astype(str)
    tissue = meta_df["tissue"].fillna("").astype(str)

    als_mask = grp.str.match(r"^ALS Spectrum MND$", na=False)
    als_ftd_mask = grp.str.match(
        r"^ALS Spectrum MND, Other Neurological Dis", na=False
    )
    ctrl_mask = grp.str.match(r"^Non-Neurological Control", na=False)
    prefals_mask = grp.str.match(r"Pre-fALS", na=False)

    print(
        f"  Pure ALS: {als_mask.sum()}, ALS-FTD: {als_ftd_mask.sum()}, "
        f"Control: {ctrl_mask.sum()}, Pre-fALS: {prefals_mask.sum()}"
    )

    # ------------------------------------------------------------------
    # 2. Map panel genes → feature indices
    # ------------------------------------------------------------------
    print("\n[2] Mapping panel genes to expression features ...")
    gene_feat: list[dict[str, Any]] = []
    for g in _PANEL_GENES:
        fname = _match_feature(g["ensg"], feature_names)
        if fname is None:
            print(f"  [WARN] {g['symbol']} ({g['ensg']}) not found")
            continue
        idx = feature_names.index(fname)
        gene_feat.append({**g, "feat": fname, "idx": idx})
        print(f"  {g['symbol']}: {fname}")

    # ------------------------------------------------------------------
    # 3. Tissue vulnerability gradient analysis
    # ------------------------------------------------------------------
    print("\n[3] Tissue vulnerability gradient ...")
    tissues_present = [t for t in _VULN if tissue.eq(t).sum() > 0]
    tissues_sorted = sorted(tissues_present, key=lambda t: _VULN[t])

    # For each gene × tissue: rank-biserial r (ALS vs Control)
    # Only include tissues with >= 5 ALS AND >= 3 Control samples
    vuln_records: list[dict] = []
    for row in gene_feat:
        expr_vec = X_log[row["idx"], :]
        rhos: list[float] = []
        vuln_ranks: list[int] = []
        tissue_names: list[str] = []

        for tiss in tissues_present:
            t_mask = tissue.eq(tiss)
            als_t = expr_vec[als_mask & t_mask]
            ctrl_t = expr_vec[ctrl_mask & t_mask]
            if len(als_t) < 5 or len(ctrl_t) < 3:
                continue
            rb = _rank_biserial(als_t, ctrl_t)
            rhos.append(rb)
            vuln_ranks.append(_VULN[tiss])
            tissue_names.append(tiss)

        if len(rhos) >= 4:
            prog_rho, prog_p = stats.spearmanr(vuln_ranks, rhos)
        else:
            prog_rho, prog_p = float("nan"), float("nan")

        vuln_records.append({
            "symbol": row["symbol"],
            "ensg": row["ensg"],
            "n_tissues": len(rhos),
            "progression_rho": prog_rho,
            "progression_p": prog_p,
            "mean_effect": float(np.nanmean(rhos)) if rhos else float("nan"),
            "_tissue_names": tissue_names,
            "_rhos": rhos,
            "_vuln_ranks": vuln_ranks,
        })

    vuln_df = pd.DataFrame(vuln_records).sort_values(
        "progression_rho", ascending=False
    )

    # Per-tissue heatmap data
    heat_data: dict[str, dict[str, float]] = {}
    for row_dict in vuln_records:
        sym = row_dict["symbol"]
        heat_data[sym] = dict(
            zip(row_dict["_tissue_names"], row_dict["_rhos"])
        )

    # ------------------------------------------------------------------
    # 4. ALS vs ALS-FTD subgroup
    # ------------------------------------------------------------------
    print("\n[4] ALS vs ALS-FTD subgroup analysis ...")
    ftd_records: list[dict] = []
    for row in gene_feat:
        expr_vec = X_log[row["idx"], :]
        als_expr = expr_vec[als_mask]
        ftd_expr = expr_vec[als_ftd_mask]

        if len(als_expr) >= 10 and len(ftd_expr) >= 10:
            rb = _rank_biserial(ftd_expr, als_expr)  # FTD > ALS → positive
            _, p = stats.mannwhitneyu(ftd_expr, als_expr, alternative="two-sided")
        else:
            rb, p = float("nan"), float("nan")

        ftd_records.append({
            "symbol": row["symbol"],
            "als_ftd_rb": rb,
            "als_ftd_p": p,
        })

    ftd_df = pd.DataFrame(ftd_records)

    # ------------------------------------------------------------------
    # 5. Pre-fALS descriptive
    # ------------------------------------------------------------------
    print("\n[5] Pre-fALS descriptive ...")
    prefals_records: list[dict] = []
    for row in gene_feat:
        expr_vec = X_log[row["idx"], :]
        pf = expr_vec[prefals_mask]
        al = expr_vec[als_mask]
        ct = expr_vec[ctrl_mask]
        prefals_records.append({
            "symbol": row["symbol"],
            "prefals_median": float(np.median(pf)) if len(pf) else float("nan"),
            "als_median": float(np.median(al)),
            "ctrl_median": float(np.median(ct)),
            "n_prefals": len(pf),
        })

    prefals_df = pd.DataFrame(prefals_records)

    # ------------------------------------------------------------------
    # 6. Plots
    # ------------------------------------------------------------------
    print("\n[6] Plotting ...")
    genes_ordered = vuln_df["symbol"].tolist()

    fig, axes = plt.subplots(
        1, 3, figsize=(18, 7),
        gridspec_kw={"width_ratios": [2.5, 1, 1]},
    )
    fig.suptitle(
        "ALS Progression Proxies — Core 25-gene Panel\nGSE153960 GPL24676",
        fontsize=12,
    )

    # (a) Heatmap: rank-biserial r per gene × tissue
    ax0 = axes[0]
    all_tissues = sorted(_VULN, key=lambda t: _VULN[t], reverse=True)
    present_tissues = [t for t in all_tissues if t in tissues_present]
    heat_matrix = np.full((len(genes_ordered), len(present_tissues)), np.nan)
    for i, sym in enumerate(genes_ordered):
        for j, tiss in enumerate(present_tissues):
            heat_matrix[i, j] = heat_data.get(sym, {}).get(tiss, np.nan)

    im = ax0.imshow(
        heat_matrix, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5,
    )
    ax0.set_xticks(range(len(present_tissues)))
    ax0.set_xticklabels(
        [t.replace("Cortex ", "Ctx ").replace("Spinal Cord ", "SC ")
         for t in present_tissues],
        rotation=45, ha="right", fontsize=7,
    )
    ax0.set_yticks(range(len(genes_ordered)))
    ax0.set_yticklabels(genes_ordered, fontsize=8)
    ax0.set_title("ALS effect (rank-biserial r)\nper tissue\n← low vuln | high vuln →")
    plt.colorbar(im, ax=ax0, shrink=0.6)

    # (b) Progression ρ bar chart
    ax1 = axes[1]
    prog_vals = vuln_df["progression_rho"].values
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in prog_vals]
    ax1.barh(range(len(genes_ordered)), prog_vals[::-1], color=colors[::-1])
    ax1.set_yticks(range(len(genes_ordered)))
    ax1.set_yticklabels(genes_ordered[::-1], fontsize=8)
    ax1.axvline(0, color="black", lw=0.8)
    ax1.set_xlabel("Spearman ρ\n(effect ~ vulnerability)")
    ax1.set_title("Progression ρ\n(+1 = tracks ALS gradient)")
    ax1.set_xlim(-1, 1)

    # (c) ALS vs ALS-FTD rank-biserial
    ax2 = axes[2]
    ftd_vals_ordered = []
    for sym in genes_ordered:
        row = ftd_df[ftd_df["symbol"] == sym]
        ftd_vals_ordered.append(
            float(row["als_ftd_rb"].iloc[0]) if len(row) else float("nan")
        )
    ftd_arr = np.array(ftd_vals_ordered)
    colors_ftd = ["#e74c3c" if v > 0 else "#3498db" for v in ftd_arr]
    ax2.barh(
        range(len(genes_ordered)),
        ftd_arr[::-1],
        color=[c for c in colors_ftd[::-1]],
    )
    ax2.set_yticks(range(len(genes_ordered)))
    ax2.set_yticklabels([""] * len(genes_ordered), fontsize=8)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel("Rank-biserial r\n(ALS-FTD > pure ALS)")
    ax2.set_title("ALS-FTD vs ALS\n(+ = higher in ALS-FTD)")
    ax2.set_xlim(-0.6, 0.6)

    plt.tight_layout()
    out_png = SCRIPT_DIR / "als_progression_proxies.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_png.name}")

    # ------------------------------------------------------------------
    # 7. Write statistics
    # ------------------------------------------------------------------
    merged = vuln_df.merge(ftd_df, on="symbol").merge(
        prefals_df.drop(columns=["als_median", "ctrl_median"]), on="symbol"
    )

    lines: list[str] = [
        "ALS Progression Proxy Analysis — Core 25-gene Panel",
        "=" * 70,
        f"Dataset: GSE153960 GPL24676 (n={len(common)})",
        f"ALS: {als_mask.sum()}, ALS-FTD: {als_ftd_mask.sum()}, "
        f"Control: {ctrl_mask.sum()}, Pre-fALS: {prefals_mask.sum()}",
        "",
        "TISSUE VULNERABILITY GRADIENT",
        "progression_rho = Spearman r between per-tissue ALS effect (rank-biserial)",
        "                  and tissue ALS vulnerability rank (1=low, 5=high)",
        "High positive ρ → gene ALS signal is strongest in most-affected tissues",
        "  (tracks neurodegeneration gradient, not just disease presence)",
        "-" * 70,
        f"{'Symbol':<10} {'prog_rho':<12} {'p':<12} {'n_tissues':<12} "
        f"{'mean_effect'}",
        "-" * 70,
    ]

    for _, r in vuln_df.iterrows():
        p_str = f"{r['progression_p']:.3e}" if not np.isnan(r["progression_p"]) else "nan"
        rho_str = f"{r['progression_rho']:.3f}" if not np.isnan(r["progression_rho"]) else "nan"
        lines.append(
            f"{r['symbol']:<10} {rho_str:<12} {p_str:<12} "
            f"{int(r['n_tissues']):<12} {r['mean_effect']:.3f}"
        )

    lines += [
        "",
        "Tissue vulnerability ranks used:",
        "  Spinal cord (all): 5  |  Motor cortex: 4  |  Frontal cortex: 3",
        "  Temporal/Hippocampus/Sensory: 2  |  Cerebellum/Occipital: 1",
        "",
        "PER-TISSUE RANK-BISERIAL r (ALS vs Control)",
        "-" * 70,
    ]

    tissue_header = "  ".join(
        t.replace("Cortex ", "Ctx ").replace("Spinal Cord ", "SC ")[:14]
        for t in present_tissues
    )
    lines.append(f"{'Symbol':<10}  {tissue_header}")
    for sym in genes_ordered:
        vals = []
        for tiss in present_tissues:
            v = heat_data.get(sym, {}).get(tiss, float("nan"))
            vals.append(f"{v:+.2f}" if not np.isnan(v) else "  nan")
        lines.append(f"{sym:<10}  {'  '.join(vals)}")

    lines += [
        "",
        "ALS vs ALS-FTD SUBGROUP COMPARISON",
        "(positive rb = ALS-FTD > pure ALS; gene elevated in FTD-component)",
        "-" * 70,
        f"{'Symbol':<10} {'rank-biserial r':<18} {'MWU p'}",
        "-" * 70,
    ]

    ftd_sorted = ftd_df.sort_values("als_ftd_rb", ascending=False)
    for _, r in ftd_sorted.iterrows():
        rb_str = f"{r['als_ftd_rb']:+.3f}" if not np.isnan(r["als_ftd_rb"]) else "nan"
        p_str = f"{r['als_ftd_p']:.3e}" if not np.isnan(r["als_ftd_p"]) else "nan"
        lines.append(f"{r['symbol']:<10} {rb_str:<18} {p_str}")

    lines += [
        "",
        "PRE-fALS DESCRIPTIVE (n=3 pre-symptomatic familial ALS carriers)",
        "-" * 70,
        f"{'Symbol':<10} {'Pre-fALS median':<18} {'ALS median':<15} {'Control median'}",
        "-" * 70,
    ]

    for _, r in prefals_df.iterrows():
        lines.append(
            f"{r['symbol']:<10} {r['prefals_median']:<18.3f} "
            f"{r['als_median']:<15.3f} {r['ctrl_median']:.3f}"
        )

    stats_path = SCRIPT_DIR / "als_progression_proxies.txt"
    stats_path.write_text("\n".join(lines) + "\n")
    print(f"  Statistics → {stats_path.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
