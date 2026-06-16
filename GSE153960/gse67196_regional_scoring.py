"""Regional ALS model scoring: GSE67196 frontal cortex vs cerebellum.

GSE67196 (Mayo Clinic; n=53 ALS):
  - 27 frontal cortex (FCX) samples  (ALS*_fcx)
  - 26 cerebellum (cereb) samples    (ALS*_cereb)
  All samples are ALS Spectrum MND; no non-neurological controls.

Strategy
--------
1. Resolve 15 protein-coding panel gene symbols via core25_annotations.tsv
   and map to versioned ENSG IDs in the GPL24676 training matrix.
2. Train LightGBM (top-500 best params, colsample_bytree=1.0) on all 874
   GPL24676 samples using the matched protein-coding genes.
3. Load GSE67196 raw counts; apply log1p; match to protein-coding panel
   genes by gene symbol.
4. Score all 53 GSE67196 samples → ALS probability (P(ALS=1 | expression)).
5. Compare FCX vs cerebellum scores:
   - Mann-Whitney U test (unpaired) + rank-biserial effect size
   - Wilcoxon signed-rank test on the 26 matched subject pairs
6. Per-gene expression comparison (FCX vs cereb) for each matched panel gene.

Outputs
-------
  gse67196_regional_scoring.png
  gse67196_regional_scoring_statistics.txt
"""

from __future__ import annotations

import gzip
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ALS_DIR = Path(__file__).parents[1]
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset  # noqa: E402

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"
_ANNOT_TSV = SCRIPT_DIR / "core25_annotations.tsv"
_RAWCOUNT = ALS_DIR / "resources" / "GSE67196" / "GSE67196_rawcount.txt.gz"
_OUT_PNG = SCRIPT_DIR / "gse67196_regional_scoring.png"
_OUT_STATS = SCRIPT_DIR / "gse67196_regional_scoring_statistics.txt"

RANDOM_STATE = 42
N_BOOTSTRAP = 2_000
PLATFORM = "GPL24676"

_PC_SYMS: set[str] = {
    "MECOM", "SERTAD1", "FCN3", "PROS1", "ANGPT2", "EMP1", "TINAGL1",
    "CKMT2", "VWF", "CLDN5", "NR4A1", "SOHLH2", "HEXB", "MCEE", "SLC37A2",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_symbol_map() -> dict[str, str]:
    """Return {ensg_base: symbol} for panel genes.

    Prefers core25_annotations.tsv; falls back to lgbm_core25_panel.csv
    symbol column which is always present.
    """
    import pandas as pd

    if _ANNOT_TSV.exists():
        df = pd.read_csv(_ANNOT_TSV, sep="\t")
        ensg_col = next((c for c in df.columns if "ensg" in c.lower()), None)
        sym_col = next(
            (c for c in df.columns if "symbol" in c.lower() or "gene_name" in c.lower()),
            None,
        )
        if ensg_col is not None and sym_col is not None:
            return {
                str(row[ensg_col]).split(".")[0]: str(row[sym_col])
                for _, row in df.iterrows()
            }

    # Fallback: use panel CSV symbol column directly
    panel_df = pd.read_csv(_PANEL_CSV)
    feat_col = next(
        (c for c in panel_df.columns if "ensg" in c.lower()),
        next(c for c in panel_df.columns if "feature" in c.lower()),
    )
    sym_col = next(c for c in panel_df.columns if "symbol" in c.lower())
    return {
        str(row[feat_col]).split(".")[0]: str(row[sym_col])
        for _, row in panel_df.iterrows()
    }


def _load_train(
    sym_map: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load GPL24676 training data; return X, y, versioned_ensg_ids, symbols.

    Only protein-coding panel genes with a symbol in _PC_SYMS are returned.
    """
    import pandas as pd

    panel_df = pd.read_csv(_PANEL_CSV)
    # Column may be named 'feature', 'ensg', etc.
    ensg_col = next(
        (c for c in panel_df.columns if "ensg" in c.lower()),
        next(c for c in panel_df.columns if "feature" in c.lower()),
    )
    panel_ensg = panel_df[ensg_col].tolist()

    (ds,) = load_dataset("GSE153960", platform=PLATFORM, resources_dir=ALS_DIR / "resources")
    X_log = np.log1p(ds.X.values.astype(np.float32))
    feat_map = {n: i for i, n in enumerate(ds.X.columns)}

    cols: list[int] = []
    versioned: list[str] = []
    symbols: list[str] = []
    for ensg_v in panel_ensg:
        base = ensg_v.split(".")[0]
        sym = sym_map.get(base, "")
        if sym in _PC_SYMS:
            idx = feat_map.get(ensg_v)
            if idx is not None:
                cols.append(idx)
                versioned.append(ensg_v)
                symbols.append(sym)

    y = ds.y.values.astype(int)
    return X_log[:, cols], y, versioned, symbols


def _load_gse67196(
    symbols: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """Load GSE67196 raw counts; return X, region_labels, sample_ids, subjects, matched_syms.

    Returns X shape (n_samples, n_matched), region_labels ('FCX'/'Cereb'),
    sample_ids (column names), subjects (patient IDs), matched_syms.
    """
    import pandas as pd

    with gzip.open(_RAWCOUNT, "rt") as fh:
        counts = pd.read_csv(fh, sep="\t", index_col="GeneID").drop(
            columns=["Chr"], errors="ignore"
        )

    sample_ids = list(counts.columns)
    region_labels = ["FCX" if "_fcx" in s else "Cereb" for s in sample_ids]
    subjects = [s.split("_")[0] for s in sample_ids]

    # Match panel genes by symbol (case-insensitive safety)
    sym_upper = {s.upper(): s for s in symbols}
    idx_map = {s.upper(): i for i, s in enumerate(symbols)}
    available_syms: list[str] = []
    matched_order: list[int] = []  # index into symbols list
    for gene in counts.index:
        key = gene.upper()
        if key in sym_upper:
            available_syms.append(sym_upper[key])
            matched_order.append(idx_map[key])

    # Build matrix in canonical symbol order
    ordered: dict[int, np.ndarray] = {}
    for gene in counts.index:
        key = gene.upper()
        if key in sym_upper:
            i = idx_map[key]
            ordered[i] = np.log1p(counts.loc[gene].values.astype(np.float32))

    matched_indices = sorted(ordered.keys())
    matched_syms = [symbols[i] for i in matched_indices]
    X = np.column_stack([ordered[i] for i in matched_indices])  # (n_samples, n_matched)

    return X, region_labels, sample_ids, subjects, matched_syms


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Bootstrap 95% CI on median(A) - median(B)."""
    diffs = []
    for _ in range(n):
        ia = rng.integers(0, len(arr_a), len(arr_a))
        ib = rng.integers(0, len(arr_b), len(arr_b))
        diffs.append(float(np.median(arr_a[ia]) - np.median(arr_b[ib])))
    diffs_arr = np.array(diffs)
    return float(np.percentile(diffs_arr, 2.5)), float(np.percentile(diffs_arr, 97.5))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _plot(
    scores_fcx: np.ndarray,
    scores_cereb: np.ndarray,
    subjects_fcx: list[str],
    subjects_cereb: list[str],
    per_gene_rb: list[tuple[str, float, float]],
    auc_mwu: float,
    p_mwu: float,
    p_wilcox: float,
    diff_ci: tuple[float, float],
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel A: violin + jitter of model scores by region ---
    ax = axes[0]
    parts = ax.violinplot(
        [scores_fcx, scores_cereb],
        positions=[0, 1],
        showmedians=True,
        showextrema=True,
    )
    for pc in parts["bodies"]:
        pc.set_alpha(0.5)
    parts["bodies"][0].set_facecolor("#d62728")
    parts["bodies"][1].set_facecolor("#1f77b4")
    rng = np.random.default_rng(42)
    ax.scatter(
        rng.uniform(-0.08, 0.08, len(scores_fcx)),
        scores_fcx,
        color="#d62728",
        alpha=0.6,
        s=20,
        zorder=3,
    )
    ax.scatter(
        1 + rng.uniform(-0.08, 0.08, len(scores_cereb)),
        scores_cereb,
        color="#1f77b4",
        alpha=0.6,
        s=20,
        zorder=3,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Frontal Cortex\n(FCX)", "Cerebellum"], fontsize=10)
    ax.set_ylabel("P(ALS) — model score")
    ax.set_title(
        f"GSE67196 regional ALS model score\n"
        f"MWU p={p_mwu:.4f}  Wilcoxon p={p_wilcox:.4f}\n"
        f"Δmedian(FCX−Cereb) = {np.median(scores_fcx) - np.median(scores_cereb):.4f} "
        f"[{diff_ci[0]:.4f}, {diff_ci[1]:.4f}]",
        fontsize=9,
    )
    ax.grid(alpha=0.3)

    # --- Panel B: paired scatter (26 subjects) ---
    ax = axes[1]
    sub_to_fcx = dict(zip(subjects_fcx, scores_fcx))
    paired_syms = sorted(set(subjects_fcx) & set(subjects_cereb))
    px = [sub_to_fcx[s] for s in paired_syms]
    py_map = dict(zip(subjects_cereb, scores_cereb))
    py = [py_map[s] for s in paired_syms]
    ax.scatter(px, py, color="#6a3d9a", alpha=0.7, s=30, zorder=3)
    lim = (
        min(min(px), min(py)) - 0.02,
        max(max(px), max(py)) + 0.02,
    )
    ax.plot(lim, lim, "k--", lw=0.8, label="FCX = Cereb")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("FCX score")
    ax.set_ylabel("Cerebellum score")
    ax.set_title(
        f"Paired scores per subject (n={len(paired_syms)})\n"
        "(above diagonal = FCX > Cereb)",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Panel C: per-gene FCX vs Cereb rank-biserial ---
    ax = axes[2]
    gene_labels = [r[0] for r in per_gene_rb]
    rb_vals = [r[1] for r in per_gene_rb]
    pvals = [r[2] for r in per_gene_rb]
    colours = ["#d62728" if r > 0 else "#1f77b4" for r in rb_vals]
    y_pos = np.arange(len(gene_labels))
    ax.barh(y_pos, rb_vals, color=colours, alpha=0.8)
    ax.axvline(0, color="black", lw=0.8)
    for i, (rb, p) in enumerate(zip(rb_vals, pvals)):
        if p < 0.05:
            x = rb + 0.01 if rb >= 0 else rb - 0.01
            ax.text(
                x, i, "*", va="center", fontsize=10,
                ha="left" if rb >= 0 else "right",
            )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(gene_labels, fontsize=8)
    ax.set_xlabel("Rank-biserial r (FCX vs Cereb expression)")
    ax.set_title(
        "Per-gene expression: FCX vs Cereb\n"
        "(red = higher in FCX; * = MWU p < 0.05)",
        fontsize=9,
    )
    ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    fig.savefig(_OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {_OUT_PNG.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import pandas as pd
    from lightgbm import LGBMClassifier
    from scipy.stats import mannwhitneyu, wilcoxon, false_discovery_control
    from sklearn.preprocessing import StandardScaler

    print("=" * 65)
    print("GSE67196 Regional ALS Model Scoring")
    print("=" * 65)

    # ------------------------------------------------------------------
    print("\n[1] Loading panel annotation and training data ...")
    sym_map = _load_symbol_map()
    X_train, y_train, _, symbols = _load_train(sym_map)
    print(f"  Training: {X_train.shape}  ALS={int(y_train.sum())}  Ctrl={int((y_train==0).sum())}")
    print(f"  Panel genes used: {symbols}")

    # ------------------------------------------------------------------
    print("\n[2] Loading GSE67196 ...")
    X_gse, region_labels, sample_ids, subjects, matched_syms = _load_gse67196(symbols)

    # Align column order with training
    # matched_syms may be a subset of symbols in different order
    # Re-order X_train columns to match matched_syms
    sym_to_train_col = {s: i for i, s in enumerate(symbols)}
    train_col_idx = [sym_to_train_col[s] for s in matched_syms]
    X_train_matched = X_train[:, train_col_idx]

    print(f"  GSE67196: {X_gse.shape} samples × genes")
    print(f"  Matched genes ({len(matched_syms)}): {matched_syms}")
    missing = [s for s in symbols if s not in matched_syms]
    print(f"  Missing from GSE67196 ({len(missing)}): {missing}")

    fcx_mask = np.array([r == "FCX" for r in region_labels])
    cereb_mask = ~fcx_mask
    print(
        f"  FCX={int(fcx_mask.sum())}  Cereb={int(cereb_mask.sum())}  "
        f"Paired subjects={len(set(np.array(subjects)[fcx_mask]) & set(np.array(subjects)[cereb_mask]))}"
    )

    # ------------------------------------------------------------------
    print("\n[3] Scaling and training model ...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_matched)
    X_gse_sc = scaler.transform(X_gse)

    params = json.loads(_PARAMS_PATH.read_text())
    params.pop("feature_names", None)
    params["colsample_bytree"] = 1.0  # override to use all matched genes
    params["random_state"] = RANDOM_STATE
    params["verbosity"] = -1
    clf = LGBMClassifier(**params)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        clf.fit(X_train_sc, y_train)

    # ------------------------------------------------------------------
    print("\n[4] Scoring GSE67196 samples ...")
    scores = clf.predict_proba(X_gse_sc)[:, 1]
    scores_fcx = scores[fcx_mask]
    scores_cereb = scores[cereb_mask]
    subjects_arr = np.array(subjects)
    subjects_fcx = list(subjects_arr[fcx_mask])
    subjects_cereb = list(subjects_arr[cereb_mask])

    print(f"  Median FCX score:   {np.median(scores_fcx):.4f}")
    print(f"  Median Cereb score: {np.median(scores_cereb):.4f}")

    # ------------------------------------------------------------------
    print("\n[5] Statistical comparison ...")
    # MWU (unpaired)
    stat_mwu, p_mwu = mannwhitneyu(scores_fcx, scores_cereb, alternative="two-sided")
    n1, n2 = len(scores_fcx), len(scores_cereb)
    rb_mwu = 2 * stat_mwu / (n1 * n2) - 1

    # Wilcoxon signed-rank (paired)
    paired_subjects = sorted(set(subjects_fcx) & set(subjects_cereb))
    sub_to_fcx_score = dict(zip(subjects_fcx, scores_fcx))
    sub_to_cereb_score = dict(zip(subjects_cereb, scores_cereb))
    diffs_paired = np.array(
        [sub_to_fcx_score[s] - sub_to_cereb_score[s] for s in paired_subjects]
    )
    stat_wilcox, p_wilcox = wilcoxon(diffs_paired, alternative="two-sided")

    rng = np.random.default_rng(RANDOM_STATE)
    ci_lo, ci_hi = _bootstrap_ci(scores_fcx, scores_cereb, N_BOOTSTRAP, rng)

    print(f"  MWU: U={stat_mwu:.0f}  p={p_mwu:.4f}  r_b={rb_mwu:.4f}")
    print(f"  Wilcoxon (paired n={len(paired_subjects)}): W={stat_wilcox:.0f}  p={p_wilcox:.4f}")
    print(f"  Δmedian(FCX−Cereb) = {np.median(scores_fcx)-np.median(scores_cereb):.4f}  "
          f"95%CI [{ci_lo:.4f}, {ci_hi:.4f}]")

    # ------------------------------------------------------------------
    print("\n[6] Per-gene expression comparison (FCX vs Cereb) ...")
    per_gene_rb: list[tuple[str, float, float]] = []
    for j, sym in enumerate(matched_syms):
        expr_fcx = X_gse[fcx_mask, j]
        expr_cereb = X_gse[cereb_mask, j]
        stat_g, p_g = mannwhitneyu(expr_fcx, expr_cereb, alternative="two-sided")
        rb_g = 2 * stat_g / (len(expr_fcx) * len(expr_cereb)) - 1
        per_gene_rb.append((sym, float(rb_g), float(p_g)))
        print(f"  {sym:<12}  r_b={rb_g:+.3f}  p={p_g:.4f}")

    # BH-FDR correction across 15 gene-level tests
    _pvals_array = np.array([p for _, _, p in per_gene_rb])
    _qvals_array = false_discovery_control(_pvals_array, method="bh")
    per_gene_rbq: list[tuple[str, float, float, float]] = [
        (sym, rb, p, float(q))
        for (sym, rb, p), q in zip(per_gene_rb, _qvals_array)
    ]

    # ------------------------------------------------------------------
    print("\n[7] Plotting ...")
    _plot(
        scores_fcx=scores_fcx,
        scores_cereb=scores_cereb,
        subjects_fcx=subjects_fcx,
        subjects_cereb=subjects_cereb,
        per_gene_rb=per_gene_rb,
        auc_mwu=rb_mwu,
        p_mwu=p_mwu,
        p_wilcox=p_wilcox,
        diff_ci=(ci_lo, ci_hi),
    )

    # ------------------------------------------------------------------
    print("\n[8] Writing statistics ...")
    lines: list[str] = [
        "GSE67196 Regional ALS Model Scoring",
        "=" * 60,
        "Dataset: GSE67196 (Mayo Clinic)",
        f"  Samples: FCX={int(fcx_mask.sum())}  Cereb={int(cereb_mask.sum())}  "
        f"Paired={len(paired_subjects)}",
        "  Phenotype: ALS Spectrum MND only (no non-neurological controls)",
        "",
        f"Model: LightGBM trained on GPL24676 (n=874; ALS={int(y_train.sum())}, "
        f"Ctrl={int((y_train==0).sum())})",
        f"Panel genes matched ({len(matched_syms)}): {', '.join(matched_syms)}",
        f"Missing genes ({len(missing)}): {', '.join(missing) if missing else 'none'}",
        "",
        "Model score = P(ALS=1 | expression), scored on ALS-only GSE67196 samples.",
        "Higher score → expression pattern more similar to ALS training tissue.",
        "",
        "Score summary:",
        f"  Median FCX:   {np.median(scores_fcx):.4f}  "
        f"(IQR [{np.percentile(scores_fcx,25):.4f}, {np.percentile(scores_fcx,75):.4f}])",
        f"  Median Cereb: {np.median(scores_cereb):.4f}  "
        f"(IQR [{np.percentile(scores_cereb,25):.4f}, {np.percentile(scores_cereb,75):.4f}])",
        f"  Δmedian (FCX−Cereb): "
        f"{np.median(scores_fcx)-np.median(scores_cereb):.4f}  "
        f"95%CI [{ci_lo:.4f}, {ci_hi:.4f}]",
        "",
        "Statistical tests:",
        f"  Mann-Whitney U (unpaired):          U={stat_mwu:.0f}  "
        f"p={p_mwu:.4f}  r_b={rb_mwu:.4f}",
        f"  Wilcoxon signed-rank (paired n={len(paired_subjects)}): "
        f"W={stat_wilcox:.0f}  p={p_wilcox:.4f}",
        "",
        f"Per-gene FCX vs Cereb (log1p expression; BH-FDR across {len(per_gene_rbq)} genes):",
        f"  {'Gene':<12}  {'r_b':>8}  {'p_MWU':>10}  {'BH-q':>10}  {'sig':>4}",
        "-" * 55,
    ]
    for sym, rb, p, q in sorted(per_gene_rbq, key=lambda x: -abs(x[1])):
        sig = "*" if q < 0.05 else ("†" if p < 0.05 else "")
        lines.append(f"  {sym:<12}  {rb:+8.4f}  {p:10.4f}  {q:10.4f}  {sig:>4}")
    lines.append("  (* = BH-q<0.05; † = nominal p<0.05 but BH-q≥0.05)")

    lines += [
        "",
        "Per-sample scores (FCX):",
        f"  {'Subject':<12}  {'Score':>8}",
        "-" * 25,
    ]
    for sub, sc in sorted(zip(subjects_fcx, scores_fcx), key=lambda x: -x[1]):
        lines.append(f"  {sub:<12}  {sc:8.4f}")

    lines += [
        "",
        "Per-sample scores (Cereb):",
        f"  {'Subject':<12}  {'Score':>8}",
        "-" * 25,
    ]
    for sub, sc in sorted(zip(subjects_cereb, scores_cereb), key=lambda x: -x[1]):
        lines.append(f"  {sub:<12}  {sc:8.4f}")

    lines += [
        "",
        "Paired differences (FCX score − Cereb score):",
        f"  {'Subject':<12}  {'Δ score':>10}",
        "-" * 28,
    ]
    for s in sorted(paired_subjects, key=lambda x: -(sub_to_fcx_score[x] - sub_to_cereb_score[x])):
        d = sub_to_fcx_score[s] - sub_to_cereb_score[s]
        lines.append(f"  {s:<12}  {d:+10.4f}")

    report = "\n".join(lines)
    _OUT_STATS.write_text(report)
    print(f"  Saved → {_OUT_STATS.name}")
    print("\n" + report)


if __name__ == "__main__":
    main()
