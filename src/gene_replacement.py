"""Gene replacement analysis for the core 25-gene ALS panel.

For each panel gene g, find substitute genes c from the full 47,822-gene
prefiltered space such that swapping g → c does not degrade panel AUC by
more than DELTA (non-inferiority tolerance).

Candidate sources (per panel gene g)
--------------------------------------
1. Correlation screen — non-panel prefiltered genes that satisfy BOTH:
   (a) Bonferroni-corrected Pearson correlation p < ALPHA_FWER (two-tailed
       t-test, df = n − 2 = 872; Bonferroni factor ≈ 47,797 per gene), AND
   (b) |r| ≥ R_MIN (minimum effect size; prevents statistically significant
       but biologically weak correlates from entering the candidate pool).
   At n=874, Bonferroni-only yields |r| ≥ 0.175 (3% variance explained) —
   too loose for surrogate identification. R_MIN = 0.5 requires ≥ 25%
   shared variance, consistent with co-expression network literature.

2. SHAP top-500 non-panel pool — the ~475 genes ranked in the top-500 by mean
   |SHAP| that are not in the core panel (established ALS discriminative signal).

Candidates from (1) and (2) are merged (union, deduplicated).

Replacement criterion
----------------------
  AUC( (P \\ {g}) ∪ {c} ) ≥ AUC(P) − DELTA

Optimisations
-------------
- Non-panel gene expression matrix (X_others) is pre-loaded ONCE outside the
  per-gene loop. Panel genes are already excluded from non_panel_cols by
  construction, so no per-gene re-loading is required.
- Correlation uses float32 BLAS; r cast to float64 for t/p computation.
- Candidate expression columns are pre-fetched per panel gene to avoid
  repeated mmap reads inside the parallel workers.
- joblib.Parallel with prefer="threads" over candidates; LightGBM n_jobs=1
  to prevent thread explosion.

Outputs
-------
  gene_replacement_statistics.txt
  gene_replacement_results.csv
  gene_replacement.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

ALS_DIR      = Path(__file__).parents[1]
SCRIPT_DIR   = Path(__file__).parent
sys.path.insert(0, str(ALS_DIR))

import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from utils import load_dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PLATFORM       = "GPL24676"
SEED           = 42
N_SPLITS       = 5
DELTA          = 0.002   # non-inferiority tolerance (AUC units)
ALPHA_FWER     = 0.01   # family-wise error rate; Bonferroni applied per gene
R_MIN          = 0.50   # minimum |r| effect-size filter applied after Bonferroni
                        # 0.50 → r² ≥ 0.25 (25% shared variance)
MAX_CORR_CANDS = 200    # take only the top-MAX_CORR_CANDS by |r| from genes
                        # passing both Bonferroni and R_MIN; handles genes
                        # (e.g. HEXB, SERTAD1) that belong to large
                        # co-expression modules with thousands of strong
                        # correlates — the most correlated subset is sufficient

_LGBM_PARAMS = {
    "n_estimators":      100,
    "learning_rate":     0.05,
    "num_leaves":        15,
    "min_child_samples": 5,
    "colsample_bytree":  0.8,
    "subsample":         0.8,
    "reg_alpha":         0.0005,
    "reg_lambda":        0.0001,
    "objective":         "binary",
    "verbose":           -1,
    "n_jobs":            1,    # single-threaded per model; outer loop is parallel
    "random_state":      SEED,
}

_PREFILTER_X     = SCRIPT_DIR / "lgbm_prefilter_X.npy"
_PREFILTER_NAMES = SCRIPT_DIR / "lgbm_prefilter_names.txt"
_TOP500_RANKING  = SCRIPT_DIR / "lgbm_top500_final_shap_ranking.csv"
_PANEL_CSV       = SCRIPT_DIR / "lgbm_core25_panel.csv"

PANEL_BASES = [
    "ENSG00000085276", "ENSG00000261599", "ENSG00000183604",
    "ENSG00000197019", "ENSG00000213860", "ENSG00000142748",
    "ENSG00000184500", "ENSG00000091879", "ENSG00000236682",
    "ENSG00000134531", "ENSG00000238622", "ENSG00000142910",
    "ENSG00000131730", "ENSG00000280893", "ENSG00000110799",
    "ENSG00000184113", "ENSG00000123358", "ENSG00000223718",
    "ENSG00000120669", "ENSG00000203616", "ENSG00000049860",
    "ENSG00000279656", "ENSG00000226308", "ENSG00000124370",
    "ENSG00000134955",
]
PANEL_SET = set(PANEL_BASES)


# ---------------------------------------------------------------------------
# CV helper
# ---------------------------------------------------------------------------

def _cv_auc(X: np.ndarray, y: np.ndarray) -> float:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    aucs = [
        roc_auc_score(
            y[te],
            lgb.LGBMClassifier(**_LGBM_PARAMS)
              .fit(X[tr], y[tr])
              .predict_proba(X[te])[:, 1],
        )
        for tr, te in skf.split(X, y)
    ]
    return float(np.mean(aucs))


# ---------------------------------------------------------------------------
# Replacement worker (called inside Parallel)
# ---------------------------------------------------------------------------

def _test_one(
    X_panel_minus: np.ndarray,  # (874, 24) pre-panel expression
    c_expr: np.ndarray,         # (874,) candidate expression
    y: np.ndarray,
    baseline_auc: float,
) -> tuple[float, bool]:
    X_test = np.column_stack([X_panel_minus, c_expr]).astype(np.float32)
    auc = _cv_auc(X_test, y)
    return auc, auc >= baseline_auc - DELTA


# ---------------------------------------------------------------------------
# Bonferroni correlation screen
# ---------------------------------------------------------------------------

def _corr_candidates(
    g_expr_f32: np.ndarray,   # (874,) float32 — panel gene expression
    X_others_f32: np.ndarray, # (874, n_non_panel) float32 — pre-loaded once
    n_tests: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised Bonferroni-corrected Pearson correlation screen.

    Returns (local_indices, r_values, p_values_raw) for significant genes
    sorted descending by |r|, capped at MAX_CORR_CANDS.
    local_indices are indices into the X_others_f32 column axis.
    """
    n = len(g_expr_f32)

    # Compute correlations in float32 (BLAS-accelerated)
    g_c = (g_expr_f32 - g_expr_f32.mean()).astype(np.float32)
    X_c = (X_others_f32 - X_others_f32.mean(axis=0)).astype(np.float32)
    r_num = g_c @ X_c                                      # (n_non_panel,)
    ss_g  = float(g_c @ g_c)
    ss_X  = (X_c * X_c).sum(axis=0)                       # (n_non_panel,)
    denom = np.sqrt(ss_g * np.maximum(ss_X, 1e-12))
    r_f32 = np.where(denom > 0, r_num / denom, 0.0)

    # Cast to float64 for t-statistic / p-value stability
    r = r_f32.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = r * np.sqrt(n - 2) / np.sqrt(np.maximum(1.0 - r ** 2, 1e-12))
    p_raw = 2.0 * stats.t.sf(np.abs(t), df=n - 2)

    alpha_bonf = ALPHA_FWER / n_tests
    sig_mask   = p_raw < alpha_bonf

    if not sig_mask.any():
        empty = np.array([], dtype=np.intp)
        return empty, np.array([]), np.array([])

    # Apply effect-size filter: require |r| >= R_MIN on top of Bonferroni
    sig_idx = np.where(sig_mask & (np.abs(r) >= R_MIN))[0]

    if sig_idx.size == 0:
        empty = np.array([], dtype=np.intp)
        return empty, np.array([]), np.array([])

    # Sort descending by |r|; cap at MAX_CORR_CANDS (focuses on strongest
    # surrogates; prevents runaway compute for large co-expression modules)
    order   = np.argsort(-np.abs(r[sig_idx]))
    sig_idx = sig_idx[order[:MAX_CORR_CANDS]]
    return sig_idx, r[sig_idx], p_raw[sig_idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    n_cpus = os.cpu_count() or 1
    print("=" * 72)
    print("Gene Replacement Analysis — Core 25-Gene ALS Panel")
    print(f"CPUs: {n_cpus}  |  DELTA={DELTA}  |  ALPHA_FWER={ALPHA_FWER}"
          f"  |  R_MIN={R_MIN}")
    print("=" * 72)

    # --- prefiltered matrix ------------------------------------------------
    print("\nLoading prefiltered feature names ...")
    pre_names = _PREFILTER_NAMES.read_text().splitlines()
    pre_bases = [n.split(".")[0] for n in pre_names]
    pre_base_to_col: dict[str, int] = {b: i for i, b in enumerate(pre_bases)}

    print("Memory-mapping prefiltered expression matrix ...")
    X_pre = np.load(_PREFILTER_X, mmap_mode="r")   # (874, 47822) float32
    n_samples = X_pre.shape[0]
    print(f"  Shape: {X_pre.shape}")

    # --- labels ------------------------------------------------------------
    (ds,) = load_dataset("GSE153960", platform=PLATFORM,
                         resources_dir=ALS_DIR / "resources")
    y = ds.y.values.astype(int)

    # --- symbol lookup from panel CSV -------------------------------------
    ensg_to_sym: dict[str, str] = {}
    try:
        pdf = pd.read_csv(_PANEL_CSV)
        feat_col = next(c for c in pdf.columns
                        if any(k in c.lower() for k in ("feat", "ensg", "gene")))
        sym_col  = next(c for c in pdf.columns if "sym" in c.lower())
        for _, row in pdf.iterrows():
            ensg_to_sym[str(row[feat_col]).split(".")[0]] = str(row[sym_col])
    except Exception:
        pass

    # --- panel column indices ----------------------------------------------
    panel_pre_cols: list[int] = []
    for b in PANEL_BASES:
        if b not in pre_base_to_col:
            raise RuntimeError(f"Panel gene {b} missing from prefiltered matrix")
        panel_pre_cols.append(pre_base_to_col[b])

    # --- baseline AUC ------------------------------------------------------
    print("\nComputing baseline AUC (full 25-gene panel) ...")
    X_panel = X_pre[:, panel_pre_cols].astype(np.float32)
    baseline_auc = _cv_auc(X_panel, y)
    threshold    = baseline_auc - DELTA
    print(f"  Baseline AUC  = {baseline_auc:.4f}")
    print(f"  Threshold AUC = {threshold:.4f}  (baseline − {DELTA})")
    del X_panel

    # --- SHAP non-panel pool -----------------------------------------------
    shap_df = pd.read_csv(_TOP500_RANKING)
    top500_bases = [f.split(".")[0] for f in shap_df["feature"].tolist()]
    shap_non_panel_cols: list[int] = [
        pre_base_to_col[b]
        for b in top500_bases
        if b not in PANEL_SET and b in pre_base_to_col
    ]
    print(f"\n  SHAP non-panel pool: {len(shap_non_panel_cols)} genes")

    # --- non-panel column indices (computed once) --------------------------
    # Panel genes are excluded by construction → X_others never needs updating
    non_panel_pre_cols: list[int] = [
        i for i, b in enumerate(pre_bases) if b not in PANEL_SET
    ]
    n_tests = len(non_panel_pre_cols)   # Bonferroni denominator

    # Bonferroni threshold (informational)
    alpha_bonf = ALPHA_FWER / n_tests
    t_crit = stats.t.ppf(1.0 - alpha_bonf / 2.0, df=n_samples - 2)
    r_crit = t_crit / np.sqrt(t_crit ** 2 + (n_samples - 2))
    print(f"  Non-panel genes: {n_tests}")
    print(f"  Bonferroni α/test: {alpha_bonf:.3e}  →  |r| ≥ {r_crit:.4f}")

    # --- PRE-LOAD X_others ONCE (key optimisation) ------------------------
    # Panel genes are already excluded from non_panel_pre_cols, so this array
    # is valid for all 25 panel-gene iterations without modification.
    print(f"\nPre-loading non-panel expression matrix ({n_tests} genes) ...")
    X_others = X_pre[:, non_panel_pre_cols].astype(np.float32)  # (874, ~47797)
    # Build local-index → global prefiltered-column mapping
    local_to_pre_col = np.array(non_panel_pre_cols, dtype=np.intp)
    print(f"  Loaded: {X_others.shape}  "
          f"({X_others.nbytes / 1e6:.0f} MB in RAM)")

    # -----------------------------------------------------------------------
    # Per-panel-gene loop
    # -----------------------------------------------------------------------
    all_rows:     list[dict] = []
    summary_rows: list[dict] = []

    for g_idx, (g_base, g_pre_col) in enumerate(zip(PANEL_BASES, panel_pre_cols)):
        symbol = ensg_to_sym.get(g_base, g_base)
        print(f"\n[{g_idx + 1:2d}/25] {symbol} ({g_base})")

        panel_minus_cols = [c for c in panel_pre_cols if c != g_pre_col]
        X_panel_minus    = X_pre[:, panel_minus_cols].astype(np.float32)  # (874,24)

        # --- Stage 1: Bonferroni correlation screen ---
        g_expr_f32 = X_pre[:, g_pre_col].astype(np.float32)

        local_sig_idx, r_vals, p_vals_raw = _corr_candidates(
            g_expr_f32, X_others, n_tests,
        )

        # Map local indices → global prefiltered column indices
        corr_pre_cols = [int(local_to_pre_col[li]) for li in local_sig_idx]
        corr_r_map    = {int(local_to_pre_col[li]): float(rv)
                         for li, rv in zip(local_sig_idx, r_vals)}
        corr_p_map    = {int(local_to_pre_col[li]): float(pv)
                         for li, pv in zip(local_sig_idx, p_vals_raw)}

        n_sig = len(corr_pre_cols)
        capped = n_sig == MAX_CORR_CANDS
        print(f"    Bonferroni+|r|≥{R_MIN} correlates "
              f"(α={ALPHA_FWER}/n={n_tests}): {n_sig}"
              + (" [capped]" if capped else ""))

        # --- Stage 2: SHAP non-panel pool ---
        shap_cands = [c for c in shap_non_panel_cols if c != g_pre_col]

        # --- Union (deduplicate, track source) ---
        candidate_meta: dict[int, dict] = {}
        for pc in corr_pre_cols:
            candidate_meta[pc] = {
                "source":    "correlation",
                "pearson_r": corr_r_map[pc],
                "p_raw":     corr_p_map[pc],
            }
        for pc in shap_cands:
            if pc in candidate_meta:
                candidate_meta[pc]["source"] = "both"
            else:
                candidate_meta[pc] = {
                    "source":    "shap",
                    "pearson_r": None,
                    "p_raw":     None,
                }

        candidate_cols = list(candidate_meta.keys())
        n_cands = len(candidate_cols)
        print(f"    Total unique candidates: {n_cands}")

        if not candidate_cols:
            print("    No candidates — skipping.")
            summary_rows.append({
                "symbol": symbol, "ensg": g_base,
                "n_corr_cands": 0, "n_shap_cands": 0,
                "n_candidates": 0, "n_replacements": 0,
                "best_ensg": None, "best_auc": None,
                "replaceability_frac": 0.0,
            })
            continue

        # Pre-fetch candidate expression columns (avoids repeated mmap reads
        # inside parallel workers)
        cand_exprs = X_pre[:, candidate_cols].astype(np.float32)  # (874, n_cands)

        # --- Parallel replacement test ---
        raw_results: list[tuple[float, bool]] = Parallel(
            n_jobs=-1, prefer="threads",
        )(
            delayed(_test_one)(
                X_panel_minus,
                cand_exprs[:, ci],
                y,
                baseline_auc,
            )
            for ci in range(n_cands)
        )

        n_valid = sum(1 for _, v in raw_results if v)
        print(f"    Valid replacements (AUC ≥ {threshold:.4f}): {n_valid}")

        # Collect rows
        for c_col, (auc, valid) in zip(candidate_cols, raw_results):
            meta = candidate_meta[c_col]
            all_rows.append({
                "panel_symbol":      symbol,
                "panel_ensg":        g_base,
                "candidate_ensg":    pre_bases[c_col],
                "source":            meta["source"],
                "pearson_r":         meta["pearson_r"],
                "pearson_p_raw":     meta["p_raw"],
                "replacement_auc":   round(auc, 6),
                "delta_vs_baseline": round(auc - baseline_auc, 6),
                "is_replacement":    valid,
            })

        valid_pairs = [
            (auc, c_col)
            for (auc, v), c_col in zip(raw_results, candidate_cols)
            if v
        ]
        best_auc  = max(a for a, _ in valid_pairs) if valid_pairs else None
        best_ensg = None
        if valid_pairs:
            best_col  = max(valid_pairs, key=lambda x: x[0])[1]
            best_ensg = pre_bases[best_col]

        n_corr_src = sum(
            1 for pc in candidate_cols
            if candidate_meta[pc]["source"] in ("correlation", "both")
        )
        n_shap_src = sum(
            1 for pc in candidate_cols
            if candidate_meta[pc]["source"] in ("shap", "both")
        )
        summary_rows.append({
            "symbol":              symbol,
            "ensg":                g_base,
            "n_corr_cands":        n_corr_src,
            "n_shap_cands":        n_shap_src,
            "n_candidates":        n_cands,
            "n_replacements":      n_valid,
            "best_ensg":           best_ensg,
            "best_auc":            round(best_auc, 4) if best_auc else None,
            "replaceability_frac": n_valid / n_cands,
        })

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    results_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summary_rows)

    out_csv = SCRIPT_DIR / "gene_replacement_results.csv"
    results_df.to_csv(out_csv, index=False)

    # Text statistics
    alpha_bonf = ALPHA_FWER / n_tests
    t_crit     = stats.t.ppf(1.0 - alpha_bonf / 2.0, df=n_samples - 2)
    r_crit     = t_crit / np.sqrt(t_crit ** 2 + (n_samples - 2))

    lines = [
        "Gene Replacement Analysis — Core 25-Gene ALS Panel",
        "=" * 72,
        f"Baseline panel AUC (5-fold CV):    {baseline_auc:.4f}",
        f"Non-inferiority tolerance DELTA:   {DELTA}",
        f"Replacement threshold AUC:         {threshold:.4f}",
        f"Correlation FWER:                  α = {ALPHA_FWER} (Bonferroni)",
        f"Bonferroni α per test:             {alpha_bonf:.3e}",
        f"|r| Bonferroni threshold:          {r_crit:.4f}",
        f"|r| effect-size threshold (R_MIN): {R_MIN}  [r² ≥ {R_MIN**2:.2f}]",
        f"MAX_CORR_CANDS cap:               {MAX_CORR_CANDS}",
        f"SHAP non-panel pool:               {len(shap_non_panel_cols)} genes",
        "",
        f"{'Gene':<16} {'n_corr':>7} {'n_shap':>7} {'n_tot':>7} "
        f"{'n_repl':>7} {'repl%':>7} {'best_replacement':>22} {'best_AUC':>10}",
        "-" * 88,
    ]
    for r in summary_rows:
        best_str = (r["best_ensg"] or "none")[:22]
        best_auc_str = f"{r['best_auc']:.4f}" if r["best_auc"] else "N/A"
        lines.append(
            f"{r['symbol']:<16} {r['n_corr_cands']:>7} {r['n_shap_cands']:>7} "
            f"{r['n_candidates']:>7} {r['n_replacements']:>7} "
            f"{r['replaceability_frac']:>6.1%} {best_str:>22} {best_auc_str:>10}"
        )

    n_replaceable   = sum(1 for r in summary_rows if r["n_replacements"] > 0)
    n_irreplaceable = 25 - n_replaceable
    lines += [
        "",
        f"Panel genes with ≥1 valid replacement: {n_replaceable}/25",
        f"Panel genes with 0 valid replacements:  {n_irreplaceable}/25"
        f"  [uniquely irreplaceable within candidate pool]",
    ]

    out_txt = SCRIPT_DIR / "gene_replacement_statistics.txt"
    out_txt.write_text("\n".join(lines))

    print(f"\n  Results CSV → {out_csv.name}  ({len(results_df)} rows)")
    print(f"  Statistics  → {out_txt.name}")
    print(f"\n  Replaceable:         {n_replaceable}/25")
    print(f"  Uniquely irreplaceable: {n_irreplaceable}/25")

    # Plot
    if not summary_df.empty:
        sdf = summary_df.sort_values(
            "n_replacements", ascending=False
        ).reset_index(drop=True)
        colors = [
            "#C62828" if n == 0 else "#1565C0"
            for n in sdf["n_replacements"]
        ]

        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        ax = axes[0]
        ax.barh(sdf["symbol"], sdf["n_replacements"],
                color=colors, edgecolor="white", height=0.7)
        ax.set_xlabel("Number of valid replacements", fontsize=11)
        ax.set_title(
            "Valid substitute genes per panel gene\n(red = uniquely irreplaceable)",
            fontsize=10,
        )
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.8)
        ax.grid(True, axis="x", alpha=0.3)

        ax = axes[1]
        ax.barh(sdf["symbol"], sdf["replaceability_frac"],
                color=colors, edgecolor="white", height=0.7)
        ax.set_xlabel(
            "Replaceability fraction (valid replacements / candidates tested)",
            fontsize=11,
        )
        ax.set_title("Replaceability score per panel gene", fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.axvline(0, color="k", lw=0.8)
        ax.grid(True, axis="x", alpha=0.3)

        plt.suptitle(
            "Gene Replacement Analysis — Core 25-Gene ALS Panel  (n=874, 5-fold CV)\n"
            f"Baseline AUC={baseline_auc:.4f}  |  Threshold={threshold:.4f}  "
            f"|  Bonferroni |r|≥{r_crit:.3f}  |  R_MIN={R_MIN}  |  DELTA={DELTA}",
            fontsize=10,
        )
        plt.tight_layout()
        out_png = SCRIPT_DIR / "gene_replacement.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot → {out_png.name}")

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
