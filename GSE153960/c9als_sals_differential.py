"""ALS subtype differential expression — ALS-FTD vs pure ALS (Step 15).

C9orf72 genetic status is not included in the public GSE153960 GEO metadata
(restricted to authorised NYGC database users).  The best available proxy is
the "ALS Spectrum MND, Other Neurological Disorders" label, which denotes
ALS-FTD: ~50 % of familial ALS-FTD cases carry C9orf72 repeat expansions,
making ALS-FTD the most C9orf72-enriched stratum in this dataset.

Four analyses:

1. Panel-gene subtype comparison (25 genes)
   ─────────────────────────────────────────
   Mann-Whitney U + rank-biserial r for ALS-FTD vs pure ALS, all 25 panel
   genes, across all tissues.  Classify each gene as:
     • FTD-enriched   — significantly different between subtypes
     • Pan-ALS        — no subtype difference
     • Mixed-signal   — trending

2. Subtype consistency (ALS-vs-Control in each subtype)
   ──────────────────────────────────────────────────────
   For each protein-coding gene compute rb(ALS vs ctrl) and rb(ALS-FTD vs
   ctrl).  Genes where the two effects are concordant are pan-ALS signals;
   genes with divergent effects encode subtype-specific biology.

3. Within-ALS subtype classification (25-gene panel)
   ─────────────────────────────────────────────────
   5-fold stratified CV within all ALS samples (pure ALS + ALS-FTD) using
   25-gene panel as features.  AUC > 0.7 would indicate embedded FTD biology
   in the panel; near 0.5 confirms subtype-agnostic utility.

4. Full 3,704-gene ALS-FTD vs pure ALS screen
   ────────────────────────────────────────────
   Rank-biserial r and Mann-Whitney U for all 3,704 SFM-selected features.
   BH-FDR correction.  Identifies novel subtype-specific genes outside the
   discriminative panel.

Outputs
-------
  c9als_sals_differential.png
  c9als_sals_differential_statistics.txt
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
from scipy.stats import mannwhitneyu

ALS_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ALS_DIR))

from utils import _load_suppl_expression, _parse_series_matrix  # noqa: E402

SCRIPT_DIR = Path(__file__).parent
RESOURCES_DIR = ALS_DIR / "resources"

_MATRIX_PATH = (
    RESOURCES_DIR / "GSE153960" / "matrix"
    / "GSE153960-GPL24676_series_matrix.txt.gz"
)
_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"
_SHAP_CSV = SCRIPT_DIR / "lgbm_iter_final_shap_ranking.csv"

RANDOM_STATE = 42
N_FOLDS = 5
FDR_THRESH = 0.05
N_BOOTSTRAP = 2_000

# 15 protein-coding panel genes
_CODING_GENES: list[dict[str, str]] = [
    {"symbol": "MECOM",   "ensg": "ENSG00000085276", "direction": "down_in_ALS"},
    {"symbol": "SERTAD1", "ensg": "ENSG00000197019", "direction": "up_in_ALS"},
    {"symbol": "FCN3",    "ensg": "ENSG00000142748", "direction": "up_in_ALS"},
    {"symbol": "PROS1",   "ensg": "ENSG00000184500", "direction": "up_in_ALS"},
    {"symbol": "ANGPT2",  "ensg": "ENSG00000091879", "direction": "down_in_ALS"},
    {"symbol": "EMP1",    "ensg": "ENSG00000134531", "direction": "up_in_ALS"},
    {"symbol": "TINAGL1", "ensg": "ENSG00000142910", "direction": "down_in_ALS"},
    {"symbol": "CKMT2",   "ensg": "ENSG00000131730", "direction": "down_in_ALS"},
    {"symbol": "VWF",     "ensg": "ENSG00000110799", "direction": "down_in_ALS"},
    {"symbol": "CLDN5",   "ensg": "ENSG00000184113", "direction": "down_in_ALS"},
    {"symbol": "NR4A1",   "ensg": "ENSG00000123358", "direction": "down_in_ALS"},
    {"symbol": "SOHLH2",  "ensg": "ENSG00000120669", "direction": "up_in_ALS"},
    {"symbol": "HEXB",    "ensg": "ENSG00000049860", "direction": "up_in_ALS"},
    {"symbol": "MCEE",    "ensg": "ENSG00000124370", "direction": "up_in_ALS"},
    {"symbol": "SLC37A2", "ensg": "ENSG00000134955", "direction": "up_in_ALS"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-biserial r (effect size for MWU). Positive = x > y."""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    u, _ = mannwhitneyu(x, y, alternative="two-sided")
    return float(2 * u / (n1 * n2) - 1)


def _bh_correct(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = np.empty(n, dtype=float)
    ranked[order] = np.arange(1, n + 1)
    q = np.minimum(1.0, pvals * n / ranked)
    # Enforce monotonicity (cummin from the right)
    for i in range(n - 2, -1, -1):
        q[order[i]] = min(q[order[i]], q[order[i + 1]])
    return q


def _match_feature(ensg_base: str, feature_names: list[str]) -> str | None:
    for fn in feature_names:
        if fn.split(".")[0] == ensg_base:
            return fn
    return None


def _bootstrap_auc(
    y: np.ndarray, scores: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    from sklearn.metrics import roc_auc_score

    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], scores[idx]))
    return np.array(aucs)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _plot(
    panel_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    auc_cv: float,
    ci_lo: float,
    ci_hi: float,
    cv_scores: np.ndarray | None = None,
    y_als: np.ndarray | None = None,
) -> None:
    from sklearn.metrics import roc_curve

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Panel 1: ALS-FTD vs pure ALS rank-biserial bar chart (panel genes) ---
    sub = panel_df.sort_values("rb_ftd_vs_als")
    colours = ["#d62728" if rb > 0 else "#1f77b4" for rb in sub["rb_ftd_vs_als"]]
    axes[0].barh(sub["symbol"], sub["rb_ftd_vs_als"], color=colours, alpha=0.8)
    axes[0].axvline(0, color="black", lw=0.8)
    for i, (_, row) in enumerate(sub.iterrows()):
        if row["bh_q"] < FDR_THRESH:
            x = row["rb_ftd_vs_als"] + 0.01 if row["rb_ftd_vs_als"] >= 0 else row["rb_ftd_vs_als"] - 0.01
            axes[0].text(x, i, "*", va="center", fontsize=10,
                         ha="left" if row["rb_ftd_vs_als"] >= 0 else "right")
    axes[0].set_xlabel("Rank-biserial r (ALS-FTD vs pure ALS)\nred = higher in ALS-FTD")
    axes[0].set_title(
        f"Panel genes: ALS-FTD vs pure ALS\n"
        f"(* = BH-FDR < {FDR_THRESH}; "
        f"n_ALS-FTD={panel_df['n_ftd'].iloc[0]}, n_ALS={panel_df['n_als'].iloc[0]})",
        fontsize=9,
    )
    axes[0].grid(alpha=0.3, axis="x")

    # --- Panel 2: Subtype consistency scatter (coding genes only) ---
    cd = consistency_df.dropna(subset=["rb_als_ctrl", "rb_ftd_ctrl"])
    colours_sc = []
    for _, row in cd.iterrows():
        ftd_v = row["rb_ftd_ctrl"]
        als_v = row["rb_als_ctrl"]
        if abs(ftd_v - als_v) < 0.1:
            colours_sc.append("#2ca02c")  # pan-ALS
        elif ftd_v * als_v < 0:
            colours_sc.append("#9467bd")  # direction reversal
        else:
            colours_sc.append("#ff7f0e")  # subtype-divergent
    axes[1].scatter(cd["rb_als_ctrl"], cd["rb_ftd_ctrl"], c=colours_sc, s=60, alpha=0.85, zorder=3)
    lim = max(abs(cd[["rb_als_ctrl", "rb_ftd_ctrl"]].values.ravel())) * 1.1
    axes[1].plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="Perfect concordance")
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].axvline(0, color="gray", lw=0.5)
    for _, row in cd.iterrows():
        axes[1].annotate(
            row["symbol"], (row["rb_als_ctrl"], row["rb_ftd_ctrl"]),
            fontsize=7, ha="center", va="bottom",
        )
    axes[1].set_xlabel("rb (pure ALS vs Control)")
    axes[1].set_ylabel("rb (ALS-FTD vs Control)")
    axes[1].set_title(
        "Subtype consistency: ALS vs Control in each subtype\n"
        "Diagonal = pan-ALS signal; off-diagonal = subtype-specific",
        fontsize=9,
    )
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # --- Panel 3: ROC curve for within-ALS subtype classification ---
    if cv_scores is not None and y_als is not None:
        fpr, tpr, _ = roc_curve(y_als, cv_scores)
        axes[2].plot(fpr, tpr, lw=2, color="#9467bd",
                     label=f"5-fold CV AUC = {auc_cv:.4f}\n95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    axes[2].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate")
    axes[2].set_title(
        "Within-ALS subtype classification\n"
        "5-fold CV: ALS-FTD vs pure ALS (25-gene panel)",
        fontsize=9,
    )
    axes[2].legend(fontsize=8, loc="lower right")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out = SCRIPT_DIR / "c9als_sals_differential.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import json
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    print("=" * 65)
    print("ALS subtype DE: ALS-FTD vs pure ALS (Step 15)")
    print("C9orf72 proxy — public GEO metadata has no genetic labels")
    print("=" * 65)

    # -----------------------------------------------------------------------
    # Load expression + metadata
    # -----------------------------------------------------------------------
    print("\n[1] Loading expression data ...")
    meta_df, _ = _parse_series_matrix(_MATRIX_PATH)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        expr_df = _load_suppl_expression("GSE153960", meta_df, RESOURCES_DIR)

    common = meta_df.index.intersection(expr_df.columns)
    meta_df = meta_df.loc[common]
    expr_df = expr_df[common]
    feature_names: list[str] = list(expr_df.index)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        X_log = np.log1p(expr_df.values.astype(np.float32))  # (genes, samples)

    grp = meta_df["group"].fillna("").astype(str)
    als_mask = grp.str.match(r"^ALS Spectrum MND$", na=False)
    als_ftd_mask = grp.str.match(r"^ALS Spectrum MND, Other Neurological Dis", na=False)
    ctrl_mask = grp.str.match(r"^Non-Neurological Control", na=False)

    n_als = int(als_mask.sum())
    n_ftd = int(als_ftd_mask.sum())
    n_ctrl = int(ctrl_mask.sum())
    print(f"  Pure ALS: {n_als} | ALS-FTD: {n_ftd} | Control: {n_ctrl}")

    # -----------------------------------------------------------------------
    # Analysis 1: Panel-gene ALS-FTD vs pure ALS comparison
    # -----------------------------------------------------------------------
    print("\n[2] Panel-gene ALS-FTD vs pure ALS comparison ...")
    panel = pd.read_csv(_PANEL_CSV)
    panel_records: list[dict] = []

    for _, row in panel.iterrows():
        sym = row["symbol"]
        feat = row["feature"]
        ensg_base = feat.split(".")[0]
        fname = _match_feature(ensg_base, feature_names)
        if fname is None:
            print(f"  [WARN] {sym} not found in expression matrix")
            continue
        idx = feature_names.index(fname)
        expr_vec = X_log[idx, :]

        als_expr = expr_vec[als_mask.values]
        ftd_expr = expr_vec[als_ftd_mask.values]
        ctrl_expr = expr_vec[ctrl_mask.values]

        rb_ftd_als = _rank_biserial(ftd_expr, als_expr)  # + = higher in ALS-FTD
        rb_als_ctrl = _rank_biserial(als_expr, ctrl_expr)  # + = higher in ALS
        rb_ftd_ctrl = _rank_biserial(ftd_expr, ctrl_expr)  # + = higher in ALS-FTD

        _, p_ftd_als = mannwhitneyu(ftd_expr, als_expr, alternative="two-sided")

        panel_records.append({
            "symbol": sym,
            "feat": feat,
            "rb_ftd_vs_als": rb_ftd_als,
            "p_ftd_vs_als": p_ftd_als,
            "rb_als_ctrl": rb_als_ctrl,
            "rb_ftd_ctrl": rb_ftd_ctrl,
            "n_als": n_als,
            "n_ftd": n_ftd,
            "n_ctrl": n_ctrl,
        })

    panel_df = pd.DataFrame(panel_records)
    panel_df["bh_q"] = _bh_correct(panel_df["p_ftd_vs_als"].values)
    panel_df = panel_df.sort_values("p_ftd_vs_als")

    print(f"\n  {'Symbol':<12} {'rb_FTD_ALS':>10} {'p':>10} {'q_BH':>10} {'rb_ALS_ctrl':>12} {'rb_FTD_ctrl':>12}")
    for _, row in panel_df.iterrows():
        sig = "*" if row["bh_q"] < FDR_THRESH else (" ~" if row["p_ftd_vs_als"] < 0.1 else "")
        print(
            f"  {row['symbol']:<12} {row['rb_ftd_vs_als']:>+10.3f} "
            f"{row['p_ftd_vs_als']:>10.3e} {row['bh_q']:>10.3e} "
            f"{row['rb_als_ctrl']:>+12.3f} {row['rb_ftd_ctrl']:>+12.3f}  {sig}"
        )

    # Subtype consistency table (coding genes only)
    coding_syms = {g["symbol"] for g in _CODING_GENES}
    consistency_df = panel_df[panel_df["symbol"].isin(coding_syms)].copy()

    # -----------------------------------------------------------------------
    # Analysis 3: Within-ALS subtype classification (25-gene panel)
    # -----------------------------------------------------------------------
    print("\n[3] Within-ALS subtype classification (5-fold CV) ...")

    als_any_mask = als_mask | als_ftd_mask
    # Build feature matrix using all 25 panel genes
    panel_cols: list[int] = []
    for _, row in panel.iterrows():
        feat = row["feature"]
        ensg_base = feat.split(".")[0]
        fname = _match_feature(ensg_base, feature_names)
        if fname is not None:
            panel_cols.append(feature_names.index(fname))

    X_panel = X_log[panel_cols, :].T  # (n_als_all, n_genes)
    X_als = X_panel[als_any_mask.values]
    y_als = als_ftd_mask[als_any_mask].astype(int).values  # 1=ALS-FTD, 0=pure ALS

    print(f"  Samples: ALS-FTD={y_als.sum()}, pure ALS={(y_als==0).sum()}")

    params = json.loads(_PARAMS_PATH.read_text())
    params["colsample_bytree"] = 1.0
    params["verbose"] = -1
    cv_params = {**params, "min_child_samples": max(1, params.get("min_child_samples", 20) // 4)}

    scaler = StandardScaler()
    X_als_sc = scaler.fit_transform(X_als).astype(np.float32)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = np.zeros(len(y_als))
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_als_sc, y_als)):
        clf = LGBMClassifier(**cv_params)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            clf.fit(X_als_sc[tr_idx], y_als[tr_idx])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            cv_scores[te_idx] = clf.predict_proba(X_als_sc[te_idx])[:, 1]
        fold_auc = float(roc_auc_score(y_als[te_idx], cv_scores[te_idx]))
        print(f"  Fold {fold+1} AUC: {fold_auc:.4f}")

    auc_cv = float(roc_auc_score(y_als, cv_scores))
    rng = np.random.default_rng(RANDOM_STATE)
    boot = _bootstrap_auc(y_als, cv_scores, N_BOOTSTRAP, rng)
    ci_lo = float(np.percentile(boot, 2.5))
    ci_hi = float(np.percentile(boot, 97.5))
    print(f"  Overall CV AUC: {auc_cv:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")

    # -----------------------------------------------------------------------
    # Analysis 4: Full 3,704-gene ALS-FTD vs pure ALS screen
    # -----------------------------------------------------------------------
    print("\n[4] Full 3,704-gene ALS-FTD vs pure ALS screen ...")
    shap_df = pd.read_csv(_SHAP_CSV)
    sfm_features: list[str] = shap_df["feature"].tolist()

    sfm_records: list[dict] = []
    n_genes = len(sfm_features)
    als_idx = np.where(als_mask.values)[0]
    ftd_idx = np.where(als_ftd_mask.values)[0]

    for i, feat in enumerate(sfm_features):
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n_genes} ...")
        ensg_base = feat.split(".")[0]
        fname = _match_feature(ensg_base, feature_names)
        if fname is None:
            continue
        idx = feature_names.index(fname)
        als_vals = X_log[idx, als_idx]
        ftd_vals = X_log[idx, ftd_idx]
        rb = _rank_biserial(ftd_vals, als_vals)
        _, p = mannwhitneyu(ftd_vals, als_vals, alternative="two-sided")
        sfm_records.append({
            "shap_rank": shap_df.iloc[i]["rank"],
            "feature": feat,
            "mean_abs_shap": shap_df.iloc[i]["mean_abs_shap"],
            "rb_ftd_vs_als": rb,
            "p": p,
        })

    sfm_df = pd.DataFrame(sfm_records)
    sfm_df["bh_q"] = _bh_correct(sfm_df["p"].values)
    sfm_df = sfm_df.sort_values("p")

    n_sig = int((sfm_df["bh_q"] < FDR_THRESH).sum())
    print(f"  Significant at BH-FDR < {FDR_THRESH}: {n_sig} / {len(sfm_df)}")

    # Map top genes to symbols via MyGeneInfo
    import time
    import mygene
    top_ensg = [f.split(".")[0] for f in sfm_df.head(50)["feature"].tolist()]
    mg = mygene.MyGeneInfo()
    sym_map: dict[str, str] = {}
    for attempt in range(3):
        try:
            hits = mg.querymany(top_ensg, scopes="ensembl.gene", fields="symbol",
                                species="human", verbose=False)
            for h in hits:
                if "symbol" in h:
                    sym_map[h["query"]] = h["symbol"]
            break
        except Exception as exc:
            print(f"  MyGeneInfo attempt {attempt+1} failed: {exc}"); time.sleep(5)

    sfm_df["symbol"] = sfm_df["feature"].apply(lambda f: sym_map.get(f.split(".")[0], f.split(".")[0]))

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    print("\n[5] Generating plot ...")
    _plot(panel_df, consistency_df, auc_cv, ci_lo, ci_hi, cv_scores, y_als)

    # -----------------------------------------------------------------------
    # Statistics file
    # -----------------------------------------------------------------------
    sig_panel = panel_df[panel_df["bh_q"] < FDR_THRESH]
    trend_panel = panel_df[(panel_df["p_ftd_vs_als"] < 0.1) & (panel_df["bh_q"] >= FDR_THRESH)]
    sig_sfm = sfm_df[sfm_df["bh_q"] < FDR_THRESH]

    lines = [
        "ALS Subtype Differential Expression — ALS-FTD vs Pure ALS (Step 15)",
        "=" * 70,
        "NOTE: C9orf72 genetic status not in public GEO metadata.",
        "ALS-FTD label used as C9orf72-enriched proxy (~50% of fALS-FTD carry C9 repeat).",
        "",
        f"Dataset: GSE153960 GPL24676 (n={n_als+n_ftd+n_ctrl})",
        f"  Pure ALS (sALS):  n={n_als}",
        f"  ALS-FTD:          n={n_ftd}  (C9-enriched proxy)",
        f"  Control:          n={n_ctrl}",
        "",
        "─" * 70,
        "ANALYSIS 1: Panel-gene ALS-FTD vs pure ALS (positive rb = higher in ALS-FTD)",
        "─" * 70,
        f"{'Symbol':<12} {'rb_FTD_ALS':>10} {'p':>10} {'BH-q':>10} {'rb_ALS_ctrl':>12} {'rb_FTD_ctrl':>12}",
        "-" * 68,
    ]
    for _, row in panel_df.iterrows():
        sig = " *" if row["bh_q"] < FDR_THRESH else (" ~" if row["p_ftd_vs_als"] < 0.1 else "")
        lines.append(
            f"{row['symbol']:<12} {row['rb_ftd_vs_als']:>+10.3f} "
            f"{row['p_ftd_vs_als']:>10.3e} {row['bh_q']:>10.3e} "
            f"{row['rb_als_ctrl']:>+12.3f} {row['rb_ftd_ctrl']:>+12.3f}{sig}"
        )

    lines += [
        "",
        f"Significant (BH-FDR < {FDR_THRESH}): {len(sig_panel)} genes",
        f"Trending (p < 0.10, not BH-significant): {len(trend_panel)} genes",
        "",
        "─" * 70,
        "ANALYSIS 2: Subtype consistency (protein-coding genes only)",
        "─" * 70,
        "Concordance: rb_ALS_ctrl ≈ rb_FTD_ctrl → pan-ALS signal",
        "Divergence:  large difference → subtype-specific biology",
        "",
        f"{'Symbol':<12} {'rb_ALS_ctrl':>12} {'rb_FTD_ctrl':>12} {'Δrb':>8} {'Classification':>20}",
        "-" * 70,
    ]
    for _, row in consistency_df.sort_values("rb_als_ctrl").iterrows():
        delta = row["rb_ftd_ctrl"] - row["rb_als_ctrl"]
        if abs(delta) < 0.1:
            cls = "pan-ALS"
        elif row["rb_als_ctrl"] * row["rb_ftd_ctrl"] < 0:
            cls = "direction-reversal"
        elif abs(delta) >= 0.2:
            cls = "subtype-divergent"
        else:
            cls = "mild-divergence"
        lines.append(
            f"{row['symbol']:<12} {row['rb_als_ctrl']:>+12.3f} "
            f"{row['rb_ftd_ctrl']:>+12.3f} {delta:>+8.3f}  {cls}"
        )

    lines += [
        "",
        "─" * 70,
        "ANALYSIS 3: Within-ALS subtype classification (25-gene panel, 5-fold CV)",
        "─" * 70,
        f"Target: ALS-FTD (1) vs pure ALS (0)",
        f"AUC (5-fold CV): {auc_cv:.4f}",
        f"Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]",
        "",
        "─" * 70,
        f"ANALYSIS 4: Full 3,704-gene screen — ALS-FTD vs pure ALS",
        "─" * 70,
        f"Significant at BH-FDR < {FDR_THRESH}: {n_sig} / {len(sfm_df)} genes",
        "",
        f"Top 30 by p-value:",
        f"{'Rank':>5} {'Symbol':<14} {'Feature':<22} {'rb_FTD_ALS':>10} {'p':>10} {'BH-q':>10}",
        "-" * 73,
    ]
    for _, row in sfm_df.head(30).iterrows():
        sig_m = " *" if row["bh_q"] < FDR_THRESH else ""
        lines.append(
            f"{int(row['shap_rank']):>5} {row['symbol']:<14} {row['feature']:<22} "
            f"{row['rb_ftd_vs_als']:>+10.3f} {row['p']:>10.3e} {row['bh_q']:>10.3e}{sig_m}"
        )

    if n_sig > 0:
        lines += [
            "",
            f"All {n_sig} BH-significant genes (sorted by p):",
            f"{'Rank':>5} {'Symbol':<14} {'rb_FTD_ALS':>10} {'p':>10} {'BH-q':>10}",
            "-" * 55,
        ]
        for _, row in sig_sfm.iterrows():
            lines.append(
                f"{int(row['shap_rank']):>5} {row['symbol']:<14} "
                f"{row['rb_ftd_vs_als']:>+10.3f} {row['p']:>10.3e} {row['bh_q']:>10.3e}"
            )

    out_txt = SCRIPT_DIR / "c9als_sals_differential_statistics.txt"
    out_txt.write_text("\n".join(lines) + "\n")
    print(f"  Statistics written: {out_txt.name}")
    print("Done.")


if __name__ == "__main__":
    main()
