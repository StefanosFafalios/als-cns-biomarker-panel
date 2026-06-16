"""Blood cohort validation: train on GPL24676 CNS, zero-shot predict on GSE234297 blood.

GSE234297 (Bjornevik et al. 2023 Ann Neurol):
  96 sporadic ALS + 48 healthy controls, peripheral blood RNA-seq.
  Features: EntrezGeneID (raw counts, HTSeq).

Strategy
--------
1. Map the 15 protein-coding panel genes (ENSG IDs → Entrez IDs).
2. Load GPL24676 RSEM raw counts; apply log1p; extract matched genes.
3. Load GSE234297 raw counts; apply log1p; extract same genes by Entrez ID.
4. Fit StandardScaler on GPL24676 training set; apply to both.
5. Train LightGBM (top-500 best params, colsample_bytree=1.0) on all 874
   GPL24676 samples with the matched genes.
6. Zero-shot predict on 144 GSE234297 samples; report AUC + bootstrap CI.
7. Within-cohort LOO-CV on GSE234297 to estimate blood-intrinsic
   discriminative signal (independent of CNS training).
8. Per-gene Mann-Whitney U (ALS vs control in blood).

Outputs
-------
  blood_validation_gse234297.png
  blood_validation_gse234297_statistics.txt
"""

from __future__ import annotations

import gzip
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ALS_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset  # noqa: E402

SCRIPT_DIR = Path(__file__).parent
_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"

_GSE234297_COUNTS = (
    ALS_DIR / "resources" / "GSE234297" / "suppl" / "GSE234297_gene_raw_counts.txt.gz"
)

RANDOM_STATE = 42
N_BOOTSTRAP = 2_000

# Protein-coding panel genes: symbol → Entrez ID (resolved via MyGeneInfo)
_ENTREZ: dict[str, int] = {
    "MECOM": 2122,
    "SERTAD1": 29950,
    "FCN3": 8547,
    "PROS1": 5627,
    "ANGPT2": 285,
    "EMP1": 2012,
    "TINAGL1": 64129,
    "CKMT2": 1160,
    "VWF": 7450,
    "CLDN5": 7122,
    "NR4A1": 3164,
    "SOHLH2": 54937,
    "HEXB": 3074,
    "MCEE": 84693,
    "SLC37A2": 219855,
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_train_panel(
    panel_features: list[str],
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """GPL24676 RSEM raw counts → log1p, return (X_matched, syms, y).

    Only columns corresponding to the 15 protein-coding panel genes are returned.
    panel_features: full versioned ENSG feature IDs from lgbm_core25_panel.csv
    """

    (ds,) = load_dataset(
        "GSE153960", platform="GPL24676", resources_dir=ALS_DIR / "resources"
    )
    X_log = np.log1p(ds.X.values.astype(np.float32))
    feat_map = {n: i for i, n in enumerate(ds.X.columns)}

    cols: list[int] = []
    syms: list[str] = []
    for feat, sym in zip(panel_features, _ENTREZ.keys()):
        idx = feat_map.get(feat)
        if idx is not None:
            cols.append(idx)
            syms.append(sym)

    return X_log[:, cols], syms, ds.y.values.astype(int)


def _load_blood(
    matched_syms: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """GSE234297 raw counts → log1p, return (X_matched, y).

    Columns: Case01..Case96 → ALS (y=1), Control01..Control48 → Control (y=0).
    Rows: EntrezGeneID.  Filter to Entrez IDs of matched_syms.
    """
    import pandas as pd

    with gzip.open(_GSE234297_COUNTS, "rt") as fh:
        counts = pd.read_csv(fh, sep="\t", index_col=0)

    # Build label vector from column names
    sample_cols = list(counts.columns)
    y = np.array([1 if c.startswith("Case") else 0 for c in sample_cols], dtype=int)

    entrez_ids = [_ENTREZ[s] for s in matched_syms]
    available = [eid for eid in entrez_ids if eid in counts.index]
    missing_entrez = [eid for eid in entrez_ids if eid not in counts.index]
    if missing_entrez:
        print(f"  WARNING: {len(missing_entrez)} Entrez IDs not found in blood matrix")

    X = np.log1p(
        counts.loc[available].values.T.astype(np.float32)
    )  # (n_samples, n_matched)

    # Return only matched symbols in the order they appear
    matched_final = [
        s for s, eid in zip(matched_syms, entrez_ids) if eid in counts.index
    ]
    return X, y, matched_final


# ---------------------------------------------------------------------------
# Bootstrap AUC
# ---------------------------------------------------------------------------


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
    y_test: np.ndarray,
    scores_zero: np.ndarray,
    scores_cv: np.ndarray,
    auc_zero: float,
    ci_lo: float,
    ci_hi: float,
    auc_cv: float,
    matched_syms: list[str],
    missing_syms: list[str],
    mwu_results: list[tuple[str, float, float]],
) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ROC: zero-shot
    fpr, tpr, _ = roc_curve(y_test, scores_zero)
    axes[0].plot(
        fpr,
        tpr,
        lw=2,
        color="#1f77b4",
        label=f"Zero-shot AUC = {auc_zero:.4f}\n95% CI [{ci_lo:.4f}, {ci_hi:.4f}]",
    )
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(
        f"GSE234297 zero-shot (trained on GPL24676 CNS n=874)\n"
        f"Blood · n=144 (ALS=96, Ctrl=48) · "
        f"{len(matched_syms)}/15 protein-coding genes",
        fontsize=9,
    )
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].grid(alpha=0.3)

    # ROC: within-cohort CV
    fpr_cv, tpr_cv, _ = roc_curve(y_test, scores_cv)
    axes[1].plot(
        fpr_cv, tpr_cv, lw=2, color="#ff7f0e", label=f"LOO-CV AUC = {auc_cv:.4f}"
    )
    axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title(
        "GSE234297 within-cohort LOO-CV\n(trained and tested entirely in blood)",
        fontsize=9,
    )
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].grid(alpha=0.3)

    # Bar chart: per-gene Mann-Whitney effect size in blood
    syms_mwu = [r[0] for r in mwu_results]
    rb_vals = [r[1] for r in mwu_results]
    pvals = [r[2] for r in mwu_results]
    colours = ["#d62728" if rb > 0 else "#1f77b4" for rb in rb_vals]
    sig_markers = ["*" if p < 0.05 else "" for p in pvals]

    y_pos = np.arange(len(syms_mwu))
    axes[2].barh(y_pos, rb_vals, color=colours, alpha=0.8)
    axes[2].axvline(0, color="black", lw=0.8)
    for i, (rb, marker) in enumerate(zip(rb_vals, sig_markers)):
        if marker:
            x = rb + 0.01 if rb >= 0 else rb - 0.01
            axes[2].text(
                x,
                i,
                marker,
                va="center",
                fontsize=10,
                ha="left" if rb >= 0 else "right",
            )
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(syms_mwu, fontsize=8)
    axes[2].set_xlabel("Rank-biserial r (ALS vs Control in blood)")
    axes[2].set_title(
        "Per-gene blood differential expression\n(* = MWU p < 0.05; red = up in ALS)",
        fontsize=9,
    )
    axes[2].grid(alpha=0.3, axis="x")

    plt.tight_layout()
    out = SCRIPT_DIR / "blood_validation_gse234297.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import matplotlib
    import pandas as pd
    from lightgbm import LGBMClassifier
    from scipy.stats import mannwhitneyu
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    matplotlib.use("Agg")

    print("\n" + "=" * 65)
    print("Blood Cohort Validation: GSE234297 (Peripheral Blood, n=144)")
    print("=" * 65)

    panel = pd.read_csv(_PANEL_CSV)
    # Protein-coding genes only (have valid Entrez ID)
    coding_mask = panel["symbol"].isin(_ENTREZ)
    panel_coding = panel[coding_mask]
    panel_features: list[str] = panel_coding["feature"].tolist()
    panel_syms: list[str] = panel_coding["symbol"].tolist()

    print(f"\nProtein-coding panel genes: {len(panel_syms)}")

    print("\nLoading GPL24676 training data ...")
    X_tr, matched_syms, y_train = _load_train_panel(panel_features)
    print(
        f"  GPL24676: n={len(y_train)}  ALS={y_train.sum()}  Ctrl={(y_train == 0).sum()}"
    )
    print(f"  Training genes matched: {len(matched_syms)}")

    print("\nLoading GSE234297 blood data ...")
    X_blood_raw, y_test, matched_syms = _load_blood(matched_syms)
    print(
        f"  GSE234297: n={len(y_test)}  ALS={y_test.sum()}  Ctrl={(y_test == 0).sum()}"
    )
    print(f"  Panel genes found in blood matrix: {len(matched_syms)}")
    print(f"  Matched: {', '.join(matched_syms)}")
    missing_syms = [s for s in panel_syms if s not in matched_syms]
    if missing_syms:
        print(f"  Missing: {', '.join(missing_syms)}")

    if len(matched_syms) < 3:
        print("  ERROR: fewer than 3 matched genes — cannot proceed")
        return

    # Subset training matrix to the genes that were also found in blood
    train_sym_order = {s: i for i, s in enumerate(matched_syms)}
    # X_tr was built in the order of panel_syms; rebuild for blood-matched only
    print("\nRealigning training matrix to blood-matched genes ...")
    (ds,) = load_dataset(
        "GSE153960", platform="GPL24676", resources_dir=ALS_DIR / "resources"
    )
    X_log_full = np.log1p(ds.X.values.astype(np.float32))
    feat_map = {n: i for i, n in enumerate(ds.X.columns)}

    train_cols_final: list[int] = []
    for sym in matched_syms:
        feat = panel_coding.loc[panel_coding["symbol"] == sym, "feature"].values[0]
        col = feat_map.get(feat)
        if col is None:
            print(f"  WARNING: {sym} ({feat}) not found in GPL24676 matrix")
            continue
        train_cols_final.append(col)

    X_tr_final = X_log_full[:, train_cols_final]

    # Standardise: fit on training, apply to blood
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_final).astype(np.float32)
    X_blood_sc = scaler.transform(X_blood_raw).astype(np.float32)

    params = json.loads(_PARAMS_PATH.read_text())
    params["colsample_bytree"] = 1.0
    params["verbose"] = -1

    # -----------------------------------------------------------------------
    # Zero-shot: train on GPL24676, predict on blood
    # -----------------------------------------------------------------------
    print(f"\nTraining on GPL24676 ({len(matched_syms)} genes) ...")
    clf = LGBMClassifier(**params)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        clf.fit(X_tr_sc, y_train)

    print("Zero-shot prediction on GSE234297 blood ...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        scores_zero = clf.predict_proba(X_blood_sc)[:, 1]
    auc_zero = float(roc_auc_score(y_test, scores_zero))
    print(f"  Zero-shot AUC: {auc_zero:.4f}")

    rng = np.random.default_rng(RANDOM_STATE)
    boot = _bootstrap_auc(y_test, scores_zero, N_BOOTSTRAP, rng)
    ci_lo = float(np.percentile(boot, 2.5))
    ci_hi = float(np.percentile(boot, 97.5))
    print(f"  Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    # -----------------------------------------------------------------------
    # Within-cohort LOO-CV on blood alone (min_child_samples=1, full params),
    # matching the leave-one-out methodology used for the other validation
    # cohorts (GSE76220, GSE122649) and the cross-cohort panel LOO.
    # -----------------------------------------------------------------------
    print("\nWithin-cohort LOO-CV on GSE234297 blood ...")
    loo_params = {**params, "min_child_samples": 1}
    cv_scores = np.zeros(len(y_test))
    for i in range(len(y_test)):
        tr_idx = [j for j in range(len(y_test)) if j != i]
        clf_cv = LGBMClassifier(**loo_params)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            clf_cv.fit(X_blood_sc[tr_idx], y_test[tr_idx])
        cv_scores[i] = clf_cv.predict_proba(X_blood_sc[[i]])[0, 1]
    auc_cv = float(roc_auc_score(y_test, cv_scores))
    boot_cv = _bootstrap_auc(y_test, cv_scores, N_BOOTSTRAP, rng)
    cv_ci_lo = float(np.percentile(boot_cv, 2.5))
    cv_ci_hi = float(np.percentile(boot_cv, 97.5))
    print(
        f"  Within-cohort LOO-CV AUC: {auc_cv:.4f}  "
        f"95% CI [{cv_ci_lo:.4f}, {cv_ci_hi:.4f}]"
    )

    # -----------------------------------------------------------------------
    # Per-gene blood differential expression (Mann-Whitney U)
    # -----------------------------------------------------------------------
    print("\nPer-gene Mann-Whitney U (ALS vs Control in blood) ...")
    als_mask = y_test == 1
    ctrl_mask = y_test == 0
    mwu_results: list[tuple[str, float, float]] = []
    for i, sym in enumerate(matched_syms):
        als_vals = X_blood_raw[als_mask, i]
        ctrl_vals = X_blood_raw[ctrl_mask, i]
        stat, p = mannwhitneyu(als_vals, ctrl_vals, alternative="two-sided")
        n1, n2 = len(als_vals), len(ctrl_vals)
        rb = float(2 * stat / (n1 * n2) - 1)  # rank-biserial r
        mwu_results.append((sym, rb, float(p)))
        sig = "*" if p < 0.05 else ""
        print(f"  {sym:<10} rb={rb:+.3f}  p={p:.3e} {sig}")

    mwu_results.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nGenerating plot ...")
    _plot(
        y_test,
        scores_zero,
        cv_scores,
        auc_zero,
        ci_lo,
        ci_hi,
        auc_cv,
        matched_syms,
        missing_syms,
        mwu_results,
    )

    # -----------------------------------------------------------------------
    # Statistics file
    # -----------------------------------------------------------------------
    lines = [
        "Blood Cohort Validation — GSE234297 (Step 14)",
        "=" * 65,
        "Dataset: GSE234297 (Bjornevik et al. 2023 Ann Neurol)",
        "Tissue: Peripheral blood (whole blood RNA-seq)",
        "Cohort: n=144 (sALS=96, Healthy Control=48)",
        "Training data: GPL24676 CNS post-mortem (n=874, ALS=684, Ctrl=190)",
        f"Panel genes used: {len(matched_syms)}/15 protein-coding",
        f"  Matched : {', '.join(matched_syms)}",
        f"  Missing : {', '.join(missing_syms) if missing_syms else 'none'}",
        "",
        "ZERO-SHOT VALIDATION (CNS-trained model → blood)",
        "-" * 65,
        f"AUC:           {auc_zero:.4f}",
        f"Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]  (n={N_BOOTSTRAP} iterations)",
        "",
        "WITHIN-COHORT LOO-CV (blood only; min_child_samples=1, full params)",
        "-" * 65,
        f"AUC:           {auc_cv:.4f}",
        f"Bootstrap 95% CI: [{cv_ci_lo:.4f}, {cv_ci_hi:.4f}]  (n={N_BOOTSTRAP} iterations)",
        "",
        "PER-GENE BLOOD DIFFERENTIAL EXPRESSION (ALS vs Control)",
        "rank-biserial r: positive = higher in ALS",
        f"{'Symbol':<12} {'rb':>7} {'MWU p':>12} {'Sig':>4}",
        "-" * 40,
    ]
    for sym, rb, p in mwu_results:
        sig = "*" if p < 0.05 else ""
        lines.append(f"{sym:<12} {rb:>+7.3f} {p:>12.3e} {sig:>4}")

    out_txt = SCRIPT_DIR / "blood_validation_gse234297_statistics.txt"
    out_txt.write_text("\n".join(lines) + "\n")
    print(f"  Statistics written: {out_txt.name}")
    print("Done.")


if __name__ == "__main__":
    main()
