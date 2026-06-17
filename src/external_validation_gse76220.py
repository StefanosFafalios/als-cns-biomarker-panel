# ruff: noqa: E402
"""Independent cohort validation on GSE76220 (lumbar spinal cord LCM, n=20).

GSE76220 uses RPKM values and gene symbols (not Ensembl IDs), and is a
completely independent study (laser-capture microdissected lumbar motor neurons).

Strategy
--------
1. Map the 25 core panel genes from Ensembl IDs to HGNC symbols using the
   annotation TSV produced by annotate_core25.py.
2. Find which panel symbols are present in GSE76220 (gene-symbol keyed).
3. For matched genes: apply log1p to both GPL24676 raw RSEM and GSE76220 RPKM.
4. Fit a StandardScaler on the GPL24676 log1p values; apply to GSE76220.
5. Train LightGBM (top-500 params, colsample_bytree=1.0) on all 874 GPL24676
   samples with the matched genes.
6. Zero-shot predict on the 20 GSE76220 samples.
7. Report AUC, LOO-CV AUC on GSE76220 alone (n=20), feature coverage.

Note: the small GSE76220 cohort (n=20, 12 ALS, 8 Control) means AUC CIs are
wide. Interpret directional signal, not absolute value.

Outputs
-------
  blood_validation_gse76220.png
  blood_validation_gse76220_statistics.txt
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ALS_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset

SCRIPT_DIR = Path(__file__).parent

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_ANNOT_TSV = SCRIPT_DIR / "core25_annotations.tsv"
_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"
_PROFILING_CSV = SCRIPT_DIR / "gene_profiling_summary.csv"

RANDOM_STATE = 42
N_BOOTSTRAP = 2_000

# De-duplicated replacement genes for panel genes missing from GSE76220.
# Selected from gene_replacement_results.csv (clean pool, artifacts excluded).
# Assignment sorted by |Pearson r| descending, then replacement AUC; each
# replacement gene used at most once across all missing panel positions.
# De-duplicated clean surrogates (confirmed artifacts AND artifact gene
# families excluded), matching the canonical assignment in
# adaptive_panel_validation.py. The keratin-associated artifact KRTAP6-2 is
# filtered, so MAP3K2-DT takes its next clean candidate (C1orf185) and
# RPL15P11 cascades to MAGEB2.
REPLACEMENTS_76220: dict[str, tuple[str, str]] = {
    "MECOM": ("MYCT1", "ENSG00000120279"),  # |r|=0.576
    "HERC2P8": ("HERC2P4", "ENSG00000230267"),  # |r|=0.872
    "SMG1P5": ("SMG1", "ENSG00000157106"),  # |r|=0.606
    "RPL21P75": ("SLC6A18", "ENSG00000164363"),
    "MAP3K2-DT": ("C1orf185", "ENSG00000204006"),
    "ENSG00000280893": ("INSL6", "ENSG00000120210"),
    "RPL15P11": ("MAGEB2", "ENSG00000099399"),
    "RHOT1P2": ("DHCR24", "ENSG00000116133"),
    "ENSG00000279656": ("OAS2", "ENSG00000111335"),
    "LOC112268270": ("PPP1R1B", "ENSG00000131771"),  # |r|=0.573
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _load_train() -> tuple[np.ndarray, list[str], np.ndarray]:
    """GPL24676 raw RSEM → log1p, return (X, feature_names, y)."""
    (ds,) = load_dataset(
        "GSE153960", platform="GPL24676", resources_dir=ALS_DIR / "resources"
    )
    X = np.log1p(ds.X.values.astype(np.float32))
    return X, list(ds.X.columns), ds.y.values.astype(int)


def _load_gse76220() -> tuple[np.ndarray, list[str], np.ndarray]:
    """GSE76220 RPKM → log1p, return (X, gene_symbols, y).

    Negative RPKM values produce NaN via log1p; LightGBM handles NaN
    natively as missing, so they are preserved intentionally.
    """
    (ds,) = load_dataset("GSE76220", resources_dir=ALS_DIR / "resources")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        X = np.log1p(ds.X.values.astype(np.float32))
    return X, list(ds.X.columns), ds.y.values.astype(int)


def _build_gene_map(annot_tsv: Path, panel_csv: Path) -> dict[str, str]:
    """Return {ensembl_feature_id → hgnc_symbol} from annotation TSV."""
    import pandas as pd

    ann = pd.read_csv(annot_tsv, sep="\t")
    panel = pd.read_csv(panel_csv)
    # Use the symbol column in panel CSV directly (already resolved)
    result: dict[str, str] = {}
    for _, row in panel.iterrows():
        feat = str(row["feature"])
        sym = str(row["symbol"])
        result[feat] = sym
    return result


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


def _loo_auc(X: np.ndarray, y: np.ndarray, params: dict) -> tuple[float, np.ndarray]:
    """Leave-one-out AUC within the GSE76220 cohort (n=20).

    Overrides min_child_samples=1 so LightGBM can split on n_train=19.
    """
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score

    loo_params = {**params, "min_child_samples": 1, "verbose": -1}
    scores = np.zeros(len(y))
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        clf = LGBMClassifier(**loo_params)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            clf.fit(X[tr], y[tr])
        scores[i] = clf.predict_proba(X[[i]])[0, 1]
    return float(roc_auc_score(y, scores)), scores


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _plot(
    y_test: np.ndarray,
    scores_zeroshot: np.ndarray,
    scores_loo: np.ndarray,
    auc_zero: float,
    ci_lo: float,
    ci_hi: float,
    auc_loo: float,
    train_auc: float,
    matched_genes: list[str],
    missing_genes: list[str],
) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, scores, auc, label, colour in [
        (
            axes[0],
            scores_zeroshot,
            auc_zero,
            f"Zero-shot AUC = {auc_zero:.4f}\n95% CI [{ci_lo:.4f}, {ci_hi:.4f}]",
            "#1f77b4",
        ),
        (
            axes[1],
            scores_loo,
            auc_loo,
            f"Within-cohort LOO AUC = {auc_loo:.4f}",
            "#ff7f0e",
        ),
    ]:
        fpr, tpr, _ = roc_curve(y_test, scores)
        ax.plot(fpr, tpr, lw=2, color=colour, label=label)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)

    axes[0].set_title(
        f"GSE76220 zero-shot validation (train GPL24676 n=874)\n"
        f"Lumbar SC LCM · n=20 (ALS=12, Ctrl=8) · "
        f"{len(matched_genes)}/{len(matched_genes) + len(missing_genes)} panel genes matched",
        fontsize=9,
    )
    axes[1].set_title(
        "GSE76220 within-cohort LOO-CV\n"
        "(trained on n−1 GSE76220 samples per fold, matched genes only)",
        fontsize=9,
    )

    plt.tight_layout()
    out = SCRIPT_DIR / "blood_validation_gse76220.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import json

    import matplotlib
    import pandas as pd
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    matplotlib.use("Agg")

    print("\n" + "=" * 65)
    print("Independent Cohort Validation: GSE76220 (Lumbar SC LCM, n=20)")
    print("=" * 65)

    panel = pd.read_csv(_PANEL_CSV)
    panel_features: list[str] = panel["feature"].tolist()
    panel_symbols: list[str] = panel["symbol"].tolist()

    print("\nLoading GPL24676 training data ...")
    X_train_log, feat_train, y_train = _load_train()
    feat_train_base = {n.split(".")[0]: n for n in feat_train}
    print(
        f"  GPL24676: n={len(y_train)}  ALS={y_train.sum()}  Ctrl={(y_train == 0).sum()}"
    )

    print("\nLoading GSE76220 ...")
    X_test_log, feat_test_sym, y_test = _load_gse76220()
    test_sym_to_col = {s: i for i, s in enumerate(feat_test_sym)}
    print(
        f"  GSE76220: n={len(y_test)}  ALS={y_test.sum()}  Ctrl={(y_test == 0).sum()}"
    )

    # Match panel genes by symbol
    gene_map = _build_gene_map(_ANNOT_TSV, _PANEL_CSV)

    matched: list[tuple[str, str, int, int]] = []  # (symbol, feat, train_col, test_col)
    missing: list[str] = []
    train_name_to_col = {n: i for i, n in enumerate(feat_train)}

    for feat, sym in zip(panel_features, panel_symbols):
        train_col = train_name_to_col.get(feat)
        test_col = test_sym_to_col.get(sym)
        if train_col is not None and test_col is not None:
            matched.append((sym, feat, train_col, test_col))
        else:
            missing.append(sym)

    matched_syms = [m[0] for m in matched]
    print(f"\nPanel gene coverage: {len(matched)}/25 matched in GSE76220")
    print(f"  Matched : {', '.join(matched_syms)}")
    print(f"  Missing : {', '.join(missing)}")

    if len(matched) < 3:
        print("  ERROR: fewer than 3 matched genes — cannot proceed")
        return

    # -----------------------------------------------------------------------
    # Condition A: native matches only
    # -----------------------------------------------------------------------
    X_tr = np.column_stack([X_train_log[:, c] for _, _, c, _ in matched])
    X_te = np.column_stack([X_test_log[:, c] for _, _, _, c in matched])

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_sc = scaler.transform(X_te).astype(np.float32)

    params = json.loads(_PARAMS_PATH.read_text())
    params["colsample_bytree"] = 1.0

    print(f"\nZero-shot A: {len(matched)} native genes ...")
    clf = LGBMClassifier(**params)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        clf.fit(X_tr_sc, y_train)

    TRAIN_CV_AUC = 0.9621
    test_scores_zero = clf.predict_proba(X_te_sc)[:, 1]
    auc_zero = float(roc_auc_score(y_test, test_scores_zero))
    rng = np.random.default_rng(RANDOM_STATE)
    boot = _bootstrap_auc(y_test, test_scores_zero, N_BOOTSTRAP, rng)
    ci_lo = float(np.percentile(boot, 2.5))
    ci_hi = float(np.percentile(boot, 97.5))
    print(f"  AUC = {auc_zero:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]")

    # -----------------------------------------------------------------------
    # Condition B: native + replacements for missing genes
    # -----------------------------------------------------------------------
    train_ensg_base_to_col = {n.split(".")[0]: i for i, n in enumerate(feat_train)}
    ext_matched = list(matched)  # copy
    repl_pairs: list[str] = []
    for sym in missing:
        if sym not in REPLACEMENTS_76220:
            continue
        repl_sym, repl_ensg = REPLACEMENTS_76220[sym]
        tr_col = train_ensg_base_to_col.get(repl_ensg)
        te_col = test_sym_to_col.get(repl_sym)
        if tr_col is not None and te_col is not None:
            # Use replacement ENSG full name from feat_train for the tuple
            repl_feat = feat_train[tr_col]
            ext_matched.append((f"{sym}→{repl_sym}", repl_feat, tr_col, te_col))
            repl_pairs.append(f"{sym} → {repl_sym} ({repl_ensg})")

    n_repl = len(ext_matched) - len(matched)
    print(
        f"\nZero-shot B: {len(ext_matched)} genes ({len(matched)} native + {n_repl} replacements) ..."
    )

    X_tr_b = np.column_stack([X_train_log[:, c] for _, _, c, _ in ext_matched])
    X_te_b = np.column_stack([X_test_log[:, c] for _, _, _, c in ext_matched])
    scaler_b = StandardScaler()
    X_tr_b_sc = scaler_b.fit_transform(X_tr_b).astype(np.float32)
    X_te_b_sc = scaler_b.transform(X_te_b).astype(np.float32)

    clf_b = LGBMClassifier(**params)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        clf_b.fit(X_tr_b_sc, y_train)
    scores_b = clf_b.predict_proba(X_te_b_sc)[:, 1]
    auc_b = float(roc_auc_score(y_test, scores_b))
    boot_b = _bootstrap_auc(y_test, scores_b, N_BOOTSTRAP, rng)
    ci_b_lo = float(np.percentile(boot_b, 2.5))
    ci_b_hi = float(np.percentile(boot_b, 97.5))
    print(f"  AUC = {auc_b:.4f}  [{ci_b_lo:.4f}, {ci_b_hi:.4f}]")

    # -----------------------------------------------------------------------
    # Within-cohort LOO-CV: native genes only (small n — StandardScaler only)
    # -----------------------------------------------------------------------
    print("\nWithin-cohort LOO-CV on GSE76220 ...")
    auc_loo, scores_loo = _loo_auc(X_te_sc, y_test, params)
    print(f"  LOO AUC: {auc_loo:.4f}")

    # -----------------------------------------------------------------------
    # Plot: 3 panels
    # -----------------------------------------------------------------------
    ext_syms = [m[0] for m in ext_matched]
    print("\nGenerating plot ...")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, scores, auc, lo, hi, colour, title in [
        (
            axes[0],
            test_scores_zero,
            auc_zero,
            ci_lo,
            ci_hi,
            "#1f77b4",
            f"Zero-Shot A: {len(matched)} native genes\nAUC = {auc_zero:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]",
        ),
        (
            axes[1],
            scores_b,
            auc_b,
            ci_b_lo,
            ci_b_hi,
            "#2ca02c",
            f"Zero-Shot B: {len(ext_matched)} genes (+{n_repl} replacements)\nAUC = {auc_b:.4f}  [{ci_b_lo:.4f}, {ci_b_hi:.4f}]",
        ),
        (
            axes[2],
            scores_loo,
            auc_loo,
            None,
            None,
            "#ff7f0e",
            f"Within-cohort LOO-CV\nAUC = {auc_loo:.4f}",
        ),
    ]:
        fpr, tpr, _ = roc_curve(y_test, scores)
        ax.plot(fpr, tpr, lw=2, color=colour, label=title)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)

    axes[0].set_title(
        f"GSE76220 zero-shot (GPL24676 n=874 → n=20 LCM lumbar SC)\n{len(matched)}/25 native genes",
        fontsize=9,
    )
    axes[1].set_title(
        f"GSE76220 zero-shot with replacements\n{len(ext_matched)}/25 genes", fontsize=9
    )
    axes[2].set_title(
        "GSE76220 within-cohort LOO-CV\n(trained on n−1 per fold, native genes only)",
        fontsize=9,
    )

    plt.tight_layout()
    out = SCRIPT_DIR / "blood_validation_gse76220.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {out.name}")

    # -----------------------------------------------------------------------
    # Statistics file
    # -----------------------------------------------------------------------
    repl_lines_txt = [f"  {p}" for p in repl_pairs]
    lines = [
        "Independent Cohort Validation: GSE76220 (Lumbar SC LCM)",
        "=" * 65,
        f"Train : GSE153960 GPL24676  n={len(y_train)}  ALS={y_train.sum()}  Ctrl={(y_train == 0).sum()}",
        f"Test  : GSE76220 (LCM lumbar SC)  n={len(y_test)}  ALS={y_test.sum()}  Ctrl={(y_test == 0).sum()}",
        f"Panel gene coverage: {len(matched)}/25 native  |  {len(ext_matched)}/25 with replacements",
        f"Matched natively : {', '.join(matched_syms)}",
        f"Missing          : {', '.join(missing)}",
        "",
        "Replacement gene mapping:",
        *repl_lines_txt,
        "",
        "Preprocessing: log1p → StandardScaler fit on GPL24676",
        "",
        f"Train 5-fold CV AUC              : {TRAIN_CV_AUC:.4f}",
        f"Zero-shot AUC (A, {len(matched)} native)      : {auc_zero:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]",
        f"Zero-shot AUC (B, {len(ext_matched)} with repl): {auc_b:.4f}  [{ci_b_lo:.4f}, {ci_b_hi:.4f}]",
        f"Within-cohort LOO AUC            : {auc_loo:.4f}",
        "",
        "Caution: n=20 cohort — CIs are wide; interpret directional signal only.",
    ]
    stat_out = SCRIPT_DIR / "blood_validation_gse76220_statistics.txt"
    stat_out.write_text("\n".join(lines))
    print(f"  Saved → {stat_out.name}")

    print("\n" + "=" * 65)
    print("DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
