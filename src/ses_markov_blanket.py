"""Markov Blanket discovery on the top-500 SHAP feature set via IAMB.

Runs IAMBSelector (Fisher-Z partial-correlation CI tests) on the top-500
SHAP-ranked features from the GSE153960 pre-filtered matrix.  Alpha is
tuned by 5-fold stratified CV AUC over the grid {0.01, 0.05, 0.10, 0.20}.

Outputs
-------
  ses_markov_blanket_statistics.txt   — full report (MB genes, panel overlap,
                                        CV AUC per alpha, novel MB-only genes)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ALS_DIR = Path(__file__).parents[1]
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset  # noqa: E402

PLATFORM = "GPL24676"
RANDOM_STATE = 42
N_FOLDS = 5
TOP_N = 500
ALPHA_GRID = [0.01, 0.05, 0.10, 0.20]

_PREFILTER_X = SCRIPT_DIR / "lgbm_prefilter_X.npy"
_PREFILTER_NAMES = SCRIPT_DIR / "lgbm_prefilter_names.txt"
_SHAP_RANKING = SCRIPT_DIR / "lgbm_iter_final_shap_ranking.csv"
_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_ANNOT_TSV = SCRIPT_DIR / "core25_annotations.tsv"
_OUT_STATS = SCRIPT_DIR / "ses_markov_blanket_statistics.txt"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_top500() -> tuple[np.ndarray, list[str]]:
    """Slice pre-filter cache to the top-500 SHAP-ranked features."""
    shap_df = pd.read_csv(_SHAP_RANKING)
    top_features: list[str] = shap_df["feature"].head(TOP_N).tolist()
    all_names = _PREFILTER_NAMES.read_text().splitlines()
    name_to_col = {n: i for i, n in enumerate(all_names)}
    col_idx = [name_to_col[f] for f in top_features]
    X_full = np.load(_PREFILTER_X)
    return X_full[:, col_idx].astype(np.float32), top_features


def _load_labels() -> np.ndarray:
    (ds,) = load_dataset("GSE153960", platform=PLATFORM, resources_dir=ALS_DIR / "resources")
    return ds.y.values.astype(int)


# ---------------------------------------------------------------------------
# Alpha tuning via stratified k-fold CV
# ---------------------------------------------------------------------------


def _cv_auc(X: np.ndarray, y: np.ndarray, alpha: float) -> float:
    """Mean 5-fold stratified CV AUC for IAMBSelector + LightGBM."""
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    from model.fs.iamb import IAMBSelector

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs: list[float] = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        # Feature selection on training fold
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            sel = IAMBSelector(alpha=alpha).fit(X_tr, y_tr)
        mask = sel.get_support()
        n_sel = int(mask.sum())
        if n_sel == 0:
            continue

        X_tr_sel = X_tr[:, mask]
        X_val_sel = X_val[:, mask]

        scaler = StandardScaler()
        X_tr_sel = scaler.fit_transform(X_tr_sel)
        X_val_sel = scaler.transform(X_val_sel)

        clf = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            colsample_bytree=1.0,
            random_state=RANDOM_STATE,
            verbosity=-1,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            clf.fit(X_tr_sel, y_tr)

        prob = clf.predict_proba(X_val_sel)[:, 1]
        aucs.append(roc_auc_score(y_val, prob))
        print(f"    Fold {fold + 1}/{N_FOLDS}: n_MB={n_sel}  AUC={aucs[-1]:.4f}")

    return float(np.mean(aucs)) if aucs else 0.0


# ---------------------------------------------------------------------------
# Full IAMB run at best alpha
# ---------------------------------------------------------------------------


def _run_iamb(X: np.ndarray, y: np.ndarray, alpha: float, feature_names: list[str]) -> list[str]:
    """Run IAMB on the full dataset and return selected feature names."""
    from model.fs.iamb import IAMBSelector

    print(f"\nRunning IAMB on full dataset (n={len(y)}, p={X.shape[1]}, alpha={alpha}) ...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        sel = IAMBSelector(alpha=alpha).fit(X, y.astype(float))
    mask = sel.get_support()
    return [feature_names[i] for i in np.where(mask)[0]]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _load_panel_ensg() -> list[str]:
    df = pd.read_csv(_PANEL_CSV)
    ensg_col = next(
        (c for c in df.columns if "ensg" in c.lower() or c.upper().startswith("ENSG")),
        df.columns[1],
    )
    return [g.split(".")[0] for g in df[ensg_col].tolist()]


def _load_panel_symbols() -> dict[str, str]:
    """Return {ensg_base: symbol} mapping from annotation TSV (if available)."""
    if not _ANNOT_TSV.exists():
        return {}
    df = pd.read_csv(_ANNOT_TSV, sep="\t")
    ensg_col = next((c for c in df.columns if "ensg" in c.lower()), None)
    sym_col = next((c for c in df.columns if "symbol" in c.lower() or "gene_name" in c.lower()), None)
    if ensg_col is None or sym_col is None:
        return {}
    return {
        str(row[ensg_col]).split(".")[0]: str(row[sym_col])
        for _, row in df.iterrows()
    }


def main() -> None:
    print("=" * 65)
    print("IAMB Markov Blanket Discovery — GSE153960 top-500 SHAP features")
    print("=" * 65)

    print(f"\n[1] Loading top-{TOP_N} SHAP features ...")
    X, feature_names = _load_top500()
    print(f"  Matrix: {X.shape}")

    print("\n[2] Loading labels ...")
    y = _load_labels()
    print(f"  ALS={int(y.sum())}  Control={int((y == 0).sum())}  Total={len(y)}")

    # Standardize once (CV scaler is per-fold)
    from sklearn.preprocessing import StandardScaler
    X_std = StandardScaler().fit_transform(X)

    # ------------------------------------------------------------------
    # Alpha tuning
    # ------------------------------------------------------------------
    print(f"\n[3] Alpha tuning ({N_FOLDS}-fold CV) over {ALPHA_GRID} ...")
    cv_results: dict[float, float] = {}
    for alpha in ALPHA_GRID:
        print(f"\n  alpha={alpha}:")
        auc = _cv_auc(X_std, y, alpha)
        cv_results[alpha] = auc
        print(f"  => mean CV AUC = {auc:.4f}")

    best_alpha = max(cv_results, key=lambda a: cv_results[a])
    best_auc = cv_results[best_alpha]
    print(f"\n  Best alpha = {best_alpha}  (CV AUC = {best_auc:.4f})")

    # ------------------------------------------------------------------
    # Full IAMB at best alpha
    # ------------------------------------------------------------------
    mb_features = _run_iamb(X_std, y, best_alpha, feature_names)
    mb_base = [f.split(".")[0] for f in mb_features]
    print(f"  Markov Blanket size: {len(mb_features)} genes")

    # ------------------------------------------------------------------
    # Panel comparison
    # ------------------------------------------------------------------
    panel_base = _load_panel_ensg()
    sym_map = _load_panel_symbols()

    panel_set = set(panel_base)
    mb_set = set(mb_base)

    overlap = panel_set & mb_set
    mb_only = mb_set - panel_set
    panel_only = panel_set - mb_set

    def _sym(ensg: str) -> str:
        return sym_map.get(ensg, ensg)

    print(f"\n  Panel ∩ MB (in both):  {len(overlap)}")
    print(f"  MB only (novel):       {len(mb_only)}")
    print(f"  Panel only (not in MB): {len(panel_only)}")

    # ------------------------------------------------------------------
    # Write statistics file
    # ------------------------------------------------------------------
    lines: list[str] = []
    lines += [
        "IAMB Markov Blanket Discovery — GSE153960",
        "=" * 60,
        f"Input: top-{TOP_N} SHAP-ranked features (n_samples={len(y)})",
        f"  ALS: {int(y.sum())}  Control: {int((y == 0).sum())}",
        f"CI test: Fisher-Z partial correlation",
        f"Alpha tuning: {N_FOLDS}-fold stratified CV, grid={ALPHA_GRID}",
        "",
        "Alpha tuning results:",
    ]
    for a, auc in sorted(cv_results.items()):
        marker = " <-- best" if a == best_alpha else ""
        lines.append(f"  alpha={a:.2f}  CV AUC={auc:.4f}{marker}")
    lines += [
        "",
        f"Selected alpha: {best_alpha}",
        f"Best CV AUC:    {best_auc:.4f}",
        "",
        f"Markov Blanket: {len(mb_features)} genes",
        "-" * 60,
    ]
    for feat in sorted(mb_features, key=lambda f: _sym(f.split(".")[0])):
        lines.append(f"  {_sym(feat.split('.')[0]):<20}  {feat}")

    lines += [
        "",
        f"Panel vs Markov Blanket comparison",
        f"  25-gene panel size: {len(panel_base)}",
        f"  Markov Blanket size: {len(mb_features)}",
        f"  Overlap (in both):  {len(overlap)}",
        f"  MB-only (novel):    {len(mb_only)}",
        f"  Panel-only:         {len(panel_only)}",
        "",
        f"Overlap genes ({len(overlap)}):",
    ]
    for ensg in sorted(overlap, key=_sym):
        lines.append(f"  {_sym(ensg):<20}  {ensg}")

    lines += ["", f"MB-only genes — not in 25-gene panel ({len(mb_only)}):"]
    for ensg in sorted(mb_only, key=_sym):
        lines.append(f"  {_sym(ensg):<20}  {ensg}")

    lines += ["", f"Panel-only genes — not in MB ({len(panel_only)}):"]
    for ensg in sorted(panel_only, key=_sym):
        lines.append(f"  {_sym(ensg):<20}  {ensg}")

    report = "\n".join(lines)
    _OUT_STATS.write_text(report)
    print(f"\n[4] Statistics written to {_OUT_STATS.name}")
    print("\n" + report)


if __name__ == "__main__":
    main()
