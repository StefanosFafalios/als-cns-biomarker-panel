"""Multi-cohort LR baseline + Brier/calibration across all validation cohorts.

Extends the GPL16791-only LR baseline to GSE76220 (LCM SC), GSE122649
(motor cortex), SRP064478 (cervical SC), GSE234297 (blood). For each cohort,
reports zero-shot AUC + Brier + log-loss + mean control score for both
LightGBM and L2-LR on the 25-gene panel.

The training pipeline is the existing one (with CTD residualisation).
Test cohorts that aren't matched by Ensembl ID (HGNC-only) use
symbol-based matching with column-mean imputation for missing genes,
matching the validation_panel_loo.py pipeline.

Outputs: multicohort_baselines.csv, .txt
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

from external_validation_gpl16791 import (  # noqa: E402
    _load_platform,
)
from utils import load_dataset  # noqa: E402
from utils import selected_lgbm_params  # noqa: E402

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"


def _fit_lgbm(X, y):
    from utils import PanelLGBMClassifier as LGBMClassifier

    params = selected_lgbm_params()
    params["colsample_bytree"] = 1.0
    clf = LGBMClassifier(**params)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        clf.fit(X, y)
    return clf


def _fit_lr(X, y):
    from sklearn.linear_model import LogisticRegressionCV

    clf = LogisticRegressionCV(
        cv=5,
        solver="liblinear",
        penalty="l2",
        scoring="roc_auc",
        max_iter=2000,
        random_state=42,
        refit=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        clf.fit(X, y)
    return clf


def _extract_panel(X, feat_names, panel_features):
    fmap = {f: i for i, f in enumerate(feat_names)}
    return X[:, [fmap[f] for f in panel_features if f in fmap]]


def _score_cohort(name, X_tr, y_train, X_te, y_te):
    """Fit both baselines for one cohort and score them.

    Both models are fitted here rather than once up front, because the correct
    standardisation depends on the *test* cohort: GSE76220 is log2 RPKM against
    log1p-count training data, so train and test must each be standardised within
    themselves (utils.zero_shot_scale). Scoring a model trained on raw log1p
    counts with raw log2 RPKM puts every test sample below all of the tree's
    split thresholds -- the model emits one constant probability and AUC is
    exactly 0.5000. That is what this function used to do for LGBM, which is why
    the fit was moved inside.
    """
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    from utils import zero_shot_scale

    X_tr_sc, X_te_sc = zero_shot_scale(X_tr, X_te, name)
    lgbm = _fit_lgbm(X_tr_sc, y_train)
    lr = _fit_lr(X_tr_sc, y_train)
    s_lgbm = lgbm.predict_proba(X_te_sc)[:, 1]
    s_lr = lr.predict_proba(X_te_sc)[:, 1]
    out = []
    for model_name, s in [("LGBM", s_lgbm), ("LR_L2", s_lr)]:
        out.append(
            {
                "cohort": name,
                "model": model_name,
                "n": len(y_te),
                "n_pos": int(y_te.sum()),
                "AUC": round(float(roc_auc_score(y_te, s)), 4),
                "brier": round(float(brier_score_loss(y_te, s)), 4),
                "log_loss": round(float(log_loss(y_te, np.clip(s, 1e-7, 1 - 1e-7))), 4),
                "mean_pos": round(float(s[y_te == 1].mean()), 4),
                "mean_neg": round(float(s[y_te == 0].mean()), 4)
                if (y_te == 0).any()
                else float("nan"),
            }
        )
    return out


def _load_external_by_symbol(gse_id: str, platform: str, panel_symbols: list[str]):
    """Load external cohort and match panel by HGNC symbol (with column-mean fill)."""
    try:
        (ds,) = load_dataset(
            gse_id, platform=platform, resources_dir=ALS_DIR / "resources"
        )
    except Exception as exc:
        print(f"  [warn] {gse_id}/{platform}: {exc}")
        return None
    cols = list(ds.X.columns)
    # symbol-based matching: external cohort columns may already be symbols
    # Try direct symbol match first
    sym_lookup = {c: i for i, c in enumerate(cols)}
    X = ds.X.values.astype(np.float32)
    y = ds.y.values.astype(int)
    matched = []
    matched_cols = []
    missing = []
    for sym in panel_symbols:
        if sym in sym_lookup:
            matched.append(sym)
            matched_cols.append(sym_lookup[sym])
        else:
            missing.append(sym)
    if missing:
        print(
            f"  [{gse_id}] missing {len(missing)} symbols: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    if not matched_cols:
        return None
    from utils import needs_log1p

    _sel = np.nan_to_num(X[:, matched_cols], nan=0.0)
    # GSE76220 is already log2-scale; a second log1p would null values <= -1.
    X_panel = np.log1p(_sel) if needs_log1p(gse_id) else _sel
    # Build full 25-column matrix with col-mean fill for missing
    result = np.zeros((X_panel.shape[0], len(panel_symbols)), dtype=np.float32)
    col_means = X_panel.mean(axis=0) if X_panel.shape[0] else None
    fill_value = (
        float(np.nan_to_num(col_means.mean(), nan=0.0))
        if col_means is not None
        else 0.0
    )
    j = 0
    for i, sym in enumerate(panel_symbols):
        if sym in sym_lookup:
            result[:, i] = np.nan_to_num(X_panel[:, j], nan=fill_value)
            j += 1
        else:
            result[:, i] = fill_value
    return result, y


def _parity_note(df: "pd.DataFrame", tol: float = 0.02) -> list[str]:
    """Describe LR-vs-LGBM agreement from the data, not from a fixed sentence.

    This used to be a static claim that parity held and the signal was therefore
    linearly extractable. It printed unchanged regardless of the numbers above
    it, and after the k=18 redefinition it sat directly beneath a table refuting
    it (SRP064478 differs by 0.125). A stale conclusion is worse than a stale
    number: it reads as an interpretation of whatever it happens to sit beside.

    Args:
        df: Combined results with cohort/panel/model/AUC columns.
        tol: Absolute AUC difference counted as parity.

    Returns:
        Lines describing how many comparisons agree within ``tol``.
    """
    # multicohort_baselines has no "panel" column; the extended run adds one.
    idx = [c for c in ("cohort", "panel") if c in df.columns]
    wide = df.pivot_table(index=idx, columns="model", values="AUC")
    lgbm_col = next((c for c in wide.columns if "LGBM" in c.upper()), None)
    lr_col = next((c for c in wide.columns if c.upper().startswith("LR")), None)
    if lgbm_col is None or lr_col is None:
        return ["  - LR/LGBM comparison unavailable (a model is missing)."]
    d = (wide[lgbm_col] - wide[lr_col]).dropna()
    if d.empty:
        return ["  - LR/LGBM comparison unavailable (no paired rows)."]
    n_par = int((d.abs() <= tol).sum())
    worst = d.abs().idxmax()
    out = [
        f"  - LR-L2 vs LGBM agree within {tol:.2f} AUC in {n_par}/{len(d)} "
        f"cohort-panel comparisons.",
        f"    Largest gap: {' / '.join(map(str, worst)) if isinstance(worst, tuple) else worst}"
        f" = {d[worst]:+.4f} ({'LGBM' if d[worst] > 0 else 'LR-L2'} higher).",
    ]
    if n_par == len(d):
        out.append(
            "    Parity throughout: the panel's discriminative signal is "
            "largely linearly extractable."
        )
    else:
        out.append(
            "    Parity does NOT hold throughout, so the signal is not purely "
            "linear; treat model-specific results as model-specific."
        )
    return out

def main() -> None:
    panel = pd.read_csv(_PANEL_CSV)
    panel_features = panel["feature"].tolist()
    panel_symbols = panel["symbol"].tolist()

    print("Loading GPL24676 training set ...")
    X_train_raw, feat_train, y_train, _, _ = _load_platform("GPL24676")
    X_train_log = np.log1p(X_train_raw)
    # CTD is discovery-side only; baselines use raw log1p (uniform with all cohorts)
    X_tr = _extract_panel(X_train_log, feat_train, panel_features)
    print(f"  shape: {X_tr.shape}")

    print("\nBaselines are fitted per cohort (standardisation depends on the")
    print("test cohort's expression scale -- see _score_cohort).")

    all_results = []

    # Cohort 1: GPL16791 (Ensembl ID match, full pipeline)
    print("\nGPL16791 ...")
    X_test_raw, feat_test, y_test, _, _ = _load_platform("GPL16791")
    X_test_log = np.log1p(X_test_raw)
    X_te = _extract_panel(X_test_log, feat_test, panel_features)
    all_results.extend(_score_cohort("GPL16791", X_tr, y_train, X_te, y_test))

    # External cohorts via symbol matching (no CTD residualisation possible)
    for gse, plat in [("GSE76220", None), ("GSE122649", None), ("GSE234297", None)]:
        print(f"\n{gse} ...")
        loaded = _load_external_by_symbol(gse, plat, panel_symbols)
        if loaded is None:
            print("  skipped")
            continue
        X_ext, y_ext = loaded
        all_results.extend(_score_cohort(gse, X_tr, y_train, X_ext, y_ext))

    df = pd.DataFrame(all_results)
    df.to_csv(SCRIPT_DIR / "multicohort_baselines.csv", index=False)
    print("\n" + df.to_string(index=False))

    lines = [
        "Multi-cohort LR-L2 vs LGBM baseline + calibration",
        "=" * 60,
        df.to_string(index=False),
        "",
        "Interpretation:",
        *_parity_note(df),
        "  - Brier and log-loss directly compare calibration: lower is better.",
        "  - Mean-neg score gives the average ALS probability assigned to control",
        "    samples; a value near the training prevalence indicates good calibration.",
    ]
    (SCRIPT_DIR / "multicohort_baselines.txt").write_text("\n".join(lines))
    print("\nSaved -> multicohort_baselines.{csv,txt}")


if __name__ == "__main__":
    main()
