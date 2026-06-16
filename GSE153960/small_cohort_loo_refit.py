"""Small-cohort within-cohort LOO with sane hyperparameters.

Reviewer M6: the within-cohort LOO numbers in the manuscript were computed
with n_estimators=1500, min_child_samples=105 (tuned for n=874). Applied to
n=20 (GSE76220) and n=38 (GSE122649) cohorts, this regime overfits — each
fold trains 1500 trees on 19/37 samples respectively.

This script refits within-cohort LOO with hyperparameters appropriate for
small n (n_estimators=100, min_child_samples=5) and reports whether the
headline LOO AUC numbers change.

The within-cohort LOO uses HGNC-symbol matching with column-mean fill
(same as multicohort_baselines.py).

Outputs: small_cohort_loo_refit.csv, .txt
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

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"


def _loo_auc(X: np.ndarray, y: np.ndarray, params: dict) -> float:
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score

    n = len(y)
    scores = np.zeros(n)
    for i in range(n):
        mask = np.arange(n) != i
        X_tr, y_tr = X[mask], y[mask]
        X_te = X[i:i + 1]
        clf = LGBMClassifier(**params)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            clf.fit(X_tr, y_tr)
            scores[i] = clf.predict_proba(X_te)[0, 1]
    return float(roc_auc_score(y, scores))


def _load_panel_by_symbol(gse_id: str, panel_symbols: list[str]):
    try:
        (ds,) = load_dataset(gse_id, platform=None,
                             resources_dir=ALS_DIR / "resources")
    except Exception as exc:
        print(f"  [warn] {gse_id}: {exc}")
        return None
    cols = list(ds.X.columns)
    sym_lookup = {c: i for i, c in enumerate(cols)}
    X = ds.X.values.astype(np.float32)
    y = ds.y.values.astype(int)
    matched_cols = [sym_lookup[s] for s in panel_symbols if s in sym_lookup]
    matched_syms = [s for s in panel_symbols if s in sym_lookup]
    X_p = np.log1p(np.nan_to_num(X[:, matched_cols], nan=0.0))
    # build full 25-col with col-mean fill
    out = np.zeros((X.shape[0], len(panel_symbols)), dtype=np.float32)
    fill = float(X_p.mean()) if X_p.size else 0.0
    j = 0
    for i, sym in enumerate(panel_symbols):
        if sym in sym_lookup:
            out[:, i] = np.nan_to_num(X_p[:, j], nan=fill)
            j += 1
        else:
            out[:, i] = fill
    print(f"  [{gse_id}] n={X.shape[0]} matched={len(matched_syms)}/{len(panel_symbols)}")
    return out, y


def main() -> None:
    panel = pd.read_csv(_PANEL_CSV)
    panel_symbols = panel["symbol"].tolist()

    # Two hyperparameter regimes
    paranoid = {  # tuned for n=874 — what the manuscript currently uses
        "n_estimators": 1500, "learning_rate": 0.04046, "num_leaves": 21,
        "min_child_samples": 1,  # paper says min_child_samples=1 for within-cohort LOO
        "colsample_bytree": 1.0, "subsample": 0.454,
        "reg_alpha": 0.0005, "reg_lambda": 0.0001,
        "objective": "binary", "random_state": 42, "verbosity": -1,
    }
    sane = {  # appropriate for small n
        "n_estimators": 100, "learning_rate": 0.05, "num_leaves": 15,
        "min_child_samples": 5, "colsample_bytree": 1.0, "subsample": 1.0,
        "reg_alpha": 0.0, "reg_lambda": 0.0,
        "objective": "binary", "random_state": 42, "verbosity": -1,
    }

    rows = []
    for gse in ["GSE76220", "GSE122649", "GSE234297"]:
        print(f"\n{gse}:")
        loaded = _load_panel_by_symbol(gse, panel_symbols)
        if loaded is None:
            continue
        X, y = loaded
        for label, params in [("paper_hyperparams_LOO", paranoid),
                              ("sane_small_n_LOO", sane)]:
            auc = _loo_auc(X, y, params)
            rows.append({
                "cohort": gse, "n": len(y), "n_pos": int(y.sum()),
                "regime": label,
                "n_estimators": params["n_estimators"],
                "min_child_samples": params["min_child_samples"],
                "AUC_LOO": round(auc, 4),
            })
            print(f"  {label:<30} AUC={auc:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(SCRIPT_DIR / "small_cohort_loo_refit.csv", index=False)
    print("\n" + df.to_string(index=False))

    lines = [
        "Small-cohort within-cohort LOO — hyperparameter sensitivity (M6)",
        "=" * 70,
        "Compares within-cohort LOO AUC under the n=874-tuned hyperparameter",
        "regime (currently in main text) vs a regime appropriate for small n.",
        "",
        df.to_string(index=False),
        "",
        "Interpretation:",
        "  - If both regimes give similar AUC, the LOO numbers are robust to",
        "    the hyperparameter choice and the reviewer's concern is bounded.",
        "  - If the sane-regime AUC is notably different, the original numbers",
        "    over- or under-fit and the main-text values need a caveat.",
    ]
    (SCRIPT_DIR / "small_cohort_loo_refit.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
