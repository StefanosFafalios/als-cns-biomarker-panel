"""Nested donor-grouped CV of the panel-discovery procedure (selection optimism).

The published within-cohort AUCs (0.9470 for the 25-gene panel, donor-grouped) score
a panel that was *chosen* on all 874 samples, so every test fold influenced which
genes were kept. This script repeats the label-dependent discovery steps inside each
outer fold, on the training donors only, and scores the held-out donors:

1. exclude known-ALS and contamination genes (annotation-defined, label-free; the
   same set the iterative pipeline's flagging converges to);
2. SelectFromModel(LightGBM, top-500 params, threshold="mean");
3. LightGBM on the SFM set, SHAP ranking, top 500 (as ``lgbm_iterative_pipeline.py``);
4. LightGBM on those 500, SHAP re-ranking (as ``lgbm_top500_final_analysis.py``);
5. top 25 and top k_crit (the critical-panel size) by that ranking, fitted with ``PanelLGBMClassifier`` and
   ``adjust_colsample`` exactly as the published panel CV, scored on the held-out donors.

The published fixed panels are scored on the same folds, so ``fixed - nested`` is a
paired estimate of the selection optimism. Held fixed, and disclosed: the LightGBM
hyperparameters (tuned once on all samples) and the greedy 25 -> k_crit trim (it
selects on the external cohorts; "top k_crit by SHAP" stands in for it here).

Outputs ``nested_panel_cv.csv`` (one row per seed x fold),
``nested_panel_cv_selection.csv`` (per-gene re-selection frequency) and
``nested_panel_cv_statistics.txt``.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

SCRIPT_DIR = Path(__file__).parent
ALS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ALS_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from lgbm_iterative_pipeline import (  # noqa: E402
    KNOWN_ALS_FEATURES,
    _compute_shap,
    _resolve_contamination_pool,
)
from utils import (  # noqa: E402
    PanelLGBMClassifier,
    adjust_colsample,
    critical_panel_features,
    critical_panel_size,
    load_dataset,
)

from utils import selected_lgbm_params  # noqa: E402
PARAMS = selected_lgbm_params()
SEEDS = (42, 43, 44)
FOLDS = 5
TOP_SFM = 500
K_CRIT = critical_panel_size()
PANEL_SIZES = (25, K_CRIT)
OUT_CSV = SCRIPT_DIR / "nested_panel_cv.csv"
OUT_SEL = SCRIPT_DIR / "nested_panel_cv_selection.csv"
OUT_TXT = SCRIPT_DIR / "nested_panel_cv_statistics.txt"


def _fit_lgbm(X: np.ndarray, y: np.ndarray, params: dict):
    from utils import TunedLGBMClassifier as LGBMClassifier

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LGBMClassifier(**params).fit(X, y)


def _discover(X: np.ndarray, y: np.ndarray, names: list[str]) -> list[str]:
    """Steps 2-4 on training rows only; return all 500 names, best first."""
    from sklearn.feature_selection import SelectFromModel
    from utils import TunedLGBMClassifier as LGBMClassifier

    t0 = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sfm = SelectFromModel(LGBMClassifier(**PARAMS), threshold="mean").fit(X, y)
    sfm_idx = np.flatnonzero(sfm.get_support())
    print(f"    SFM kept {len(sfm_idx)} genes ({time.monotonic() - t0:.0f}s)", flush=True)

    reranked = _rank(X[:, sfm_idx], y, [names[i] for i in sfm_idx])
    print(f"    discovery done ({time.monotonic() - t0:.0f}s)", flush=True)
    return reranked


def _rank(X: np.ndarray, y: np.ndarray, names: list[str]) -> list[str]:
    """Steps 3-4 on a given gene pool: SHAP top 500, then SHAP re-rank; best first."""
    _, ranked, _ = _compute_shap(_fit_lgbm(X, y, PARAMS), X, names)
    top = ranked[:TOP_SFM]
    col = {n: i for i, n in enumerate(names)}
    X_top = X[:, [col[n] for n in top]]
    model = _fit_lgbm(X_top, y, adjust_colsample(PARAMS, len(top)))
    _, reranked, _ = _compute_shap(model, X_top, top)
    return reranked


def _score(X_tr, y_tr, X_te, y_te) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = PanelLGBMClassifier(**adjust_colsample(PARAMS, X_tr.shape[1])).fit(X_tr, y_tr)
    return float(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))


def main(max_features: int | None = None, seeds: tuple[int, ...] = SEEDS) -> None:
    """Run the nested CV and write the three outputs.

    Args:
        max_features: Keep only the first N candidate genes (smoke tests; outputs
            are then written with a ``_smoke`` suffix).
        seeds: Outer-fold shuffling seeds.
    """
    (ds,) = load_dataset("GSE153960", platform="GPL24676", resources_dir=ALS_DIR / "resources")
    y = ds.y.to_numpy().astype(int)
    groups = ds.meta.loc[ds.y.index, "subject id"].to_numpy()
    all_names = (SCRIPT_DIR / "lgbm_prefilter_names.txt").read_text().splitlines()
    X_all = np.load(SCRIPT_DIR / "lgbm_prefilter_X.npy", mmap_mode="r")
    assert X_all.shape == (len(y), len(all_names)), X_all.shape

    published = {
        25: pd.read_csv(SCRIPT_DIR / "lgbm_core25_panel.csv").feature.tolist(),
        K_CRIT: critical_panel_features(),
    }
    pool, _ = _resolve_contamination_pool({n.split(".")[0]: n for n in all_names})
    discard = KNOWN_ALS_FEATURES | pool
    keep = [i for i, n in enumerate(all_names) if n not in discard]
    n_discarded = len(all_names) - len(keep)
    if max_features:  # smoke test: a slice, plus the published genes so both arms run
        must = set(published[25]) | set(published[K_CRIT])
        keep = sorted(set(keep[:max_features]) | {i for i in keep if all_names[i] in must})
    names = [all_names[i] for i in keep]
    X = np.asarray(X_all[:, keep], dtype=np.float32)
    col = {n: i for i, n in enumerate(names)}
    print(f"n={len(y)} donors={len(set(groups))} candidate genes={len(names)} "
          f"(known-ALS/contamination discarded: {n_discarded})")

    rows, picks = [], {k: [] for k in PANEL_SIZES}
    for seed in seeds:
        for fold, (tr, te) in enumerate(
            StratifiedGroupKFold(FOLDS, shuffle=True, random_state=seed).split(X, y, groups)
        ):
            assert not set(groups[tr]) & set(groups[te])
            print(f"seed {seed} fold {fold}: {len(tr)} train / {len(te)} test", flush=True)
            ranked = _discover(X[tr], y[tr], names)
            row = dict(seed=seed, fold=fold, n_test=len(te))
            for k in PANEL_SIZES:
                nested = ranked[:k]
                picks[k].append(nested)
                idx = [col[n] for n in nested]
                row[f"nested_{k}"] = _score(X[np.ix_(tr, idx)], y[tr], X[np.ix_(te, idx)], y[te])
                fixed = [col[n] for n in published[k] if n in col]
                row[f"fixed_{k}"] = _score(
                    X[np.ix_(tr, fixed)], y[tr], X[np.ix_(te, fixed)], y[te]
                )
                row[f"overlap_{k}"] = len(set(nested) & set(published[k]))
            print(f"    AUC nested25 {row['nested_25']:.4f} fixed25 {row['fixed_25']:.4f} "
                  f"nested{K_CRIT} {row[f'nested_{K_CRIT}']:.4f} fixed{K_CRIT} {row[f'fixed_{K_CRIT}']:.4f}",
                  flush=True)
            rows.append(row)

    suffix = "_smoke" if max_features else ""
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV.with_stem(OUT_CSV.stem + suffix), index=False)

    n_fits = len(rows)
    sel = pd.DataFrame(
        [
            dict(panel_size=k, feature=g, frequency=sum(g in p for p in picks[k]) / n_fits,
                 in_published=g in published[k])
            for k in PANEL_SIZES
            for g in sorted({g for p in picks[k] for g in p} | set(published[k]))
        ]
    ).sort_values(["panel_size", "frequency"], ascending=[True, False])
    sel.to_csv(OUT_SEL.with_stem(OUT_SEL.stem + suffix), index=False)

    lines = [
        "Nested donor-grouped CV of the panel-discovery procedure",
        "=" * 60,
        f"GPL24676: {len(y)} samples, {len(set(groups))} donors; {len(names)} candidate genes; "
        f"StratifiedGroupKFold({FOLDS}) x seeds {list(seeds)} = {n_fits} outer folds",
        "Inside each fold: SFM -> SHAP top 500 -> SHAP re-rank -> top k (training donors only).",
        f"Fixed: LightGBM hyperparameters (tuned on all samples); 25->{K_CRIT} greedy trim "
        f"(top {K_CRIT} by SHAP stands in).",
        "",
    ]
    for k in PANEL_SIZES:
        nested, fixed = df[f"nested_{k}"], df[f"fixed_{k}"]
        gap = fixed - nested
        freq = sel[(sel.panel_size == k) & sel.in_published].frequency
        lines += [
            f"{k}-gene panel",
            f"  nested (re-selected per fold): AUC {nested.mean():.4f} (SD {nested.std():.4f})",
            f"  fixed published panel, same folds: AUC {fixed.mean():.4f} (SD {fixed.std():.4f})",
            f"  selection optimism (fixed - nested): {gap.mean():+.4f} "
            f"(range {gap.min():+.4f} to {gap.max():+.4f})",
            f"  overlap with published panel: mean {df[f'overlap_{k}'].mean():.1f}/{k} per fold; "
            f"published genes re-selected in >=50% of folds: {int((freq >= 0.5).sum())}/{len(freq)}",
            "",
        ]
    text = "\n".join(lines)
    OUT_TXT.with_stem(OUT_TXT.stem + suffix).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(max_features=int(sys.argv[1]), seeds=(42,))
    else:
        main()
