"""Decompose the nested-CV selection optimism (``nested_panel_cv.py``) by discovery step.

On the same donor-grouped outer folds as ``nested_panel_cv.py``, five arms score a
k-gene LightGBM panel on the held-out donors:

* ``fixed``: the published panel (chosen on all 874 samples);
* ``full_data``: this script's discovery approximation run once on all samples. Its
  gap to ``fixed`` and its overlap with the published panel measure how faithful the
  approximation is, since both arms see every label;
* ``pool_fixed``: the published 3,704-gene pool (iterative decontamination on all
  samples) is kept, and only the SHAP ranking and top-k are redone on training donors;
* ``nested``: every label-dependent step on training donors (as ``nested_panel_cv.py``);
* ``random_pool`` / ``random_all``: random k-gene panels drawn from the 3,704 pool or
  from all candidate genes (the baseline an arbitrary panel reaches).

Differences between adjacent arms attribute the ``fixed - nested`` gap:
``fixed - full_data`` (approximation mismatch), ``full_data - pool_fixed`` (SHAP
ranking and top-k on test labels), ``pool_fixed - nested`` (the pool was chosen on
test labels, plus iterative vs single-pass SFM).

Outputs ``nested_cv_decomposition.csv`` (one row per seed x fold) and
``nested_cv_decomposition_statistics.txt``. Smoke: ``python nested_cv_decomposition.py 300``.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from nested_panel_cv import (
    CORE_PANEL_SIZE,
    ALS_DIR,
    FOLDS,
    K_CRIT,
    known_als_features,
    PANEL_SIZES,
    SCRIPT_DIR,
    SEEDS,
    _discover,
    _rank,
    _resolve_contamination_pool,
    _score,
    critical_panel_features,
    load_dataset,
)
from sklearn.model_selection import StratifiedGroupKFold

N_RANDOM = 20  # random panels per fold and pool
ARMS = ("fixed", "full_data", "pool_fixed", "nested", "random_pool", "random_all")
OUT_CSV = SCRIPT_DIR / "nested_cv_decomposition.csv"
OUT_TXT = SCRIPT_DIR / "nested_cv_decomposition_statistics.txt"


def main(
    max_features: int | None = None, seeds: tuple[int, ...] = SEEDS, n_random: int = N_RANDOM
) -> None:
    """Run all arms on the shared outer folds and write the two outputs.

    Args:
        max_features: Keep only the first N candidate genes (smoke test; outputs get
            a ``_smoke`` suffix).
        seeds: Outer-fold shuffling seeds (same as ``nested_panel_cv.py``).
        n_random: Random panels per fold and pool.
    """
    (ds,) = load_dataset("GSE153960", platform="GPL24676", resources_dir=ALS_DIR / "resources")
    y = ds.y.to_numpy().astype(int)
    groups = ds.meta.loc[ds.y.index, "subject id"].to_numpy()
    all_names = (SCRIPT_DIR / "lgbm_prefilter_names.txt").read_text().splitlines()
    X_all = np.load(SCRIPT_DIR / "lgbm_prefilter_X.npy", mmap_mode="r")
    assert X_all.shape == (len(y), len(all_names)), X_all.shape

    published = {
        CORE_PANEL_SIZE: pd.read_csv(SCRIPT_DIR / "lgbm_core25_panel.csv").feature.tolist(),
        K_CRIT: critical_panel_features(),
    }
    pub_pool = pd.read_csv(SCRIPT_DIR / "lgbm_iter_final_shap_ranking.csv").feature.tolist()
    pool, _ = _resolve_contamination_pool({n.split(".")[0]: n for n in all_names})
    discard = known_als_features(all_names) | pool
    keep = [i for i, n in enumerate(all_names) if n not in discard]
    if max_features:  # smoke: a slice plus the published genes and pool
        must = set(published[CORE_PANEL_SIZE]) | set(published[K_CRIT]) | set(pub_pool[:600])
        keep = sorted(set(keep[:max_features]) | {i for i in keep if all_names[i] in must})
    names = [all_names[i] for i in keep]
    X = np.asarray(X_all[:, keep], dtype=np.float32)
    col = {n: i for i, n in enumerate(names)}
    pool_names = [n for n in pub_pool if n in col]
    pool_idx = [col[n] for n in pool_names]
    print(f"n={len(y)} donors={len(set(groups))} candidate genes={len(names)} "
          f"published pool {len(pub_pool)} ({len(pool_names)} among candidates)", flush=True)

    print("full-data discovery (all samples)", flush=True)
    full_ranked = _discover(X, y, names)
    rng = np.random.default_rng(0)

    rows = []
    for seed in seeds:
        for fold, (tr, te) in enumerate(
            StratifiedGroupKFold(FOLDS, shuffle=True, random_state=seed).split(X, y, groups)
        ):
            assert not set(groups[tr]) & set(groups[te])
            print(f"seed {seed} fold {fold}: {len(tr)} train / {len(te)} test", flush=True)
            nested_ranked = _discover(X[tr], y[tr], names)
            pool_ranked = _rank(X[np.ix_(tr, pool_idx)], y[tr], pool_names)

            def auc(panel: list[str], tr=tr, te=te) -> float:
                idx = [col[n] for n in panel if n in col]
                return _score(X[np.ix_(tr, idx)], y[tr], X[np.ix_(te, idx)], y[te])

            row = dict(seed=seed, fold=fold, n_test=len(te))
            for k in PANEL_SIZES:
                row[f"fixed_{k}"] = auc(published[k])
                row[f"full_data_{k}"] = auc(full_ranked[:k])
                row[f"pool_fixed_{k}"] = auc(pool_ranked[:k])
                row[f"nested_{k}"] = auc(nested_ranked[:k])
                row[f"random_pool_{k}"] = np.mean(
                    [auc(list(rng.choice(pool_names, k, replace=False))) for _ in range(n_random)]
                )
                row[f"random_all_{k}"] = np.mean(
                    [auc(list(rng.choice(names, k, replace=False))) for _ in range(n_random)]
                )
                row[f"overlap_pool_fixed_{k}"] = len(set(pool_ranked[:k]) & set(published[k]))
            print("    " + " ".join(f"{a} {row[f'{a}_{CORE_PANEL_SIZE}']:.4f}" for a in ARMS), flush=True)
            rows.append(row)

    suffix = "_smoke" if max_features else ""
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV.with_stem(OUT_CSV.stem + suffix), index=False)

    lines = [
        "Decomposition of the nested-CV selection optimism",
        "=" * 60,
        f"GPL24676: {len(y)} samples, {len(set(groups))} donors; {len(names)} candidate genes; "
        f"StratifiedGroupKFold({FOLDS}) x seeds {list(seeds)} = {len(df)} outer folds; "
        f"random arms: {n_random} panels per fold",
        f"Published pool: {len(pub_pool)} genes ({len(pool_names)} among candidates)",
        "",
    ]
    for k in PANEL_SIZES:
        m = {a: df[f"{a}_{k}"].mean() for a in ARMS}
        full_overlap = len(set(full_ranked[:k]) & set(published[k]))
        lines += [f"{k}-gene panel: mean AUC over folds (SD)"]
        lines += [f"  {a:<12} {m[a]:.4f} ({df[f'{a}_{k}'].std():.4f})" for a in ARMS]
        lines += [
            f"  full-data panel overlap with published: {full_overlap}/{k}; "
            f"pool_fixed overlap per fold: {df[f'overlap_pool_fixed_{k}'].mean():.1f}/{k}",
            f"  total gap fixed - nested:            {m['fixed'] - m['nested']:+.4f}",
            f"    fixed - full_data (approximation): {m['fixed'] - m['full_data']:+.4f}",
            f"    full_data - pool_fixed (ranking):  {m['full_data'] - m['pool_fixed']:+.4f}",
            f"    pool_fixed - nested (pool):        {m['pool_fixed'] - m['nested']:+.4f}",
            "",
        ]
    text = "\n".join(lines)
    OUT_TXT.with_stem(OUT_TXT.stem + suffix).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(max_features=int(sys.argv[1]), seeds=(42,), n_random=2)
    else:
        main()
