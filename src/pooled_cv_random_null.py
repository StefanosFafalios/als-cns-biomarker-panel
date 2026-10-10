"""Does the critical panel beat random pre-filtered genes under pooled cross-validation? (Step 105i)

Compares the critical panel with random gene sets of the same size drawn from the
3,001-gene pre-filtered (SFM) pool, all scored identically:

  * the nine trimming datasets merged (merge_cohorts), per-cohort control-referenced
    z-scores refitted per fold (ctrl_z), unit-equal weights, one grouped 10-fold split
    shared by the panel and every random set;
  * the split is cv_folds' sixth repeat (seed 47); the trimming used repeats 1-5
    (seeds 42-46), so the panel is not scored on the splits that selected it;
  * score = _W-weighted mean of the per-cohort out-of-fold AUCs;
  * each random gene is drawn from the pool genes measured in exactly the same datasets as
    the panel gene it stands in for (matched_random_set), so a random set has the panel's
    platform gaps, no more and no fewer. With every panel gene measured everywhere this is
    a draw from the genes measured in all nine datasets (the rule before 2026-10-08, when
    ENSG00000280893, which has no HGNC symbol, joined the panel and is absent from the arrays).

The critical panel was chosen on these datasets, so the comparison favours it: a
fresh split removes split-selection optimism, not gene-selection optimism.

Empirical p = (#random >= panel + 1) / (B + 1).
Output: pooled_cv_random_null_statistics.txt
"""

from __future__ import annotations

import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
ALS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ALS_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

B = 1000
RANDOM_STATE = 42
MIN_STRATUM = 20  # pool genes needed per measurement pattern for a meaningful draw


Full = dict[str, tuple[np.ndarray, dict[str, int], np.ndarray, np.ndarray]]


def load_full() -> tuple[Full, dict[str, str]]:
    """The nine trimming datasets over all their genes, and an Ensembl-base -> symbol map.

    Returns:
        ({name: (matrix, {ensembl base: column}, labels, donors)}, symbols). Symbol-keyed
        datasets are mapped to Ensembl bases through the symbols, as the panel is.
    """
    import iterative_panel_elimination as ipe
    from new_cohort_validation import _emtab, _gse26927_arms, _gse255683, _gse287256
    from random_panel_null_cohorts import _symbols
    from utils import load_dataset

    panel = pd.read_csv(SCRIPT_DIR / "lgbm_core25_panel.csv")
    clean = [n.split(".")[0] for n in (SCRIPT_DIR / "lgbm_prefilter_names.txt").read_text().splitlines()]
    sym = _symbols(clean)
    sym.update(dict(zip([f.split(".")[0] for f in panel.feature], panel.symbol, strict=True)))

    full: Full = {}
    (d16,) = load_dataset("GSE153960", platform="GPL16791", resources_dir=ALS_DIR / "resources")
    sid = d16.meta.loc[d16.y.index, "subject id"].astype(str).to_numpy()
    keep = ~np.isin(sid, ipe._subject_ids("GPL24676"))
    full["GPL16791"] = (
        np.log1p(d16.X.to_numpy(dtype=np.float32)[keep]),
        {str(c).split(".")[0]: j for j, c in enumerate(d16.X.columns)},
        d16.y.to_numpy().astype(int)[keep], sid[keep],
    )
    del d16

    def add(name: str, x: np.ndarray, vocab: list[str], y: np.ndarray, by_symbol: bool, donor=None) -> None:
        vm: dict[str, int] = {}
        for j, v in enumerate(vocab):
            vm.setdefault(str(v) if by_symbol else str(v).split(".")[0], j)
        if by_symbol:
            vm = {g: vm[s] for g, s in sym.items() if s in vm}
        y = np.asarray(y).astype(int)
        full[name] = (x.astype(np.float32), vm, y,
                      donor if donor is not None else np.array([f"{name}_{i}" for i in range(len(y))]))

    add("GSE76220", *ipe._load_gse76220(), by_symbol=True)
    add("GSE122649", *ipe._load_gse122649(), by_symbol=True)
    add("SRP064478", *ipe._load_srp064478(), by_symbol=False)
    for c in (_gse255683(), _gse287256(), _gse26927_arms()["ALS"], _emtab("E-MTAB-8635"), _emtab("E-MTAB-2325")):
        add(c.scale_id, c.X, c.genes, c.y, c.by_symbol, ipe._emtab_donors(c))
    assert set(full) == set(ipe._W), set(full) ^ set(ipe._W)
    return full, sym


def pooled_score(full: Full, bases: list[str], params: dict, fold: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Weighted mean pooled-CV AUC of *bases* and the fold ids used (default: the unseen split, seed 47)."""
    import iterative_panel_elimination as ipe
    from sklearn.metrics import roc_auc_score

    cohorts = {}
    for name in ipe._W:  # merge order = _W order
        x, vm, y, donor = full[name]
        av = [i for i, g in enumerate(bases) if g in vm]
        cohorts[name] = (x[:, [vm[bases[i]] for i in av]], av, y, donor)
    m = ipe.merge_cohorts(cohorts, len(bases))
    f = ipe.cv_folds(m, n_repeats=ipe.N_REPEATS + 1)[-1:] if fold is None else fold
    s = ipe.pooled_oof(m, list(range(len(bases))), f, params)
    return sum(w * roc_auc_score(m.y[m.cid == c], s[m.cid == c]) for c, w in ipe._W.items()), f


def matched_random_set(rng: np.random.Generator, strata: dict[tuple, list[str]], need: dict[tuple, int]) -> list[str]:
    """One random gene set: *need[p]* genes drawn without replacement from *strata[p]* per pattern.

    Args:
        rng: Random generator (patterns are drawn in *need* order, so one pattern is one choice call).
        strata: Measurement pattern (one bool per dataset) -> pool genes with that pattern.
        need: Pattern -> number of panel genes with that pattern.
    """
    return [g for p, n in need.items() for g in rng.choice(strata[p], n, replace=False)]


def main(b: int = B) -> None:
    warnings.filterwarnings("ignore")
    import iterative_panel_elimination as ipe
    from utils import critical_panel_features

    from utils import selected_lgbm_params  # noqa: E402
    params = selected_lgbm_params()
    pbases = [f.split(".")[0] for f in critical_panel_features()]
    pool = [f.split(".")[0] for f in pd.read_csv(SCRIPT_DIR / "lgbm_iter_final_shap_ranking.csv").feature]
    full, _ = load_full()

    def score(bases: list[str], fold: np.ndarray | None = None) -> tuple[float, np.ndarray]:
        return pooled_score(full, bases, params, fold)

    def pattern(g: str) -> tuple[bool, ...]:
        return tuple(g in full[n][1] for n in full)

    need = dict(Counter(pattern(g) for g in pbases))
    strata: dict[tuple, list[str]] = {}
    for g in pool:
        strata.setdefault(pattern(g), []).append(g)
    for p, n in need.items():
        assert len(strata.get(p, [])) >= max(n, MIN_STRATUM), f"too few pool genes measured like {p}"
    real, fold = score(pbases)  # donors/labels are gene-independent, so one split serves all sets
    print(f"critical panel W.mean {real:.4f}", flush=True)
    rng = np.random.default_rng(RANDOM_STATE)
    k = len(pbases)
    null = np.array([score(matched_random_set(rng, strata, need), fold)[0] for _ in range(b)])
    names = list(full)
    gaps = "; ".join(f"{n} gene(s) missing from {', '.join(c for c, ok in zip(names, p) if not ok)}: "
                     f"{len(strata[p])} pool genes alike" for p, n in need.items() if not all(p))
    ge = int((null >= real).sum())
    lines = [
        f"Pooled 10-fold CV (nine trimming datasets) of the {k}-gene critical panel vs random {k}-gene sets",
        "=" * 88,
        f"Score = _W-weighted mean of per-cohort out-of-fold AUCs; one grouped 10-fold split (seed "
        f"{ipe.RANDOM_STATE + ipe.N_REPEATS}, unseen by the trimming); B={b}.",
        "Random genes: SFM pool genes measured in the same datasets as each panel gene. In-sample: the panel was",
        "chosen on these data." + (f" Platform gaps matched: {gaps}." if gaps else ""),
        "",
        f"Critical panel W.mean = {real:.4f}",
        f"  random SFM ({len(strata.get(tuple([True] * len(full)), []))} of {len(pool)} pool genes measured everywhere): mean {null.mean():.4f}  "
        f"95th pct {np.percentile(null, 95):.4f}  #>= {ge}  p = {(ge + 1) / (b + 1):.4f}",
    ]
    (SCRIPT_DIR / "pooled_cv_random_null_statistics.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else B)
