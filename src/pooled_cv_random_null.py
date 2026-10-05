"""Is the 25-gene panel's pooled cross-validated AUC panel-specific? (Step 105g)

The pooled CV (iterative_panel_elimination.pooled_oof) learns inside the nine external
datasets, so any gene set that tracks ALS in tissue can score well there. This compares
the 25-gene panel with random gene sets of the same size scored identically:

  * the nine trimming datasets merged (merge_cohorts), per-cohort control-referenced
    z-scores refitted per fold (ctrl_z), unit-equal weights, one grouped 10-fold split
    (cv_folds, first repeat) shared by the panel and every random set;
  * score = _W-weighted mean of the per-cohort out-of-fold AUCs;
  * random sets of 25 Ensembl genes drawn from (a) the clean candidate genes and
    (b) the SFM pool, each mapped into every cohort the way the panel is (Ensembl base
    ID, or HGNC symbol for the symbol-keyed cohorts); a cohort lacking a gene leaves
    it missing, as for the panel.

The 25-gene panel was chosen on GPL24676 alone, so this is a clean test. The critical
panel was chosen on these datasets and is not tested here.

Empirical p = (#random >= panel + 1) / (B + 1).
Output: pooled_cv_random_null_statistics.txt
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
ALS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ALS_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

B = 1000
RANDOM_STATE = 42


def main(b: int = B) -> None:
    warnings.filterwarnings("ignore")
    import iterative_panel_elimination as ipe
    from new_cohort_validation import _emtab, _gse26927_arms, _gse255683, _gse287256
    from random_panel_null_cohorts import _symbols
    from utils import load_dataset

    params = json.loads((SCRIPT_DIR / "lgbm_top500_best_params.json").read_text())
    panel = pd.read_csv(SCRIPT_DIR / "lgbm_core25_panel.csv")
    pbases = [f.split(".")[0] for f in panel.feature]
    clean = [n.split(".")[0] for n in (SCRIPT_DIR / "lgbm_prefilter_names.txt").read_text().splitlines()]
    pool = [f.split(".")[0] for f in pd.read_csv(SCRIPT_DIR / "lgbm_iter_final_shap_ranking.csv").feature]
    sym = _symbols(clean)
    sym.update(dict(zip(pbases, panel.symbol, strict=True)))

    # name -> (full matrix, {ensg base: column}, labels, donors)
    full: dict[str, tuple[np.ndarray, dict[str, int], np.ndarray, np.ndarray]] = {}
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
    emtab = ipe._emtab_donors()
    for c in (_gse255683(), _gse287256(), _gse26927_arms()["ALS"], _emtab("E-MTAB-8635"), _emtab("E-MTAB-2325")):
        add(c.scale_id, c.X, c.genes, c.y, c.by_symbol, emtab.get(c.scale_id))
    assert set(full) == set(ipe._W), set(full) ^ set(ipe._W)

    def score(bases: list[str], fold: np.ndarray | None = None) -> tuple[float, np.ndarray]:
        cohorts = {}
        for name in ipe._W:  # merge order = _W order
            x, vm, y, donor = full[name]
            av = [i for i, g in enumerate(bases) if g in vm]
            cohorts[name] = (x[:, [vm[bases[i]] for i in av]], av, y, donor)
        m = ipe.merge_cohorts(cohorts, len(bases))
        f = ipe.cv_folds(m, n_repeats=1) if fold is None else fold
        s = ipe.pooled_oof(m, list(range(len(bases))), f, params)
        from sklearn.metrics import roc_auc_score

        return sum(w * roc_auc_score(m.y[m.cid == c], s[m.cid == c]) for c, w in ipe._W.items()), f

    real, fold = score(pbases)  # donors/labels are gene-independent, so one split serves all sets
    print(f"panel W.mean {real:.4f}", flush=True)
    rng = np.random.default_rng(RANDOM_STATE)
    lines = [
        "Pooled 10-fold CV (nine trimming datasets) of the 25-gene panel vs random 25-gene sets",
        "=" * 82,
        f"Score = _W-weighted mean of per-cohort out-of-fold AUCs; one grouped 10-fold split; B={b} per pool.",
        "",
        f"25-gene panel W.mean = {real:.4f}",
    ]
    for label, src in (("broad", clean), ("SFM", pool)):
        cands = [g for g in src if g in full["GPL16791"][1]]
        null = np.array([score(list(rng.choice(cands, len(pbases), replace=False)), fold)[0] for _ in range(b)])
        ge = int((null >= real).sum())
        lines.append(
            f"  random {label:<5} (n={len(cands)} candidates): mean {null.mean():.4f}  "
            f"95th pct {np.percentile(null, 95):.4f}  #>= {ge}  p = {(ge + 1) / (b + 1):.4f}"
        )
        print(lines[-1], flush=True)
    (SCRIPT_DIR / "pooled_cv_random_null_statistics.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else B)
