"""Held-out test of the protein-coding replacement panel.

panel_variant_greedy.py accepts a swap only if it raises GSE122649 zero-shot AUC,
so GSE122649 cannot measure the result. This script scores the swapped panel, and
each accepted swap on its own, against the original 25-gene panel on the cohorts
the search never used:

  GPL16791 donor-disjoint  (NYGC HiSeq, the 518 samples of donors absent from the
                           GPL24676 training set)  primary test
  GPL16791 all             (n=636, reproduces the paper's 25-gene AUC)
  SRP064478 (n=15), GSE76220 (n=20)  secondary
  GSE122649 (n=38)         the selection cohort, reported for reference only

GPL16791 rows resample donors in the paired bootstrap; the others resample samples.

Rule, fixed before the run: a variant is non-inferior to the original panel if the
lower 95% bound of its paired donor-disjoint GPL16791 dAUC is above -MARGIN, the
margin of the greedy stop rule (greedy_significance_stop.py).

It also measures why the chosen replacement genes are not identifiable: for each
swapped slot it counts the protein-coding candidates whose replacement-screen AUC
(gene_replacement.py) is within MARGIN of the chosen gene, and reports the
training-set Spearman correlation between the panel gene and its replacement.

Model: the paper's zero-shot LightGBM (top-500 params, colsample rescaled to ~25 genes per tree,
utils.zero_shot_scale), trained on GPL24676 log1p with the genes each cohort carries.

Outputs:
  replacement_panel_heldout_statistics.txt
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ALS_DIR = Path(__file__).parents[1]
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(ALS_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

import pandas as pd  # noqa: E402
from panel_critical_eval import (  # noqa: E402
    _load_gse76220,
    _load_gse122649,
    _load_srp064478,
)
from panel_variant_cv import (  # noqa: E402
    _AUDIT_CSV,
    _REPL_CSV,
    _build_gene_info,
    _build_replacement_lookup,
)
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from combined_validation_roc import N_SEEDS as FIG3_SEEDS  # noqa: E402
from cross_cohort_substitution import _ensemble_scores  # noqa: E402
from utils import load_dataset, zero_shot_scale  # noqa: E402
from utils import CORE_PANEL_SIZE  # noqa: E402
from utils import selected_lgbm_params  # noqa: E402

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_SWAPS_JSON = SCRIPT_DIR / "panel_variant_greedy_swaps.json"
_REFERENCE_TXT = SCRIPT_DIR / "combined_validation_roc_statistics.txt"
_OUT = SCRIPT_DIR / "replacement_panel_heldout_statistics.txt"

MARGIN = 0.01  # greedy_significance_stop.py primary margin
N_BOOTSTRAP = 2_000
RANDOM_STATE = 42
PRIMARY = "GPL16791 donor-disjoint"
COHORTS = (PRIMARY, "GPL16791", "SRP064478", "GSE76220", "GSE122649")


def _paired_bootstrap(
    y: np.ndarray,
    s_base: np.ndarray,
    s_var: np.ndarray,
    clusters: list[np.ndarray],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """95% CIs of the variant AUC and of dAUC = variant - base, resampling clusters."""
    rng = np.random.default_rng(RANDOM_STATE)
    auc_var, delta = [], []
    for _ in range(N_BOOTSTRAP):
        idx = np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))])
        if len(np.unique(y[idx])) < 2:
            continue
        a_var = roc_auc_score(y[idx], s_var[idx])
        auc_var.append(a_var)
        delta.append(a_var - roc_auc_score(y[idx], s_base[idx]))
    ci = lambda a: tuple(np.percentile(a, [2.5, 97.5]))  # noqa: E731
    return ci(auc_var), ci(delta)


def _scores(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    cohort: str,
    params: dict,
) -> np.ndarray:
    Z_tr, Z_te = zero_shot_scale(X_train, X_test, cohort)
    # Figure 3's seed ensemble, so the original panel reproduces combined_validation_roc.py
    return _ensemble_scores(Z_tr, y_train, Z_te, params, seeds=tuple(range(FIG3_SEEDS)))


def _reference_auc25() -> dict[str, float]:
    """25-gene zero-shot AUCs quoted in the paper (combined_validation_roc.py)."""
    rows = re.findall(r"^(\S+)\s+\d+\s+([01]\.\d{4})", _REFERENCE_TXT.read_text(), re.M)
    return {c: float(a) for c, a in rows}


def main() -> None:
    params = dict(selected_lgbm_params(), n_jobs=-1, verbose=-1)
    panel = pd.read_csv(_PANEL_CSV)
    base_panel = list(zip(panel["symbol"], panel["feature"].str.split(".").str[0]))
    swaps: dict[str, list[str]] = json.loads(_SWAPS_JSON.read_text())

    def _swap(sw: dict[str, list[str]]) -> list[tuple[str, str]]:
        return [tuple(sw[s]) if s in sw else (s, b) for s, b in base_panel]

    variants = {f"original {CORE_PANEL_SIZE}": base_panel}
    variants |= {f"{s}->{r} only": _swap({s: [r, b]}) for s, (r, b) in swaps.items()}
    variants[f"all {len(swaps)} swaps"] = _swap(swaps)

    print("Loading GPL24676 (training) and GPL16791 ...")
    (ds24,) = load_dataset("GSE153960", platform="GPL24676", resources_dir=ALS_DIR / "resources")
    (ds16,) = load_dataset("GSE153960", platform="GPL16791", resources_dir=ALS_DIR / "resources")
    assert list(ds16.X.columns) == list(ds24.X.columns)
    tr_map = {f.split(".")[0]: j for j, f in enumerate(ds24.X.columns)}
    X24 = np.log1p(ds24.X.to_numpy(np.float32))
    y24 = ds24.y.to_numpy(int)
    donors24 = set(ds24.meta.loc[ds24.y.index, "subject id"])
    donors16 = ds16.meta.loc[ds16.y.index, "subject id"].to_numpy()
    disjoint = np.array([d not in donors24 for d in donors16])
    X16, y16 = np.log1p(ds16.X.to_numpy(np.float32)), ds16.y.to_numpy(int)

    print("Loading SRP064478, GSE76220, GSE122649 ...")
    # name -> (X, vocabulary -> column, y, match key 'base'/'symbol', scaling cohort, donors)
    cohorts: dict[str, tuple] = {
        PRIMARY: (X16[disjoint], tr_map, y16[disjoint], "base", "GPL16791", donors16[disjoint]),
        "GPL16791": (X16, tr_map, y16, "base", "GPL16791", donors16),
    }
    for name, loader, key in (
        ("SRP064478", _load_srp064478, "base"),
        ("GSE76220", _load_gse76220, "symbol"),
        ("GSE122649", _load_gse122649, "symbol"),
    ):
        X, vocab, y = loader()
        cohorts[name] = (X, {v: i for i, v in enumerate(vocab)}, y, key, name, np.arange(len(y)))

    results = []
    for cname in COHORTS:
        X, vmap, y, key, gse, donors = cohorts[cname]
        _, inv = np.unique(donors, return_inverse=True)
        clusters = [np.flatnonzero(inv == k) for k in range(inv.max() + 1)]
        scores, n_genes = {}, {}
        for vname, genes in variants.items():
            avail = [(s, b) for s, b in genes if (b if key == "base" else s) in vmap and b in tr_map]
            X_tr = X24[:, [tr_map[b] for _, b in avail]]
            X_te = X[:, [vmap[b if key == "base" else s] for s, b in avail]].astype(np.float32)
            scores[vname] = _scores(X_tr, y24, X_te, gse, params)
            n_genes[vname] = len(avail)
        for vname in variants:
            (lo, hi), (dlo, dhi) = _paired_bootstrap(y, scores[f"original {CORE_PANEL_SIZE}"], scores[vname], clusters)
            auc = roc_auc_score(y, scores[vname])
            results.append(dict(
                cohort=cname, variant=vname, genes=n_genes[vname], auc=auc, lo=lo, hi=hi,
                delta=auc - roc_auc_score(y, scores[f"original {CORE_PANEL_SIZE}"]), dlo=dlo, dhi=dhi,
            ))
        print(f"  {cname} done")
    res = pd.DataFrame(results)

    ref = _reference_auc25()
    base_rows = res[res.variant == f"original {CORE_PANEL_SIZE}"].set_index("cohort").auc
    mismatch = {c: (round(base_rows[c], 4), ref[c]) for c in ref if round(base_rows[c], 4) != ref[c]}

    print("Replacement-screen near-ties ...")
    lookup = _build_replacement_lookup(_REPL_CSV, _AUDIT_CSV)
    info = _build_gene_info(_AUDIT_CSV, {b for s in swaps for _, b in lookup.get(s, [])})
    sym_to_base = dict(base_panel)
    ties = []
    for s, (r, rb) in swaps.items():
        coding = [(a, b) for a, b in lookup[s] if info[b]["biotype"] == "protein-coding"]
        chosen = next(a for a, b in coding if b == rb)
        near = [info[b]["symbol"] or b for a, b in coding if chosen - a <= MARGIN]
        rho = spearmanr(X24[:, tr_map[sym_to_base[s]]], X24[:, tr_map[rb]])[0]
        ties.append((s, r, chosen, len(coding), near, rho))

    lines = [
        "Held-out test of the protein-coding replacement panel",
        "=" * 78,
        "Swaps (panel_variant_greedy.py, selected on GSE122649 zero-shot AUC): "
        + ", ".join(f"{s}->{r}" for s, (r, _) in swaps.items()),
        "Model: zero-shot LightGBM, top-500 params, colsample rescaled to ~25 genes per tree, trained on",
        "  GPL24676 log1p (n=874) with the genes each cohort carries; zero_shot_scale.",
        f"Paired bootstrap B={N_BOOTSTRAP}, seed {RANDOM_STATE}: GPL16791 rows resample donors, "
        "the others resample samples.",
        f"GPL16791 donor-disjoint: {int(disjoint.sum())} samples, {len(set(donors16[disjoint]))} donors "
        f"({int((~disjoint).sum())} samples of {len(set(donors16[~disjoint]))} shared donors removed).",
        f"Rule (fixed before the run): non-inferior if the {PRIMARY} dAUC lower bound > -{MARGIN}.",
        "GSE122649 is the selection cohort; its values are not evidence.",
        "",
        f"{'cohort':<25}{'variant':<22}{'genes':>6}{'AUC':>8}{'95% CI':>17}{'dAUC':>9}{'dAUC 95% CI':>20}",
        "-" * 107,
    ]
    for r in res.itertuples():
        lines.append(
            f"{r.cohort:<25}{r.variant:<22}{r.genes:>6}{r.auc:>8.4f}  [{r.lo:.3f}, {r.hi:.3f}]"
            f"{r.delta:>+9.4f}  [{r.dlo:+.4f}, {r.dhi:+.4f}]"
        )
    lines += ["", f"Verdict on {PRIMARY}:"]
    for r in res[(res.cohort == PRIMARY) & (res.variant != f"original {CORE_PANEL_SIZE}")].itertuples():
        verdict = "non-inferior" if r.dlo > -MARGIN else "not shown non-inferior"
        lines.append(f"  {r.variant:<22} dAUC {r.delta:+.4f} [{r.dlo:+.4f}, {r.dhi:+.4f}]  {verdict}")
    lines += [
        "",
        "Original-panel AUCs vs combined_validation_roc_statistics.txt: "
        + ("match" if not mismatch else f"MISMATCH {mismatch}"),
        "",
        f"Replacement-screen near-ties (protein-coding candidates within {MARGIN} AUC of the chosen gene):",
    ]
    for s, r, chosen, n_coding, near, rho in ties:
        lines.append(
            f"  {s}->{r}: screen AUC {chosen:.4f}, {len(near)} of {n_coding} coding candidates within "
            f"margin; Spearman rho({s},{r}) in GPL24676 = {rho:+.3f}"
        )
        lines.append(f"    {', '.join(near[:12])}{' ...' if len(near) > 12 else ''}")
    _OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
