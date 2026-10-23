# ruff: noqa: E402
"""Random-panel null per external cohort, with GPL16791 split by donor overlap.

``random_panel_null.py`` tests the 25-gene panel against random size-matched panels
on GPL16791 only. GPL16791 shares 38 of its 159 donors with the GPL24676 training
cohort, and the nested CV (``nested_cv_decomposition.py``) showed that in-cohort
SHAP ranking favours genes that do not carry over to new donors. This script repeats
the null where no training donor can reach the test set:

* GPL16791, all samples (reproduces the published null under this pipeline);
* GPL16791, donor-disjoint (the 118 samples of shared donors removed);
* GSE76220, GSE122649, SRP064478 (separate cohorts).

Per cohort, the real panel is its genes available in that cohort, and each random
panel has the same size, drawn from (a) the 47,822 clean candidate genes or (b) the
3,704-gene SFM pool, restricted to genes available in that cohort. Every panel, real or
random, is scored with the manuscript's zero-shot model: the top-500 LightGBM parameters
with ``colsample_bytree`` set by ``utils.PanelLGBMClassifier`` (as in ``panel_critical_eval.py`` and
``external_validation_gpl16791.py``), trained on all GPL24676 samples (raw log1p), with
``zero_shot_scale``. The panel defaults to the published 25 genes; a second argument names
another panel CSV (``feature``/``symbol`` columns), e.g. ``lgbm_core25_panel_annotated.csv``. The critical panel
was selected on these cohorts, so a null on them is not a clean test.

Empirical p = (#null AUC >= real AUC + 1) / (B + 1).

Outputs ``random_panel_null_cohorts_statistics.txt``, the null AUCs
(``random_panel_null_cohorts_distributions.csv``), the bar plot
``random_panel_null_cohorts.png`` (published panel only) and ``ensg_symbol_cache.json``
(MyGeneInfo Ensembl -> symbol map, reused on reruns).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ALS_DIR = Path(__file__).parents[1]
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(ALS_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from adaptive_panel_validation import _resolve_mygeneinfo
from panel_critical_eval import _load_gse76220, _load_gse122649, _load_srp064478
from utils import load_dataset, zero_shot_scale
from utils import selected_lgbm_params  # noqa: E402

B = 1000
# the manuscript's zero-shot model; one thread per fit, parallel across random panels
_ZS_PARAMS = dict(
    selected_lgbm_params(),
    verbose=-1, n_jobs=1,
)
RANDOM_STATE = 42
_SYMBOL_CACHE = SCRIPT_DIR / "ensg_symbol_cache.json"
OUT_TXT = SCRIPT_DIR / "random_panel_null_cohorts_statistics.txt"
OUT_CSV = SCRIPT_DIR / "random_panel_null_cohorts_distributions.csv"
OUT_PNG = SCRIPT_DIR / "random_panel_null_cohorts.png"
# figure: donor-disjoint GPL16791 and the separate cohorts, larger first
FIG_COHORTS = ("GPL16791 donor-disjoint", "GSE122649", "SRP064478", "GSE76220")


def _auc(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, yte: np.ndarray, gse: str) -> float:
    from sklearn.metrics import roc_auc_score
    from utils import PanelLGBMClassifier as LGBMClassifier

    ztr, zte = zero_shot_scale(xtr, xte, gse)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        clf = LGBMClassifier(**_ZS_PARAMS).fit(ztr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(zte)[:, 1]))


def _score_panels(xtr, ytr, xte, yte, gse: str, panels: list[np.ndarray]) -> list[float]:
    """AUC per panel; each panel indexes the shared columns of ``xtr`` and ``xte``."""
    return [_auc(xtr[:, p], ytr, xte[:, p], yte, gse) for p in panels]


def _null(xtr_all, tr_col, xte, vmap, ytr, yte, gse, cands, draws, n_chunks=48) -> np.ndarray:
    """Score the random panels in parallel chunks that carry only their own columns."""
    from joblib import Parallel, delayed

    jobs = []
    for chunk in np.array_split(np.arange(len(draws)), n_chunks):
        genes = sorted({cands[i] for d in chunk for i in draws[d]})
        pos = {g: j for j, g in enumerate(genes)}
        panels = [np.array([pos[cands[i]] for i in draws[d]]) for d in chunk]
        jobs.append(delayed(_score_panels)(
            np.ascontiguousarray(xtr_all[:, [tr_col[g] for g in genes]]), ytr,
            np.ascontiguousarray(xte[:, [vmap[g] for g in genes]]), yte, gse, panels,
        ))
    return np.array([a for part in Parallel(n_jobs=-1)(jobs) for a in part])


def _symbols(bases: list[str]) -> dict[str, str]:
    cache = json.loads(_SYMBOL_CACHE.read_text()) if _SYMBOL_CACHE.exists() else {}
    missing = [b for b in bases if b not in cache]
    if missing:
        found = _resolve_mygeneinfo(missing)
        cache.update({b: found.get(b, "") for b in missing})
        _SYMBOL_CACHE.write_text(json.dumps(cache))
    return {b: cache[b] for b in bases if cache.get(b)}


def main(b: int = B, panel_csv: str = "lgbm_core25_panel.csv") -> None:
    """Run the per-cohort nulls and write the statistics file.

    Args:
        b: Random panels per cohort and pool.
        panel_csv: Panel file in the script directory.
    """
    (tr,) = load_dataset("GSE153960", platform="GPL24676", resources_dir=ALS_DIR / "resources")
    (d16,) = load_dataset("GSE153960", platform="GPL16791", resources_dir=ALS_DIR / "resources")
    assert list(d16.X.columns) == list(tr.X.columns)
    ytr = tr.y.to_numpy().astype(int)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        xtr_all = np.log1p(tr.X.to_numpy(dtype=np.float32))
        x16 = np.log1p(d16.X.to_numpy(dtype=np.float32))
    y16 = d16.y.to_numpy().astype(int)
    tr_col = {str(c).split(".")[0]: j for j, c in enumerate(tr.X.columns)}

    subj_tr = set(tr.meta.loc[tr.y.index, "subject id"])
    subj16 = d16.meta.loc[d16.y.index, "subject id"].to_numpy()
    disjoint = np.array([s not in subj_tr for s in subj16])

    clean = [n.split(".")[0] for n in (SCRIPT_DIR / "lgbm_prefilter_names.txt").read_text().splitlines()]
    pool = [
        f.split(".")[0]
        for f in pd.read_csv(SCRIPT_DIR / "lgbm_iter_final_shap_ranking.csv").feature
    ]
    panel_df = pd.read_csv(SCRIPT_DIR / panel_csv)
    panel = [f.split(".")[0] for f in panel_df.feature]
    sym = _symbols(clean)
    sym_col = next(c for c in panel_df.columns if "symbol" in c.lower())
    sym.update(dict(zip(panel, panel_df[sym_col], strict=True)))

    # cohort -> (test matrix, labels, {ensg base: test column}, scaling id)
    cohorts: dict[str, tuple[np.ndarray, np.ndarray, dict[str, int], str]] = {}
    cohorts["GPL16791 all"] = (x16, y16, tr_col, "GPL16791")
    cohorts["GPL16791 donor-disjoint"] = (x16[disjoint], y16[disjoint], tr_col, "GPL16791")
    for name, loader, by_symbol in (
        ("GSE76220", _load_gse76220, True),
        ("GSE122649", _load_gse122649, True),
        ("SRP064478", _load_srp064478, False),
    ):
        x, vocab, y = loader()
        vmap: dict[str, int] = {}
        for j, v in enumerate(vocab):
            vmap.setdefault(str(v) if by_symbol else str(v).split(".")[0], j)
        if by_symbol:
            vmap = {g: vmap[s] for g, s in sym.items() if s in vmap}
        cohorts[name] = (x, y, vmap, name)

    rng = np.random.default_rng(RANDOM_STATE)
    lines = [
        f"Random-panel null per cohort ({panel_csv}, {len(panel)} genes), GPL16791 split by donor overlap",
        "=" * 72,
        f"Train GPL24676 n={len(ytr)} (raw log1p, all samples); manuscript zero-shot LightGBM "
        f"(top-500 params, colsample rescaled to ~25 genes per tree); zero_shot_scale; B={b} per cohort and pool.",
        f"GPL16791: {int((~disjoint).sum())} samples from "
        f"{len(set(subj16[~disjoint]))} shared donors removed in the donor-disjoint row.",
        "",
        f"{'cohort':<24} {'n':>4} {'ALS/ctl':>8} {'genes':>5} {'real':>7} | "
        f"{'null':<5} {'mean':>6} {'95pct':>6} {'#>=':>5} {'p':>7} {'z':>5}",
    ]
    res: dict[str, dict] = {}
    for name, (xte, yte, vmap, gse) in cohorts.items():
        avail = [g for g in panel if g in vmap and g in tr_col]
        k = len(avail)

        def score(genes: list[str], xte=xte, yte=yte, vmap=vmap, gse=gse) -> float:
            return _auc(
                xtr_all[:, [tr_col[g] for g in genes]], ytr,
                xte[:, [vmap[g] for g in genes]], yte, gse,
            )

        real = score(avail)
        res[name] = dict(real=real, n=len(yte), k=k, null={})
        head = f"{name:<24} {len(yte):>4} {f'{yte.sum()}/{len(yte) - yte.sum()}':>8} {k:>5} {real:>7.4f}"
        for label, src in (("broad", clean), ("SFM", pool)):
            cands = [g for g in src if g in vmap and g in tr_col]
            draws = [rng.choice(len(cands), k, replace=False) for _ in range(b)]
            null = _null(xtr_all, tr_col, xte, vmap, ytr, yte, gse, cands, draws)
            res[name]["null"][label] = null
            ge = int((null >= real).sum())
            z = (real - null.mean()) / (null.std() + 1e-12)
            lines.append(
                f"{head} | {label:<5} {null.mean():>6.4f} {np.percentile(null, 95):>6.4f} "
                f"{ge:>5} {(ge + 1) / (b + 1):>7.4f} {z:>5.2f}"
            )
            head = " " * len(head)
            print(lines[-1], flush=True)
    text = "\n".join(lines)
    tag = "" if panel_csv == "lgbm_core25_panel.csv" else "_" + Path(panel_csv).stem
    sfx = tag + ("_smoke" if b < B else "")
    OUT_TXT.with_stem(OUT_TXT.stem + sfx).write_text(text + "\n")
    pd.DataFrame(
        [dict(cohort=c, pool=pl, auc=a) for c, r in res.items() for pl, v in r["null"].items() for a in v]
    ).to_csv(OUT_CSV.with_stem(OUT_CSV.stem + sfx), index=False)
    _plot(res, OUT_PNG.with_stem(OUT_PNG.stem + sfx))
    print(text)


def _plot(res: dict[str, dict], path: Path) -> None:
    """Panel AUC vs the two random-panel nulls (mean, 2.5-97.5 percentile), per cohort."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [c for c in FIG_COHORTS if c in res]
    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x - w, [res[c]["real"] for c in names], w, color="#d62728", label="Panel")
    for off, (label, color, text) in zip(
        (0, w), (("broad", "#9467bd", "Random genes (all candidates)"),
                 ("SFM", "#ff7f0e", "Random genes (pre-filtered pool)")), strict=True
    ):
        nulls = [res[c]["null"][label] for c in names]
        mean = np.array([v.mean() for v in nulls])
        lo = mean - np.array([np.percentile(v, 2.5) for v in nulls])
        hi = np.array([np.percentile(v, 97.5) for v in nulls]) - mean
        ax.bar(x + off, mean, w, color=color, alpha=0.75, yerr=[lo, hi], capsize=3, label=text)
    for i, c in enumerate(names):
        ps = []
        for label in ("broad", "SFM"):
            ge = int((res[c]["null"][label] >= res[c]["real"]).sum())
            b = len(res[c]["null"][label])
            ps.append(f"{(ge + 1) / (b + 1):.3f}")
        ax.text(x[i], 1.02, f"p = {ps[0]} / {ps[1]}", ha="center", fontsize=8)
    ax.axhline(0.5, color="grey", lw=1, ls="--")
    ax.set_xticks(x, [f"{c.replace(' donor-disjoint', chr(10) + 'donor-disjoint')}\n"
                      f"n={res[c]['n']}, {res[c]['k']} genes" for c in names], fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Zero-shot AUC-ROC")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(b=int(sys.argv[1]), panel_csv=(sys.argv[2:] or ["lgbm_core25_panel.csv"])[0])
    else:  # the manuscript's two runs: the 25-gene panel and its 17 annotated genes
        main()
        main(panel_csv="lgbm_core25_panel_annotated.csv")
