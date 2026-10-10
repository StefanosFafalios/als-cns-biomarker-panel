"""Leave-one-dataset-out trimming: a clean external test of the critical panel (Step 105i).

The critical panel was trimmed on all nine external datasets, so every external result
for it is in-sample. This repeats the whole trimming once per held-out unit and tests
the resulting panel on data that took no part in choosing it:

  * for each trimming unit (iterative_panel_elimination._UNITS; the two E-MTAB series
    share donors and are held out together), drop its datasets;
  * on the rest, run step 24's greedy backward elimination from 25 genes to 1 with
    step 24's score (merged datasets, ctrl_z, pooled grouped 5x10-fold CV, unit-equal
    weights renormalised over the remaining units);
  * pick the size with step 24a's rule: the smallest k whose class-stratified paired
    bootstrap (B=2000) upper 95% bound of the loss against the peak lies below 0.01;
  * score that panel on the held-out datasets zero-shot (discovery model trained on all
    GPL24676 samples, utils.zero_shot_scale), next to the 25-gene panel and the final
    (in-sample) critical panel on the same data.

Resumable: lodo_trim_ckpt.json is rewritten after every greedy step.
``python lodo_trim.py smoke`` runs one unit for two steps on one split, no checkpoint.

Outputs: lodo_trim.csv, lodo_trim_statistics.txt, lodo_trim_progress.log
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

from greedy_significance_stop import _boot_indices  # noqa: E402
from utils import (  # noqa: E402
    PanelLGBMClassifier,
    critical_panel_symbols,
    zero_shot_scale,
)

from utils import selected_lgbm_params  # noqa: E402
MARGIN, N_BOOT, ALPHA, RANDOM_STATE = 0.01, 2000, 0.05, 42
CKPT = SCRIPT_DIR / "lodo_trim_ckpt.json"
PROGRESS = SCRIPT_DIR / "lodo_trim_progress.log"


def _log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    print(line, flush=True)
    with PROGRESS.open("a") as fh:
        fh.write(line + "\n")


def greedy(
    score: Callable[[list[int]], tuple[float, np.ndarray]],
    n_genes: int,
    state: dict | None = None,
    on_step: Callable[[dict], None] | None = None,
    max_steps: int | None = None,
) -> dict:
    """Greedy backward elimination; each step drops the gene whose removal scores best.

    Args:
        score: Maps gene indices to (weighted mean AUC, out-of-fold scores).
        n_genes: Starting panel size.
        state: A previous return value to resume from.
        on_step: Called with the state after every step (checkpointing).
        max_steps: Stop after this many drops (smoke runs).

    Returns:
        ``{"active": [...], "path": [{"k", "dropped", "wmean", "oof"}, ...]}``.
    """
    st = state or {"active": list(range(n_genes)), "path": []}
    if not st["path"]:
        v, s = score(st["active"])
        st["path"].append({"k": n_genes, "dropped": None, "wmean": v, "oof": list(map(float, s))})
    steps = 0
    while len(st["active"]) > 1 and (max_steps is None or steps < max_steps):
        trials = [(score([i for i in st["active"] if i != g]), g) for g in st["active"]]
        (v, s), g = max(trials, key=lambda t: t[0][0])  # first maximum, as in step 24
        st["active"].remove(g)
        st["path"].append({"k": len(st["active"]), "dropped": g, "wmean": v, "oof": list(map(float, s))})
        steps += 1
        if on_step:
            on_step(st)
    return st


def stopping_rule(path: list[dict], y: np.ndarray, cid: np.ndarray, w: dict[str, float]) -> tuple[int, int]:
    """Step 24a's non-inferiority rule on a greedy path.

    Args:
        path: Greedy path (each entry's ``oof`` aligned with *y* and *cid*).
        y: Labels.
        cid: Cohort per sample.
        w: Cohort weights summing to 1.

    Returns:
        (chosen k, peak k).
    """
    idx = {c: np.flatnonzero(cid == c) for c in w}
    boots = {c: _boot_indices(y[idx[c]], N_BOOT, RANDOM_STATE + i) for i, c in enumerate(w)}

    def wboot(oof: np.ndarray) -> np.ndarray:
        return sum(
            w[c] * np.array([roc_auc_score(y[idx[c]][b], oof[idx[c]][b]) for b in boots[c]]) for c in w
        )

    point = {p["k"]: sum(w[c] * roc_auc_score(y[idx[c]], np.asarray(p["oof"])[idx[c]]) for c in w) for p in path}
    peak = max(point, key=point.get)
    wb = {p["k"]: wboot(np.asarray(p["oof"])) for p in path}
    hi = {k: np.percentile(wb[peak] - wb[k], 100 * (1 - ALPHA / 2)) for k in wb}
    return min(k for k in hi if hi[k] < MARGIN), peak


def _panel_at(path: list[dict], n_genes: int, k: int) -> list[int]:
    dropped = [p["dropped"] for p in path if p["dropped"] is not None and p["k"] >= k]
    return [i for i in range(n_genes) if i not in dropped]


def main(smoke: bool = False) -> None:
    warnings.filterwarnings("ignore")
    params = dict(selected_lgbm_params(),
                  colsample_bytree=1.0, n_jobs=-1, verbose=-1)
    ipe = SCRIPT_DIR / "iterative_panel_elimination.py"
    ns: dict = {"__name__": "notmain", "__file__": str(ipe)}
    exec(compile(ipe.read_text().split("def main(")[0], "ipe", "exec"), ns)  # noqa: S102

    panel_df = pd.read_csv(SCRIPT_DIR / "lgbm_core25_panel.csv")
    sym25 = list(panel_df.symbol)
    feat25_bases = [f.split(".")[0] for f in panel_df.feature]
    X_log, y_train, _, feat_train = ns["_load_gpl24676_ctd"]()
    base_map = {f.split(".")[0]: j for j, f in enumerate(feat_train)}
    panel_train_cols = [base_map[b] for b in feat25_bases]
    Xtr25 = X_log[:, panel_train_cols].astype(np.float32)
    del X_log
    cohorts = ns["_trim_cohorts"](sym25, feat25_bases, panel_train_cols)
    units: dict[str, list[str]] = ns["_UNITS"]
    crit = critical_panel_symbols()
    ckpt = {} if smoke or not CKPT.exists() else json.loads(CKPT.read_text())
    if smoke:
        units = {"GSE287256": units["GSE287256"]}

    def zero_shot(cname: str, pidx: list[int]) -> tuple[float, float, float, int]:
        M, av, yc, _ = cohorts[cname]
        common = [pi for pi in av if pi in set(pidx)]
        if not common:
            return np.nan, np.nan, np.nan, 0
        Ztr, Zte = zero_shot_scale(Xtr25[:, common], M[:, [av.index(pi) for pi in common]], cname)
        sc = PanelLGBMClassifier(**params).fit(Ztr, y_train).predict_proba(Zte)[:, 1]
        lo, hi = ns["_bootstrap_auc"](yc, sc)
        return roc_auc_score(yc, sc), lo, hi, len(common)

    rows = []
    for unit, held in units.items():
        train = {c: v for c, v in cohorts.items() if c not in held}
        m = ns["merge_cohorts"](train, len(sym25))
        fold = ns["cv_folds"](m, n_repeats=1 if smoke else ns["N_REPEATS"])
        wsum = sum(ns["_W"][c] for c in train)
        w = {c: ns["_W"][c] / wsum for c in train}

        def score(idx: list[int], m=m, fold=fold, w=w) -> tuple[float, np.ndarray]:
            s = ns["pooled_oof"](m, sorted(idx), fold, params)
            return sum(w[c] * roc_auc_score(m.y[m.cid == c], s[m.cid == c]) for c in w), s

        def save(st: dict, unit=unit) -> None:
            if smoke:
                return
            ckpt[unit] = st
            CKPT.write_text(json.dumps(ckpt))
            p = st["path"][-1]
            _log(f"{unit} held out: k={p['k']} dropped {sym25[p['dropped']]}  W.mean {p['wmean']:.4f}")

        st = ckpt.get(unit)
        if not (st and len(st["active"]) == 1):
            _log(f"{unit} held out: trimming on {len(train)} datasets ({len(m.y)} samples)"
                 + (" [resumed]" if st else ""))
            st = greedy(score, len(sym25), st, save, max_steps=2 if smoke else None)
        core_k, peak = stopping_rule(st["path"], m.y, m.cid, w)
        pidx = _panel_at(st["path"], len(sym25), core_k)
        genes = [sym25[i] for i in pidx]
        _log(f"{unit} held out: peak k={peak}, rule k={core_k}: {', '.join(genes)}")
        for cname in held:
            for pname, idx in (("LODO panel", pidx), ("25-gene", list(range(len(sym25)))),
                               ("final critical (in-sample)", [sym25.index(g) for g in crit])):
                auc, lo, hi, n_used = zero_shot(cname, idx)
                rows.append({"held_out": unit, "cohort": cname, "panel": pname, "k": len(idx),
                             "genes_used": n_used, "auc": auc, "ci_lo": lo, "ci_hi": hi,
                             "members": ";".join(sym25[i] for i in idx) if pname == "LODO panel" else "",
                             "peak_k": peak if pname == "LODO panel" else np.nan})

    df = pd.DataFrame(rows)
    if smoke:
        print(df.to_string())
        return
    df.to_csv(SCRIPT_DIR / "lodo_trim.csv", index=False)
    _report(df, ns["_W"], sym25, crit)


def _report(df: pd.DataFrame, W: dict[str, float], sym25: list[str], crit: list[str]) -> None:
    """Per held-out unit AUCs, weighted means and how often each gene is re-selected."""
    lines = ["Leave-one-dataset-out trimming: zero-shot AUC on the held-out datasets", "=" * 72,
             "Each panel was trimmed (step 24 greedy + step 24a rule, margin 0.01) without the held-out",
             "unit; scored zero-shot from the discovery model. 'final critical' is in-sample (reference).", "",
             f"{'held out':<11}{'cohort':<13}{'k':>3}  {'LODO panel':<22}{'25-gene':<22}{'final critical':<16}"]
    for (unit, cname), g in df.groupby(["held_out", "cohort"], sort=False):
        r = g.set_index("panel")
        cell = lambda p: f"{r.loc[p, 'auc']:.4f} [{r.loc[p, 'ci_lo']:.3f}, {r.loc[p, 'ci_hi']:.3f}]"  # noqa: E731
        lines.append(f"{unit:<11}{cname:<13}{int(r.loc['LODO panel', 'k']):>3}  {cell('LODO panel'):<22}"
                     f"{cell('25-gene'):<22}{r.loc['final critical (in-sample)', 'auc']:.4f}")
    lines.append("")
    wm = {p: sum(W[c] * a for c, a in zip(g.cohort, g.auc)) for p, g in df.groupby("panel")}
    lines += [f"Weighted mean (unit-equal) {p:<28} {v:.4f}" for p, v in wm.items()]
    piv = df.pivot_table(index="cohort", columns="panel", values="auc")
    better = int((piv["LODO panel"] > piv["25-gene"]).sum())
    lines += [f"LODO panel above the 25-gene panel in {better} of {len(piv)} datasets", ""]
    lodo = df[df.panel == "LODO panel"].drop_duplicates("held_out")
    freq = {g: sum(g in m.split(";") for m in lodo.members) for g in sym25}
    lines += [f"Gene re-selection over {len(lodo)} held-out units (* = final critical panel):"]
    lines += [f"  {g:<16}{n}/{len(lodo)}{'  *' if g in crit else ''}"
              for g, n in sorted(freq.items(), key=lambda t: -t[1]) if n or g in crit]
    lines += ["", "Panel sizes (rule / peak): " + ", ".join(
        f"{u} {int(k)}/{int(p)}" for u, k, p in zip(lodo.held_out, lodo.k, lodo.peak_k))]
    (SCRIPT_DIR / "lodo_trim_statistics.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(smoke=len(sys.argv) > 1 and sys.argv[1] == "smoke")
