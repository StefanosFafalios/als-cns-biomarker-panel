"""Significance-based stopping rule for the greedy backward elimination.

The greedy trajectory's argmax is not a defensible panel-size criterion once the
GSE76220 confound is removed: W.mean varies by only ~0.02 across k=6..21, so the
peak sits well inside sampling noise (findings.md Step 44/45).

This script replaces "take the argmax" with an explicit equivalence test. For
every panel size along the greedy path it asks: *is this panel significantly
worse than the best panel on the trajectory?* The core panel is the **smallest**
one that is not.

Method
------
1. Score every panel size once with the greedy's own criterion (pooled grouped
   5x10-fold CV over the merged trimming cohorts, iterative_panel_elimination.pooled_oof)
   and store the per-sample out-of-fold scores. The model is not refit inside the bootstrap: the question is how much
   the *evaluation* would move on a different sample of subjects, not how much
   training would move.
2. Paired bootstrap over subjects, stratified by class within each cohort so a
   resample can never lose a class. Every panel is scored on the *same* resample
   indices, which makes the comparison paired and much tighter than comparing two
   independent CIs.
3. For each k, take the percentile CI of dW = W.mean(peak) - W.mean(k).
   Not significantly worse  <=>  that CI contains 0.

Outputs
-------
greedy_significance_stop.csv   per-k W.mean, dW vs peak, CI, verdict
greedy_significance_stop.txt   human-readable summary + the chosen panel
greedy_significance_stop.png   W.mean vs k with the equivalence band
panel_critical.csv             **the canonical panel definition** — every
                               downstream script reads it through
                               utils.critical_panel_symbols/_features rather
                               than hardcoding members. This script is the only
                               writer; regenerating it here is what makes the
                               panel definition reproducible.
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).parent
ALS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ALS_DIR))


warnings.filterwarnings("ignore")

_CKPT = SCRIPT_DIR / "iterative_panel_elimination_ckpt.json"
_PARAMS = SCRIPT_DIR / "lgbm_top500_best_params.json"
N_BOOT = 2000
RANDOM_STATE = 42
ALPHA = 0.05
# Margins reported in the sensitivity table (the primary one is set in main()).
MARGIN_SWEEP = (0.005, 0.01, 0.02, 0.03)
# Cohort held out for the leave-one-cohort-out sensitivity column. SRP064478 is
# n=15 and saturates at AUC 1.0000 on the critical panel, yet carries the same
# 0.25 weight as n=636 GPL16791, so the headline W.mean needs a figure computed
# without it (findings.md Step 52 SS C1).
SENSITIVITY_DROP_COHORT = "SRP064478"


def _trajectory() -> tuple[list[str], list[str], dict[int, float]]:
    """Return (full 25-gene panel, drop order, {k: W.mean}) from the greedy run."""
    # The greedy indexes the panel in lgbm_core25_panel.csv row order. This was parsed
    # from a manual-run log that run_all.sh never writes, so a rerun read a stale panel.
    full = pd.read_csv(SCRIPT_DIR / "lgbm_core25_panel.csv")["symbol"].tolist()
    hist = json.loads(_CKPT.read_text())["history"]
    drops = [h["dropped"] for h in hist if h["dropped"]]
    wmean = {h["panel_size"]: h["wmean"] for h in hist if h["panel_size"] is not None}
    return full, drops, wmean


def _panel_at(full: list[str], drops: list[str], k: int) -> list[str]:
    """The surviving gene set when the greedy has been run down to size *k*."""
    return [g for g in full if g not in set(drops[: len(full) - k])]


def _boot_indices(y: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Class-stratified bootstrap indices, shape (n_boot, len(y)).

    Stratifying keeps both classes present in every resample, so AUC is always
    defined for cohorts as small as n=15.
    """
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    out = np.empty((n_boot, len(y)), dtype=np.int64)
    for b in range(n_boot):
        out[b] = np.concatenate(
            [
                rng.choice(pos, len(pos), replace=True),
                rng.choice(neg, len(neg), replace=True),
            ]
        )
    return out


def _write_panel_critical(core_panel: list[str] | set[str]) -> None:
    """Write panel_critical.csv, the canonical definition of the critical panel.

    Joins the surviving symbols back onto the 25-gene panel (for feature ID and
    SHAP rank) and onto core25_annotations.tsv (for gene biotype), so the file
    carries everything downstream scripts need and no script has to re-resolve
    biotypes against mygene.

    Args:
        core_panel: Symbols retained by the non-inferiority stopping rule.
    """
    keep = set(core_panel)
    panel25 = pd.read_csv(SCRIPT_DIR / "lgbm_core25_panel.csv")
    annot_path = SCRIPT_DIR / "core25_annotations.tsv"
    annot = pd.read_csv(annot_path, sep="\t")

    # mygene spells these "protein-coding" / "pseudo"; the panel file and every
    # consumer use Ensembl's spelling.
    _CANON = {"protein-coding": "protein_coding", "pseudo": "pseudogene"}
    biotype = {
        s: _CANON.get(t, t)
        for s, t in zip(annot["symbol"], annot["type_of_gene"], strict=False)
    }

    rows = []
    for _, r in panel25.iterrows():
        if r["symbol"] not in keep:
            continue
        ensg = str(r["feature"]).split(".")[0]
        bt = biotype.get(r["symbol"])
        if not isinstance(bt, str):
            # mygene returns no type_of_gene for some pseudogenes (e.g. SMG1P5).
            # HGNC suffixes parent-derived pseudogenes with P<n>; nothing else in
            # this panel matches that pattern.
            bt = "pseudogene" if re.search(r"P\d+$", r["symbol"]) else None
            if bt is None:
                raise SystemExit(f"no biotype for {r['symbol']} in {annot_path.name}")
        rows.append(
            {
                "symbol": r["symbol"],
                "feature": r["feature"],
                "ensg": ensg,
                "biotype": bt,
                "protein_coding": bt == "protein_coding",
                "shap_rank_in_25": int(r["rank"]),
                "mean_abs_shap": float(r["mean_abs_shap"]),
            }
        )

    missing = keep - {r["symbol"] for r in rows}
    if missing:
        raise SystemExit(f"core panel members absent from the 25-gene panel: {missing}")

    out = SCRIPT_DIR / "panel_critical.csv"
    n_pc = sum(r["protein_coding"] for r in rows)
    payload = pd.DataFrame(rows).to_csv(index=False)

    # Only touch the file when membership actually changes. Downstream staleness is
    # judged by comparing output mtimes against this file (panel_staleness_audit.py),
    # and an unconditional rewrite makes every correctly-rerun script look stale.
    if out.exists() and out.read_text() == payload:
        print(f"\n{out.name} unchanged: {len(rows)} genes ({n_pc} protein-coding)")
        return
    out.write_text(payload)
    print(f"\nWrote {out.name}: {len(rows)} genes ({n_pc} protein-coding)")


_SCORES_CSV = SCRIPT_DIR / "greedy_significance_stop.csv"


def _cached_scores(ks: list[int]) -> "pd.DataFrame | None":
    """Return the per-k bootstrap table from a previous run, if it is complete.

    Scoring 25 panel sizes across the trimming cohorts is ~8 h under the corrected
    colsample_bytree, and it depends only on the greedy trajectory and the
    cohort matrices -- not on the reporting parameters (margin, sweep) that get
    revised far more often. Reusing a complete table makes regenerating the
    report seconds rather than hours. Delete the CSV, or set
    ``GREEDY_STOP_REFIT=1``, to force a refit.

    Args:
        ks: Panel sizes the current run expects.

    Returns:
        The cached table, or ``None`` if absent, incomplete, or refit is forced.
    """
    import os

    if os.environ.get("GREEDY_STOP_REFIT") or not _SCORES_CSV.exists():
        return None
    cached = pd.read_csv(_SCORES_CSV)
    missing = set(ks) - set(cached.k)
    if missing:
        print(f"  cached table is missing k={sorted(missing)}; refitting")
        return None
    # Schema check, not just coverage: a table written before the per-cohort
    # columns existed would otherwise be reused silently and the report would
    # quietly lose the leave-one-cohort-out analysis.
    required = {"wmean_drop1", "ci_lo_drop1"}
    if not required.issubset(cached.columns):
        print(f"  cached table predates {sorted(required)}; refitting")
        return None
    print(f"Reusing cached scores for k={min(ks)}..{max(ks)} ({_SCORES_CSV.name})")
    return cached


def main() -> None:
    full, drops, wmean_traj = _trajectory()
    params = json.loads(_PARAMS.read_text())

    # Panel sizes come from the greedy trajectory alone, so the cache can be
    # checked before any cohort is loaded.
    ks = sorted([k for k in wmean_traj if k >= 1], reverse=True)
    cached = _cached_scores(ks)
    if cached is not None:
        point = dict(zip(cached.k, cached.wmean, strict=False))
        _report(cached, full, drops, point, max(point, key=point.get))
        return

    # Mirror the greedy's own data prep exactly (same loaders, same index maps)
    ipe = SCRIPT_DIR / "iterative_panel_elimination.py"
    src = ipe.read_text()
    ns: dict = {"__name__": "notmain", "__file__": str(ipe)}
    exec(compile(src.split("def main(")[0], "ipe", "exec"), ns)  # noqa: S102

    panel_df = pd.read_csv(SCRIPT_DIR / "lgbm_core25_panel.csv")
    sym25 = list(panel_df.symbol)
    feat25_bases = [f.split(".")[0] for f in panel_df.feature]

    print("Loading cohorts (greedy's own loaders) ...")
    _, _, _, feat_train = ns["_load_gpl24676_ctd"]()
    base_map = {f.split(".")[0]: j for j, f in enumerate(feat_train)}
    panel_train_cols = [base_map[b] for b in feat25_bases]

    # The greedy's own cohorts and weights (every CNS cohort except discovery, one
    # equal-weight unit each), so the stopping rule scores exactly what it trimmed on.
    cohorts = ns["_trim_cohorts"](sym25, feat25_bases, panel_train_cols)
    weights = ns["_W"]
    merged = ns["merge_cohorts"](cohorts, len(sym25))
    fold = ns["cv_folds"](merged)

    print(f"Scoring {len(ks)} panel sizes on {len(cohorts)} cohorts ...")

    scores: dict[int, dict[str, np.ndarray]] = {}
    labels: dict[str, np.ndarray] = {c: v[2] for c, v in cohorts.items()}
    sym_index = {s: i for i, s in enumerate(sym25)}

    for k in ks:
        panel_idx = sorted(sym_index[g] for g in _panel_at(full, drops, k))
        oof = ns["pooled_oof"](merged, panel_idx, fold, params)
        scores[k] = {c: oof[merged.cid == c] for c in cohorts}
        print(f"  k={k:<3} done", flush=True)

    # One shared set of resample indices per cohort -> paired comparison.
    boots = {
        c: _boot_indices(labels[c], N_BOOT, RANDOM_STATE + i)
        for i, c in enumerate(cohorts)
    }

    def cohort_boot_aucs(k: int) -> dict[str, np.ndarray]:
        """Per-cohort AUC for every bootstrap replicate at panel size *k*.

        Kept per-cohort rather than pre-summed so the report can show which cohort
        supplies the variance behind each confidence interval. The k=15/16/17
        intervals are 10-40x tighter than every other panel size and the stopping
        rule turns on exactly that; without this breakdown the cliff cannot be
        attributed and the panel size cannot be defended.
        """
        out = {}
        for c in cohorts:
            y, sc_k, bi = labels[c], scores[k][c], boots[c]
            out[c] = np.array([roc_auc_score(y[i], sc_k[i]) for i in bi])
        return out

    print("Bootstrapping ...")
    cboot = {k: cohort_boot_aucs(k) for k in ks}
    wb = {k: sum(weights[c] * cboot[k][c] for c in cohorts) for k in ks}

    keep3 = [c for c in cohorts if c != SENSITIVITY_DROP_COHORT]
    w3 = {c: weights[c] / sum(weights[d] for d in keep3) for c in keep3}
    wb3 = {k: sum(w3[c] * cboot[k][c] for c in keep3) for k in ks}

    cohort_point = {
        k: {c: roc_auc_score(labels[c], scores[k][c]) for c in cohorts} for k in ks
    }
    point = {k: sum(weights[c] * cohort_point[k][c] for c in cohorts) for k in ks}
    point3 = {k: sum(w3[c] * cohort_point[k][c] for c in keep3) for k in ks}
    peak = max(point, key=point.get)
    peak3 = max(point3, key=point3.get)
    print(f"  peak panel: k={peak}  W.mean={point[peak]:.4f}")

    pct = [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)]
    rows = []
    for k in ks:
        lo, hi = np.percentile(wb[peak] - wb[k], pct)
        lo3, hi3 = np.percentile(wb3[peak3] - wb3[k], pct)
        row = {
            "k": k,
            "wmean": point[k],
            "wmean_greedy": wmean_traj.get(k, np.nan),
            "delta_vs_peak": point[peak] - point[k],
            "ci_lo": lo,
            "ci_hi": hi,
            "significantly_worse": bool(lo > 0),
            # Leave-one-cohort-out sensitivity (SENSITIVITY_DROP_COHORT excluded).
            "wmean_drop1": point3[k],
            "delta_vs_peak_drop1": point3[peak3] - point3[k],
            "ci_lo_drop1": lo3,
            "ci_hi_drop1": hi3,
        }
        # Per-cohort point AUC and bootstrap SD. The SD is the diagnostic: a
        # cohort pinned at 1.0000 across a run of panel sizes contributes zero
        # variance there and can manufacture a narrow interval on its own.
        for c in cohorts:
            row[f"auc_{c}"] = cohort_point[k][c]
            row[f"sd_{c}"] = float(cboot[k][c].std())
        row["panel"] = ";".join(_panel_at(full, drops, k))
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("k", ascending=False)
    df.to_csv(SCRIPT_DIR / "greedy_significance_stop.csv", index=False)

    _report(df, full, drops, point, peak)


def _cohort_section(df: "pd.DataFrame", core_k: int, peak: int) -> list[str]:
    """Per-cohort AUC/SD breakdown and the leave-one-cohort-out sensitivity.

    Written because the stopping rule turns entirely on interval width, and width
    is a property of which cohorts still have variance left at a given panel size.
    A cohort saturated at AUC 1.0000 across a run of panel sizes contributes none,
    and can therefore manufacture a narrow interval by itself.

    Args:
        df: The scored per-k table, including the ``auc_*``/``sd_*`` columns.
        core_k: Panel size selected by the non-inferiority rule.
        peak: Panel size with the highest W.mean.

    Returns:
        Report lines. Empty if the table predates the per-cohort columns.
    """
    cols = sorted(c[4:] for c in df.columns if c.startswith("auc_"))
    if not cols:
        return []
    out = [
        "",
        "Per-cohort AUC and bootstrap SD (the CI width is built from these):",
        "  " + f"{'k':>3}  " + "  ".join(f"{c:>21}" for c in cols),
        "  " + "-" * (5 + 23 * len(cols)),
    ]
    for _, r in df.iterrows():
        cells = [f"{r['auc_' + c]:.4f} (sd {r['sd_' + c]:.4f})" for c in cols]
        out.append(f"  {int(r.k):>3}  " + "  ".join(f"{x:>21}" for x in cells))

    zero_sd = [
        c for c in cols if float(df.loc[df.k == core_k, "sd_" + c].iloc[0]) == 0.0
    ]
    if zero_sd:
        out += [
            "",
            f"  NOTE: at the core panel k={core_k}, {', '.join(zero_sd)} has zero bootstrap",
            "  variance -- it is saturated and contributes nothing to the interval that",
            "  the stopping rule tests. Treat the panel-size choice accordingly.",
        ]

    if "wmean_drop1" in df.columns:
        peak3 = int(df.loc[df.wmean_drop1.idxmax(), "k"])
        out += [
            "",
            f"Leave-one-cohort-out sensitivity (excluding {SENSITIVITY_DROP_COHORT}):",
            f"  {'k':>3}  {'W.mean(3)':>10}  {'dW':>8}  {'95% CI':>20}",
            "  " + "-" * 46,
        ]
        for _, r in df.iterrows():
            out.append(
                f"  {int(r.k):>3}  {r.wmean_drop1:>10.4f}  "
                f"{r.delta_vs_peak_drop1:>+8.4f}  "
                f"[{r.ci_lo_drop1:>+7.4f},{r.ci_hi_drop1:>+7.4f}]"
            )
        w4_core = float(df.loc[df.k == core_k, "wmean"].iloc[0])
        w3_core = float(df.loc[df.k == core_k, "wmean_drop1"].iloc[0])
        w4_full = float(df.loc[df.k == df.k.max(), "wmean"].iloc[0])
        w3_full = float(df.loc[df.k == df.k.max(), "wmean_drop1"].iloc[0])
        out += [
            "",
            f"  peak without {SENSITIVITY_DROP_COHORT}: k={peak3} (all-unit peak was k={peak})",
            f"  Gain of the core panel over the full {int(df.k.max())}-gene panel:",
            f"    all units, equal weight      : {w4_core:.4f} - {w4_full:.4f} = {w4_core - w4_full:+.4f}",
            f"    without {SENSITIVITY_DROP_COHORT}, reweighted : {w3_core:.4f} - {w3_full:.4f} = {w3_core - w3_full:+.4f}",
        ]
    return out


def _report(
    df: "pd.DataFrame",
    full: list[str],
    drops: list[str],
    point: dict[int, float],
    peak: int,
) -> None:
    """Apply the stopping rule to a scored table and write every artifact.

    Split out from the scoring so the report can be regenerated from a cached
    bootstrap table: the margin and the sensitivity sweep get revised often,
    the ~8 h scoring almost never.

    Args:
        df: Per-k table with wmean, delta_vs_peak and the bootstrap CI.
        full: The full 25-gene panel, in order.
        drops: Greedy drop order.
        point: Point-estimate W.mean per panel size.
        peak: Panel size with the highest W.mean.
    """
    # ------------------------------------------------------------------
    # Choosing the rule.
    #
    # The naive rule -- "smallest k not significantly worse than the peak" --
    # is invalid here. The paired bootstrap's precision depends on how many
    # genes separate the two panels, not on how good they are: k=16 differs
    # from the peak by one gene, so dW=+0.0024 has a CI of width 0.005 and is
    # "significant", while k=14 differs by three genes, so a LARGER dW=+0.0084
    # has a CI of width 0.055 and is not. Scanning for the smallest
    # "equivalent" panel therefore rewards low power and runs away to k=3.
    #
    # Non-inferiority inverts that: panel k is acceptable only when the upper
    # confidence bound on the loss sits below a margin, which demands
    # precision. Wide-CI panels fail, as they should.
    # ------------------------------------------------------------------
    # Primary analysis. Reported alongside the sensitivity sweep below rather
    # than claimed as pre-specified: the margin was chosen after the greedy
    # trajectory was in hand, and the sweep is what shows the choice is not
    # load-bearing.
    MARGIN = 0.01  # AUC points; largest loss vs the peak we would accept

    df["non_inferior"] = df.ci_hi < MARGIN
    acceptable = df[df.non_inferior]
    core_k = int(acceptable.k.min())
    core_panel = _panel_at(full, drops, core_k)

    # Sensitivity to the margin. Derived from the CIs already computed, so this
    # costs nothing extra -- no refit, no second bootstrap.
    sweep = []
    for m in MARGIN_SWEEP:
        acc_m = df[df.ci_hi < m]
        if not len(acc_m):
            sweep.append((m, None, []))
            continue
        k_m = int(acc_m.k.min())
        sweep.append((m, k_m, _panel_at(full, drops, k_m)))

    seq = df[(df.k <= peak) & df.significantly_worse]
    seq_k = int(seq.k.max()) + 1 if len(seq) else int(df.k.min())

    # Cross-check numbers, derived rather than written down: they moved every time
    # the trajectory was rescored and a stale literal in this block would be
    # invisible (findings.md Step 53).
    seq_dw = (
        float(df.loc[df.k == seq_k, "delta_vs_peak"].iloc[0])
        if seq_k in set(df.k)
        else float("nan")
    )
    naive = df[~df.significantly_worse]
    naive_k = int(naive.k.min()) if len(naive) else core_k
    core_ciw = float(df.loc[df.k == core_k, "ci_hi"].iloc[0]) - float(
        df.loc[df.k == core_k, "ci_lo"].iloc[0]
    )
    below = df[
        (df.k < core_k)
        & (df.delta_vs_peak > df.loc[df.k == core_k, "delta_vs_peak"].iloc[0])
    ]
    if len(below) and core_ciw > 0:
        wr = below.assign(w=below.ci_hi - below.ci_lo).sort_values("k", ascending=False)
        wide_k = int(wr.k.iloc[0])
        wide_dw = float(wr.delta_vs_peak.iloc[0])
        wide_ratio = float(wr.w.iloc[0]) / core_ciw
    else:
        wide_k, wide_dw, wide_ratio = core_k, seq_dw, 1.0

    lines = [
        "Significance-based stopping rule for greedy backward elimination",
        "=" * 66,
        "",
        f"Peak panel: k={peak}  W.mean={point[peak]:.4f}",
        f"Paired bootstrap B={N_BOOT}, class-stratified within cohort, alpha={ALPHA}",
        f"Non-inferiority margin: {MARGIN} AUC",
        "",
        f"{'k':>3}  {'W.mean':>8}  {'dW':>8}  {'95% CI':>20}  {'CIw':>7}  verdict",
        "-" * 78,
    ]
    for _, r in df.iterrows():
        v = "non-inferior" if r.non_inferior else "not established"
        mark = "  <== CORE" if int(r.k) == core_k else ""
        lines.append(
            f"{int(r.k):>3}  {r.wmean:>8.4f}  {r.delta_vs_peak:>+8.4f}  "
            f"[{r.ci_lo:>+7.4f},{r.ci_hi:>+7.4f}]  {r.ci_hi - r.ci_lo:>7.4f}  {v}{mark}"
        )
    lines += [
        "",
        f"CORE PANEL (non-inferiority, margin {MARGIN}): k={core_k}",
        f"  smallest panel whose loss vs the k={peak} optimum is provably under {MARGIN} AUC.",
        f"  Members ({len(core_panel)}): {', '.join(sorted(core_panel))}",
        "",
        "Sensitivity to the non-inferiority margin:",
        f"  {'delta':>7}  {'k':>3}  membership vs the primary panel",
        "  " + "-" * 62,
        *[
            f"  {m:>7.3f}  {'--' if k is None else k:>3}  "
            + (
                "no panel qualifies"
                if k is None
                else (
                    "identical"
                    if set(p) == set(core_panel)
                    else f"differs: {', '.join(sorted(set(p) ^ set(core_panel)))}"
                )
            )
            for m, k, p in sweep
        ],
        "",
        "Cross-checks:",
        f"  sequential rule (stop at first significant drop below peak): k={seq_k}",
        f"    -- driven by a dW of {seq_dw:+.4f} AUC: statistically real, "
        + ("practically trivial." if abs(seq_dw) < MARGIN else f"larger than the {MARGIN} margin."),
        f"  naive 'smallest not significantly worse': k={naive_k} -- REJECTED, rewards",
        *(
            [
                f"    low power (k={wide_k} has a LARGER dW of {wide_dw:+.4f} yet is 'not significant'",
                f"    only because its CI is {wide_ratio:.0f}x wider than the core panel's).",
            ]
            if wide_k != core_k
            else ["    low power (its CI on the loss vs the peak is too wide to show non-inferiority)."]
        ),
    ]
    lines += _cohort_section(df, core_k, peak)
    (SCRIPT_DIR / "greedy_significance_stop.txt").write_text("\n".join(lines) + "\n")
    df.to_csv(SCRIPT_DIR / "greedy_significance_stop.csv", index=False)
    print("\n".join(lines[3:]))

    _write_panel_critical(core_panel)

    fig, ax = plt.subplots(figsize=(9, 5))
    d = df.sort_values("k")
    ax.plot(d.k, d.wmean, "o-", color="#2c3e50", lw=1.6, ms=4, label="W.mean")
    band = df[~df.significantly_worse]
    ax.axvspan(
        band.k.min(),
        band.k.max(),
        color="#27ae60",
        alpha=0.12,
        label=f"not sig. worse than k={peak}",
    )
    ax.axvline(core_k, color="#27ae60", ls="--", lw=1.4, label=f"core panel k={core_k}")
    ax.axvline(peak, color="#c0392b", ls=":", lw=1.4, label=f"peak k={peak}")
    ax.set_xlabel("panel size k")
    ax.set_ylabel("weighted mean pooled-CV AUC")
    ax.set_title("Greedy trajectory with significance-based stopping rule")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / "greedy_significance_stop.png", dpi=150)
    print("\nWrote greedy_significance_stop.{csv,txt,png}")


if __name__ == "__main__":
    main()
