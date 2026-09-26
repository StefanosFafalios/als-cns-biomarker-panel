"""Explainability figures for the cross-disease specificity result (Step 37).

Three panels, all reading persisted per-sample data rather than recomputing:

  A. explainability_covariates.png
     Panel score against each post-mortem covariate (PMI, RIN, age at death) and
     against Braak stage, as small multiples. The aggregate table reports one
     Spearman rho per (cohort, covariate, group); a rho near zero is consistent
     with no relationship *or* with two opposing ones, and only the points
     distinguish those. Cases and controls are drawn separately for the same
     reason.

  B. explainability_crossdisease.png
     Per-cohort score distributions for cases against their own controls, with
     the control distribution repeated as a constant grey reference in every
     facet so the disease-to-disease comparison is read against a fixed
     baseline rather than against a moving one.

  C. explainability_deconv.png
     Cell-type assignment for each panel gene as a block heatmap: top-1
     correlation, the gap to top-2, and whether the assignment survived
     bootstrap resampling. Asterisks mark genes whose top-1-vs-top-2 gap has a
     bootstrap CI excluding zero, i.e. an assignment that is not a coin flip
     between two cell types.

Inputs (all produced upstream; nothing is recomputed here):
  cross_disease_covariates_per_sample.csv   (step 36)
  cross_disease_specificity_scores.npz      (step 35)
  deconv_robustness.csv                     (step 11a)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).parent
ALS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ALS_DIR))

from utils import critical_panel_label  # noqa: E402

_CRIT_LABEL = critical_panel_label()

_PER_SAMPLE = SCRIPT_DIR / "cross_disease_covariates_per_sample.csv"
_SCORES_NPZ = SCRIPT_DIR / "cross_disease_specificity_scores.npz"
_DECONV = SCRIPT_DIR / "deconv_robustness.csv"

PRIMARY_MODEL = "LGBM"
_CASE_C = "#c44e52"
_CTRL_C = "#7f7f7f"

_DISEASE_ORDER = [
    ("GSE153960-GPL16791", "ALS (positive control)"),
    ("GSE13162", "FTLD-TDP"),
    ("GSE64810", "Huntington's"),
    ("GSE125583", "Alzheimer's"),
    ("GSE68719", "Parkinson's"),
]


def _require(path: Path, step: str) -> None:
    if not path.exists():
        raise SystemExit(
            f"missing {path.name}; run `bash reproduce/run_all.sh only {step}` first"
        )


def _fig_covariates() -> None:
    """Panel A: score vs each covariate, cases and controls drawn separately."""
    _require(_PER_SAMPLE, "36")
    df = pd.read_csv(_PER_SAMPLE)
    df = df[df.model == PRIMARY_MODEL]

    panels: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = []
    for (cohort, cov), g in df.groupby(["cohort", "covariate"], sort=False):
        panels.append(
            (
                cohort,
                cov,
                g["covariate_value"].values,
                g["score"].values,
                g["is_case"].values,
            )
        )

    braak = _braak_points()
    if braak is not None:
        panels.append(("GSE125583", "braak", *braak))

    if not panels:
        raise SystemExit("no covariate points to plot")

    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, (cohort, cov, x, y, is_case) in zip(axes, panels):
        notes: list[str] = []
        for m, c, lab in ((is_case == 0, _CTRL_C, "control"), (is_case == 1, _CASE_C, "case")):
            if not m.sum():
                continue
            ax.scatter(x[m], y[m], s=16, alpha=0.65, c=c, label=lab, edgecolors="none")
            if m.sum() >= 6 and np.ptp(x[m]) > 0:
                b, a = np.polyfit(x[m], y[m], 1)
                xs = np.linspace(x[m].min(), x[m].max(), 20)
                # The score is a probability; an OLS line is free to leave [0,1]
                # and drawing it there would assert values the model cannot emit.
                ax.plot(xs, np.clip(a + b * xs, 0.0, 1.0), color=c, lw=1.4)
                r, pv = spearmanr(x[m], y[m])
                notes.append(f"{lab}s $\\rho$={r:+.3f} p={pv:.2g} n={int(m.sum())}")
        # Within-group rho is the quantity this figure argues about; the pooled
        # rho mostly restates the case/control contrast and is reported in the
        # aggregate table instead.
        ax.set_title(f"{cohort} — {cov.upper()}", fontsize=9)
        ax.set_xlabel(cov, fontsize=8)
        ax.set_ylabel("panel score", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_ylim(-0.05, 1.08)
        ax.text(
            0.03,
            0.98,
            "\n".join(notes),
            transform=ax.transAxes,
            va="top",
            fontsize=7,
        )
        ax.grid(alpha=0.25)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    axes[0].legend(fontsize=7, loc="lower right", framealpha=0.9)

    fig.suptitle(
        f"Panel score ({_CRIT_LABEL}, {PRIMARY_MODEL}) against post-mortem covariates "
        "and Braak stage\ncases and controls fitted separately — an aggregate "
        "correlation near zero can hide two opposing trends",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = SCRIPT_DIR / "explainability_covariates.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out.name} ({len(panels)} facets)")


def _braak_points() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Braak stage, score and case flag for GSE125583, or None if unavailable."""
    if not _SCORES_NPZ.exists():
        return None
    z = np.load(_SCORES_NPZ, allow_pickle=False)
    bkey = "__braak__|GSE125583"
    skey = f"GSE125583|{_CRIT_LABEL}|zscore|{PRIMARY_MODEL}"
    ykey = "__y__|GSE125583"
    if not all(k in z for k in (bkey, skey, ykey)):
        return None
    b, s, y = z[bkey].astype(float), z[skey], z[ykey]
    ok = np.isfinite(b)
    return b[ok], s[ok], y[ok]


def _auc(case: np.ndarray, ctrl: np.ndarray) -> float:
    """Mann-Whitney AUC of cases against controls (ties counted as half)."""
    from scipy.stats import rankdata

    n1, n0 = len(case), len(ctrl)
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(np.concatenate([case, ctrl]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _fig_crossdisease() -> None:
    """Panel B: per-cohort case scores against a constant control reference."""
    _require(_SCORES_NPZ, "35")
    z = np.load(_SCORES_NPZ, allow_pickle=False)

    facets = []
    for gse, label in _DISEASE_ORDER:
        skey = f"{gse}|{_CRIT_LABEL}|zscore|{PRIMARY_MODEL}"
        ykey = f"__y__|{gse}"
        if skey not in z or ykey not in z:
            continue
        s, y = z[skey], z[ykey]
        facets.append((label, s[y == 1], s[y == 0]))
    if not facets:
        raise SystemExit(
            f"no cohorts found in {_SCORES_NPZ.name} for panel {_CRIT_LABEL}; "
            "was step 35 run after the panel was redefined?"
        )

    # Scores concentrate hard against 1.0 on the transfer cohorts, so histograms
    # collapse into a single bar and hide everything. A jittered strip shows each
    # sample, and the grey control band repeats unchanged in every facet as the
    # fixed reference the diseases are read against.
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(facets), figsize=(2.6 * len(facets), 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (label, case, ctrl) in zip(axes, facets):
        lo, hi = np.percentile(ctrl, [25, 75])
        ax.axhspan(lo, hi, color=_CTRL_C, alpha=0.18, zorder=0)
        ax.axhline(np.median(ctrl), color=_CTRL_C, lw=1.3, ls="--", zorder=1)
        for xpos, vals, c, lab in ((0.0, ctrl, _CTRL_C, "own controls"),
                                   (1.0, case, _CASE_C, "cases")):
            jit = xpos + rng.uniform(-0.16, 0.16, len(vals))
            ax.scatter(jit, vals, s=14, c=c, alpha=0.7, edgecolors="none",
                       label=lab, zorder=3)
            ax.plot([xpos - 0.26, xpos + 0.26], [np.median(vals)] * 2,
                    color=c, lw=2.2, zorder=4)
        auc = _auc(case, ctrl)
        ax.set_title(f"{label}\nAUC = {auc:.3f}   n = {len(case)}/{len(ctrl)}", fontsize=8.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["ctrl", "case"], fontsize=8)
        ax.set_xlim(-0.5, 1.5)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.22, axis="y")
    axes[0].set_ylabel("panel score", fontsize=9)
    axes[0].legend(fontsize=7, framealpha=0.9, loc="lower left")
    fig.suptitle(
        f"Cross-disease panel scores ({_CRIT_LABEL}, {PRIMARY_MODEL}, z-score mode) — "
        "each disease against its own controls",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = SCRIPT_DIR / "explainability_crossdisease.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out.name} ({len(facets)} cohorts)")


def _fig_deconv() -> None:
    """Panel C: cell-type assignment block heatmap with bootstrap markers."""
    _require(_DECONV, "11a")
    d = pd.read_csv(_DECONV)
    need = {"gene", "top1_ct", "top1_rho", "top2_rho", "gap_CI_lo", "top1_argmax_freq"}
    missing = need - set(d.columns)
    if missing:
        raise SystemExit(f"{_DECONV.name} lacks columns: {sorted(missing)}")

    d = d.sort_values(["top1_ct", "top1_rho"], ascending=[True, False]).reset_index(drop=True)
    cts = sorted(d["top1_ct"].unique())
    ct_idx = {c: i for i, c in enumerate(cts)}
    mat = np.full((len(d), len(cts)), np.nan)
    for i, r in d.iterrows():
        mat[i, ct_idx[r["top1_ct"]]] = r["top1_rho"]

    fig, ax = plt.subplots(figsize=(1.5 + 1.05 * len(cts), 0.34 * len(d) + 2.0))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cts)))
    ax.set_xticklabels(cts, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["gene"], fontsize=7.5)
    for i, r in d.iterrows():
        j = ct_idx[r["top1_ct"]]
        # Asterisk = the top1-vs-top2 gap's bootstrap CI excludes zero, so the
        # winning cell type is separated from the runner-up rather than tied.
        star = "*" if float(r["gap_CI_lo"]) > 0 else ""
        ax.text(
            j,
            i,
            f"{r['top1_rho']:.2f}{star}",
            ha="center",
            va="center",
            fontsize=6.8,
            color="white" if abs(r["top1_rho"]) > 0.55 else "black",
        )
    ax.set_xlabel("assigned cell type", fontsize=9)
    n_star = int((d["gap_CI_lo"] > 0).sum())
    # Only protein-coding members carry cell-type marker correlations, so this is
    # a subset of the panel, not all of it -- say so rather than implying 18.
    ax.set_title(
        f"Cell-type assignment, {len(d)} protein-coding members of the "
        f"{_CRIT_LABEL} panel\nvalue = Spearman $\\rho$;  * = top1-vs-top2 gap CI "
        f"excludes 0 ({n_star}/{len(d)})",
        fontsize=9,
    )
    fig.colorbar(im, ax=ax, shrink=0.6, label=r"Spearman $\rho$")
    fig.tight_layout()
    out = SCRIPT_DIR / "explainability_deconv.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out.name} ({len(d)} genes, {n_star} separated)")


def main() -> None:
    print("=" * 62)
    print(f"Explainability figures — {_CRIT_LABEL} panel")
    print("=" * 62)
    _fig_covariates()
    _fig_crossdisease()
    _fig_deconv()
    print("done")


if __name__ == "__main__":
    main()
