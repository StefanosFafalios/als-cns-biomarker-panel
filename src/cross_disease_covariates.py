# ruff: noqa: E402
"""Covariate sensitivity of the cross-disease panel scores (Step 36).

Step 35 (`cross_disease_specificity.py`) found the ALS panel scores Huntington's
disease above chance and higher than either Alzheimer's or Parkinson's disease
under both LightGBM (primary) and LR-L2 (sensitivity baseline).  The obvious
reviewer objection to the Huntington's result is confounding by post-mortem
tissue quality: HD brains could differ from their controls in post-mortem
interval (PMI), RNA integrity (RIN) or age at death, and a panel containing
vascular and microglial transcripts could track that rather than disease.

GSE64810 (HD) and GSE68719 (PD) both publish per-sample PMI, RIN and age at
death, and they re-analyse the SAME neurologically-normal control samples, which
makes them an unusually clean pair for this test.  This script asks:

1. Is the panel score correlated with PMI / RIN / age within each cohort
   (overall, and separately within controls and within cases)?
2. Do the cohorts differ in those covariates between cases and controls?
3. Does the panel score remain associated with disease status after adjusting
   for PMI, RIN and age in a logistic regression, and how much of the
   association survives?

The plotting layout follows the small-multiples-with-constant-baseline idea of
Zhou et al. 2026 Fig 2A: one panel per covariate, with the shared control
samples drawn in grey in every panel as a fixed reference.

Outputs
-------
  cross_disease_covariates_statistics.txt
  cross_disease_covariates.csv
  cross_disease_covariates.png
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ALS_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ALS_DIR))

import cross_disease_specificity as cds
from utils import critical_panel_label, load_dataset

# Panel size is derived, so every reader-facing label must be too.
_CRIT_LABEL = critical_panel_label()


SCRIPT_DIR = Path(__file__).parent
PREFIX = "cross_disease_covariates"
RANDOM_STATE = 42

# Covariate keys as they appear in the GEO characteristics rows
_COVARIATES = {
    "pmi": ("pmi", "post-mortem interval"),
    "rin": ("rin", "rna integrity"),
    "age": ("age of death", "age at death", "age"),
}

_COHORTS = {
    "GSE64810": {
        "disease": "Huntington's disease",
        "matrix_url": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE64nnn/GSE64810/matrix/"
            "GSE64810_series_matrix.txt.gz"
        ),
        "matrix_name": "GSE64810_series_matrix.txt.gz",
    },
    "GSE68719": {
        "disease": "Parkinson's disease",
        "matrix_url": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE68nnn/GSE68719/matrix/"
            "GSE68719_series_matrix.txt.gz"
        ),
        "matrix_name": "GSE68719_series_matrix.txt.gz",
    },
}


def _parse_covariates(gse: str, url: str, fname: str) -> dict[str, dict[str, float]]:
    """Return {sample_id: {covariate: value}} keyed by the C_/H_/P_ sample id.

    The counts matrices are keyed by short sample ids (``C_0002``) while the
    series matrix is keyed by GSM; ``!Sample_title`` carries the short id, so it
    is used to bridge the two.
    """
    path = cds._download(
        url, cds.RESOURCES_DIR / gse / "matrix" / fname, f"{gse} series matrix"
    )
    hdr = cds._series_matrix_meta(path)
    titles = hdr["Sample_title"]
    short = [t.split(" ")[0] for t in titles]

    out: dict[str, dict[str, float]] = {s: {} for s in short}
    for row in cds._sample_char_rows(path):
        if not row:
            continue
        key_raw = row[0].split(":", 1)[0].strip().lower()
        target: str | None = None
        for canon, aliases in _COVARIATES.items():
            if any(key_raw.startswith(a) for a in aliases):
                target = canon
                break
        if target is None:
            continue
        for sid, cell in zip(short, row):
            if ":" not in cell:
                continue
            val = cell.split(":", 1)[1].strip()
            try:
                out[sid][target] = float(val)
            except ValueError:
                continue
    return out


def _panel_scores_for(gse: str, cfg: dict, params: dict, X_tr_all, y_tr, panel_def):
    """Return (sample_ids, y, scores_by_model) for one DESeq2 cohort."""
    disease_prefix = "H_" if gse == "GSE64810" else "P_"
    suppl = {
        "GSE64810": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE64nnn/GSE64810/suppl/"
            "GSE64810_mlhd_DESeq2_norm_counts_adjust.txt.gz",
            "GSE64810_mlhd_DESeq2_norm_counts_adjust.txt.gz",
            False,
        ),
        "GSE68719": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE68nnn/GSE68719/suppl/"
            "GSE68719_mlpd_PCG_DESeq2_norm_counts.txt.gz",
            "GSE68719_mlpd_PCG_DESeq2_norm_counts.txt.gz",
            True,
        ),
    }[gse]
    expr, y_te, _meta = cds._load_deseq_cohort(
        gse, suppl[0], suppl[1], disease_prefix, has_symbol_col=suppl[2]
    )
    feats = [f for f, _ in panel_def]
    syms = [s for _, s in panel_def]
    tr_keys, te_keys, _matched = cds._match_panel(
        feats, syms, set(expr.columns), "ensembl"
    )
    Xtr = np.log1p(X_tr_all[tr_keys].values.astype(np.float64))
    Xte = np.log1p(np.clip(expr[te_keys].values.astype(np.float64), 0, None))
    # cds._score returns {model: {"scores", "oof_auc", "fold_aucs"}} and is
    # fold-averaged over the same 5-fold CV used in Step 35.
    res = cds._score(Xtr, y_tr, Xte, params, "zscore", y_te=y_te)
    preds = {m: r["scores"] for m, r in res.items()}
    return list(expr.index), y_te, preds, len(tr_keys), len(feats)


def main() -> None:
    """Run the covariate-sensitivity analysis and write all outputs."""
    import json

    import pandas as pd
    from scipy.stats import mannwhitneyu, spearmanr

    print("=" * 74)
    print("Cross-disease covariate sensitivity (PMI / RIN / age) — HD and PD")
    print("=" * 74)

    params = json.loads(cds._PARAMS_PATH.read_text())
    panel = pd.read_csv(cds._PANEL_CSV)
    panel_def = [
        (f, s)
        for f, s in zip(panel["feature"], panel["symbol"])
        if s in cds.CRITICAL_PANEL
    ]

    (ds_tr,) = load_dataset(
        "GSE153960", platform="GPL24676", resources_dir=cds.RESOURCES_DIR
    )
    y_tr = ds_tr.y.values.astype(int)
    X_tr_all = ds_tr.X

    rows: list[dict] = []
    store: dict[str, dict] = {}

    for gse, cfg in _COHORTS.items():
        print(f"\n[{gse}] {cfg['disease']}")
        covs = _parse_covariates(gse, cfg["matrix_url"], cfg["matrix_name"])
        sids, y_te, preds, n_match, n_tot = _panel_scores_for(
            gse, cfg, params, X_tr_all, y_tr, panel_def
        )
        cov_df = pd.DataFrame(
            {c: [covs.get(s, {}).get(c, np.nan) for s in sids] for c in _COVARIATES},
            index=sids,
        )
        n_cov = {c: int(cov_df[c].notna().sum()) for c in _COVARIATES}
        print(f"  covariates parsed: {n_cov}  (n={len(sids)})")
        store[gse] = {
            "disease": cfg["disease"],
            "sids": sids,
            "y": y_te,
            "preds": preds,
            "cov": cov_df,
            "matched": f"{n_match}/{n_tot}",
        }

        for model, s in preds.items():
            for cov in _COVARIATES:
                v = cov_df[cov].values.astype(float)
                for grp_name, mask in (
                    ("all", np.ones(len(y_te), bool)),
                    ("controls", y_te == 0),
                    ("cases", y_te == 1),
                ):
                    ok = mask & ~np.isnan(v)
                    if ok.sum() < 6:
                        continue
                    rho, p = spearmanr(v[ok], s[ok])
                    rows.append(
                        {
                            "cohort": gse,
                            "disease": cfg["disease"],
                            "model": model,
                            "covariate": cov,
                            "group": grp_name,
                            "n": int(ok.sum()),
                            "spearman_rho": float(rho),
                            "p": float(p),
                        }
                    )

    df = pd.DataFrame(rows)
    df.to_csv(SCRIPT_DIR / f"{PREFIX}.csv", index=False)

    # Per-sample long form. The aggregate table above reports one rho per
    # (cohort, covariate, group), which is enough to test for confounding but not
    # to *show* it: a rho near zero can come from no relationship or from two
    # opposing ones. The explainability figures need the points themselves.
    per_sample: list[dict] = []
    for gse, st in store.items():
        for model, s in st["preds"].items():
            for cov in _COVARIATES:
                v = st["cov"][cov].values.astype(float)
                for i, sid in enumerate(st["sids"]):
                    if np.isnan(v[i]):
                        continue
                    per_sample.append(
                        {
                            "cohort": gse,
                            "disease": st["disease"],
                            "model": model,
                            "sample": sid,
                            "covariate": cov,
                            "covariate_value": float(v[i]),
                            "score": float(s[i]),
                            "is_case": int(st["y"][i]),
                        }
                    )
    ps = pd.DataFrame(per_sample)
    ps.to_csv(SCRIPT_DIR / f"{PREFIX}_per_sample.csv", index=False)
    print(f"  wrote {PREFIX}_per_sample.csv ({len(ps)} rows)")

    # --- covariate imbalance + adjusted logistic regression ---------------
    adjusted: list[dict] = []
    for gse, d in store.items():
        y_te = d["y"]
        cov_df = d["cov"]
        imbalance: dict[str, tuple[float, float, float]] = {}
        for cov in _COVARIATES:
            v = cov_df[cov].values.astype(float)
            ok = ~np.isnan(v)
            if (ok & (y_te == 1)).sum() < 3 or (ok & (y_te == 0)).sum() < 3:
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                p = float(
                    mannwhitneyu(
                        v[ok & (y_te == 1)],
                        v[ok & (y_te == 0)],
                        alternative="two-sided",
                    )[1]
                )
            imbalance[cov] = (
                float(np.median(v[ok & (y_te == 1)])),
                float(np.median(v[ok & (y_te == 0)])),
                p,
            )
        d["imbalance"] = imbalance

        for model, s in d["preds"].items():
            adjusted.append(_adjusted_fit(gse, d["disease"], model, s, y_te, cov_df))
    adj_df = pd.DataFrame([a for a in adjusted if a])
    if not adj_df.empty:
        adj_df.to_csv(SCRIPT_DIR / f"{PREFIX}_adjusted.csv", index=False)

    _write_report(df, adj_df, store)
    _plot(store)
    print(
        f"\nWrote {PREFIX}.csv, {PREFIX}_adjusted.csv, "
        f"{PREFIX}_statistics.txt, {PREFIX}.png"
    )


def _adjusted_fit(gse, disease, model, s, y_te, cov_df) -> dict | None:
    """Logistic regression of disease status on panel score +/- covariates."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    cols = [c for c in _COVARIATES if cov_df[c].notna().sum() >= len(cov_df) * 0.8]
    sub = pd.DataFrame({"score": s}, index=cov_df.index)
    for c in cols:
        sub[c] = cov_df[c].values.astype(float)
    keep = sub.notna().all(axis=1).values
    if keep.sum() < 20 or len(np.unique(y_te[keep])) < 2:
        return None
    yk = y_te[keep]

    def _auc(mat: np.ndarray) -> float:
        Xz = StandardScaler().fit_transform(mat)
        lr = LogisticRegression(max_iter=5_000, random_state=RANDOM_STATE)
        lr.fit(Xz, yk)
        return float(roc_auc_score(yk, lr.predict_proba(Xz)[:, 1]))

    score_only = sub.loc[keep, ["score"]].values
    covs_only = sub.loc[keep, cols].values
    both = sub.loc[keep, ["score"] + cols].values

    # Coefficient on the panel score in the adjusted model
    Xz = StandardScaler().fit_transform(both)
    lr = LogisticRegression(max_iter=5_000, random_state=RANDOM_STATE)
    lr.fit(Xz, yk)

    return {
        "cohort": gse,
        "disease": disease,
        "model": model,
        "n": int(keep.sum()),
        "covariates_used": ",".join(cols),
        "auc_score_only": _auc(score_only),
        "auc_covariates_only": _auc(covs_only),
        "auc_score_plus_covariates": _auc(both),
        "beta_score_adjusted": float(lr.coef_[0][0]),
        "beta_covariates_adjusted": ",".join(
            f"{c}={b:+.3f}" for c, b in zip(cols, lr.coef_[0][1:])
        ),
    }


def _write_report(df, adj_df, store: dict) -> None:
    """Write the plain-text statistics report."""
    lines: list[str] = []
    add = lines.append
    add("Cross-disease covariate sensitivity — PMI / RIN / age at death")
    add("=" * 74)
    add("Question: is the Huntington's-disease panel signal from Step 35 driven")
    add("by post-mortem tissue quality rather than by disease?")
    add("")
    add("GSE64810 (HD) and GSE68719 (PD) re-analyse the SAME control samples and")
    add("both publish per-sample PMI, RIN and age at death, which makes them a")
    add("clean pair: any covariate artefact in the shared controls applies")
    add("equally to both, yet the two cohorts score very differently.")
    add("")

    for gse, d in store.items():
        add("-" * 74)
        add(f"{gse} — {d['disease']}   (panel genes matched {d['matched']})")
        add("-" * 74)
        y = d["y"]
        add(f"  n cases={int(y.sum())}  n controls={int((y == 0).sum())}")
        if d.get("imbalance"):
            add("  Covariate balance (median cases vs controls, MWU p):")
            for cov, (mc, mk, p) in d["imbalance"].items():
                flag = "  <-- IMBALANCED" if p < 0.05 else ""
                add(
                    f"    {cov:<5} cases={mc:>8.2f}  controls={mk:>8.2f}  "
                    f"p={p:.3e}{flag}"
                )
        add("")
        add("  Spearman rho of panel score vs covariate:")
        sub = df[df["cohort"] == gse]
        for model in sorted(sub["model"].unique()):
            add(f"    [{model}]")
            for cov in _COVARIATES:
                cells = []
                for grp in ("all", "controls", "cases"):
                    r = sub[
                        (sub["model"] == model)
                        & (sub["covariate"] == cov)
                        & (sub["group"] == grp)
                    ]
                    if r.empty:
                        cells.append(f"{grp}=n/a")
                    else:
                        rr = r.iloc[0]
                        star = "*" if rr["p"] < 0.05 else " "
                        cells.append(
                            f"{grp}: rho={rr['spearman_rho']:+.3f}{star} "
                            f"(p={rr['p']:.2f}, n={int(rr['n'])})"
                        )
                add(f"      {cov:<5} " + "  ".join(cells))
        add("")

    if not adj_df.empty:
        add("-" * 74)
        add(f"ADJUSTED ASSOCIATION (logistic regression, {_CRIT_LABEL} / zscore)")
        add("-" * 74)
        add("In-sample AUC of nested models. If the panel score were a proxy for")
        add("tissue quality, 'covariates only' would match 'score only', and the")
        add("adjusted score coefficient would collapse toward zero.")
        add("")
        add(
            f"{'cohort':<11}{'model':<7}{'n':>4}{'score':>8}{'covars':>8}"
            f"{'both':>8}{'beta_score':>12}"
        )
        for _, r in adj_df.iterrows():
            add(
                f"{r['cohort']:<11}{r['model']:<7}{int(r['n']):>4}"
                f"{r['auc_score_only']:>8.4f}{r['auc_covariates_only']:>8.4f}"
                f"{r['auc_score_plus_covariates']:>8.4f}"
                f"{r['beta_score_adjusted']:>+12.3f}"
            )
        add("")
        add("  covariates used: " + str(adj_df["covariates_used"].iloc[0]))
        add("  Note these are in-sample (not cross-validated) AUCs, used only to")
        add("  compare nested models on the same samples; they are not")
        add("  comparable to the zero-shot AUCs of Step 35.")
        add("")

    add("-" * 74)
    add("CAVEATS")
    add("-" * 74)
    add("1. n is small in both cohorts (HD 20 cases, PD 29 cases), so covariate")
    add("   adjustment is underpowered and absence of confounding is not proven.")
    add("2. GSE68719 is all-male; sex cannot be assessed as a covariate there.")
    add("3. The two cohorts share controls, so their covariate distributions are")
    add("   not independent.")
    add("4. Adjusted AUCs are in-sample and serve only for nested-model")
    add("   comparison.")
    add("")

    (SCRIPT_DIR / f"{PREFIX}_statistics.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _plot(store: dict) -> None:
    """Small multiples: panel score vs each covariate, per cohort."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    # LightGBM is the primary model in Step 35, so the figure shows it too.
    model = "LGBM"
    covs = list(_COVARIATES)
    gses = list(store)
    fig, axes = plt.subplots(
        len(gses),
        len(covs),
        figsize=(4.6 * len(covs), 4.1 * len(gses)),
        squeeze=False,
    )

    for i, gse in enumerate(gses):
        d = store[gse]
        if model not in d["preds"]:
            continue
        s = d["preds"][model]
        y = d["y"]
        for j, cov in enumerate(covs):
            ax = axes[i][j]
            v = d["cov"][cov].values.astype(float)
            ok = ~np.isnan(v)
            # Controls in grey as a constant reference in every panel
            mc = ok & (y == 0)
            md = ok & (y == 1)
            ax.scatter(
                v[mc],
                s[mc],
                s=34,
                c="#9aa0a6",
                edgecolor="black",
                linewidth=0.4,
                label="controls (shared)",
                zorder=3,
            )
            ax.scatter(
                v[md],
                s[md],
                s=40,
                c="#d65f5f",
                edgecolor="black",
                linewidth=0.4,
                label=d["disease"],
                zorder=4,
            )
            if ok.sum() >= 6:
                z = np.polyfit(v[ok], s[ok], 1)
                xs = np.linspace(np.nanmin(v[ok]), np.nanmax(v[ok]), 50)
                ax.plot(xs, np.polyval(z, xs), ls="--", c="black", lw=1.1, zorder=5)
                rho, p = spearmanr(v[ok], s[ok])
                p_str = f"{p:.2f}" if p >= 0.01 else f"{p:.1e}"
                ax.set_title(
                    f"{gse} — {cov.upper()}\n"
                    f"$\\rho$={rho:+.3f}, p={p_str} (n={int(ok.sum())})",
                    fontsize=10,
                )
            ax.set_xlabel(cov.upper())
            ax.set_ylabel("ALS panel score" if j == 0 else "")
            ax.grid(alpha=0.25)
            if i == 0 and j == 0:
                ax.legend(fontsize=7.5, loc="lower left")

    fig.suptitle(
        "Panel score vs post-mortem covariates — controls (grey) are the same "
        "samples in both rows",
        fontsize=12.5,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / f"{PREFIX}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
