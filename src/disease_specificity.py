# ruff: noqa: E402
"""Disease specificity: does the ALS panel score FTD as ALS-like?

GSE153960 GPL24676 contains four disease groups accessible via raw metadata:

  ALS Spectrum MND                              : ~684  (training positives)
  Non-Neurological Control                      : ~190  (training negatives)
  Other Neurological Disorders                  : 125 samples, 49 donors,
                                                  clinical FTD (NYGC donor table)
  ALS Spectrum MND, Other Neurological Disorders: ~172  ALS-FTD comorbid
                                                         (in training as ALS)

Strategy
--------
1. Parse the GPL24676 series matrix directly (all samples, no label filter).
2. Load supplementary RSEM counts for all samples.
3. Assign group labels from the "group" characteristic field.
4. Define training set: ALS (pure + ALS-FTD) vs Control.
5. Extract the 25 core panel genes (log1p).
6. Donor-grouped 5-fold CV with the within-cohort LightGBM: out-of-fold scores
   for ALS and controls, fold-averaged scores for the held-out FTD group.
7. Donor-level tests, and a genotype test inside the FTD group.
8. Plot score distributions (violin) + ROC curve (ALS vs FTD).
9. Mann-Whitney U tests: FTD vs ALS and FTD vs Control.

Outputs
-------
  disease_specificity.png
  disease_specificity_statistics.txt
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ALS_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ALS_DIR))
sys.path.insert(0, str(Path(__file__).parent))

from utils import _load_suppl_expression, _parse_series_matrix

SCRIPT_DIR = Path(__file__).parent
RESOURCES_DIR = ALS_DIR / "resources"

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"
_MATRIX_PATH = (
    RESOURCES_DIR / "GSE153960" / "matrix" / "GSE153960-GPL24676_series_matrix.txt.gz"
)

N_BOOTSTRAP = 2_000
RANDOM_STATE = 42

# Group label patterns (from series matrix "group" characteristic field)
_ALS_RE = "^ALS Spectrum MND$"
_ALS_FTD_RE = "^ALS Spectrum MND, Other Neurological Dis"
_FTLD_RE = r"^Other Neurological Dis"  # covers typo "DIsorders"
_CTRL_RE = "^Non-Neurological Control"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_all_samples() -> tuple[
    np.ndarray,  # X_log  (n_all, n_genes)  log1p RSEM
    list[str],  # feature_names
    dict[str, list[int]],  # group_idx: group_name → list of row indices
    "pd.DataFrame",  # meta_df aligned to X_log rows (donor, tissue)
]:
    """Parse GPL24676 series matrix and load expression for ALL samples."""

    print(f"  Parsing series matrix: {_MATRIX_PATH.name} ...")
    meta_df, _ = _parse_series_matrix(_MATRIX_PATH)
    print(f"  Total samples in matrix: {len(meta_df)}")

    print("  Loading supplementary RSEM counts ...")
    expr_df = _load_suppl_expression("GSE153960", meta_df, RESOURCES_DIR)

    # Align to samples present in both metadata and expression file
    common = meta_df.index.intersection(expr_df.columns)
    meta_df = meta_df.loc[common]
    expr_df = expr_df[common]
    print(f"  Samples with expression data: {len(common)}")

    group_col = meta_df["group"].fillna("").astype(str)

    als_mask = group_col.str.match(_ALS_RE, na=False)
    alsftd_mask = group_col.str.match(_ALS_FTD_RE, na=False)
    ftld_mask = group_col.str.match(_FTLD_RE, na=False) & ~alsftd_mask
    ctrl_mask = group_col.str.match(_CTRL_RE, na=False)

    counts = {
        "ALS (pure)": als_mask.sum(),
        "ALS-FTD": alsftd_mask.sum(),
        "FTD": ftld_mask.sum(),
        "Control": ctrl_mask.sum(),
        "Other/excluded": (~als_mask & ~alsftd_mask & ~ftld_mask & ~ctrl_mask).sum(),
    }
    for grp, n in counts.items():
        print(f"    {grp:<25} n={n}")


    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        X_log = np.log1p(expr_df.values.T.astype(np.float32))

    group_idx: dict[str, list[int]] = {
        "ALS (pure)": [i for i, m in enumerate(als_mask) if m],
        "ALS-FTD": [i for i, m in enumerate(alsftd_mask) if m],
        "FTD": [i for i, m in enumerate(ftld_mask) if m],
        "Control": [i for i, m in enumerate(ctrl_mask) if m],
    }

    return X_log, list(expr_df.index), group_idx, meta_df


def _ftd_genotypes(donors: np.ndarray) -> "pd.Series":
    """Classify FTD donors by the pathology their mutated gene implies.

    TDP-43: C9orf72 expansion (ExpansionHunter/RP-PCR call), GRN or TBK1 variant.
    tau: MAPT variant. Donors with neither are left out. Source: NYGC ALS
    Consortium donor table (ST1, ST2; Zenodo 20025792).
    """
    import pandas as pd
    from nygc_clinical_association import ST1, _st1

    if not ST1.exists():  # donor table not downloaded: no genotype test
        print(f"  [skip] genotype test needs {ST1.name}")
        return pd.Series(dtype=str)
    st1 = _st1().reindex(donors)
    raw = pd.read_excel(ST1, sheet_name="ST2", header=None)
    h = raw.index[raw.iloc[:, 0].astype(str).str.strip().eq("donor_id")][0]
    st2 = raw.iloc[h + 1 :].copy()
    st2.columns = raw.iloc[h].astype(str).str.strip()
    st2_genes = st2.groupby(st2["donor_id"].astype(str))["annotated_gene"].apply(
        lambda g: " ".join(g.astype(str))
    )
    site = st1["site_reported_mutations"].astype(str)
    genes = st2_genes.reindex(donors).fillna("")
    tdp = (
        (st1["c9"] == 1)
        | site.str.contains("Positive for (?:C9orf72|GRN|TBK1)", regex=True)
        | genes.str.contains(r"(?:^|\s)C9orf72(?:\s|$)|GRN|TBK1", regex=True)
    )
    tau = site.str.contains("MAPT") | genes.str.contains("MAPT")
    out = pd.Series(np.where(tdp & ~tau, "TDP-43", np.where(tau & ~tdp, "tau", "")), index=donors)
    return out[out != ""]


# ---------------------------------------------------------------------------
# Bootstrap AUC
# ---------------------------------------------------------------------------


_FPR_GRID = np.linspace(0, 1, 200)


def _bootstrap_roc(
    y: np.ndarray, scores: np.ndarray, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (aucs, tpr_lo, tpr_hi) where tpr bands are on _FPR_GRID."""
    from sklearn.metrics import roc_auc_score, roc_curve

    aucs: list[float] = []
    tprs: list[np.ndarray] = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(y[idx], scores[idx])
        tprs.append(np.interp(_FPR_GRID, fpr_b, tpr_b))
        aucs.append(roc_auc_score(y[idx], scores[idx]))
    a = np.array(aucs)
    t = np.array(tprs)
    return a, np.percentile(t, 2.5, axis=0), np.percentile(t, 97.5, axis=0)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _plot(
    group_scores: dict[str, np.ndarray],
    y_ftld: np.ndarray,
    scores_ftld: np.ndarray,
    auc_als_vs_ftld: float,
    ci_lo_ftld: float,
    ci_hi_ftld: float,
    tpr_lo_ftld: np.ndarray,
    tpr_hi_ftld: np.ndarray,
    fpr_cv: np.ndarray,
    tpr_cv: np.ndarray,
    auc_cv: float,
    ci_lo_cv: float,
    ci_hi_cv: float,
    tpr_lo_cv: np.ndarray,
    tpr_hi_cv: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Violin plot of score distributions ---
    ax = axes[0]
    order = ["Control", "FTD", "ALS-FTD", "ALS (pure)"]
    colours = ["#4878d0", "#ee854a", "#d65f5f", "#6acc65"]
    data = [group_scores[g] for g in order]
    parts = ax.violinplot(
        data, positions=range(len(order)), showmedians=True, showextrema=True
    )
    for i, (pc, col) in enumerate(zip(parts["bodies"], colours)):
        pc.set_facecolor(col)
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    parts["cmaxes"].set_color("black")
    parts["cmins"].set_color("black")
    parts["cbars"].set_color("black")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=9)
    ax.set_ylabel("ALS probability (model score)")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="grey", linestyle="--", lw=0.8, label="Decision boundary")
    ax.set_title(
        "ALS panel score distributions — GPL24676\n"
        "(train: ALS+ALS-FTD vs Control; scored: all groups)",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for i, g in enumerate(order):
        med = float(np.median(group_scores[g]))
        ax.text(i, med + 0.04, f"{med:.2f}", ha="center", va="bottom", fontsize=7)

    # --- ROC: ALS vs FTD + 5-fold CV reference ---
    ax2 = axes[1]
    fpr_ftld, tpr_ftld, _ = roc_curve(y_ftld, scores_ftld)

    ax2.fill_between(_FPR_GRID, tpr_lo_ftld, tpr_hi_ftld, alpha=0.2, color="#ee854a")
    ax2.plot(
        fpr_ftld,
        tpr_ftld,
        lw=2,
        color="#ee854a",
        label=f"ALS vs FTD  AUC={auc_als_vs_ftld:.4f}\n"
        f"95% CI [{ci_lo_ftld:.4f}, {ci_hi_ftld:.4f}]",
    )
    ax2.fill_between(_FPR_GRID, tpr_lo_cv, tpr_hi_cv, alpha=0.2, color="#4878d0")
    ax2.plot(
        fpr_cv,
        tpr_cv,
        lw=2,
        color="#4878d0",
        label=f"ALS vs Control   AUC={auc_cv:.4f}  (donor-grouped CV)\n"
        f"95% CI [{ci_lo_cv:.4f}, {ci_hi_cv:.4f}]",
    )
    ax2.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title(
        "ROC: ALS vs FTD  (zero-shot scoring of held-out groups)\n"
        "FTD = 'Other Neurological Disorders' in GPL24676",
        fontsize=9,
    )
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = SCRIPT_DIR / "disease_specificity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import json

    import matplotlib
    import pandas as pd
    from scipy.stats import mannwhitneyu
    from sklearn.metrics import roc_auc_score
    from utils import PanelLGBMClassifier as LGBMClassifier

    matplotlib.use("Agg")

    print("\n" + "=" * 65)
    print("Disease Specificity: ALS panel on FTD (GPL24676)")
    print("=" * 65)

    panel = pd.read_csv(_PANEL_CSV)
    panel_features: list[str] = panel["feature"].tolist()

    print("\nLoading all GPL24676 samples ...")
    X_log, feat_names, group_idx, meta_df = _load_all_samples()
    donor_all = meta_df["subject id"].astype(str).to_numpy()
    tissue_all = meta_df["tissue"].astype(str).to_numpy()

    # Map panel features to column indices (strip version suffix)
    feat_base = {n.split(".")[0]: i for i, n in enumerate(feat_names)}
    panel_cols: list[int] = []
    missing: list[str] = []
    for feat in panel_features:
        base = feat.split(".")[0]
        if base in feat_base:
            panel_cols.append(feat_base[base])
        else:
            missing.append(feat)
    print(f"\nPanel coverage: {len(panel_cols)}/25 genes found in expression matrix")
    if missing:
        print(f"  Missing: {', '.join(missing)}")

    # Extract panel features for all samples
    X_panel = X_log[:, panel_cols]  # (n_all, n_panel)

    # Training mask: ALS (pure + ALS-FTD) vs Control
    tr_als_idx = group_idx["ALS (pure)"] + group_idx["ALS-FTD"]
    tr_ctrl_idx = group_idx["Control"]
    tr_idx = tr_als_idx + tr_ctrl_idx
    y_train = np.array([1] * len(tr_als_idx) + [0] * len(tr_ctrl_idx), dtype=int)

    print(
        f"\nTraining set: n={len(tr_idx)}  "
        f"ALS={len(tr_als_idx)}  Ctrl={len(tr_ctrl_idx)}"
    )

    # the paper's within-cohort model (top-500 params; PanelLGBMClassifier rescales
    # colsample to the panel size), folds grouped by donor
    params = {**json.loads(_PARAMS_PATH.read_text()), "verbose": -1}

    # Donor-grouped 5-fold CV: OOF scores for training groups AND fold-averaged
    # scores for the held-out FTD group. Both are scored by the same 5 fold models
    # (none of which ever saw an FTD sample), so the two are symmetric.
    from sklearn.metrics import roc_curve as _roc_curve
    from sklearn.model_selection import StratifiedGroupKFold

    print("\nRunning donor-grouped 5-fold CV ...")
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    X_tr = X_panel[tr_idx]
    X_ftld = X_panel[group_idx["FTD"]]
    oof_scores = np.zeros(len(tr_idx))
    ftld_fold_scores = np.zeros((5, len(group_idx["FTD"])))
    for k, (fold_tr, fold_val) in enumerate(skf.split(X_tr, y_train, donor_all[tr_idx])):
        clf_fold = LGBMClassifier(**params)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            clf_fold.fit(X_tr[fold_tr], y_train[fold_tr])
        oof_scores[fold_val] = clf_fold.predict_proba(X_tr[fold_val])[:, 1]
        ftld_fold_scores[k] = clf_fold.predict_proba(X_ftld)[:, 1]
    ftld_cv_scores = ftld_fold_scores.mean(axis=0)  # fold-averaged FTD scores

    # Honest scores: OOF for training groups, fold-averaged CV for FTD
    # All scores come from the same 5 fold models — no asymmetry.
    n_als_pure = len(group_idx["ALS (pure)"])
    honest_scores: dict[str, np.ndarray] = {
        "ALS (pure)": oof_scores[:n_als_pure],
        "ALS-FTD": oof_scores[n_als_pure : len(tr_als_idx)],
        "Control": oof_scores[len(tr_als_idx) :],
        "FTD": ftld_cv_scores,
    }
    print("\nHonest (OOF / fold-averaged CV) score distributions:")
    for g, sc in honest_scores.items():
        print(f"  {g:<25}  median={np.median(sc):.4f}  mean={np.mean(sc):.4f}")

    rng = np.random.default_rng(RANDOM_STATE)

    # AUC: ALS (pure, OOF) vs FTD (zero-shot) — fully honest
    y_als_ftld = np.concatenate(
        [
            np.ones(n_als_pure, dtype=int),
            np.zeros(len(group_idx["FTD"]), dtype=int),
        ]
    )
    scores_als_ftld = np.concatenate(
        [
            honest_scores["ALS (pure)"],
            honest_scores["FTD"],
        ]
    )
    auc_als_vs_ftld = float(roc_auc_score(y_als_ftld, scores_als_ftld))
    print(f"\nAUC (ALS pure OOF vs FTD zero-shot): {auc_als_vs_ftld:.4f}")
    boot_ftld, tpr_lo_ftld, tpr_hi_ftld = _bootstrap_roc(
        y_als_ftld, scores_als_ftld, N_BOOTSTRAP, rng
    )
    ci_lo = float(np.percentile(boot_ftld, 2.5))
    ci_hi = float(np.percentile(boot_ftld, 97.5))
    print(f"  Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    # AUC: ALS vs Control (OOF) — reference curve
    auc_als_vs_ctrl = float(roc_auc_score(y_train, oof_scores))
    fpr_cv, tpr_cv, _ = _roc_curve(y_train, oof_scores)
    boot_cv, tpr_lo_cv, tpr_hi_cv = _bootstrap_roc(
        y_train, oof_scores, N_BOOTSTRAP, rng
    )
    ci_lo_cv = float(np.percentile(boot_cv, 2.5))
    ci_hi_cv = float(np.percentile(boot_cv, 97.5))
    print(
        f"AUC (ALS vs Control, OOF):         {auc_als_vs_ctrl:.4f}  "
        f"95% CI [{ci_lo_cv:.4f}, {ci_hi_cv:.4f}]"
    )

    # Donor-level tests: several regions per donor, so a sample-level p-value would
    # count one donor several times. Scores are z-scored within tissue (over the
    # four groups) and averaged per donor.
    idx_all = np.concatenate([np.asarray(group_idx[g], dtype=int) for g in honest_scores])
    sc_all = np.concatenate(list(honest_scores.values()))
    lab_all = np.concatenate([[g] * len(v) for g, v in honest_scores.items()])
    dfs = pd.DataFrame({"s": sc_all, "tissue": tissue_all[idx_all],
                        "donor": donor_all[idx_all], "group": lab_all})
    dfs["z"] = dfs.groupby("tissue")["s"].transform(lambda c: (c - c.mean()) / (c.std(ddof=0) or 1.0))
    dmean = dfs.groupby(["group", "donor"])["z"].mean()
    n_donors = {g: int(dmean.loc[g].size) for g in honest_scores}
    _, p_ftld_als = mannwhitneyu(dmean.loc["FTD"], dmean.loc["ALS (pure)"], alternative="less")
    _, p_ftld_ctrl = mannwhitneyu(dmean.loc["FTD"], dmean.loc["Control"], alternative="greater")
    _, p_alsftd_als = mannwhitneyu(dmean.loc["ALS-FTD"], dmean.loc["ALS (pure)"], alternative="two-sided")
    print(f"\nDonors: {n_donors}")
    print(f"Donor-level MWU (FTD < ALS pure):     p={p_ftld_als:.3e}")
    print(f"Donor-level MWU (FTD > Control):      p={p_ftld_ctrl:.3e}")
    print(f"Donor-level MWU (ALS-FTD ≠ ALS pure): p={p_alsftd_als:.3e}")

    # Genotype test inside the FTD group: donors carrying a variant in a gene whose
    # pathology is TDP-43 (C9orf72, GRN, TBK1) vs tau (MAPT), from the NYGC
    # donor table (ST1 C9 call and site-reported mutations, ST2 annotated gene).
    geno = _ftd_genotypes(np.unique(donor_all[group_idx["FTD"]]))
    ftd_d = dfs[dfs.group == "FTD"].groupby("donor")["s"].mean()
    tdp = ftd_d[ftd_d.index.isin(geno[geno == "TDP-43"].index)]
    tau = ftd_d[ftd_d.index.isin(geno[geno == "tau"].index)]
    unk = ftd_d[~ftd_d.index.isin(geno.index)]
    if len(tdp) and len(tau):
        _, p_geno = mannwhitneyu(tdp, tau, alternative="two-sided")
        auc_geno = float(roc_auc_score(np.r_[np.ones(len(tdp)), np.zeros(len(tau))], np.r_[tdp, tau]))
    else:
        p_geno, auc_geno = float("nan"), float("nan")
    print(f"FTD genotype: TDP-43 genes n={len(tdp)} median={tdp.median():.3f} | "
          f"MAPT n={len(tau)} median={tau.median():.3f} | unknown n={len(unk)} "
          f"median={unk.median():.3f} | AUC {auc_geno:.3f} p={p_geno:.3g}")

    print("\nGenerating plot ...")
    _plot(
        honest_scores,
        y_als_ftld,
        scores_als_ftld,
        auc_als_vs_ftld,
        ci_lo,
        ci_hi,
        tpr_lo_ftld,
        tpr_hi_ftld,
        fpr_cv,
        tpr_cv,
        auc_als_vs_ctrl,
        ci_lo_cv,
        ci_hi_cv,
        tpr_lo_cv,
        tpr_hi_cv,
    )

    # Statistics file
    lines = [
        "Disease Specificity Analysis — GSE153960 GPL24676",
        "=" * 65,
        "Training set : ALS Spectrum MND (pure + ALS-FTD) vs Non-Neurological Control",
        "Model: within-cohort LightGBM (top-500 params, colsample rescaled), donor-grouped 5-fold CV",
        f"  ALS in training : {len(tr_als_idx)} (pure ALS={len(group_idx['ALS (pure)'])},"
        f" ALS-FTD={len(group_idx['ALS-FTD'])})",
        f"  Control         : {len(tr_ctrl_idx)}",
        f"Panel genes used  : {len(panel_cols)}/25",
        "",
        "Group score distributions (honest: OOF for training groups, zero-shot for FTD):",
    ]
    for g, sc in honest_scores.items():
        lines.append(
            f"  {g:<30} n={len(sc):3d}  median={np.median(sc):.4f}"
            f"  mean={np.mean(sc):.4f}  IQR=[{np.percentile(sc, 25):.4f},"
            f"{np.percentile(sc, 75):.4f}]"
        )
    lines += [
        "",
                f"AUC ALS (pure) vs FTD   : {auc_als_vs_ftld:.4f}",
        f"Bootstrap 95% CI             : [{ci_lo:.4f}, {ci_hi:.4f}]  (n={N_BOOTSTRAP})",
        f"AUC ALS vs Control (donor-grouped OOF): {auc_als_vs_ctrl:.4f}  "
        f"Bootstrap 95% CI [{ci_lo_cv:.4f}, {ci_hi_cv:.4f}]  (reference)",
        "",
        f"Donors: {n_donors}",
        "Donor-level Mann-Whitney U (score z-scored within tissue, averaged per donor):",
        f"  FTD < ALS pure     : p={p_ftld_als:.3e}",
        f"  FTD > Control      : p={p_ftld_ctrl:.3e}",
        f"  ALS-FTD ≠ ALS pure : p={p_alsftd_als:.3e}",
        "",
        "FTD donors by genotype (NYGC ST1/ST2; donor mean of the fold-averaged score):",
        f"  TDP-43 genes (C9orf72, GRN, TBK1): n={len(tdp)}  median={tdp.median():.4f}",
        f"  MAPT (tau)                       : n={len(tau)}  median={tau.median():.4f}",
        f"  no known mutation                : n={len(unk)}  median={unk.median():.4f}",
        f"  TDP-43 vs MAPT: AUC={auc_geno:.4f}  MWU p={p_geno:.3g}",
        "",
        "The 'Other Neurological Disorders' group is clinical FTD (47 of 49 donors in",
        "the NYGC donor table; one FTD with PSP), not a PSP series.",
    ]
    stat_out = SCRIPT_DIR / "disease_specificity_statistics.txt"
    stat_out.write_text("\n".join(lines))
    print(f"  Saved → {stat_out.name}")

    print("\n" + "=" * 65)
    print("DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
