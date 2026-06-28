"""Combined external-validation ROC grid for all 4 CNS cohorts.

Replaces the previous stacked Figures 4 (blood_validation_gpl16791 +
blood_validation_gse76220 + additional_cohort_gse122649) and 5
(srp064478_validation.png two-panel). Produces:

  combined_validation_roc.png — 2x2 grid: GPL16791, GSE76220, GSE122649,
                                SRP064478. Each panel: 25-gene + 15-crit
                                LGBM zero-shot ROC overlaid.

  substitution_roc_all.png   — substitution-augmented ROC overlay across
                                cohorts where substitution is informative
                                (GSE76220 and GSE122649; SRP064478 is at
                                ceiling so excluded).
"""
from __future__ import annotations

import gzip
import json
import re
import sys
import tarfile
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ALS_DIR = Path(__file__).parents[1]
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset  # noqa: E402

from cross_cohort_substitution import (  # noqa: E402
    _load_gpl24676_ctd,
    _load_gse76220,
    _load_gse122649,
    _load_srp064478,
    _ensemble_scores,
)

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"

_CRITICAL_IDX = [1, 2, 3, 5, 6, 7, 10, 11, 12, 15, 16, 18, 20, 23, 24]

_COMPARTMENTS: dict[str, tuple[str, ...]] = {
    "erythrocyte": (
        "ENSG00000206172", "ENSG00000188536", "ENSG00000244734",
        "ENSG00000158578", "ENSG00000170180", "ENSG00000133742",
        "ENSG00000159111", "ENSG00000105610", "ENSG00000179364",
        "ENSG00000223609",
    ),
    "platelet": (
        "ENSG00000163736", "ENSG00000163737", "ENSG00000185245",
    ),
    "endothelial": (
        "ENSG00000261371", "ENSG00000179776", "ENSG00000110799",
    ),
}

RANDOM_STATE = 42
N_BOOTSTRAP = 2000
N_SEEDS = 5

# Substitution wins from cross_cohort_substitution.py
_SUBSTITUTIONS = {
    "GSE76220":  [("HERC2P8", "ZNF586",  "ENSG00000083828")],
    "GSE122649": [("HERC2P8", "XAF1",    "ENSG00000132530")],
}


def _bootstrap_tpr(y, scores, n=N_BOOTSTRAP, seed=RANDOM_STATE):
    """Return mean fpr grid + 2.5/97.5 percentile TPR bands + AUC CI."""
    from sklearn.metrics import roc_auc_score, roc_curve
    rng = np.random.default_rng(seed)
    fpr_grid = np.linspace(0, 1, 200)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    aucs, tprs = [], []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        try:
            fpr_b, tpr_b, _ = roc_curve(y[idx], scores[idx])
            tprs.append(np.interp(fpr_grid, fpr_b, tpr_b))
            aucs.append(roc_auc_score(y[idx], scores[idx]))
        except ValueError:
            continue
    aucs = np.array(aucs)
    tprs = np.array(tprs)
    return (fpr_grid,
            np.percentile(tprs, 2.5, axis=0),
            np.percentile(tprs, 97.5, axis=0),
            float(np.percentile(aucs, 2.5)),
            float(np.percentile(aucs, 97.5)))


def _load_gpl16791_ctd(X_train_log, feat_train):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    (ds16,) = load_dataset("GSE153960", platform="GPL16791",
                            resources_dir=ALS_DIR / "resources")
    X16_raw = ds16.X.values.astype(np.float32)
    y16 = ds16.y.values.astype(int)
    feat16 = list(ds16.X.columns)
    assert feat16 == feat_train
    X16_log = np.log1p(X16_raw)
    base_ids = [n.split(".")[0] for n in feat16]
    pcs_tr, pcs_te = [], []
    for comp_bases in _COMPARTMENTS.values():
        base_set = set(comp_bases)
        idx = [i for i, b in enumerate(base_ids) if b in base_set]
        if not idx:
            continue
        pca = PCA(n_components=1, random_state=0)
        pcs_tr.append(pca.fit_transform(X_train_log[:, idx]))
        pcs_te.append(pca.transform(X16_log[:, idx]))
    Z_tr = np.hstack(pcs_tr)
    Z_te = np.hstack(pcs_te)
    reg = LinearRegression().fit(Z_tr, X_train_log)
    X16_ctd = (X16_log - reg.predict(Z_te)).astype(np.float32)
    return X16_ctd, y16


def _train_predict(Xtr, y_tr, Xte, params):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    scores = _ensemble_scores(sc.transform(Xtr), y_tr, sc.transform(Xte),
                                params, seeds=tuple(range(N_SEEDS)))
    return scores


def main() -> None:
    from sklearn.metrics import roc_auc_score, roc_curve
    warnings.filterwarnings("ignore")

    params_raw = json.loads(_PARAMS_PATH.read_text())
    params = dict(params_raw, colsample_bytree=1.0, n_jobs=-1, verbose=-1)

    df_panel = pd.read_csv(_PANEL_CSV)
    feat_col = next(c for c in df_panel.columns
                     if "feature" in c.lower() or "ensg" in c.lower())
    sym_col = next(c for c in df_panel.columns if "symbol" in c.lower())
    feat25 = df_panel[feat_col].tolist()
    sym25 = df_panel[sym_col].tolist()
    feat25_bases = [f.split(".")[0] for f in feat25]
    panel_syms = [sym25[i] for i in _CRITICAL_IDX]
    panel_bases = [feat25_bases[i] for i in _CRITICAL_IDX]

    # Load training data (twice — CTD for GPL16791, raw for others)
    print("Loading GPL24676 (CTD + log1p) ...")
    X_train_ctd, y_train, X_train_log, feat_train = _load_gpl24676_ctd()
    feat_base_to_col = {f.split(".")[0]: j for j, f in enumerate(feat_train)}

    # Cohort spec list:
    # (cohort_name, y_test, X_test, vocab_map, by_symbol, preprocessing)
    cohort_data = []

    print("Loading GPL16791 (raw log1p; CTD is discovery-side only) ...")
    (ds16,) = load_dataset("GSE153960", platform="GPL16791", resources_dir=ALS_DIR / "resources")
    assert list(ds16.X.columns) == feat_train
    X16_raw = np.log1p(ds16.X.values.astype(np.float32))
    y16 = ds16.y.values.astype(int)
    cohort_data.append({
        "name": "GPL16791", "tissue": "Multi-region CNS", "n": len(y16),
        "y": y16, "X_te": X16_raw, "vocab": None, "by_symbol": False,
        "X_train_use": X_train_log,
    })

    print("Loading GSE76220 ...")
    X76, vocab76, y76 = _load_gse76220()
    cohort_data.append({
        "name": "GSE76220", "tissue": "Lumbar SC (LCM)", "n": len(y76),
        "y": y76, "X_te": X76, "vocab": {s: i for i, s in enumerate(vocab76)},
        "by_symbol": True, "X_train_use": X_train_log,
    })

    print("Loading GSE122649 ...")
    X122, vocab122, y122 = _load_gse122649()
    cohort_data.append({
        "name": "GSE122649", "tissue": "Motor cortex", "n": len(y122),
        "y": y122, "X_te": X122,
        "vocab": {s: i for i, s in enumerate(vocab122)},
        "by_symbol": True, "X_train_use": X_train_log,
    })

    print("Loading SRP064478 ...")
    Xsrp, vocab_srp, y_srp = _load_srp064478()
    cohort_data.append({
        "name": "SRP064478", "tissue": "Cervical SC", "n": len(y_srp),
        "y": y_srp, "X_te": Xsrp,
        "vocab": {v: i for i, v in enumerate(vocab_srp)},
        "by_symbol": False, "X_train_use": X_train_log,
    })

    # Helper to select panel columns for a cohort
    def _panel_cols(cohort, indices):
        v = cohort["vocab"]
        if cohort["by_symbol"]:
            te_syms = [sym25[i] for i in indices]
            tr_cols = [feat_base_to_col[feat25_bases[i]] for i in indices
                       if sym25[i] in v]
            te_cols = [v[s] for s in te_syms if s in v]
        else:
            te_bases = [feat25_bases[i] for i in indices]
            if cohort["name"] == "GPL16791":
                # GPL16791 uses same feat_train ordering as training
                tr_cols = [feat_base_to_col[b] for b in te_bases]
                te_cols = [feat_base_to_col[b] for b in te_bases]
            else:
                tr_cols = [feat_base_to_col[b] for b in te_bases
                            if b in cohort["vocab"]]
                te_cols = [cohort["vocab"][b] for b in te_bases
                            if b in cohort["vocab"]]
        return tr_cols, te_cols

    # ---- Figure 1: combined 2x2 ROC grid (25-gene + 15-crit) ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes_flat = axes.flatten()
    summary = []
    for ax, cohort in zip(axes_flat, cohort_data):
        # 25-gene
        tr25, te25 = _panel_cols(cohort, list(range(25)))
        scores25 = _train_predict(cohort["X_train_use"][:, tr25], y_train,
                                     cohort["X_te"][:, te25], params)
        fpr25, tpr25, _ = roc_curve(cohort["y"], scores25)
        auc25 = roc_auc_score(cohort["y"], scores25)
        _, lo25_band, hi25_band, lo25, hi25 = _bootstrap_tpr(cohort["y"], scores25)

        # 15-crit
        tr15, te15 = _panel_cols(cohort, _CRITICAL_IDX)
        scores15 = _train_predict(cohort["X_train_use"][:, tr15], y_train,
                                     cohort["X_te"][:, te15], params)
        fpr15, tpr15, _ = roc_curve(cohort["y"], scores15)
        auc15 = roc_auc_score(cohort["y"], scores15)
        _, lo15_band, hi15_band, lo15, hi15 = _bootstrap_tpr(cohort["y"], scores15)

        # Plot
        ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=0.7)
        # CI bands
        fpr_grid = np.linspace(0, 1, 200)
        ax.fill_between(fpr_grid, lo25_band, hi25_band,
                          color="#1f77b4", alpha=0.15)
        ax.fill_between(fpr_grid, lo15_band, hi15_band,
                          color="#d62728", alpha=0.15)
        ax.plot(fpr25, tpr25, color="#1f77b4", lw=2.0,
                  label=f"25-gene ({len(te25)}/25)  AUC = {auc25:.3f} "
                        f"[{lo25:.2f}, {hi25:.2f}]")
        ax.plot(fpr15, tpr15, color="#d62728", lw=2.0,
                  label=f"15-crit ({len(te15)}/15)  AUC = {auc15:.3f} "
                        f"[{lo15:.2f}, {hi15:.2f}]")
        ax.set_xlabel("False positive rate", fontsize=9)
        ax.set_ylabel("True positive rate", fontsize=9)
        ax.set_title(f"{cohort['name']} — {cohort['tissue']} "
                       f"($n = {cohort['n']}$)", fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
        summary.append({"cohort": cohort["name"], "auc25": auc25, "auc15": auc15})

    fig.suptitle(
        "External-validation ROC across four CNS cohorts — 25-gene panel "
        "(blue) vs 15-gene critical panel (red)\n"
        "GPL24676-trained LGBM, zero-shot transfer; shaded bands = 95% paired bootstrap CI ($B = 2000$).",
        fontsize=10,
    )
    plt.tight_layout()
    out = SCRIPT_DIR / "combined_validation_roc.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    for s in summary:
        print(f"  {s['cohort']}: 25g AUC={s['auc25']:.4f}  15c AUC={s['auc15']:.4f}")

    # ---- Figure 2: substitution ROC across GSE76220 + GSE122649 ----
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 5))

    for ax, cohort_name in zip(axes2, ["GSE76220", "GSE122649"]):
        cohort = next(c for c in cohort_data if c["name"] == cohort_name)
        # Baseline = drop missing genes (14/15 for GSE122649, 13/15 for GSE76220)
        v = cohort["vocab"]
        included = [i for i, sym in zip(_CRITICAL_IDX,
                                            [sym25[k] for k in _CRITICAL_IDX])
                    if sym in v]
        tr_b, te_b = _panel_cols(cohort, included)
        scores_b = _train_predict(cohort["X_train_use"][:, tr_b], y_train,
                                      cohort["X_te"][:, te_b], params)
        fpr_b, tpr_b, _ = roc_curve(cohort["y"], scores_b)
        auc_b = roc_auc_score(cohort["y"], scores_b)
        _, blo, bhi, _, _ = _bootstrap_tpr(cohort["y"], scores_b)

        # Substitution panel: included + substitution columns
        sub_info = _SUBSTITUTIONS[cohort_name]
        tr_extra = [feat_base_to_col[eg] for _, _, eg in sub_info
                    if eg in feat_base_to_col]
        te_extra = [v[sym] for _, sym, _ in sub_info if sym in v]
        Xtr_sub = np.hstack([cohort["X_train_use"][:, tr_b],
                              cohort["X_train_use"][:, tr_extra]])
        Xte_sub = np.hstack([cohort["X_te"][:, te_b],
                              cohort["X_te"][:, te_extra]])
        scores_s = _train_predict(Xtr_sub, y_train, Xte_sub, params)
        fpr_s, tpr_s, _ = roc_curve(cohort["y"], scores_s)
        auc_s = roc_auc_score(cohort["y"], scores_s)
        _, slo, shi, _, _ = _bootstrap_tpr(cohort["y"], scores_s)

        fpr_grid = np.linspace(0, 1, 200)
        ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=0.7)
        ax.fill_between(fpr_grid, blo, bhi, color="#888", alpha=0.15)
        ax.fill_between(fpr_grid, slo, shi, color="#1565C0", alpha=0.15)
        sub_label = ", ".join(f"{old}→{new}" for old, new, _ in sub_info)
        ax.plot(fpr_b, tpr_b, color="#888", lw=2.0,
                  label=f"Baseline ({len(te_b)}/15)  AUC = {auc_b:.3f}")
        ax.plot(fpr_s, tpr_s, color="#1565C0", lw=2.0,
                  label=f"With sub: {sub_label}  AUC = {auc_s:.3f}")
        ax.set_xlabel("False positive rate", fontsize=9)
        ax.set_ylabel("True positive rate", fontsize=9)
        ax.set_title(f"{cohort_name} ($n = {cohort['n']}$)", fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)

    fig2.suptitle(
        "Cross-cohort surrogate substitution effect — pseudogene gaps in the 15-crit "
        "panel filled by protein-coding alternatives from the replaceability screen.\n"
        "SRP064478 not shown (already at AUC = 1.000 with the 14/15 baseline).",
        fontsize=10,
    )
    plt.tight_layout()
    out2 = SCRIPT_DIR / "substitution_roc_all.png"
    plt.savefig(out2, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
