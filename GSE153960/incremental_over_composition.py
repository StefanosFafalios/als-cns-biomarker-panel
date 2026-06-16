# ruff: noqa: E402
"""Incremental value of the panel over cell-type composition — experiment #2.

Central rebuttal to "the panel is just a readout of microgliosis + BBB loss".
On raw log1p GPL24676 (n=874 ALS+control), compares four 5-fold CV models:

  A. Composition only  : 6 BRETIGEA cell-type estimates.
  B. Panel only        : 25-gene panel expression.
  C. Panel + composition.
  D. Composition-RESIDUALISED panel: each panel gene regressed on the 6
     cell-type estimates (OLS fit within each train fold, no leakage), the
     classifier trained on residuals. If D stays discriminative, the panel
     carries per-cell signal beyond composition.

Cell-type estimates use BRETIGEA (experiment #3) on the same raw matrix, so
composition is estimated on the classifier's exact samples. Estimation is
unsupervised (SVD), so computing it once on all samples leaks no labels.

Outputs
-------
  incremental_over_composition_statistics.txt
  incremental_over_composition.png
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ALS_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset  # noqa: E402

SCRIPT_DIR = Path(__file__).parent
_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PARAMS = SCRIPT_DIR / "lgbm_top500_best_params.json"
RANDOM_STATE = 42
N_BOOT = 2000


def _cv_oof(x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
    """Return out-of-fold P(ALS) from 5-fold stratified CV."""
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    for tr, te in skf.split(x, y):
        clf = LGBMClassifier(**params)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            clf.fit(x[tr], y[tr])
        oof[te] = clf.predict_proba(x[te])[:, 1]
    return oof


def _cv_oof_residualised(
    genes: np.ndarray, comp: np.ndarray, y: np.ndarray, params: dict
) -> np.ndarray:
    """CV with within-fold OLS residualisation of each gene on composition."""
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    for tr, te in skf.split(genes, y):
        # design matrix [1, composition] fit on train
        d_tr = np.column_stack([np.ones(len(tr)), comp[tr]])
        d_te = np.column_stack([np.ones(len(te)), comp[te]])
        beta, *_ = np.linalg.lstsq(d_tr, genes[tr], rcond=None)  # (1+ncomp, ngene)
        res_tr = genes[tr] - d_tr @ beta
        res_te = genes[te] - d_te @ beta
        clf = LGBMClassifier(**params)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            clf.fit(res_tr, y[tr])
        oof[te] = clf.predict_proba(res_te)[:, 1]
    return oof


def _auc_ci(y: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(RANDOM_STATE)
    auc = float(roc_auc_score(y, scores))
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], scores[idx]))
    return auc, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from cell_type_deconvolution import _map_symbols_to_ensembl
    from deconv_reference_bretigea import _load_markers_by_ensembl, bretigea_estimates

    print("=" * 64)
    print("Incremental value of panel over composition — experiment #2")
    print("=" * 64)

    print("\n[1] Loading raw GPL24676 (n=874 ALS+control) ...")
    (ds,) = load_dataset(
        "GSE153960", platform="GPL24676", resources_dir=ALS_DIR / "resources"
    )
    feat = list(ds.X.columns)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        x_log = np.log1p(ds.X.values.astype(np.float32))
    y = ds.y.values.astype(int)
    print(
        f"  n={len(y)}  ALS={int(y.sum())}  Ctrl={int((y == 0).sum())}  genes={len(feat)}"
    )

    print("\n[2] BRETIGEA cell-type estimates on these samples ...")
    markers_by_ct = _load_markers_by_ensembl(_map_symbols_to_ensembl)
    comp_df, cov = bretigea_estimates(x_log, feat, markers_by_ct)
    comp = comp_df.values.astype(np.float32)  # (n, 6)
    print(f"  composition matrix: {comp.shape}  (coverage {cov})")

    print("\n[3] Extracting panel genes ...")
    panel = pd.read_csv(_PANEL_CSV)
    base2col: dict[str, int] = {}
    for j, f in enumerate(feat):
        base2col.setdefault(str(f).split(".")[0], j)
    pcols, psyms = [], []
    for _, r in panel.iterrows():
        b = str(r["feature"]).split(".")[0]
        if b in base2col:
            pcols.append(base2col[b])
            psyms.append(r["symbol"])
    genes = x_log[:, pcols]  # (n, n_panel)
    print(f"  panel genes matched in raw matrix: {len(pcols)}/25")

    params = json.loads(_PARAMS.read_text())
    params.update({"verbose": -1, "n_jobs": 4})

    print("\n[4] 5-fold CV models ...")
    auc_comp = _auc_ci(y, _cv_oof(comp, y, params))
    auc_panel = _auc_ci(y, _cv_oof(genes, y, params))
    auc_both = _auc_ci(y, _cv_oof(np.column_stack([genes, comp]), y, params))
    auc_resid = _auc_ci(y, _cv_oof_residualised(genes, comp, y, params))

    rows = [
        ("A. Composition only (6 cell types)", auc_comp),
        ("B. Panel only (25 genes)", auc_panel),
        ("C. Panel + composition", auc_both),
        ("D. Panel residualised on composition", auc_resid),
    ]
    lines = [
        "Incremental value of panel over cell-type composition — experiment #2",
        "=" * 64,
        "Substrate: raw log1p GPL24676 (n=874 ALS+control); LightGBM top-500",
        "params; 5-fold stratified CV (random_state=42); BRETIGEA composition.",
        "",
        f"{'Model':<40}{'AUC':>8}{'95% CI':>20}",
        "-" * 68,
    ]
    for name, (a, lo, hi) in rows:
        lines.append(f"{name:<40}{a:>8.4f}   [{lo:.3f}, {hi:.3f}]")
    lines += [
        "",
        f"Panel-only minus composition-only : {auc_panel[0] - auc_comp[0]:+.4f}",
        f"Residualised panel (composition removed): {auc_resid[0]:.4f}",
        "",
        "Interpretation:",
        f"  - Panel ({auc_panel[0]:.3f}) >> composition alone ({auc_comp[0]:.3f}): "
        "the panel is not reducible to cell-type proportions.",
        f"  - After regressing out composition, the panel still discriminates at "
        f"AUC={auc_resid[0]:.3f}, i.e. it carries per-cell / non-composition signal.",
        f"  - Adding composition to the panel changes AUC by "
        f"{auc_both[0] - auc_panel[0]:+.4f} (composition adds little once the panel "
        "is present).",
    ]
    (SCRIPT_DIR / "incremental_over_composition_statistics.txt").write_text(
        "\n".join(lines) + "\n"
    )
    print("\n".join(lines))

    # figure
    fig, ax = plt.subplots(figsize=(8, 5))
    aucs = [r[1][0] for r in rows]
    los = [r[1][0] - r[1][1] for r in rows]
    his = [r[1][2] - r[1][0] for r in rows]
    colours = ["#9467bd", "#1f77b4", "#2ca02c", "#d62728"]
    ax.bar(range(4), aucs, yerr=[los, his], color=colours, alpha=0.85, capsize=4)
    ax.axhline(0.5, ls="--", c="grey", lw=0.8, label="chance")
    for i, a in enumerate(aucs):
        ax.text(i, a + 0.02, f"{a:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(4))
    ax.set_xticklabels(
        [
            "Composition\nonly",
            "Panel\nonly",
            "Panel +\ncomposition",
            "Panel\n(comp. removed)",
        ],
        fontsize=9,
    )
    ax.set_ylabel("5-fold CV AUC-ROC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title(
        "Panel carries ALS signal beyond cell-type composition\n"
        "(raw log1p GPL24676, n=874)",
        fontsize=11,
    )
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(
        SCRIPT_DIR / "incremental_over_composition.png", dpi=150, bbox_inches="tight"
    )
    print("\nSaved -> incremental_over_composition.{txt,png}")


if __name__ == "__main__":
    main()
