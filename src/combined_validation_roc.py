"""Combined external-validation ROC grid for all 4 CNS cohorts.

Replaces the previous stacked Figures 4 (blood_validation_gpl16791 +
blood_validation_gse76220 + additional_cohort_gse122649) and 5
(srp064478_validation.png two-panel). Produces:

  combined_validation_roc.png — 2x2 grid: GPL16791, GSE76220, GSE122649,
                                SRP064478. Each panel: 25-gene + critical panel
                                LGBM zero-shot ROC overlaid.

The substitution ROC overlay (substitution_roc_all.png) is drawn by
replaceability_figures.py, which runs after cross_cohort_substitution.py and
reads its surrogates.
"""

from __future__ import annotations

import json
import sys
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

from cross_cohort_substitution import (  # noqa: E402
    _ensemble_scores,
    _load_gpl24676_ctd,
    _load_gse76220,
    _load_gse122649,
    _load_srp064478,
)
from utils import (  # noqa: E402
    critical_panel_features,
    critical_panel_label,
    critical_panel_size,
    load_dataset,
    zero_shot_scale,
)

# Panel size is derived, so every reader-facing denominator must be too.
_K_CRIT = critical_panel_size()
_CRIT_LABEL = critical_panel_label()

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"


_COMPARTMENTS: dict[str, tuple[str, ...]] = {
    "erythrocyte": (
        "ENSG00000206172",
        "ENSG00000188536",
        "ENSG00000244734",
        "ENSG00000158578",
        "ENSG00000170180",
        "ENSG00000133742",
        "ENSG00000159111",
        "ENSG00000105610",
        "ENSG00000179364",
        "ENSG00000223609",
    ),
    "platelet": (
        "ENSG00000163736",
        "ENSG00000163737",
        "ENSG00000185245",
    ),
    "endothelial": (
        "ENSG00000261371",
        "ENSG00000179776",
        "ENSG00000110799",
    ),
}

RANDOM_STATE = 42
N_BOOTSTRAP = 2000
# Point estimates in Figure 3 are averaged over this many seeds. That averaging is a
# no-op in practice: with colsample_bytree=1 and no row subsampling LightGBM is
# deterministic, so every seed fits the same trees and these AUCs match
# panel_critical_eval.py's single-model values exactly. A caption that once showed
# them differing (0.8075 vs 0.8269 on GPL16791) was stale, not a second estimator.
N_SEEDS = 5



def _bootstrap_tpr(y, scores, n=N_BOOTSTRAP, seed=RANDOM_STATE):
    """Return mean fpr grid + 2.5/97.5 percentile TPR bands + AUC CI."""
    from sklearn.metrics import roc_auc_score, roc_curve

    rng = np.random.default_rng(seed)
    fpr_grid = np.linspace(0, 1, 200)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    aucs, tprs = [], []
    for _ in range(n):
        idx = np.concatenate(
            [
                rng.choice(pos, len(pos), replace=True),
                rng.choice(neg, len(neg), replace=True),
            ]
        )
        try:
            fpr_b, tpr_b, _ = roc_curve(y[idx], scores[idx])
            tprs.append(np.interp(fpr_grid, fpr_b, tpr_b))
            aucs.append(roc_auc_score(y[idx], scores[idx]))
        except ValueError:
            continue
    aucs = np.array(aucs)
    tprs = np.array(tprs)
    return (
        fpr_grid,
        np.percentile(tprs, 2.5, axis=0),
        np.percentile(tprs, 97.5, axis=0),
        float(np.percentile(aucs, 2.5)),
        float(np.percentile(aucs, 97.5)),
    )


def _train_predict(Xtr, y_tr, Xte, params, cohort: str):
    """Zero-shot scores for one cohort.

    Scaling is delegated to utils.zero_shot_scale so cohorts on a different
    expression scale (GSE76220) are standardised within themselves rather than
    through the training scaler, which would collapse them to a constant
    prediction (AUC exactly 0.5).
    """
    Ztr, Zte = zero_shot_scale(Xtr, Xte, cohort)
    return _ensemble_scores(Ztr, y_tr, Zte, params, seeds=tuple(range(N_SEEDS)))


def main() -> None:
    from sklearn.metrics import roc_auc_score, roc_curve

    warnings.filterwarnings("ignore")

    params_raw = json.loads(_PARAMS_PATH.read_text())
    params = dict(params_raw, colsample_bytree=1.0, n_jobs=-1, verbose=-1)

    df_panel = pd.read_csv(_PANEL_CSV)
    feat_col = next(
        c for c in df_panel.columns if "feature" in c.lower() or "ensg" in c.lower()
    )
    sym_col = next(c for c in df_panel.columns if "symbol" in c.lower())
    feat25 = df_panel[feat_col].tolist()
    sym25 = df_panel[sym_col].tolist()
    feat25_bases = [f.split(".")[0] for f in feat25]

    # Panel membership from the canonical definition, never hardcoded.
    _crit_bases = {f.split(".")[0] for f in critical_panel_features()}
    _CRITICAL_IDX = [i for i, b in enumerate(feat25_bases) if b in _crit_bases]

    # Every cohort is scored against the raw log1p training matrix: CTD
    # residualisation is discovery-side only, and applying it here would make
    # these AUCs incomparable with panel_critical_eval.py's. _load_gpl24676_ctd
    # still returns the residualised matrix; we deliberately discard it.
    print("Loading GPL24676 (raw log1p; CTD is discovery-side only) ...")
    _, y_train, X_train_log, feat_train = _load_gpl24676_ctd()
    feat_base_to_col = {f.split(".")[0]: j for j, f in enumerate(feat_train)}

    # Cohort spec list:
    # (cohort_name, y_test, X_test, vocab_map, by_symbol, preprocessing)
    cohort_data = []

    print("Loading GPL16791 (raw log1p; CTD is discovery-side only) ...")
    (ds16,) = load_dataset(
        "GSE153960", platform="GPL16791", resources_dir=ALS_DIR / "resources"
    )
    assert list(ds16.X.columns) == feat_train
    X16_raw = np.log1p(ds16.X.values.astype(np.float32))
    y16 = ds16.y.values.astype(int)
    cohort_data.append(
        {
            "name": "GPL16791",
            "tissue": "Multi-region CNS",
            "n": len(y16),
            "y": y16,
            "X_te": X16_raw,
            "vocab": None,
            "by_symbol": False,
            "X_train_use": X_train_log,
        }
    )

    print("Loading GSE76220 ...")
    X76, vocab76, y76 = _load_gse76220()
    cohort_data.append(
        {
            "name": "GSE76220",
            "tissue": "Lumbar SC (LCM)",
            "n": len(y76),
            "y": y76,
            "X_te": X76,
            "vocab": {s: i for i, s in enumerate(vocab76)},
            "by_symbol": True,
            "X_train_use": X_train_log,
        }
    )

    print("Loading GSE122649 ...")
    X122, vocab122, y122 = _load_gse122649()
    cohort_data.append(
        {
            "name": "GSE122649",
            "tissue": "Motor cortex",
            "n": len(y122),
            "y": y122,
            "X_te": X122,
            "vocab": {s: i for i, s in enumerate(vocab122)},
            "by_symbol": True,
            "X_train_use": X_train_log,
        }
    )

    print("Loading SRP064478 ...")
    Xsrp, vocab_srp, y_srp = _load_srp064478()
    cohort_data.append(
        {
            "name": "SRP064478",
            "tissue": "Cervical SC",
            "n": len(y_srp),
            "y": y_srp,
            "X_te": Xsrp,
            "vocab": {v: i for i, v in enumerate(vocab_srp)},
            "by_symbol": False,
            "X_train_use": X_train_log,
        }
    )

    # Helper to select panel columns for a cohort
    def _panel_cols(cohort, indices):
        v = cohort["vocab"]
        if cohort["by_symbol"]:
            te_syms = [sym25[i] for i in indices]
            tr_cols = [
                feat_base_to_col[feat25_bases[i]] for i in indices if sym25[i] in v
            ]
            te_cols = [v[s] for s in te_syms if s in v]
        else:
            te_bases = [feat25_bases[i] for i in indices]
            if cohort["name"] == "GPL16791":
                # GPL16791 uses same feat_train ordering as training
                tr_cols = [feat_base_to_col[b] for b in te_bases]
                te_cols = [feat_base_to_col[b] for b in te_bases]
            else:
                tr_cols = [
                    feat_base_to_col[b] for b in te_bases if b in cohort["vocab"]
                ]
                te_cols = [cohort["vocab"][b] for b in te_bases if b in cohort["vocab"]]
        return tr_cols, te_cols

    # ---- Figure 1: combined 2x2 ROC grid (25-gene + critical panel) ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes_flat = axes.flatten()
    summary = []
    for ax, cohort in zip(axes_flat, cohort_data):
        # 25-gene
        tr25, te25 = _panel_cols(cohort, list(range(25)))
        scores25 = _train_predict(
            cohort["X_train_use"][:, tr25], y_train, cohort["X_te"][:, te25], params,
            cohort["name"]
        )
        fpr25, tpr25, _ = roc_curve(cohort["y"], scores25)
        auc25 = roc_auc_score(cohort["y"], scores25)
        _, lo25_band, hi25_band, lo25, hi25 = _bootstrap_tpr(cohort["y"], scores25)

        # critical panel
        tr15, te15 = _panel_cols(cohort, _CRITICAL_IDX)
        scores15 = _train_predict(
            cohort["X_train_use"][:, tr15], y_train, cohort["X_te"][:, te15], params,
            cohort["name"]
        )
        fpr15, tpr15, _ = roc_curve(cohort["y"], scores15)
        auc15 = roc_auc_score(cohort["y"], scores15)
        _, lo15_band, hi15_band, lo15, hi15 = _bootstrap_tpr(cohort["y"], scores15)

        # Plot
        ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=0.7)
        # CI bands
        fpr_grid = np.linspace(0, 1, 200)
        ax.fill_between(fpr_grid, lo25_band, hi25_band, color="#1f77b4", alpha=0.15)
        ax.fill_between(fpr_grid, lo15_band, hi15_band, color="#d62728", alpha=0.15)
        ax.plot(
            fpr25,
            tpr25,
            color="#1f77b4",
            lw=2.0,
            label=f"25-gene ({len(te25)}/25)  AUC = {auc25:.3f} "
            f"[{lo25:.2f}, {hi25:.2f}]",
        )
        ax.plot(
            fpr15,
            tpr15,
            color="#d62728",
            lw=2.0,
            label=f"{_CRIT_LABEL} ({len(te15)}/{_K_CRIT})  AUC = {auc15:.3f} "
            f"[{lo15:.2f}, {hi15:.2f}]",
        )
        ax.set_xlabel("False positive rate", fontsize=9)
        ax.set_ylabel("True positive rate", fontsize=9)
        ax.set_title(
            f"{cohort['name']} — {cohort['tissue']} ($n = {cohort['n']}$)", fontsize=10
        )
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
        summary.append(
            {
                "cohort": cohort["name"],
                "n": len(cohort["y"]),
                "auc25": auc25,
                "lo25": lo25,
                "hi25": hi25,
                "auc15": auc15,
                "lo15": lo15,
                "hi15": hi15,
            }
        )

    fig.suptitle(
        "External-validation ROC across four CNS cohorts — 25-gene panel "
        f"(blue) vs {_CRIT_LABEL} critical panel (red)\n"
        "GPL24676-trained LGBM, zero-shot transfer; shaded bands = 95% paired bootstrap CI ($B = 2000$).",
        fontsize=10,
    )
    plt.tight_layout()
    out = SCRIPT_DIR / "combined_validation_roc.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # Persist the AUCs. Until 2026-08-11 this script wrote only the PNG, so the
    # eight values in the Figure 3 caption existed nowhere but inside the image
    # and hand-transcribed in the .tex -- unverifiable by any guard, and exactly
    # the defect class of findings.md Step 65. Writing them makes the caption
    # checkable (see test_manuscript_numbers.py).
    stats_out = SCRIPT_DIR / "combined_validation_roc_statistics.txt"
    lines = [
        "Combined external-validation ROC — 25-gene vs critical panel",
        "=" * 66,
        "Estimator: 5-seed LightGBM ensemble, zero-shot (no retraining).",
        "These are the values quoted in the Figure 3 caption. They match",
        "panel_critical_eval_statistics.txt exactly (seeding is a no-op here:",
        "colsample_bytree=1 and no row subsampling make LightGBM deterministic),",
        "and their equal-weight mean is that file's W.mean.",
        "",
        f"{'cohort':<14}{'n':>5}{'25-gene AUC':>14}{'95% CI':>18}"
        f"{_CRIT_LABEL + ' AUC':>14}{'95% CI':>18}",
        "-" * 83,
    ]
    for s in summary:
        ci25 = f"[{s['lo25']:.3f}, {s['hi25']:.3f}]"
        ci15 = f"[{s['lo15']:.3f}, {s['hi15']:.3f}]"
        lines.append(
            f"{s['cohort']:<14}{s['n']:>5}{s['auc25']:>14.4f}{ci25:>18}"
            f"{s['auc15']:>14.4f}{ci15:>18}"
        )
    stats_out.write_text("\n".join(lines) + "\n")
    print(f"Saved: {stats_out}")

    for s in summary:
        print(
            f"  {s['cohort']}: 25-gene AUC={s['auc25']:.4f}  "
            f"{_CRIT_LABEL} AUC={s['auc15']:.4f}"
        )


if __name__ == "__main__":
    main()
