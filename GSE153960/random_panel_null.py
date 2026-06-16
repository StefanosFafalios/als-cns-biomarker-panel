# ruff: noqa: E402
"""Random-panel null for cross-cohort transfer — experiment #4.

Tests whether the SHAP-selected 25-gene panel transfers zero-shot
(GPL24676 -> GPL16791) better than random size-matched panels, i.e. that the
panel's generalisation is not what any 25 genes would achieve.

Two nulls (B random 25-gene panels each; identical train/test pipeline):
  Null-1 (broad)  : 25 genes drawn from all genes common to both cohorts.
  Null-2 (strict) : 25 genes drawn from the 3,704 SFM-preselected pool.

Empirical p = (#null AUC >= real AUC + 1) / (B + 1).

Outputs
-------
  random_panel_null_statistics.txt
  random_panel_null.png
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ALS_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset  # noqa: E402

SCRIPT_DIR = Path(__file__).parent
_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_SFM_CSV = SCRIPT_DIR / "lgbm_iter_final_shap_ranking.csv"
RANDOM_STATE = 42
B = 1000
# fast, fixed LightGBM config (identical for real and random panels)
_PARAMS = {
    "n_estimators": 200,
    "num_leaves": 21,
    "learning_rate": 0.05,
    "min_child_samples": 105,
    "subsample": 0.45,
    "colsample_bytree": 1.0,
    "verbose": -1,
    "n_jobs": 4,
    "random_state": RANDOM_STATE,
}


def _zeroshot_auc(
    xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, yte: np.ndarray
) -> float:
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score

    clf = LGBMClassifier(**_PARAMS)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        clf.fit(xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(xte)[:, 1]))


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    print("=" * 64)
    print("Random-panel null for cross-cohort transfer — experiment #4")
    print("=" * 64)

    print("\n[1] Loading GPL24676 (train) and GPL16791 (test) ...")
    (tr,) = load_dataset(
        "GSE153960", platform="GPL24676", resources_dir=ALS_DIR / "resources"
    )
    (te,) = load_dataset(
        "GSE153960", platform="GPL16791", resources_dir=ALS_DIR / "resources"
    )
    tr_base = {str(c).split(".")[0]: c for c in tr.X.columns}
    te_base = {str(c).split(".")[0]: c for c in te.X.columns}
    common = sorted(set(tr_base) & set(te_base))
    print(f"  train n={len(tr.y)}, test n={len(te.y)}, common genes={len(common)}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        xtr_full = np.log1p(
            tr.X[[tr_base[b] for b in common]].values.astype(np.float32)
        )
        xte_full = np.log1p(
            te.X[[te_base[b] for b in common]].values.astype(np.float32)
        )
    ytr = tr.y.values.astype(int)
    yte = te.y.values.astype(int)
    col_of = {b: j for j, b in enumerate(common)}

    # real panel
    panel = pd.read_csv(_PANEL_CSV)
    panel_bases = [
        str(f).split(".")[0] for f in panel["feature"] if str(f).split(".")[0] in col_of
    ]
    pidx = [col_of[b] for b in panel_bases]
    real_auc = _zeroshot_auc(xtr_full[:, pidx], ytr, xte_full[:, pidx], yte)
    print(f"\n[2] Real panel ({len(pidx)} genes) zero-shot AUC = {real_auc:.4f}")

    # SFM pool (strict null)
    sfm_bases: list[str] = []
    if _SFM_CSV.exists():
        sfm = pd.read_csv(_SFM_CSV)
        fcol = "feature" if "feature" in sfm.columns else sfm.columns[0]
        sfm_bases = [
            b for b in (str(f).split(".")[0] for f in sfm[fcol]) if b in col_of
        ]
    print(f"  SFM pool genes available in common space: {len(sfm_bases)}")

    rng = np.random.default_rng(RANDOM_STATE)
    n_panel = len(pidx)

    def run_null(pool: list[str], label: str) -> np.ndarray:
        pool_idx = [col_of[b] for b in pool]
        aucs = np.zeros(B)
        for i in range(B):
            sel = rng.choice(pool_idx, size=n_panel, replace=False)
            aucs[i] = _zeroshot_auc(xtr_full[:, sel], ytr, xte_full[:, sel], yte)
        ge = int(np.sum(aucs >= real_auc))
        p = (ge + 1) / (B + 1)
        print(
            f"  {label}: null mean={aucs.mean():.4f} 95pct={np.percentile(aucs, 95):.4f} "
            f"max={aucs.max():.4f}  p={p:.4g}  (#>=real={ge})"
        )
        return aucs

    print(f"\n[3] Null-1 (broad: all {len(common)} common genes), B={B} ...")
    null1 = run_null(common, "Null-1 broad")
    null2 = None
    if len(sfm_bases) >= n_panel:
        print(f"\n[4] Null-2 (strict: {len(sfm_bases)} SFM genes), B={B} ...")
        null2 = run_null(sfm_bases, "Null-2 SFM")

    def stat_block(aucs: np.ndarray, label: str) -> list[str]:
        ge = int(np.sum(aucs >= real_auc))
        p = (ge + 1) / (B + 1)
        z = (real_auc - aucs.mean()) / (aucs.std() + 1e-12)
        return [
            f"{label}",
            f"  null mean   : {aucs.mean():.4f}",
            f"  null sd     : {aucs.std():.4f}",
            f"  null 95 pct : {np.percentile(aucs, 95):.4f}",
            f"  null max    : {aucs.max():.4f}",
            f"  # null >= real ({real_auc:.4f}) : {ge}/{B}",
            f"  empirical p : {p:.4g}",
            f"  z vs null   : {z:.2f}",
            "",
        ]

    lines = [
        "Random-panel null for cross-cohort transfer — experiment #4",
        "=" * 64,
        "Train GPL24676 (n=%d) -> zero-shot test GPL16791 (n=%d)."
        % (len(ytr), len(yte)),
        "Identical LightGBM pipeline (raw log1p, fixed params) for all panels.",
        f"Real SHAP 25-gene panel zero-shot AUC = {real_auc:.4f}",
        f"B = {B} random size-matched panels per null.",
        "",
    ]
    lines += stat_block(null1, "Null-1 (broad: random 25 from all common genes):")
    if null2 is not None:
        lines += stat_block(null2, "Null-2 (strict: random 25 from 3,704 SFM pool):")
    (SCRIPT_DIR / "random_panel_null_statistics.txt").write_text(
        "\n".join(lines) + "\n"
    )
    print("\n" + "\n".join(lines))

    # figure
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(null1, bins=40, alpha=0.6, color="#9467bd", label="Null-1 (all genes)")
    if null2 is not None:
        ax.hist(null2, bins=40, alpha=0.6, color="#ff7f0e", label="Null-2 (SFM pool)")
    ax.axvline(real_auc, color="#d62728", lw=2.5, label=f"SHAP panel = {real_auc:.3f}")
    ax.set_xlabel("Zero-shot AUC-ROC (GPL24676 -> GPL16791)")
    ax.set_ylabel("Random panels")
    ax.set_title(
        "SHAP panel vs random size-matched panels (cross-cohort transfer)",
        fontsize=11,
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(SCRIPT_DIR / "random_panel_null.png", dpi=150, bbox_inches="tight")
    print("\nSaved -> random_panel_null.{txt,png}")


if __name__ == "__main__":
    main()
