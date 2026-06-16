"""Multi-algorithm robustness check for the 25-gene panel.

Compares SHAP-ranked feature importance from the primary LightGBM model with
equivalent rankings from logistic regression (L1-penalised) and random forest,
all trained on the top-500 SHAP features.

Outputs
-------
  multi_algorithm_robustness.png
  multi_algorithm_robustness_statistics.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ALS_DIR = Path(__file__).parents[1]
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset  # noqa: E402

_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"
_SHAP_RANK_CSV = SCRIPT_DIR / "lgbm_top500_final_shap_ranking.csv"
_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PREFILTER_X = SCRIPT_DIR / "lgbm_prefilter_X.npy"
_PREFILTER_NAMES = SCRIPT_DIR / "lgbm_prefilter_names.txt"
_OUT_PNG = SCRIPT_DIR / "multi_algorithm_robustness.png"
_OUT_TXT = SCRIPT_DIR / "multi_algorithm_robustness_statistics.txt"

PLATFORM = "GPL24676"
RANDOM_STATE = 42
TOP_K = 25  # panel size and comparison rank cut-off


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    print("=" * 65)
    print("Multi-Algorithm Robustness Check")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Load SHAP top-500 feature ranking
    # ------------------------------------------------------------------
    print("\n[1] Loading SHAP top-500 ranking ...")
    shap_df = pd.read_csv(_SHAP_RANK_CSV)
    top500_features = shap_df["feature"].tolist()[:500]

    panel_df = pd.read_csv(_PANEL_CSV)
    ensg_col = next(
        (c for c in panel_df.columns if "ensg" in c.lower()),
        next(c for c in panel_df.columns if "feature" in c.lower()),
    )
    panel_ids = set(panel_df[ensg_col].str.split(".").str[0].tolist())
    panel_symbols = dict(zip(
        panel_df[ensg_col].str.split(".").str[0],
        panel_df.get("symbol", panel_df.get("gene_symbol", panel_df[ensg_col]))
    ))

    # ------------------------------------------------------------------
    # 2. Load training data
    # ------------------------------------------------------------------
    print("\n[2] Loading training data ...")
    (ds,) = load_dataset("GSE153960", platform=PLATFORM,
                         resources_dir=ALS_DIR / "resources")
    y = ds.y.values
    print(f"  n={len(y)}  ALS={y.sum()}  Ctrl={(y==0).sum()}")

    prefilter_names = _PREFILTER_NAMES.read_text().splitlines()
    X_full = np.load(_PREFILTER_X)

    name_to_idx = {n.split(".")[0]: i for i, n in enumerate(prefilter_names)}
    top500_base = [f.split(".")[0] for f in top500_features]
    col_idx = [name_to_idx[f] for f in top500_base if f in name_to_idx]
    feature_names = [top500_base[i] for i, f in enumerate(top500_base)
                     if f in name_to_idx]
    X = X_full[:, col_idx]
    print(f"  Top-500 features loaded: {X.shape}")

    # ------------------------------------------------------------------
    # 3. LightGBM — feature importance (already ranked by SHAP; use gain)
    # ------------------------------------------------------------------
    print("\n[3] LightGBM — fitting for gain-based importance ...")
    params = json.loads(_PARAMS_PATH.read_text())
    params.pop("feature_names", None)
    params["colsample_bytree"] = 0.0667
    params["random_state"] = RANDOM_STATE
    params["verbosity"] = -1
    lgbm = LGBMClassifier(**params)
    lgbm.fit(X, y)
    lgbm_imp = lgbm.feature_importances_
    lgbm_rank = np.argsort(-lgbm_imp)
    lgbm_top25 = set(feature_names[i] for i in lgbm_rank[:TOP_K])
    print(f"  LightGBM top-{TOP_K} extracted.")

    # ------------------------------------------------------------------
    # 4. Logistic Regression (L1, cross-validated C)
    # ------------------------------------------------------------------
    print("\n[4] Logistic Regression (L1) ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegressionCV(
        Cs=10,
        cv=5,
        penalty="l1",
        solver="liblinear",
        max_iter=2000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    lr.fit(X_scaled, y)
    lr_imp = np.abs(lr.coef_[0])
    lr_rank = np.argsort(-lr_imp)
    lr_top25 = set(feature_names[i] for i in lr_rank[:TOP_K])
    n_lr_nonzero = (lr.coef_[0] != 0).sum()
    print(f"  LR non-zero features: {n_lr_nonzero}")
    print(f"  LR top-{TOP_K} extracted.")

    # ------------------------------------------------------------------
    # 5. Random Forest
    # ------------------------------------------------------------------
    print("\n[5] Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X, y)
    rf_imp = rf.feature_importances_
    rf_rank = np.argsort(-rf_imp)
    rf_top25 = set(feature_names[i] for i in rf_rank[:TOP_K])
    print(f"  RF top-{TOP_K} extracted.")

    # ------------------------------------------------------------------
    # 6. SHAP top-25 (from pre-computed ranking)
    # ------------------------------------------------------------------
    shap_top25 = set(top500_base[:TOP_K])

    # ------------------------------------------------------------------
    # 7. Overlap analysis
    # ------------------------------------------------------------------
    print("\n[6] Overlap analysis ...")

    def _overlap(a: set, b: set) -> tuple[int, float]:
        inter = len(a & b)
        jaccard = inter / len(a | b)
        return inter, jaccard

    pairs = {
        "SHAP ∩ LGBM-gain": _overlap(shap_top25, lgbm_top25),
        "SHAP ∩ LR-L1": _overlap(shap_top25, lr_top25),
        "SHAP ∩ RF": _overlap(shap_top25, rf_top25),
        "LGBM ∩ LR": _overlap(lgbm_top25, lr_top25),
        "LGBM ∩ RF": _overlap(lgbm_top25, rf_top25),
        "LR ∩ RF": _overlap(lr_top25, rf_top25),
    }

    # Panel overlap
    panel_in_shap = len(panel_ids & shap_top25)
    panel_in_lgbm = len(panel_ids & lgbm_top25)
    panel_in_lr = len(panel_ids & lr_top25)
    panel_in_rf = len(panel_ids & rf_top25)
    panel_in_all3 = len(panel_ids & shap_top25 & lgbm_top25 & rf_top25)
    panel_in_any2 = len(panel_ids & (
        (shap_top25 & lgbm_top25) | (shap_top25 & lr_top25) | (shap_top25 & rf_top25) |
        (lgbm_top25 & lr_top25) | (lgbm_top25 & rf_top25) | (lr_top25 & rf_top25)
    ))

    # ------------------------------------------------------------------
    # 8. Rank convergence — per panel gene, rank in each method
    # ------------------------------------------------------------------
    def _get_rank(feat_id: str, ranked_ids: list[str]) -> int | None:
        try:
            return ranked_ids.index(feat_id) + 1
        except ValueError:
            return None

    lgbm_ids = [feature_names[i] for i in lgbm_rank]
    lr_ids = [feature_names[i] for i in lr_rank]
    rf_ids = [feature_names[i] for i in rf_rank]

    gene_ranks: list[dict] = []
    for ensg_base in panel_ids:
        sym = panel_symbols.get(ensg_base, ensg_base)
        shap_r = _get_rank(ensg_base, top500_base[:500])
        lgbm_r = _get_rank(ensg_base, lgbm_ids)
        lr_r = _get_rank(ensg_base, lr_ids)
        rf_r = _get_rank(ensg_base, rf_ids)
        gene_ranks.append({
            "symbol": sym,
            "ensg": ensg_base,
            "shap_rank": shap_r,
            "lgbm_rank": lgbm_r,
            "lr_rank": lr_r,
            "rf_rank": rf_r,
        })

    gene_ranks.sort(key=lambda x: x["shap_rank"] or 9999)

    # ------------------------------------------------------------------
    # 9. Plot
    # ------------------------------------------------------------------
    print("\n[7] Plotting ...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Multi-Algorithm Robustness: top-500 feature importance rankings",
                 fontsize=11)

    # Panel A: top-25 overlap heatmap (Jaccard)
    ax = axes[0]
    methods = ["SHAP", "LGBM-gain", "LR-L1", "RF"]
    sets = [shap_top25, lgbm_top25, lr_top25, rf_top25]
    n = len(methods)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = len(sets[i] & sets[j]) / len(sets[i] | sets[j])
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_yticklabels(methods, fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black" if mat[i, j] < 0.6 else "white")
    ax.set_title(f"Jaccard overlap (top-{TOP_K})", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)

    # Panel B: panel gene recovery per method
    ax = axes[1]
    bars = [panel_in_shap, panel_in_lgbm, panel_in_lr, panel_in_rf]
    ax.bar(methods, bars, color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"])
    ax.axhline(TOP_K, color="gray", linestyle="--", linewidth=0.8, label=f"Max={TOP_K}")
    ax.set_ylim(0, TOP_K + 2)
    ax.set_ylabel("Panel genes in top-25")
    ax.set_title(f"25-gene panel recovery (top-{TOP_K})", fontsize=10)
    for i, (bar_h, count) in enumerate(zip(bars, bars)):
        ax.text(i, count + 0.3, str(count), ha="center", va="bottom", fontsize=10)

    # Panel C: rank heatmap for 25 panel genes
    ax = axes[2]
    syms = [g["symbol"] for g in gene_ranks]
    shap_ranks = [g["shap_rank"] or 501 for g in gene_ranks]
    lgbm_ranks = [g["lgbm_rank"] or 501 for g in gene_ranks]
    lr_ranks = [g["lr_rank"] or 501 for g in gene_ranks]
    rf_ranks = [g["rf_rank"] or 501 for g in gene_ranks]

    rank_mat = np.array([shap_ranks, lgbm_ranks, lr_ranks, rf_ranks]).T  # 25 × 4
    rank_mat_capped = np.clip(rank_mat, 1, 100)
    im2 = ax.imshow(rank_mat_capped.T, aspect="auto", cmap="RdYlGn_r",
                    vmin=1, vmax=100)
    ax.set_xticks(range(len(syms)))
    ax.set_xticklabels(syms, rotation=90, fontsize=7)
    ax.set_yticks(range(4))
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_title("Rank per method (capped at 100; green=low rank)", fontsize=9)
    fig.colorbar(im2, ax=ax, fraction=0.046, label="Rank (capped at 100)")

    plt.tight_layout()
    fig.savefig(_OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {_OUT_PNG}")

    # ------------------------------------------------------------------
    # 10. Write statistics
    # ------------------------------------------------------------------
    print("\n[8] Writing statistics ...")
    lines: list[str] = [
        "Multi-Algorithm Robustness Check",
        "=" * 60,
        f"Feature space: top-500 SHAP-ranked features",
        f"Methods compared: SHAP (primary), LightGBM-gain, LR-L1 (CV), RF",
        f"Comparison cut-off: top-{TOP_K} features per method",
        "",
        "Pairwise Jaccard overlap (top-25):",
    ]
    for label, (inter, jac) in pairs.items():
        lines.append(f"  {label:<25}  n_shared={inter}  Jaccard={jac:.3f}")

    lines += [
        "",
        "25-gene panel recovery per method:",
        f"  SHAP (primary):  {panel_in_shap}/{TOP_K}",
        f"  LGBM gain:       {panel_in_lgbm}/{TOP_K}",
        f"  LR-L1:           {panel_in_lr}/{TOP_K}",
        f"  RF:              {panel_in_rf}/{TOP_K}",
        f"  All 3 methods:   {panel_in_all3}/{TOP_K}",
        f"  Any 2 methods:   {panel_in_any2}/{TOP_K}",
        "",
        f"{'Symbol':<14}  {'SHAP':>6}  {'LGBM':>6}  {'LR':>6}  {'RF':>6}",
        "-" * 45,
    ]
    for g in gene_ranks:
        def _r(v):
            return str(v) if v else ">500"
        lines.append(
            f"  {g['symbol']:<12}  {_r(g['shap_rank']):>6}  "
            f"{_r(g['lgbm_rank']):>6}  {_r(g['lr_rank']):>6}  {_r(g['rf_rank']):>6}"
        )

    _OUT_TXT.write_text("\n".join(lines) + "\n")
    print(f"  Saved: {_OUT_TXT}")
    print("\nDone.")


if __name__ == "__main__":
    main()
