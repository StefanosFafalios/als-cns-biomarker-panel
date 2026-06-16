"""Run IAMB at each α in {0.01, 0.05, 0.10, 0.20} on the top-500 SHAP space
and report which panel genes are retained at each threshold.

The reviewer's concern (M4): α=0.20 is the most permissive option on the grid,
chosen by CV AUC; calling it "causal credentialing" is too strong. The
robustness fix is to show which panel genes are in the recovered MB *across
the α grid* — those that survive strict α (0.01–0.05) are the most defensible.

Outputs:
  iamb_alpha_grid.csv   — per-(alpha, panel_gene) membership matrix
  iamb_alpha_grid.txt   — human-readable summary with MB sizes and panel hits
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ALS_DIR = Path(__file__).parents[1]
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset  # noqa: E402
from model.fs.iamb import IAMBSelector  # noqa: E402

ALPHA_GRID = [0.01, 0.05, 0.10, 0.20]
_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_SHAP_RANKING = SCRIPT_DIR / "lgbm_top500_final_shap_ranking.csv"
_PREFILTER_X = SCRIPT_DIR / "lgbm_prefilter_X.npy"
_PREFILTER_NAMES = SCRIPT_DIR / "lgbm_prefilter_names.txt"


def _load_top500_matrix() -> tuple[np.ndarray, list[str]]:
    """Load the top-500 SHAP feature matrix (same as ses_markov_blanket.py)."""
    shap_df = pd.read_csv(_SHAP_RANKING).head(500)
    top500_features: list[str] = shap_df["feature"].tolist()

    all_names = _PREFILTER_NAMES.read_text().splitlines()
    name_to_col = {n: i for i, n in enumerate(all_names)}
    cols = [name_to_col[f] for f in top500_features if f in name_to_col]
    feature_names = [f for f in top500_features if f in name_to_col]

    X_full = np.load(_PREFILTER_X)
    X = X_full[:, cols].astype(np.float32)
    print(f"  top-500 matrix: {X.shape}")
    return X, feature_names


def _load_labels() -> np.ndarray:
    (ds,) = load_dataset("GSE153960", platform="GPL24676",
                         resources_dir=ALS_DIR / "resources")
    return ds.y.values.astype(int)


def main() -> None:
    print("Loading top-500 matrix and labels ...")
    X, feature_names = _load_top500_matrix()
    y = _load_labels()
    assert len(y) == X.shape[0]
    # Z-score per-feature for numerical stability of partial correlation
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    print(f"  labels: ALS={int(y.sum())}, Ctrl={len(y)-int(y.sum())}")

    panel_df = pd.read_csv(_PANEL_CSV)
    panel_ensg = panel_df["feature"].tolist()
    panel_sym = panel_df["symbol"].tolist()
    sym_by_ensg = dict(zip(panel_ensg, panel_sym))

    membership: dict[float, set[str]] = {}
    for alpha in ALPHA_GRID:
        print(f"\nRunning IAMB at alpha={alpha} ...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            sel = IAMBSelector(alpha=alpha).fit(X_std, y.astype(float))
        mb_idx = sel.get_support(indices=True)
        mb_features = [feature_names[i] for i in mb_idx]
        membership[alpha] = set(mb_features)
        print(f"  MB size: {len(mb_features)}")

    # Build per-panel-gene table
    rows = []
    for ensg, sym in zip(panel_ensg, panel_sym):
        row = {"panel_gene": sym, "ensg": ensg}
        for alpha in ALPHA_GRID:
            row[f"alpha_{alpha}"] = "Y" if ensg in membership[alpha] else ""
        row["robust_count"] = sum(1 for a in ALPHA_GRID if ensg in membership[a])
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("robust_count", ascending=False)
    df.to_csv(SCRIPT_DIR / "iamb_alpha_grid.csv", index=False)
    print("\nPer-panel-gene MB membership across α grid:")
    print(df.to_string(index=False))

    lines = [
        "IAMB Markov Blanket — per-α membership grid",
        "=" * 60,
        "Input: top-500 SHAP-ranked features (n=874)",
        "Conditional independence test: Fisher-Z partial correlation",
        "",
        "MB size by α:",
    ]
    for alpha in ALPHA_GRID:
        lines.append(f"  α={alpha}: |MB|={len(membership[alpha])}")

    lines.append("")
    lines.append("Per-panel-gene membership (Y = in MB at that α):")
    lines.append(df.to_string(index=False))
    lines.append("")

    # Highlight robust members
    robust = df[df["robust_count"] == 4]
    near_robust = df[df["robust_count"] == 3]
    lines += [
        f"Robust across all four α levels (n={len(robust)}):",
        "  " + ", ".join(robust["panel_gene"].tolist()) if not robust.empty else "  (none)",
        "",
        f"Robust at three of four α levels (n={len(near_robust)}):",
        "  " + ", ".join(near_robust["panel_gene"].tolist()) if not near_robust.empty else "  (none)",
    ]

    (SCRIPT_DIR / "iamb_alpha_grid.txt").write_text("\n".join(lines))
    print(f"\nSaved -> iamb_alpha_grid.{{csv,txt}}")


if __name__ == "__main__":
    main()
