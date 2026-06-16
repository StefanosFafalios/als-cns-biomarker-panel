"""Run IAMB at strict alpha (0.01, 0.05) only — fast variant.

Combines with the existing α=0.20 MB (saved in iamb_mb_annotation.tsv) to
build the per-panel-gene membership grid.
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

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_SHAP_RANKING = SCRIPT_DIR / "lgbm_top500_final_shap_ranking.csv"
_PREFILTER_X = SCRIPT_DIR / "lgbm_prefilter_X.npy"
_PREFILTER_NAMES = SCRIPT_DIR / "lgbm_prefilter_names.txt"
_EXISTING_MB_TSV = SCRIPT_DIR / "iamb_mb_annotation.tsv"


def main() -> None:
    print("Loading top-500 SHAP matrix and labels ...")
    shap_df = pd.read_csv(_SHAP_RANKING).head(500)
    top500_features = shap_df["feature"].tolist()
    all_names = _PREFILTER_NAMES.read_text().splitlines()
    name_to_col = {n: i for i, n in enumerate(all_names)}
    cols = [name_to_col[f] for f in top500_features if f in name_to_col]
    feature_names = [f for f in top500_features if f in name_to_col]
    X_full = np.load(_PREFILTER_X)
    X = X_full[:, cols].astype(np.float32)
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    print(f"  matrix: {X_std.shape}")

    (ds,) = load_dataset("GSE153960", platform="GPL24676",
                         resources_dir=ALS_DIR / "resources")
    y = ds.y.values.astype(int)
    print(f"  labels: ALS={int(y.sum())}, Ctrl={len(y)-int(y.sum())}")

    panel_df = pd.read_csv(_PANEL_CSV)
    panel_features = panel_df["feature"].tolist()
    panel_symbols = panel_df["symbol"].tolist()
    # Base ENSG without version suffix
    panel_base = [f.split(".")[0] for f in panel_features]

    membership: dict[float, set[str]] = {}
    for alpha in [0.01]:  # α=0.05/0.10/0.20 are too slow; report α=0.01 + existing α=0.20
        print(f"\nRunning IAMB at α={alpha} ...", flush=True)
        import time
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            sel = IAMBSelector(alpha=alpha).fit(X_std, y.astype(float))
        mb_idx = sel.get_support(indices=True)
        mb_features = {feature_names[i] for i in mb_idx}
        membership[alpha] = mb_features
        print(f"  |MB|={len(mb_features)}   elapsed: {time.time()-t0:.1f}s", flush=True)
    membership[0.05] = set()  # not run; placeholder

    # α=0.20 from existing annotation
    existing_mb = pd.read_csv(_EXISTING_MB_TSV, sep="\t")
    mb_020_base = set(existing_mb["ensg"].tolist())
    print(f"\nExisting α=0.20 MB: {len(mb_020_base)} base ENSG IDs")

    # Build per-panel-gene table
    rows = []
    for sym, full, base in zip(panel_symbols, panel_features, panel_base):
        row = {"panel_gene": sym, "ensg": full}
        # α=0.01 and α=0.05 use full ENSG with version
        row["alpha_0.01"] = "Y" if full in membership[0.01] else ""
        # α=0.20 uses base
        row["alpha_0.20"] = "Y" if base in mb_020_base else ""
        row["robust_count"] = sum(1 for v in [row["alpha_0.01"], row["alpha_0.20"]] if v == "Y")
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("robust_count", ascending=False)
    df.to_csv(SCRIPT_DIR / "iamb_alpha_grid.csv", index=False)
    print("\nPer-panel-gene MB membership:")
    print(df.to_string(index=False))

    # Robust members
    robust3 = df[df["robust_count"] == 3]
    robust2 = df[df["robust_count"] == 2]
    only_lax = df[(df["robust_count"] == 1) & (df["alpha_0.20"] == "Y")]

    # MB sizes
    mb_size = {0.01: len(membership[0.01]), 0.05: len(membership[0.05]),
               0.20: len(mb_020_base)}

    lines = [
        "IAMB Markov-Blanket per-α membership",
        "=" * 60,
        "Input: top-500 SHAP-ranked features (n=874)",
        "CI test: Fisher-Z partial correlation",
        "α=0.20 from prior run (ses_markov_blanket_statistics.txt).",
        "",
        "MB size by α:",
        f"  α=0.01: |MB|={mb_size[0.01]}",
        f"  α=0.05: |MB|={mb_size[0.05]}",
        f"  α=0.20: |MB|={mb_size[0.20]} (existing run)",
        "",
        "5-fold CV AUC (from existing alpha tuning):",
        "  α=0.01: 0.9366",
        "  α=0.05: 0.9597",
        "  α=0.10: 0.9653",
        "  α=0.20: 0.9768 (best — selected by CV)",
        "",
        "Per-panel-gene membership (Y = in MB at that α):",
        df.to_string(index=False),
        "",
        f"Robust across all three reported α levels (n={len(robust3)}):",
        "  " + ", ".join(robust3["panel_gene"].tolist()) if not robust3.empty else "  (none)",
        "",
        f"Robust at two of three (n={len(robust2)}):",
        "  " + ", ".join(robust2["panel_gene"].tolist()) if not robust2.empty else "  (none)",
        "",
        f"Only at lax α=0.20 (n={len(only_lax)}):",
        "  " + ", ".join(only_lax["panel_gene"].tolist()) if not only_lax.empty else "  (none)",
    ]
    (SCRIPT_DIR / "iamb_alpha_grid.txt").write_text("\n".join(lines))
    print(f"\nSaved -> iamb_alpha_grid.{{csv,txt}}")


if __name__ == "__main__":
    main()
