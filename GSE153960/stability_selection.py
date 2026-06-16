"""Bootstrap-based stability selection on the 3,704-gene SFM space.

Reviewer M5: Cross-architecture Jaccard 0.06-0.11 is weak; the standard
antidote is stability selection. This script:
  - Bootstraps 100 stratified resamples of the 874 training samples.
  - In each replicate, fits LightGBM on the 3,704-gene SFM matrix and
    records the top-25 SHAP-ranked features.
  - Reports per-gene selection frequency across replicates.
  - Reports which of the 25 panel members survive a frequency threshold.

A panel gene that appears in the top-25 across ≥80% of bootstraps is
considered stably selected; lower frequencies indicate cohort-specific or
correlated-feature substitution.

Outputs:
  stability_selection.csv          (per-gene selection frequency, top 50)
  stability_selection_statistics.txt
  stability_selection.png
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

ALS_DIR = Path(__file__).parents[1]
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset  # noqa: E402

N_BOOTSTRAP = 100
TOP_K = 25
RANDOM_STATE = 42
_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"
_SFM_RANKING = SCRIPT_DIR / "lgbm_iter_final_shap_ranking.csv"
_PREFILTER_X = SCRIPT_DIR / "lgbm_prefilter_X.npy"
_PREFILTER_NAMES = SCRIPT_DIR / "lgbm_prefilter_names.txt"
_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"


def _load_sfm_matrix() -> tuple[np.ndarray, list[str]]:
    sfm_df = pd.read_csv(_SFM_RANKING)
    sfm_features = sfm_df["feature"].tolist()
    all_names = _PREFILTER_NAMES.read_text().splitlines()
    name_to_col = {n: i for i, n in enumerate(all_names)}
    kept = [f for f in sfm_features if f in name_to_col]
    col_idx = [name_to_col[f] for f in kept]
    X_full = np.load(_PREFILTER_X)
    return X_full[:, col_idx].astype(np.float32), kept


def _bootstrap_iteration(
    X: np.ndarray, y: np.ndarray, rng: np.random.Generator,
    params: dict, sfm_features: list[str],
) -> set[str]:
    from lightgbm import LGBMClassifier
    import shap

    # Stratified resample
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
    n_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
    idx = np.concatenate([n_pos, n_neg])
    rng.shuffle(idx)
    X_b, y_b = X[idx], y[idx]

    clf = LGBMClassifier(**params)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        clf.fit(X_b, y_b)
        # SHAP top-25 on the bootstrap sample
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_b)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        mean_abs = np.abs(shap_vals).mean(axis=0)
        top_idx = np.argsort(mean_abs)[-TOP_K:]
    return {sfm_features[i] for i in top_idx}


def main() -> None:
    print("Loading SFM matrix and labels ...")
    X, sfm_features = _load_sfm_matrix()
    (ds,) = load_dataset("GSE153960", platform="GPL24676",
                         resources_dir=ALS_DIR / "resources")
    y = ds.y.values.astype(int)
    print(f"  X: {X.shape}  ALS={int(y.sum())}  Ctrl={len(y)-int(y.sum())}")

    params = json.loads(_PARAMS_PATH.read_text())
    params["colsample_bytree"] = 1.0
    params["random_state"] = RANDOM_STATE
    params["n_estimators"] = 600  # reduce for bootstrap speed
    # disable verbose
    params["verbosity"] = -1

    panel_df = pd.read_csv(_PANEL_CSV)
    panel_features = set(panel_df["feature"].tolist())
    panel_sym = dict(zip(panel_df["feature"], panel_df["symbol"]))

    rng = np.random.default_rng(RANDOM_STATE)
    counts: dict[str, int] = {}
    import time
    print(f"\nRunning {N_BOOTSTRAP} bootstrap iterations ...")
    t_start = time.time()
    for b in range(N_BOOTSTRAP):
        top = _bootstrap_iteration(X, y, rng, params, sfm_features)
        for f in top:
            counts[f] = counts.get(f, 0) + 1
        if (b + 1) % 10 == 0:
            print(f"  bootstrap {b+1}/{N_BOOTSTRAP}  elapsed: {time.time()-t_start:.1f}s", flush=True)

    # Build frequency table
    rows = []
    for f, c in counts.items():
        rows.append({
            "feature": f,
            "symbol": panel_sym.get(f, ""),
            "selection_freq": c / N_BOOTSTRAP,
            "n_selected": c,
            "is_panel": f in panel_features,
        })
    df = pd.DataFrame(rows).sort_values("selection_freq", ascending=False)
    df.to_csv(SCRIPT_DIR / "stability_selection.csv", index=False)
    print(f"\nTop-50 features by selection frequency:")
    print(df.head(50).to_string(index=False))

    # Panel coverage
    panel_freq = df[df["is_panel"]]
    print(f"\nPanel gene selection frequencies (n_panel_total={len(panel_features)}):")
    for _, row in panel_freq.iterrows():
        print(f"  {row['symbol']:<16}  freq={row['selection_freq']:.2f}  n={int(row['n_selected'])}/{N_BOOTSTRAP}")

    n_panel_seen = len(panel_freq)
    print(f"\nPanel genes ever selected in {N_BOOTSTRAP} bootstraps: {n_panel_seen}/{len(panel_features)}")
    high_stab = panel_freq[panel_freq["selection_freq"] >= 0.80]
    mid_stab = panel_freq[(panel_freq["selection_freq"] >= 0.50) & (panel_freq["selection_freq"] < 0.80)]
    low_stab = panel_freq[panel_freq["selection_freq"] < 0.50]
    print(f"  High stability (≥80%): {len(high_stab)} — {high_stab['symbol'].tolist()}")
    print(f"  Mid stability (50-80%): {len(mid_stab)} — {mid_stab['symbol'].tolist()}")
    print(f"  Low stability (<50%): {len(low_stab)} — {low_stab['symbol'].tolist()}")

    # Statistics file
    lines = [
        f"Stability selection — {N_BOOTSTRAP} stratified bootstraps × top-{TOP_K} SHAP on 3,704 SFM genes",
        "=" * 70,
        "",
        "Panel gene selection frequencies:",
    ]
    for _, row in panel_freq.iterrows():
        flag = "HIGH" if row["selection_freq"] >= 0.80 else "MID" if row["selection_freq"] >= 0.50 else "LOW"
        lines.append(f"  [{flag:<4}] {row['symbol']:<16}  freq={row['selection_freq']:.2f}")

    lines += [
        "",
        f"High stability (selected in ≥80% of bootstraps; n={len(high_stab)}):",
        "  " + ", ".join(sorted(high_stab["symbol"].tolist())) if not high_stab.empty else "  (none)",
        "",
        f"Mid stability (50-80%; n={len(mid_stab)}):",
        "  " + ", ".join(sorted(mid_stab["symbol"].tolist())) if not mid_stab.empty else "  (none)",
        "",
        f"Low stability (<50%; n={len(low_stab)}):",
        "  " + ", ".join(sorted(low_stab["symbol"].tolist())) if not low_stab.empty else "  (none)",
        "",
        "Top-30 candidates by stability across ALL bootstraps (not restricted to panel):",
    ]
    for _, row in df.head(30).iterrows():
        sym = row["symbol"] if row["symbol"] else row["feature"][:15] + "..."
        flag = "PANEL" if row["is_panel"] else "non-panel"
        lines.append(f"  {sym:<20}  freq={row['selection_freq']:.2f}  ({flag})")

    (SCRIPT_DIR / "stability_selection_statistics.txt").write_text("\n".join(lines))

    # Plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(panel_freq) * 0.3 + 2)))
    panel_freq_sorted = panel_freq.sort_values("selection_freq")
    colors = ["#d62728" if f < 0.50 else "#ff7f0e" if f < 0.80 else "#2ca02c"
              for f in panel_freq_sorted["selection_freq"]]
    ax.barh(panel_freq_sorted["symbol"], panel_freq_sorted["selection_freq"],
            color=colors, edgecolor="black", lw=0.5)
    ax.axvline(0.80, ls="--", color="black", lw=0.6, label="80% (high stability)")
    ax.axvline(0.50, ls=":", color="black", lw=0.6, label="50% (mid stability)")
    ax.set_xlabel("Selection frequency across 100 bootstrap iterations")
    ax.set_xlim(0, 1)
    ax.set_title(f"Panel gene stability — top-25 SHAP selection across {N_BOOTSTRAP} bootstraps")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(SCRIPT_DIR / "stability_selection.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("\nSaved -> stability_selection.{csv,txt,png}")


if __name__ == "__main__":
    main()
