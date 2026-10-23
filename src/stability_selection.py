"""Bootstrap-based stability selection on the SFM gene space.

Reviewer M5: Cross-architecture Jaccard 0.06-0.11 is weak; the standard
antidote is stability selection. This script:
  - Draws 100 donor-level bootstraps of the 874 training samples (donors
    resampled with replacement within each diagnosis, all of a donor's
    samples kept together, since GPL24676 has several regions per donor).
  - In each replicate, reruns the published ranking steps on the SFM
    SFM pool (nested_panel_cv._rank: SHAP top 500 with the top-500 LightGBM,
    then SHAP re-rank) and records the top 25.
  - Reports per-gene selection frequency across replicates.
  - Reports which of the 25 panel members survive a frequency threshold.

A panel gene that appears in the top-25 across ≥80% of bootstraps is
considered stably selected; lower frequencies indicate cohort-specific or
correlated-feature substitution.

Outputs:
  stability_selection.csv          (per-gene selection frequency, top 50)
  stability_selection_statistics.txt
  stability_selection.png
  stability_selection_bootstraps.json (each bootstrap's top 25; read by stability_module.py)
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

sys.path.insert(0, str(SCRIPT_DIR))

from nested_panel_cv import _rank  # noqa: E402
from utils import load_dataset  # noqa: E402
from utils import CORE_PANEL_SIZE  # noqa: E402

N_BOOTSTRAP = 100
TOP_K = CORE_PANEL_SIZE
RANDOM_STATE = 42
BOOTSTRAPS_JSON = SCRIPT_DIR / "stability_selection_bootstraps.json"
_SFM_RANKING = SCRIPT_DIR / "lgbm_iter_final_shap_ranking.csv"
_N_SFM = sum(1 for _ in _SFM_RANKING.open()) - 1  # header row
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
    X: np.ndarray,
    y: np.ndarray,
    donor_rows: dict[int, dict[str, np.ndarray]],
    rng: np.random.Generator,
    sfm_features: list[str],
) -> list[str]:
    """One donor-level stratified bootstrap; return its top-K genes, best first."""
    parts = []
    for label in (1, 0):
        rows = donor_rows[label]
        picks = rng.choice(list(rows), size=len(rows), replace=True)
        parts += [rows[d] for d in picks]
    idx = np.concatenate(parts)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        return _rank(X[idx], y[idx], sfm_features)[:TOP_K]


def main() -> None:
    print("Loading SFM matrix and labels ...")
    X, sfm_features = _load_sfm_matrix()
    (ds,) = load_dataset(
        "GSE153960", platform="GPL24676", resources_dir=ALS_DIR / "resources"
    )
    y = ds.y.values.astype(int)
    print(f"  X: {X.shape}  ALS={int(y.sum())}  Ctrl={len(y) - int(y.sum())}")

    groups = ds.meta.loc[ds.y.index, "subject id"].to_numpy()
    donor_rows = {
        label: {d: np.flatnonzero(groups == d) for d in np.unique(groups[y == label])}
        for label in (1, 0)
    }
    print(f"  donors: {len(donor_rows[1])} ALS, {len(donor_rows[0])} control")

    panel_df = pd.read_csv(_PANEL_CSV)
    panel_features = set(panel_df["feature"].tolist())
    panel_sym = dict(zip(panel_df["feature"], panel_df["symbol"]))

    rng = np.random.default_rng(RANDOM_STATE)
    counts: dict[str, int] = {}
    boots: list[list[str]] = []
    import time

    print(f"\nRunning {N_BOOTSTRAP} bootstrap iterations ...")
    t_start = time.time()
    for b in range(N_BOOTSTRAP):
        top = _bootstrap_iteration(X, y, donor_rows, rng, sfm_features)
        boots.append(top)
        for f in top:
            counts[f] = counts.get(f, 0) + 1
        if (b + 1) % 10 == 0:
            print(
                f"  bootstrap {b + 1}/{N_BOOTSTRAP}  elapsed: {time.time() - t_start:.1f}s",
                flush=True,
            )

    BOOTSTRAPS_JSON.write_text(json.dumps(boots))

    # Build frequency table
    rows = []
    for f, c in counts.items():
        rows.append(
            {
                "feature": f,
                "symbol": panel_sym.get(f, ""),
                "selection_freq": c / N_BOOTSTRAP,
                "n_selected": c,
                "is_panel": f in panel_features,
            }
        )
    df = pd.DataFrame(rows).sort_values("selection_freq", ascending=False)
    df.to_csv(SCRIPT_DIR / "stability_selection.csv", index=False)
    print("\nTop-50 features by selection frequency:")
    print(df.head(50).to_string(index=False))

    # Panel coverage
    panel_freq = df[df["is_panel"]]
    print(f"\nPanel gene selection frequencies (n_panel_total={len(panel_features)}):")
    for _, row in panel_freq.iterrows():
        print(
            f"  {row['symbol']:<16}  freq={row['selection_freq']:.2f}  n={int(row['n_selected'])}/{N_BOOTSTRAP}"
        )

    n_panel_seen = len(panel_freq)
    print(
        f"\nPanel genes ever selected in {N_BOOTSTRAP} bootstraps: {n_panel_seen}/{len(panel_features)}"
    )
    high_stab = panel_freq[panel_freq["selection_freq"] >= 0.80]
    mid_stab = panel_freq[
        (panel_freq["selection_freq"] >= 0.50) & (panel_freq["selection_freq"] < 0.80)
    ]
    low_stab = panel_freq[panel_freq["selection_freq"] < 0.50]
    print(f"  High stability (≥80%): {len(high_stab)} — {high_stab['symbol'].tolist()}")
    print(f"  Mid stability (50-80%): {len(mid_stab)} — {mid_stab['symbol'].tolist()}")
    print(f"  Low stability (<50%): {len(low_stab)} — {low_stab['symbol'].tolist()}")

    # Statistics file
    lines = [
        f"Stability selection — {N_BOOTSTRAP} donor-level bootstraps × published ranking steps, top-{TOP_K} of the {_N_SFM:,} SFM genes",
        "=" * 70,
        "",
        "Panel gene selection frequencies:",
    ]
    for _, row in panel_freq.iterrows():
        flag = (
            "HIGH"
            if row["selection_freq"] >= 0.80
            else "MID"
            if row["selection_freq"] >= 0.50
            else "LOW"
        )
        lines.append(
            f"  [{flag:<4}] {row['symbol']:<16}  freq={row['selection_freq']:.2f}"
        )

    lines += [
        "",
        f"High stability (selected in ≥80% of bootstraps; n={len(high_stab)}):",
        "  " + ", ".join(sorted(high_stab["symbol"].tolist()))
        if not high_stab.empty
        else "  (none)",
        "",
        f"Mid stability (50-80%; n={len(mid_stab)}):",
        "  " + ", ".join(sorted(mid_stab["symbol"].tolist()))
        if not mid_stab.empty
        else "  (none)",
        "",
        f"Low stability (<50%; n={len(low_stab)}):",
        "  " + ", ".join(sorted(low_stab["symbol"].tolist()))
        if not low_stab.empty
        else "  (none)",
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
    colors = [
        "#d62728" if f < 0.50 else "#ff7f0e" if f < 0.80 else "#2ca02c"
        for f in panel_freq_sorted["selection_freq"]
    ]
    ax.barh(
        panel_freq_sorted["symbol"],
        panel_freq_sorted["selection_freq"],
        color=colors,
        edgecolor="black",
        lw=0.5,
    )
    ax.axvline(0.80, ls="--", color="black", lw=0.6, label="80% (high stability)")
    ax.axvline(0.50, ls=":", color="black", lw=0.6, label="50% (mid stability)")
    ax.set_xlabel("Selection frequency across 100 bootstrap iterations")
    ax.set_xlim(0, 1)
    ax.set_title(
        f"Panel gene stability — top-25 SHAP selection across {N_BOOTSTRAP} bootstraps"
    )
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(SCRIPT_DIR / "stability_selection.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("\nSaved -> stability_selection.{csv,txt,png}")


if __name__ == "__main__":
    main()
