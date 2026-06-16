"""Module-aware stability selection.

Extends stability_selection.py: for each bootstrap iteration, records the
full top-25 set (not just aggregated counts), then computes two metrics
per panel gene:

  - single_freq: fraction of bootstraps in which the panel gene itself
    appears in the top-25 (same as stability_selection.py).
  - module_freq: fraction of bootstraps in which the panel gene OR any
    of its valid (non-inferiority) replacements appears in the top-25.

Replacement candidates are loaded from gene_replacement_results.csv
(is_replacement == True; AUC non-inferiority criterion from Step 27).

This addresses the L1/RF-style criticism: a panel gene with low single_freq
may have high module_freq if the bootstrap consistently selects a
correlated alternative from the same biological module.

Outputs:
  stability_module.csv  — per-panel-gene single_freq + module_freq
  stability_module.txt  — human-readable summary
  stability_module.png  — bar chart comparing both metrics
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
_REPLACEMENT_CSV = SCRIPT_DIR / "gene_replacement_results.csv"


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
    X, y, rng, params, sfm_features,
) -> set[str]:
    from lightgbm import LGBMClassifier
    import shap
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
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_b)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        mean_abs = np.abs(shap_vals).mean(axis=0)
        top_idx = np.argsort(mean_abs)[-TOP_K:]
    return {sfm_features[i] for i in top_idx}


def _build_module_map(panel_df: pd.DataFrame) -> dict[str, set[str]]:
    """For each panel gene, the set of valid replacement ENSG base IDs."""
    rep_df = pd.read_csv(_REPLACEMENT_CSV)
    rep_df = rep_df[rep_df["is_replacement"]]
    modules: dict[str, set[str]] = {}
    panel_base = {sym: ensg for sym, ensg in zip(panel_df["symbol"], panel_df["feature"])}
    for sym in panel_df["symbol"]:
        sub = rep_df[rep_df["panel_symbol"] == sym]
        bases = {panel_base[sym].split(".")[0]}
        bases.update(c.split(".")[0] for c in sub["candidate_ensg"].astype(str))
        modules[sym] = bases
    return modules


def _to_base(feature_id: str) -> str:
    return feature_id.split(".")[0]


def main() -> None:
    print("Loading SFM matrix and labels ...", flush=True)
    X, sfm_features = _load_sfm_matrix()
    (ds,) = load_dataset("GSE153960", platform="GPL24676",
                         resources_dir=ALS_DIR / "resources")
    y = ds.y.values.astype(int)
    print(f"  X: {X.shape}  ALS={int(y.sum())}  Ctrl={len(y)-int(y.sum())}", flush=True)

    params = json.loads(_PARAMS_PATH.read_text())
    params["colsample_bytree"] = 1.0
    params["random_state"] = RANDOM_STATE
    params["n_estimators"] = 600
    params["verbosity"] = -1

    panel_df = pd.read_csv(_PANEL_CSV)
    panel_features = set(panel_df["feature"].tolist())

    print("Building module map from gene_replacement_results.csv ...", flush=True)
    modules = _build_module_map(panel_df)
    for sym, bases in modules.items():
        print(f"  {sym:<16}  |module|={len(bases)}", flush=True)

    rng = np.random.default_rng(RANDOM_STATE)
    iterations: list[set[str]] = []
    import time
    t0 = time.time()
    print(f"\nRunning {N_BOOTSTRAP} bootstrap iterations ...", flush=True)
    for b in range(N_BOOTSTRAP):
        top = _bootstrap_iteration(X, y, rng, params, sfm_features)
        iterations.append({_to_base(f) for f in top})
        if (b + 1) % 10 == 0:
            print(f"  bootstrap {b+1}/{N_BOOTSTRAP}  elapsed: {time.time()-t0:.1f}s", flush=True)

    # Compute single + module stability per panel gene
    rows = []
    for sym in panel_df["symbol"]:
        panel_base = panel_df.loc[panel_df["symbol"] == sym, "feature"].iloc[0].split(".")[0]
        single = sum(1 for it in iterations if panel_base in it)
        mod = sum(1 for it in iterations if it & modules[sym])
        rows.append({
            "panel_gene": sym,
            "panel_base": panel_base,
            "module_size": len(modules[sym]),
            "single_freq": single / N_BOOTSTRAP,
            "module_freq": mod / N_BOOTSTRAP,
            "delta": (mod - single) / N_BOOTSTRAP,
        })
    df = pd.DataFrame(rows).sort_values("module_freq", ascending=False)
    df.to_csv(SCRIPT_DIR / "stability_module.csv", index=False)
    print("\n" + df.to_string(index=False))

    lines = [
        "Module-aware stability selection",
        "=" * 60,
        "single_freq = fraction of bootstraps where the panel gene itself is in top-25.",
        "module_freq = fraction of bootstraps where the panel gene OR ANY of its",
        "              valid replacements (Step 27, non-inferiority criterion) is in top-25.",
        "",
        df.to_string(index=False),
        "",
        "Stability tiers (module_freq):",
    ]
    high = df[df["module_freq"] >= 0.80]
    mid  = df[(df["module_freq"] >= 0.50) & (df["module_freq"] < 0.80)]
    low  = df[df["module_freq"] < 0.50]
    lines.append(f"  HIGH (≥80%): n={len(high)} — " + (", ".join(high["panel_gene"]) if not high.empty else "(none)"))
    lines.append(f"  MID  (50-80%): n={len(mid)} — " + (", ".join(mid["panel_gene"]) if not mid.empty else "(none)"))
    lines.append(f"  LOW  (<50%): n={len(low)} — " + (", ".join(low["panel_gene"]) if not low.empty else "(none)"))

    (SCRIPT_DIR / "stability_module.txt").write_text("\n".join(lines))

    # Plot
    fig, ax = plt.subplots(figsize=(11, max(6, len(df) * 0.3 + 2)))
    df_sorted = df.sort_values("module_freq")
    y_pos = np.arange(len(df_sorted))
    ax.barh(y_pos - 0.2, df_sorted["single_freq"], height=0.4,
            color="#d62728", label="single-gene stability")
    ax.barh(y_pos + 0.2, df_sorted["module_freq"], height=0.4,
            color="#2ca02c", label="module stability (gene or valid replacement)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["panel_gene"], fontsize=9)
    ax.axvline(0.80, ls="--", color="black", lw=0.6)
    ax.axvline(0.50, ls=":", color="black", lw=0.6)
    ax.set_xlabel("Selection frequency across 100 bootstraps")
    ax.set_xlim(0, 1.05)
    ax.set_title("Single-gene vs module-aware stability for each panel member")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(SCRIPT_DIR / "stability_module.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("\nSaved -> stability_module.{csv,txt,png}")


if __name__ == "__main__":
    main()
