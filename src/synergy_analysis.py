# ruff: noqa: E402
"""Combinatorial synergy analysis — find non-predictive gene pairs/triples/quadruplets
that become discriminative when combined.

Pool: genes whose univariate AUC < AUC_THRESHOLD from gene_profiling_summary.csv.
Tests all C(n,2), C(n,3), C(n,4) combinations of pool genes.

Model: LightGBM with top-500 hyperparams, but colsample_bytree=1.0 so that all
features in a k-gene combination participate in every tree split.

Synergy score = combo_AUC − max(individual AUC of genes in combo).
A positive synergy score means the combination adds information beyond the best
individual gene — i.e. the genes exhibit epistatic / interaction signal.

Outputs
-------
  synergy_results.csv          — all tested combos with AUC and synergy score
  synergy_top_pairs.png        — top synergistic pairs
  synergy_top_triples.png      — top synergistic triples
  synergy_top_quads.png        — top synergistic quadruplets
"""

from __future__ import annotations

import itertools
import sys
import warnings
from pathlib import Path

import numpy as np

ALS_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset

SCRIPT_DIR = Path(__file__).parent
PLATFORM = "GPL24676"

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PROFILING_CSV = SCRIPT_DIR / "gene_profiling_summary.csv"
_PREFILTER_X_NPY = SCRIPT_DIR / "lgbm_prefilter_X.npy"
_PREFILTER_NAMES_TXT = SCRIPT_DIR / "lgbm_prefilter_names.txt"
_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"

AUC_THRESHOLD = 0.60  # genes with univariate AUC < this form the pool
SYNERGY_DELTA = 0.04  # min gain over best individual to count as synergistic
CV_FOLDS = 5
CV_RANDOM_STATE = 42
TOP_DISPLAY = 15  # how many combos to show per rank


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_data() -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    """Return X (874 × 25), features, symbols, y for the full 25-gene panel."""

    import pandas as pd

    panel = pd.read_csv(_PANEL_CSV)
    features: list[str] = panel["feature"].tolist()
    symbols: list[str] = panel["symbol"].tolist()

    all_names = _PREFILTER_NAMES_TXT.read_text().splitlines()
    name_to_col = {n: i for i, n in enumerate(all_names)}
    col_idx = [name_to_col[f] for f in features]
    X_full = np.load(_PREFILTER_X_NPY)
    X = X_full[:, col_idx]

    (ds,) = load_dataset(
        "GSE153960", platform=PLATFORM, resources_dir=ALS_DIR / "resources"
    )
    y_int = ds.y.values.astype(int)
    return X, features, symbols, y_int


def _get_pool(symbols: list[str]) -> list[int]:
    """Return column indices of genes with univariate AUC < AUC_THRESHOLD."""
    import pandas as pd

    prof = pd.read_csv(_PROFILING_CSV)
    sym_to_auc = dict(zip(prof["symbol"], prof["univariate_auc"]))
    pool = [i for i, s in enumerate(symbols) if sym_to_auc.get(s, 1.0) < AUC_THRESHOLD]
    pool_syms = [symbols[i] for i in pool]
    pool_aucs = [sym_to_auc[s] for s in pool_syms]
    print(f"  Pool ({len(pool)} genes, univariate AUC < {AUC_THRESHOLD}):")
    for s, a in sorted(zip(pool_syms, pool_aucs), key=lambda x: x[1]):
        print(f"    {s:<20} AUC={a:.4f}")
    return pool


# ---------------------------------------------------------------------------
# CV scoring
# ---------------------------------------------------------------------------


def _cv_auc(X_sub: np.ndarray, y: np.ndarray, params: dict) -> float:
    """5-fold stratified CV AUC for a sub-matrix."""
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    aucs = []
    for tr, te in skf.split(X_sub, y):
        clf = LGBMClassifier(**params)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            clf.fit(X_sub[tr], y[tr])
        aucs.append(float(roc_auc_score(y[te], clf.predict_proba(X_sub[te])[:, 1])))
    return float(np.mean(aucs))


# ---------------------------------------------------------------------------
# Exhaustive combinatorial search
# ---------------------------------------------------------------------------


def run_combos(
    X: np.ndarray,
    y: np.ndarray,
    pool: list[int],
    symbols: list[str],
    individual_aucs: dict[str, float],
    params: dict,
    k: int,
) -> list[dict]:
    """Test all C(len(pool), k) combinations and return sorted results."""
    combos = list(itertools.combinations(pool, k))
    n = len(combos)
    print(f"\n  Testing C({len(pool)},{k}) = {n} combinations ...")
    results = []
    for idx, cols in enumerate(combos, 1):
        syms = tuple(symbols[c] for c in cols)
        best_indiv = max(individual_aucs[s] for s in syms)
        X_sub = X[:, list(cols)]
        auc = _cv_auc(X_sub, y, params)
        synergy = auc - best_indiv
        results.append(
            {
                "k": k,
                "genes": " | ".join(syms),
                "combo_auc": round(auc, 4),
                "best_individual_auc": round(best_indiv, 4),
                "synergy_score": round(synergy, 4),
            }
        )
        if idx % max(1, n // 10) == 0 or idx == n:
            print(
                f"    [{idx:4d}/{n}] {' + '.join(syms):<55}  "
                f"AUC={auc:.4f}  Δ={synergy:+.4f}"
            )
    results.sort(key=lambda r: r["combo_auc"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_combo_bar(results: list[dict], k: int, title_suffix: str) -> None:
    import matplotlib.pyplot as plt

    top = [r for r in results if r["synergy_score"] >= SYNERGY_DELTA][:TOP_DISPLAY]
    if not top:
        top = results[:TOP_DISPLAY]

    best_indivs = [r["best_individual_auc"] for r in top]
    synergies = [r["synergy_score"] for r in top]

    n = len(top)
    fig, ax = plt.subplots(figsize=(10, max(0.65 * n, 5)))
    y_pos = list(range(n - 1, -1, -1))

    # Stacked bar: best individual (grey) + synergy gain (colour)
    ax.barh(
        y_pos,
        best_indivs[::-1],
        color="#bbbbbb",
        edgecolor="none",
        label="Best individual AUC",
    )
    ax.barh(
        y_pos,
        synergies[::-1],
        left=best_indivs[::-1],
        color=["#2ca02c" if s >= SYNERGY_DELTA else "#ff7f0e" for s in synergies[::-1]],
        edgecolor="none",
        label=f"Synergy gain (Δ ≥ {SYNERGY_DELTA}: green, else orange)",
    )

    ax.axvline(
        AUC_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=0.9,
        label=f"Individual AUC threshold ({AUC_THRESHOLD})",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{r['genes']}  [Δ={r['synergy_score']:+.3f}]" for r in top[::-1]],
        fontsize=8,
    )
    ax.set_xlabel(
        "5-fold CV AUC-ROC  (grey = best individual baseline; coloured = synergy gain)\n"
        "Green bar = synergy score ≥ threshold; orange = positive but below threshold"
    )
    ax.set_title(
        f"Top synergistic {title_suffix} — non-predictive gene pool "
        f"(individual AUC < {AUC_THRESHOLD})\n"
        "ALS Spectrum MND vs Non-Neurological Control · GSE153960 GPL24676 · n=874",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0.45, 1.0)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fname = SCRIPT_DIR / f"synergy_top_{title_suffix.lower().replace(' ', '_')}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {fname.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import json

    import matplotlib
    import pandas as pd

    matplotlib.use("Agg")

    print("\n" + "=" * 65)
    print("Synergy Analysis — Non-predictive Gene Combinations")
    print("=" * 65)

    # Load params — override colsample_bytree so all k features participate
    params = json.loads(_PARAMS_PATH.read_text())
    params["colsample_bytree"] = 1.0
    print("\nModel: top-500 params with colsample_bytree=1.0")

    print("\nLoading data ...")
    X, features, symbols, y_int = _load_data()
    print(f"  X={X.shape}  ALS={y_int.sum()}  Control={(y_int == 0).sum()}")

    print("\nIdentifying non-predictive gene pool ...")
    pool = _get_pool(symbols)

    # Individual AUCs from profiling output
    prof = pd.read_csv(_PROFILING_CSV)
    individual_aucs: dict[str, float] = dict(
        zip(prof["symbol"], prof["univariate_auc"])
    )

    all_results: list[dict] = []
    for k in (2, 3, 4):
        res = run_combos(X, y_int, pool, symbols, individual_aucs, params, k)
        all_results.extend(res)

        synergistic = [r for r in res if r["synergy_score"] >= SYNERGY_DELTA]
        print(
            f"\n  k={k}: {len(synergistic)}/{len(res)} combos with "
            f"synergy_score ≥ {SYNERGY_DELTA}"
        )
        label = {2: "pairs", 3: "triples", 4: "quadruplets"}[k]
        print("  Top-5 by combo AUC:")
        for r in res[:5]:
            flag = "★" if r["synergy_score"] >= SYNERGY_DELTA else " "
            print(
                f"   {flag} {r['genes']:<55}  "
                f"AUC={r['combo_auc']:.4f}  "
                f"best_indiv={r['best_individual_auc']:.4f}  "
                f"Δ={r['synergy_score']:+.4f}"
            )

    # Save CSV
    out_csv = SCRIPT_DIR / "synergy_results.csv"
    pd.DataFrame(all_results).to_csv(out_csv, index=False)
    print(f"\nSaved all results → {out_csv}")

    # Plots
    print("\n--- Plots ---")
    for k, label in ((2, "pairs"), (3, "triples"), (4, "quadruplets")):
        subset = [r for r in all_results if r["k"] == k]
        _plot_combo_bar(subset, k, label)

    print("\n" + "=" * 65)
    print("DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
