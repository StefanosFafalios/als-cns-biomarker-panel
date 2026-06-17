# ruff: noqa: E402
"""Synergy analysis over the full top-500 SHAP feature space.

Strategy (avoids brute-force CV over 500 features):
  1. Rank all 500 features by univariate AUC (rank-based, O(n log n)).
  2. Pool = features with AUC < AUC_THRESHOLD.
  3. Screen all C(pool,2) pairs with logistic regression AUC (fast proxy).
     Interaction score  =  LR_pair_AUC  −  max(indiv_AUC_i, indiv_AUC_j).
  4. Shortlist top-SHORTLIST_K pairs → validate with LightGBM 5-fold CV.
  5. Greedy triple extension: for each top-GREEDY_SEED pair, try adding every
     remaining pool gene → screen with LR → shortlist top-SHORTLIST_K triples
     → validate with LightGBM CV.
  6. Repeat for quadruplets.
  Only combinations where ALL members are in the weak pool are reported.

Outputs
-------
  synergy500_pairs.csv           — LightGBM-validated pairs
  synergy500_triples.csv         — LightGBM-validated triples
  synergy500_quads.csv           — LightGBM-validated quadruplets
  synergy500_pairs.png
  synergy500_triples.png
  synergy500_quads.png
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ALS_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset

SCRIPT_DIR = Path(__file__).parent
PLATFORM = "GPL24676"

_SHAP_RANKING_CSV = SCRIPT_DIR / "lgbm_top500_final_shap_ranking.csv"
_PREFILTER_X_NPY = SCRIPT_DIR / "lgbm_prefilter_X.npy"
_PREFILTER_NAMES_TXT = SCRIPT_DIR / "lgbm_prefilter_names.txt"
_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"

AUC_THRESHOLD = 0.60   # defines "individually weak" pool
SHORTLIST_K = 100      # pairs/triples/quads validated with LightGBM CV
GREEDY_SEED = 30       # top pairs used as seeds for triple/quad extension
SYNERGY_DELTA = 0.04   # min gain over best individual to flag as synergistic
CV_FOLDS = 5
CV_RANDOM_STATE = 42
TOP_DISPLAY = 20


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _load_top500() -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return X (874 × 500), feature_names, y."""
    import pandas as pd

    shap_df = pd.read_csv(_SHAP_RANKING_CSV)
    top500: list[str] = shap_df["feature"].head(500).tolist()

    all_names = _PREFILTER_NAMES_TXT.read_text().splitlines()
    name_to_col = {n: i for i, n in enumerate(all_names)}
    col_idx = [name_to_col[f] for f in top500]
    X_full = np.load(_PREFILTER_X_NPY)
    X = X_full[:, col_idx]

    (ds,) = load_dataset(
        "GSE153960", platform=PLATFORM, resources_dir=ALS_DIR / "resources"
    )
    y_int = ds.y.values.astype(int)
    print(f"  X={X.shape}  ALS={y_int.sum()}  Control={(y_int == 0).sum()}")
    return X, top500, y_int


def _resolve_symbols(features: list[str]) -> dict[str, str]:
    """Map Ensembl feature IDs to gene symbols via MyGeneInfo (batched, with fallback)."""
    import mygene

    bases = [f.split(".")[0] for f in features]
    mg = mygene.MyGeneInfo()
    base_map: dict[str, str] = {}
    batch_size = 50
    for start in range(0, len(bases), batch_size):
        batch = bases[start : start + batch_size]
        try:
            hits = mg.querymany(
                batch,
                scopes="ensembl.gene",
                fields="symbol",
                species="human",
                verbose=False,
            )
            for h in hits:
                q = h.get("query", "")
                base_map[q] = h.get("symbol", q)
        except Exception as exc:
            print(f"    [warn] MyGeneInfo batch {start}–{start+len(batch)} failed: {exc}")
            for b in batch:
                base_map.setdefault(b, b)
        print(f"    Resolved {min(start + batch_size, len(bases))}/{len(bases)} ...")
    return {f: base_map.get(b, b) for f, b in zip(features, bases)}


# ---------------------------------------------------------------------------
# Fast scoring primitives
# ---------------------------------------------------------------------------


def _univariate_auc(x: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    a = float(roc_auc_score(y, x))
    return max(a, 1.0 - a)


def _lr_auc(X_sub: np.ndarray, y: np.ndarray) -> float:
    """Logistic regression AUC — fast proxy for interaction screening."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    aucs = []
    for tr, te in skf.split(X_sub, y):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_sub[tr])
        Xte = sc.transform(X_sub[te])
        clf = LogisticRegression(max_iter=300, C=1.0)
        clf.fit(Xtr, y[tr])
        aucs.append(float(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1])))
    return float(np.mean(aucs))


def _lgbm_auc(X_sub: np.ndarray, y: np.ndarray, params: dict) -> float:
    """LightGBM 5-fold CV AUC — final validation."""
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
# Screening helpers
# ---------------------------------------------------------------------------


def _screen_pairs(
    X: np.ndarray,
    y: np.ndarray,
    pool: list[int],
    indiv_aucs: np.ndarray,
) -> list[tuple[float, float, tuple[int, ...]]]:
    """Return (lr_auc, synergy, cols) for all C(pool,2) pairs, sorted desc by synergy."""
    import itertools

    results = []
    combos = list(itertools.combinations(pool, 2))
    n = len(combos)
    print(f"    Screening {n} pairs with logistic regression ...")
    for idx, (i, j) in enumerate(combos, 1):
        auc = _lr_auc(X[:, [i, j]], y)
        best = max(indiv_aucs[i], indiv_aucs[j])
        results.append((auc, auc - best, (i, j)))
        if idx % max(1, n // 5) == 0 or idx == n:
            print(f"      [{idx}/{n}] done")
    results.sort(key=lambda r: r[1], reverse=True)
    return results


def _greedy_extend(
    X: np.ndarray,
    y: np.ndarray,
    seed_combos: list[tuple[int, ...]],
    pool: list[int],
    indiv_aucs: np.ndarray,
) -> list[tuple[float, float, tuple[int, ...]]]:
    """Extend each seed combo by one pool gene; screen with LR; return all candidates."""
    seen: set[frozenset] = set()
    candidates = []
    n_seeds = len(seed_combos)
    for s_idx, seed in enumerate(seed_combos, 1):
        seed_set = set(seed)
        for g in pool:
            if g in seed_set:
                continue
            combo = tuple(sorted(seed_set | {g}))
            fs = frozenset(combo)
            if fs in seen:
                continue
            seen.add(fs)
            auc = _lr_auc(X[:, list(combo)], y)
            best = max(indiv_aucs[c] for c in combo)
            candidates.append((auc, auc - best, combo))
        if s_idx % max(1, n_seeds // 5) == 0 or s_idx == n_seeds:
            print(f"      [{s_idx}/{n_seeds} seeds] {len(candidates)} candidates")
    candidates.sort(key=lambda r: r[1], reverse=True)
    return candidates


def _validate_top(
    X: np.ndarray,
    y: np.ndarray,
    candidates: list[tuple[float, float, tuple[int, ...]]],
    indiv_aucs: np.ndarray,
    symbols: list[str],
    params: dict,
    k: int,
) -> list[dict]:
    """Run LightGBM CV on the top-SHORTLIST_K candidates and return results."""
    top = candidates[: SHORTLIST_K]
    print(f"    Validating top {len(top)} candidates with LightGBM CV ...")
    results = []
    for rank, (lr_a, lr_syn, cols) in enumerate(top, 1):
        auc = _lgbm_auc(X[:, list(cols)], y, params)
        best = max(indiv_aucs[c] for c in cols)
        syms = " | ".join(symbols[c] for c in cols)
        results.append(
            {
                "k": k,
                "genes": syms,
                "lgbm_auc": round(auc, 4),
                "lr_auc": round(lr_a, 4),
                "best_individual_auc": round(float(best), 4),
                "synergy_score": round(auc - best, 4),
            }
        )
        if rank % max(1, len(top) // 5) == 0 or rank == len(top):
            print(
                f"      [{rank}/{len(top)}] {syms:<55}  "
                f"LGBM={auc:.4f}  Δ={auc - best:+.4f}"
            )
    results.sort(key=lambda r: r["lgbm_auc"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _plot(results: list[dict], k: int, label: str) -> None:
    import matplotlib.pyplot as plt

    top = [r for r in results if r["synergy_score"] >= SYNERGY_DELTA][:TOP_DISPLAY]
    if not top:
        top = results[:TOP_DISPLAY]

    best_indivs = [r["best_individual_auc"] for r in top]
    synergies = [r["synergy_score"] for r in top]
    n = len(top)

    fig, ax = plt.subplots(figsize=(11, max(0.55 * n, 5)))
    y_pos = list(range(n - 1, -1, -1))

    ax.barh(y_pos, best_indivs[::-1], color="#bbbbbb", edgecolor="none",
            label="Best individual AUC")
    ax.barh(
        y_pos, synergies[::-1], left=best_indivs[::-1],
        color=["#2ca02c" if s >= SYNERGY_DELTA else "#ff7f0e" for s in synergies[::-1]],
        edgecolor="none",
        label=f"Synergy gain (Δ ≥ {SYNERGY_DELTA}: green, else orange)",
    )
    ax.axvline(AUC_THRESHOLD, color="red", linestyle="--", linewidth=0.9,
               label=f"Individual AUC threshold ({AUC_THRESHOLD})")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{r['genes']}  [Δ={r['synergy_score']:+.3f}]" for r in top[::-1]],
        fontsize=7.5,
    )
    ax.set_xlabel(
        "5-fold CV AUC-ROC  (LightGBM; grey = best individual baseline; "
        "coloured = synergy gain)\nGreen = Δ ≥ threshold; orange = positive but below threshold"
    )
    ax.set_title(
        f"Top synergistic {label} — top-500 SHAP feature pool  "
        f"(individual AUC < {AUC_THRESHOLD})\n"
        "ALS Spectrum MND vs Non-Neurological Control · GSE153960 GPL24676 · n=874\n"
        f"Screened with logistic regression; top-{SHORTLIST_K} validated with LightGBM CV",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0.45, 1.0)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    out = SCRIPT_DIR / f"synergy500_{label.replace(' ', '_')}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import json

    import matplotlib
    import pandas as pd

    matplotlib.use("Agg")

    print("\n" + "=" * 68)
    print("Synergy Analysis — Top-500 SHAP Features (MI Pre-screening)")
    print("=" * 68)

    params = json.loads(_PARAMS_PATH.read_text())
    params["colsample_bytree"] = 1.0
    print("Model: top-500 params, colsample_bytree=1.0")

    print("\nLoading data ...")
    X, features, y_int = _load_top500()

    print("\nResolving gene symbols ...")
    sym_map = _resolve_symbols(features)
    symbols = [sym_map.get(f, f.split(".")[0]) for f in features]

    print("\nComputing univariate AUC for all 500 features ...")
    indiv_aucs = np.array([_univariate_auc(X[:, i], y_int) for i in range(X.shape[1])])
    pool = [i for i in range(len(features)) if indiv_aucs[i] < AUC_THRESHOLD]
    print(
        f"  Pool: {len(pool)} features with univariate AUC < {AUC_THRESHOLD}  "
        f"(range {indiv_aucs[pool].min():.3f}–{indiv_aucs[pool].max():.3f})"
    )
    print(f"  Top-5 pool members by AUC:")
    sorted_pool = sorted(pool, key=lambda i: indiv_aucs[i], reverse=True)
    for i in sorted_pool[:5]:
        print(f"    {symbols[i]:<22} AUC={indiv_aucs[i]:.4f}")

    # ------------------------------------------------------------------
    # Pairs
    # ------------------------------------------------------------------
    print("\n--- PAIRS ---")
    pair_candidates = _screen_pairs(X, y_int, pool, indiv_aucs)
    pair_results = _validate_top(X, y_int, pair_candidates, indiv_aucs, symbols, params, k=2)
    pd.DataFrame(pair_results).to_csv(SCRIPT_DIR / "synergy500_pairs.csv", index=False)
    synergistic_pairs = [r for r in pair_results if r["synergy_score"] >= SYNERGY_DELTA]
    print(
        f"  {len(synergistic_pairs)}/{len(pair_results)} validated pairs "
        f"with synergy Δ ≥ {SYNERGY_DELTA}"
    )
    print("  Top-5:")
    for r in pair_results[:5]:
        flag = "★" if r["synergy_score"] >= SYNERGY_DELTA else " "
        print(
            f"   {flag} {r['genes']:<55}  AUC={r['lgbm_auc']:.4f}  "
            f"best={r['best_individual_auc']:.4f}  Δ={r['synergy_score']:+.4f}"
        )

    # ------------------------------------------------------------------
    # Triples (greedy extension of top pairs)
    # ------------------------------------------------------------------
    print(f"\n--- TRIPLES (greedy extend top-{GREEDY_SEED} pairs) ---")
    seed_pairs = [c[2] for c in pair_candidates[:GREEDY_SEED]]
    triple_candidates = _greedy_extend(X, y_int, seed_pairs, pool, indiv_aucs)
    triple_results = _validate_top(
        X, y_int, triple_candidates, indiv_aucs, symbols, params, k=3
    )
    pd.DataFrame(triple_results).to_csv(SCRIPT_DIR / "synergy500_triples.csv", index=False)
    synergistic_triples = [r for r in triple_results if r["synergy_score"] >= SYNERGY_DELTA]
    print(
        f"  {len(synergistic_triples)}/{len(triple_results)} validated triples "
        f"with synergy Δ ≥ {SYNERGY_DELTA}"
    )
    print("  Top-5:")
    for r in triple_results[:5]:
        flag = "★" if r["synergy_score"] >= SYNERGY_DELTA else " "
        print(
            f"   {flag} {r['genes']:<55}  AUC={r['lgbm_auc']:.4f}  "
            f"best={r['best_individual_auc']:.4f}  Δ={r['synergy_score']:+.4f}"
        )

    # ------------------------------------------------------------------
    # Quadruplets (greedy extension of top triples)
    # ------------------------------------------------------------------
    print(f"\n--- QUADS (greedy extend top-{GREEDY_SEED} triples) ---")
    top_triple_cols = [c[2] for c in triple_candidates[:GREEDY_SEED]]
    quad_candidates = _greedy_extend(X, y_int, top_triple_cols, pool, indiv_aucs)
    quad_results = _validate_top(
        X, y_int, quad_candidates, indiv_aucs, symbols, params, k=4
    )
    pd.DataFrame(quad_results).to_csv(SCRIPT_DIR / "synergy500_quads.csv", index=False)
    synergistic_quads = [r for r in quad_results if r["synergy_score"] >= SYNERGY_DELTA]
    print(
        f"  {len(synergistic_quads)}/{len(quad_results)} validated quads "
        f"with synergy Δ ≥ {SYNERGY_DELTA}"
    )
    print("  Top-5:")
    for r in quad_results[:5]:
        flag = "★" if r["synergy_score"] >= SYNERGY_DELTA else " "
        print(
            f"   {flag} {r['genes']:<55}  AUC={r['lgbm_auc']:.4f}  "
            f"best={r['best_individual_auc']:.4f}  Δ={r['synergy_score']:+.4f}"
        )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\n--- Plots ---")
    _plot(pair_results, 2, "pairs")
    _plot(triple_results, 3, "triples")
    _plot(quad_results, 4, "quadruplets")

    print("\n" + "=" * 68)
    print("DONE")
    print("=" * 68)


if __name__ == "__main__":
    main()
