"""Iterative decontamination pipeline — LightGBM throughout.

Feature selection uses SelectFromModel(LGBMClassifier, threshold="mean").
SHAP values computed via shap.TreeExplainer (exact).

Outputs (prefix: lgbm_iter_)
------------------------------
  lgbm_iter_log.csv
  lgbm_iter_performance.png
  lgbm_iter_contamination_pool.json
  lgbm_iter_{N}_sfm_features.txt
  lgbm_iter_final_shap_ranking.csv
  lgbm_iter_final_shap_values.npy
  lgbm_iter_final_curve.csv
  lgbm_iter_final_loo_summary.csv
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path

import numpy as np

ALS_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ALS_DIR))
SCRIPT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PLATFORM = "GPL24676"
CV_FOLDS = 5
CV_RANDOM_STATE = 42
TOP_N = 50
MAX_ITERATIONS = 20

_BEST_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"
if _BEST_PARAMS_PATH.exists():
    import json as _json

    LGBM_PARAMS: dict = _json.loads(_BEST_PARAMS_PATH.read_text())
    print(f"Loaded LightGBM params from {_BEST_PARAMS_PATH}")
else:
    LGBM_PARAMS: dict = dict(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        random_state=0,
        verbose=-1,
    )
    print("Using default LightGBM params (lgbm_best_params.json not found)")

PREFIX = "lgbm_iter_"

ITER_LOG_CSV = SCRIPT_DIR / f"{PREFIX}log.csv"
ITER_PLOT_PNG = SCRIPT_DIR / f"{PREFIX}performance.png"
CONTAMINATION_POOL_JSON = SCRIPT_DIR / f"{PREFIX}contamination_pool.json"

# Pre-filter cache written by biomarker_discovery_lgbm_bayesopt.py
_PREFILTER_X_NPY = SCRIPT_DIR / "lgbm_prefilter_X.npy"
_PREFILTER_NAMES_TXT = SCRIPT_DIR / "lgbm_prefilter_names.txt"
FINAL_SHAP_CSV = SCRIPT_DIR / f"{PREFIX}final_shap_ranking.csv"
FINAL_SHAP_NPY = SCRIPT_DIR / f"{PREFIX}final_shap_values.npy"
FINAL_CURVE_CSV = SCRIPT_DIR / f"{PREFIX}final_curve.csv"
FINAL_CURVE_CHECKPOINT = SCRIPT_DIR / f"{PREFIX}final_curve_ckpt.csv"
FINAL_LOO_CSV = SCRIPT_DIR / f"{PREFIX}final_loo_summary.csv"
FINAL_LOO_CHECKPOINT = SCRIPT_DIR / f"{PREFIX}final_loo_ckpt.csv"

# CTD compartments — used for decontamination regression only
COMPARTMENTS: dict[str, tuple[str, ...]] = {
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

# Known ALS-associated features removed at iteration 0
KNOWN_ALS_FEATURES: frozenset[str] = frozenset(
    {
        "ENSG00000133063.16",  # CHIT1
        "ENSG00000165030.4",  # NFIL3
        "ENSG00000168314.17",  # MOBP
        "ENSG00000114698.15",  # PLSCR4
        "ENSG00000272821.1",  # SCO2-AS1
        "ENSG00000197181.12",  # PIWIL2
        "ENSG00000169248.12",  # CXCL11
        "ENSG00000004939.15",  # SLC4A1
        "ENSG00000281721.1",  # LINC01080
        "ENSG00000003989.17",  # SLC7A2
        "ENSG00000188993.3",  # LRRC66
        "ENSG00000137714.3",  # FDX1
        "ENSG00000203804.4",  # ADAMTSL4-AS1
        "ENSG00000011198.9",  # ABHD5
        "ENSG00000137880.6",  # GCHFR
        "ENSG00000158578.20",  # ALAS2
        "ENSG00000225972.1",  # MTND1P23
        "ENSG00000248527.1",  # MTATP6P1
    }
)

KNOWN_ALS_SYMBOLS: dict[str, str] = {
    "ENSG00000133063.16": "CHIT1",
    "ENSG00000165030.4": "NFIL3",
    "ENSG00000168314.17": "MOBP",
    "ENSG00000114698.15": "PLSCR4",
    "ENSG00000272821.1": "SCO2-AS1",
    "ENSG00000197181.12": "PIWIL2",
    "ENSG00000169248.12": "CXCL11",
    "ENSG00000004939.15": "SLC4A1",
    "ENSG00000281721.1": "LINC01080",
    "ENSG00000003989.17": "SLC7A2",
    "ENSG00000188993.3": "LRRC66",
    "ENSG00000137714.3": "FDX1",
    "ENSG00000203804.4": "ADAMTSL4-AS1",
    "ENSG00000011198.9": "ABHD5",
    "ENSG00000137880.6": "GCHFR",
    "ENSG00000158578.20": "ALAS2",
    "ENSG00000225972.1": "MTND1P23",
    "ENSG00000248527.1": "MTATP6P1",
}

CONTAMINATION_SYMBOLS: list[str] = [
    # Erythrocyte / reticulocyte
    "HBB",
    "HBA1",
    "HBA2",
    "ALAS2",
    "GYPA",
    "GYPB",
    "GYPC",
    "CA1",
    "CA2",
    "AHSP",
    "KLF1",
    "HEMGN",
    "HBD",
    "BNIP3L",
    "EPB41",
    "SPTA1",
    "SPTB",
    "BPGM",
    "SLC4A1",
    # Platelet / megakaryocyte
    "PPBP",
    "PF4",
    "PF4V1",
    "GP1BA",
    "GP1BB",
    "GP9",
    "ITGA2B",
    "ITGB3",
    "SELP",
    "CLEC1B",
    "TUBB1",
    # Neutrophil / myeloid
    "DEFA1",
    "DEFA1B",
    "DEFA3",
    "DEFA4",
    "DEFA5",
    "DEFA6",
    "ELANE",
    "MPO",
    "PRTN3",
    "AZU1",
    "CTSG",
    "S100A8",
    "S100A9",
    "S100A12",
    "CAMP",
    "LTF",
    "LCN2",
    "MMP8",
    "MMP9",
    "CEACAM8",
    "FCGR3B",
    # T cell / NK
    "CD3D",
    "CD3E",
    "CD3G",
    "CD247",
    "TRBC1",
    "TRBC2",
    "TRAC",
    "CD8A",
    "CD8B",
    "GNLY",
    "NKG7",
    "GZMA",
    "GZMB",
    "PRF1",
    # B cell / immunoglobulin
    "IGHG1",
    "IGHG2",
    "IGHG3",
    "IGHG4",
    "IGHA1",
    "IGHA2",
    "IGHM",
    "IGHD",
    "IGHE",
    "IGKC",
    "IGLC1",
    "IGLC2",
    "IGLC3",
    "IGLC6",
    "IGLC7",
    "IGHV1-2",
    "IGHV3-23",
    "IGKV1-5",
    "IGLV2-14",
    # Muscle contamination
    "TNNI2",
    "TNNT3",
    "MYH1",
    "MYH2",
    "MYH4",
    "MYBPC2",
    "ACTN3",
    # Mitochondrial / RNA quality
    "MT-ND1",
    "MT-ND2",
    "MT-ND3",
    "MT-ND4",
    "MT-ND4L",
    "MT-ND5",
    "MT-ND6",
    "MT-CO1",
    "MT-CO2",
    "MT-CO3",
    "MT-ATP6",
    "MT-ATP8",
    "MT-CYB",
    "MT-RNR1",
    "MT-RNR2",
    # Sex chromosome / pseudoautosomal
    "XIST",
    "TSIX",
    "RPS4Y1",
    "EIF1AY",
    "DDX3Y",
    "USP9Y",
    "KDM5D",
    "UTY",
    "TMSB4Y",
    "NLGN4Y",
    "ZFY",
    "PRKY",
    "AMELY",
]


# ---------------------------------------------------------------------------
# Data loading helpers (identical to original)
# ---------------------------------------------------------------------------


def _load_raw() -> tuple[np.ndarray, np.ndarray, list[str]]:
    from utils import load_dataset

    print("Loading GSE153960 ...")
    (ds,) = load_dataset(
        "GSE153960", platform=PLATFORM, resources_dir=ALS_DIR / "resources"
    )
    X_log = np.log1p(ds.X.values.astype(float))
    y_int = ds.y.values.astype(int)
    feature_names: list[str] = ds.X.columns.tolist()
    print(f"  Loaded: {X_log.shape[0]} samples, {X_log.shape[1]} features")
    return X_log, y_int, feature_names


def _ctd_decontaminate(X_log: np.ndarray, feature_names: list[str]) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression

    print("CTD compartment regression ...")
    pc_scores: list[np.ndarray] = []
    for comp, bases in COMPARTMENTS.items():
        idx = [i for i, n in enumerate(feature_names) if n.split(".")[0] in set(bases)]
        if not idx:
            print(f"  WARNING: no features found for compartment '{comp}'")
            continue
        pca = PCA(n_components=1, random_state=0)
        score = pca.fit_transform(X_log[:, idx])
        pc_scores.append(score)
        print(
            f"  {comp}: {len(idx)} markers, PC1 var={pca.explained_variance_ratio_[0]:.3f}"
        )

    Z = np.hstack(pc_scores)
    reg = LinearRegression()
    reg.fit(Z, X_log)
    X_decontam = (X_log - reg.predict(Z)).astype(np.float32)
    print(f"  Residual matrix: {X_decontam.shape}")
    return X_decontam


def _apply_constant_remover(
    X: np.ndarray, feature_names: list[str]
) -> tuple[np.ndarray, list[str]]:
    from sklearn.feature_selection import VarianceThreshold

    vt = VarianceThreshold(threshold=0.0)
    X_out = vt.fit_transform(X)
    names_out = [feature_names[i] for i in vt.get_support(indices=True)]
    removed = len(feature_names) - len(names_out)
    print(
        f"  ConstantRemover: removed {removed} zero-variance features → {len(names_out)}"
    )
    return X_out, names_out


def _resolve_contamination_pool(
    base_to_full: dict[str, str],
) -> tuple[frozenset[str], dict[str, str]]:
    """Resolve contamination symbols → versioned Ensembl IDs via REST API."""
    if CONTAMINATION_POOL_JSON.exists():
        data = json.loads(CONTAMINATION_POOL_JSON.read_text())
        pool = frozenset(data["pool"])
        sym_map = data["symbol_map"]
        print(f"  Loaded contamination pool from cache: {len(pool)} features")
        return pool, sym_map

    print(
        f"  Resolving {len(CONTAMINATION_SYMBOLS)} contamination symbols via Ensembl REST API ..."
    )
    pool: set[str] = set()
    sym_map: dict[str, str] = {}

    for symbol in CONTAMINATION_SYMBOLS:
        url = f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{symbol}?content-type=application/json"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=10) as resp:
                    obj = json.loads(resp.read())
                ensembl_base = obj.get("id", "")
                if ensembl_base and ensembl_base in base_to_full:
                    versioned = base_to_full[ensembl_base]
                    pool.add(versioned)
                    sym_map[versioned] = symbol
                break
            except (urllib.error.URLError, json.JSONDecodeError):
                if attempt < 2:
                    time.sleep(1.0)
        else:
            print(f"    WARNING: could not resolve {symbol}")

    CONTAMINATION_POOL_JSON.write_text(
        json.dumps({"pool": sorted(pool), "symbol_map": sym_map}, indent=2)
    )
    print(f"  Resolved {len(pool)} contamination features → cached")
    return frozenset(pool), sym_map


# ---------------------------------------------------------------------------
# Feature selection — SelectFromModel (LightGBM, best params)
# ---------------------------------------------------------------------------


def _sfm_cache_path(iteration: int) -> Path:
    return SCRIPT_DIR / f"{PREFIX}{iteration}_sfm_features.txt"


def _run_sfm_iteration(
    X_iter: np.ndarray,
    y: np.ndarray,
    iter_names: list[str],
    iteration: int,
) -> list[str]:
    """Select features via SelectFromModel with the optimised LightGBM params."""
    cache_path = _sfm_cache_path(iteration)
    if cache_path.exists():
        selected = [s.strip() for s in cache_path.read_text().splitlines() if s.strip()]
        print(f"  Loaded {len(selected)} SFM features from cache (iter {iteration})")
        return selected

    from lightgbm import LGBMClassifier
    from sklearn.feature_selection import SelectFromModel

    print(
        f"  Running SelectFromModel (LightGBM) on {X_iter.shape[1]} features"
        f" (iter {iteration}) ..."
    )
    clf = LGBMClassifier(**LGBM_PARAMS)
    sfm = SelectFromModel(clf, threshold="mean")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        sfm.fit(X_iter, y)

    support = sfm.get_support()
    selected = [iter_names[i] for i, m in enumerate(support) if m]
    cache_path.write_text("\n".join(selected) + "\n")
    print(f"  SelectFromModel selected {len(selected)} features (iter {iteration})")
    return selected


# ---------------------------------------------------------------------------
# Model helpers — LightGBM replaces SingleOutputNeuralNet
# ---------------------------------------------------------------------------


def _cv_auc(
    X: np.ndarray,
    y_int: np.ndarray,
    col_mask: np.ndarray,
) -> tuple[float, list[float]]:
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    X_sub = X[:, col_mask]
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    fold_aucs: list[float] = []

    for train_idx, test_idx in skf.split(X_sub, y_int):
        clf = LGBMClassifier(**LGBM_PARAMS)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            clf.fit(X_sub[train_idx], y_int[train_idx])
        proba = clf.predict_proba(X_sub[test_idx])[:, 1]
        fold_aucs.append(float(roc_auc_score(y_int[test_idx], proba)))

    return float(np.mean(fold_aucs)), fold_aucs


def _train_model(X: np.ndarray, y_int: np.ndarray) -> object:
    from lightgbm import LGBMClassifier

    clf = LGBMClassifier(**LGBM_PARAMS)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        clf.fit(X, y_int)
    return clf


def _compute_shap(
    model: object,
    X: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return (shap_class1, ranked_names, ranked_mean_abs_shap) via TreeExplainer."""
    import shap

    print("  Computing SHAP values (TreeExplainer) ...")
    explainer = shap.TreeExplainer(model)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        shap_vals = explainer.shap_values(X)

    # LightGBM binary: shap_values returns list [class0, class1] or (n, p, 2)
    if isinstance(shap_vals, list):
        shap_class1 = np.array(shap_vals[1])
    elif shap_vals.ndim == 3:
        shap_class1 = shap_vals[:, :, 1]
    else:
        shap_class1 = shap_vals

    mean_abs = np.abs(shap_class1).mean(axis=0)
    rank_idx = np.argsort(mean_abs)[::-1]
    ranked_names = [feature_names[i] for i in rank_idx]
    return shap_class1, ranked_names, mean_abs[rank_idx]


def _run_incremental_cv(
    X: np.ndarray,
    y_int: np.ndarray,
    ranked_names: list[str],
    feature_names: list[str],
    checkpoint_path: Path,
    label: str = "",
    max_k: int | None = None,
) -> list[float]:
    import pandas as pd
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    n_features = min(len(ranked_names), max_k) if max_k else len(ranked_names)
    name_to_col = {name: i for i, name in enumerate(feature_names)}
    rank_col_order = [name_to_col[n] for n in ranked_names]

    auc_scores: list[float] = []
    start_k = 1
    if checkpoint_path.exists():
        prev = pd.read_csv(checkpoint_path)
        auc_scores = prev["mean_auc"].tolist()
        start_k = len(auc_scores) + 1
        print(f"  {label} Resuming from k={start_k}")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)

    for k in range(start_k, n_features + 1):
        col_idx = rank_col_order[:k]
        X_k = X[:, col_idx]
        fold_aucs: list[float] = []

        for train_idx, test_idx in skf.split(X_k, y_int):
            clf = LGBMClassifier(**LGBM_PARAMS)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                clf.fit(X_k[train_idx], y_int[train_idx])
            proba = clf.predict_proba(X_k[test_idx])[:, 1]
            fold_aucs.append(float(roc_auc_score(y_int[test_idx], proba)))

        mean_auc = float(np.mean(fold_aucs))
        auc_scores.append(mean_auc)
        print(
            f"  {label} k={k:3d}/{n_features}  {ranked_names[k - 1][:28]:<28}  "
            f"AUC={mean_auc:.4f}"
        )
        pd.DataFrame(
            {"n_features": list(range(1, len(auc_scores) + 1)), "mean_auc": auc_scores}
        ).to_csv(checkpoint_path, index=False)

    return auc_scores


def _run_loo(
    X: np.ndarray,
    y_int: np.ndarray,
    top30_names: list[str],
    feature_names: list[str],
    baseline_auc: float,
    checkpoint_path: Path,
    label: str = "",
) -> list[dict]:
    import pandas as pd

    name_to_col = {name: i for i, name in enumerate(feature_names)}
    all_mask = np.ones(X.shape[1], dtype=bool)

    done: dict[str, dict] = {}
    if checkpoint_path.exists():
        prev = pd.read_csv(checkpoint_path)
        for _, row in prev.iterrows():
            done[row["feature"]] = row.to_dict()
        print(f"  {label} Resuming LOO ({len(done)} cached)")

    results: list[dict] = []
    for rank_i, feat_name in enumerate(top30_names, start=1):
        if feat_name in done:
            results.append(done[feat_name])
            print(
                f"  {label} [{rank_i:2d}/{TOP_N}]  {feat_name[:36]:<36}  "
                f"AUC={done[feat_name]['loo_auc']:.4f}  "
                f"Δ={done[feat_name]['delta_auc']:+.4f}  (cached)"
            )
            continue

        col_idx = name_to_col[feat_name]
        loo_mask = all_mask.copy()
        loo_mask[col_idx] = False
        loo_auc, fold_aucs = _cv_auc(X, y_int, loo_mask)
        delta = baseline_auc - loo_auc

        row = {
            "rank": rank_i,
            "feature": feat_name,
            "baseline_auc": round(baseline_auc, 4),
            "loo_auc": round(loo_auc, 4),
            "delta_auc": round(delta, 4),
            **{f"fold_{j + 1}_auc": round(a, 4) for j, a in enumerate(fold_aucs)},
        }
        results.append(row)
        done[feat_name] = row

        print(
            f"  {label} [{rank_i:2d}/{TOP_N}]  {feat_name[:36]:<36}  "
            f"AUC={loo_auc:.4f}  Δ={delta:+.4f}  "
            f"folds=[{', '.join(f'{a:.3f}' for a in fold_aucs)}]"
        )
        pd.DataFrame(list(done.values())).to_csv(checkpoint_path, index=False)

    return results


# ---------------------------------------------------------------------------
# Plotting (identical to original)
# ---------------------------------------------------------------------------


def _plot_iter_performance(iter_log: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iterations = [int(r["iteration"]) for r in iter_log]
    aucs = [float(r["mean_auc"]) for r in iter_log]
    n_sfm = [int(r["n_sfm_selected"]) for r in iter_log]

    fig, ax1 = plt.subplots(figsize=(max(8, len(iterations) * 1.5 + 2), 5))
    color_auc = "steelblue"
    color_feat = "darkorange"

    ax1.plot(
        iterations,
        aucs,
        "o-",
        color=color_auc,
        linewidth=2,
        markersize=7,
        label="5-fold CV AUC (LightGBM)",
    )
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Mean AUC-ROC (5-fold CV)", color=color_auc)
    ax1.tick_params(axis="y", labelcolor=color_auc)
    ax1.set_xticks(iterations)
    ax1.set_ylim(max(0.7, min(aucs) - 0.02), 1.01)
    ax1.grid(True, alpha=0.3)

    for r, auc in zip(iter_log, aucs):
        flagged = str(r.get("newly_flagged_symbols", "") or "").strip()
        label_text = f"▲ {flagged}" if flagged else "✓ clean"
        ax1.annotate(
            label_text,
            xy=(int(r["iteration"]), auc),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="crimson" if flagged else "green",
        )

    ax2 = ax1.twinx()
    ax2.bar(
        iterations,
        n_sfm,
        alpha=0.18,
        color=color_feat,
        width=0.4,
        label="SFM features",
    )
    ax2.set_ylabel("SFM-selected features", color=color_feat)
    ax2.tick_params(axis="y", labelcolor=color_feat)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)

    final_iter = iterations[-1]
    final_auc = aucs[-1]
    ax1.axvline(
        final_iter,
        color="green",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label=f"Final clean iter (AUC={final_auc:.4f})",
    )
    ax1.set_title(
        "Iterative decontamination — AUC vs iteration (LightGBM)\n"
        "GSE153960 GPL24676 · SelectFromModel (LightGBM)",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(ITER_PLOT_PNG, dpi=150)
    plt.close("all")
    print(f"  Saved performance plot → {ITER_PLOT_PNG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import pandas as pd

    print("\n" + "=" * 65)
    print("LightGBM Iterative Decontamination Pipeline")
    print("=" * 65)

    if _PREFILTER_X_NPY.exists() and _PREFILTER_NAMES_TXT.exists():
        print("Loading pre-filtered dataset from cache ...")
        X_std = np.load(_PREFILTER_X_NPY)
        names_cr = _PREFILTER_NAMES_TXT.read_text().splitlines()
        _, y_int, _ = _load_raw()
        print(f"  {X_std.shape[0]} samples, {X_std.shape[1]} features")
    else:
        X_log, y_int, feature_names = _load_raw()
        X_decontam = _ctd_decontaminate(X_log, feature_names)
        X_std, names_cr = _apply_constant_remover(X_decontam, feature_names)
        X_std = X_std.astype(np.float32)
        print(f"  Feature matrix: {X_std.shape}")

    base_to_full: dict[str, str] = {n.split(".")[0]: n for n in names_cr}

    print("\n" + "=" * 65)
    print("Resolving contamination pool")
    print("=" * 65)
    contamination_pool, symbol_map = _resolve_contamination_pool(base_to_full)
    symbol_map.update(KNOWN_ALS_SYMBOLS)
    discard_candidates: frozenset[str] = KNOWN_ALS_FEATURES | contamination_pool

    # =========================================================================
    # Load iteration log
    # =========================================================================
    iter_log: list[dict] = []
    discard: set[str] = set(KNOWN_ALS_FEATURES)

    if ITER_LOG_CSV.exists():
        log_df = pd.read_csv(ITER_LOG_CSV)
        for _, row in log_df.iterrows():
            iter_log.append(row.to_dict())
            flagged_str = str(row.get("newly_flagged", "") or "").strip()
            if flagged_str:
                discard.update(f.strip() for f in flagged_str.split(",") if f.strip())
        print(f"\n  Loaded {len(iter_log)} completed iterations from log")
        print(f"  Reconstructed DISCARD: {len(discard)} features")

    # =========================================================================
    # Iterative decontamination loop
    # =========================================================================
    final_sfm_names: list[str] | None = None
    final_iter_idx: int = -1

    for iteration in range(len(iter_log), MAX_ITERATIONS):
        print("\n" + "=" * 65)
        print(
            f"ITERATION {iteration}  |  DISCARD={len(discard)}  "
            f"(ALS={len(KNOWN_ALS_FEATURES)}, "
            f"artifacts={len(discard) - len(KNOWN_ALS_FEATURES)})"
        )
        print("=" * 65)

        iter_col_idx = [i for i, n in enumerate(names_cr) if n not in discard]
        iter_names = [names_cr[i] for i in iter_col_idx]
        X_iter = X_std[:, iter_col_idx]
        print(f"  Features available (after DISCARD): {len(iter_names)}")

        sfm_names = _run_sfm_iteration(X_iter, y_int, iter_names, iteration)

        sfm_col_map = {n: i for i, n in enumerate(iter_names)}
        sfm_idx_in_iter = [sfm_col_map[n] for n in sfm_names]
        X_sfm = X_iter[:, sfm_idx_in_iter]

        print(f"  Computing 5-fold CV AUC on {len(sfm_names)} SFM features ...")
        all_mask = np.ones(X_sfm.shape[1], dtype=bool)
        mean_auc, fold_aucs = _cv_auc(X_sfm, y_int, all_mask)
        print(
            f"  AUC: {mean_auc:.4f}  folds=[{', '.join(f'{a:.3f}' for a in fold_aucs)}]"
        )

        sfm_set = set(sfm_names)
        newly_flagged: set[str] = (sfm_set & discard_candidates) - discard
        newly_flagged_symbols = sorted(symbol_map.get(f, f) for f in newly_flagged)

        row: dict = {
            "iteration": iteration,
            "n_discard": len(discard),
            "n_features_available": len(iter_names),
            "n_sfm_selected": len(sfm_names),
            "mean_auc": round(mean_auc, 4),
            **{f"fold_{j + 1}_auc": round(a, 4) for j, a in enumerate(fold_aucs)},
            "n_newly_flagged": len(newly_flagged),
            "newly_flagged": ",".join(sorted(newly_flagged)),
            "newly_flagged_symbols": ",".join(newly_flagged_symbols),
        }
        iter_log.append(row)
        pd.DataFrame(iter_log).to_csv(ITER_LOG_CSV, index=False)
        print(f"  Logged iteration {iteration} → {ITER_LOG_CSV}")

        if newly_flagged:
            print(
                f"  Flagged {len(newly_flagged)} new artifact(s): {newly_flagged_symbols}"
            )
            discard.update(newly_flagged)
        else:
            print("  No new artifacts — this is the final clean iteration.")
            final_sfm_names = sfm_names
            final_iter_idx = iteration
            break

    _plot_iter_performance(iter_log)

    if final_sfm_names is None:
        print(
            f"\nWARNING: MAX_ITERATIONS ({MAX_ITERATIONS}) reached without convergence. "
            "Using last iteration as final."
        )
        last_iter = len(iter_log) - 1
        final_sfm_names = [
            s.strip()
            for s in _sfm_cache_path(last_iter).read_text().splitlines()
            if s.strip()
        ]
        final_iter_idx = last_iter

    # =========================================================================
    # Final iteration: SHAP + incremental curve + LOO
    # =========================================================================
    print("\n" + "=" * 65)
    print(
        f"FINAL ITERATION {final_iter_idx} — {len(final_sfm_names)} clean SFM features"
    )
    print("SHAP + Incremental Curve + LOO")
    print("=" * 65)

    final_col_idx = [i for i, n in enumerate(names_cr) if n not in discard]
    final_names = [names_cr[i] for i in final_col_idx]
    X_final_iter = X_std[:, final_col_idx]

    final_sfm_col_map = {n: i for i, n in enumerate(final_names)}
    final_sfm_idx = [final_sfm_col_map[n] for n in final_sfm_names]
    X_final = X_final_iter[:, final_sfm_idx]

    # SHAP ranking (cached)
    if FINAL_SHAP_CSV.exists() and FINAL_SHAP_NPY.exists():
        print(f"  Loading cached SHAP from {FINAL_SHAP_CSV}")
        shap_class1 = np.load(FINAL_SHAP_NPY)
        shap_df = pd.read_csv(FINAL_SHAP_CSV)
        final_ranked_names = shap_df["feature"].tolist()
        final_ranked_shap = shap_df["mean_abs_shap"].values
    else:
        print("  Training LightGBM on final SFM features for SHAP ranking ...")
        model = _train_model(X_final, y_int)
        shap_class1, final_ranked_names, final_ranked_shap = _compute_shap(
            model, X_final, final_sfm_names
        )
        np.save(FINAL_SHAP_NPY, shap_class1)
        pd.DataFrame(
            {
                "rank": range(1, len(final_ranked_names) + 1),
                "feature": final_ranked_names,
                "mean_abs_shap": final_ranked_shap,
            }
        ).to_csv(FINAL_SHAP_CSV, index=False)
        print(f"  Saved SHAP ranking → {FINAL_SHAP_CSV}")

    print("  Top 10 final clean features by SHAP:")
    for i, (n, v) in enumerate(zip(final_ranked_names[:10], final_ranked_shap[:10]), 1):
        sym = symbol_map.get(n, "?")
        print(f"    {i:2d}. {n}  ({sym})  shap={v:.6f}")

    final_log_row = next(r for r in iter_log if int(r["iteration"]) == final_iter_idx)
    final_baseline_auc = float(final_log_row["mean_auc"])

    # Incremental CV curve (cached)
    print("\n" + "=" * 65)
    print("Incremental CV curve")
    print("=" * 65)

    if FINAL_CURVE_CSV.exists() and not FINAL_CURVE_CHECKPOINT.exists():
        final_auc_scores = pd.read_csv(FINAL_CURVE_CSV)["mean_auc"].tolist()
        print(f"  Loaded completed final curve ({len(final_auc_scores)} points)")
    else:
        final_auc_scores = _run_incremental_cv(
            X_final,
            y_int,
            final_ranked_names,
            final_sfm_names,
            FINAL_CURVE_CHECKPOINT,
            label="[FINAL-CURVE]",
            max_k=500,
        )
        pd.DataFrame(
            {
                "n_features": list(range(1, len(final_auc_scores) + 1)),
                "mean_auc": final_auc_scores,
            }
        ).to_csv(FINAL_CURVE_CSV, index=False)
        FINAL_CURVE_CHECKPOINT.unlink(missing_ok=True)
        print(f"  Saved incremental curve → {FINAL_CURVE_CSV}")

    # LOO on top-30 SHAP features (cached)
    print("\n" + "=" * 65)
    print(f"LOO on top-{TOP_N} SHAP features (LightGBM)")
    print("=" * 65)

    top30_names = final_ranked_names[:TOP_N]

    if FINAL_LOO_CSV.exists() and not FINAL_LOO_CHECKPOINT.exists():
        print(f"  Loaded completed LOO from {FINAL_LOO_CSV}")
        loo_results = pd.read_csv(FINAL_LOO_CSV).to_dict("records")
    else:
        loo_results = _run_loo(
            X_final,
            y_int,
            top30_names,
            final_sfm_names,
            final_baseline_auc,
            FINAL_LOO_CHECKPOINT,
            label="[LOO]",
        )
        pd.DataFrame(loo_results).to_csv(FINAL_LOO_CSV, index=False)
        FINAL_LOO_CHECKPOINT.unlink(missing_ok=True)
        print(f"  Saved LOO summary → {FINAL_LOO_CSV}")

    print("\n  LOO summary (top 30, sorted by delta):")
    for r in sorted(loo_results, key=lambda x: -float(x["delta_auc"])):
        sym = symbol_map.get(str(r["feature"]), "?")
        print(
            f"    rank={int(r['rank']):2d}  {str(r['feature'])[:36]:<36}  "
            f"({sym:<12})  Δ={float(r['delta_auc']):+.4f}"
        )

    print("\n" + "=" * 65)
    print("DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
