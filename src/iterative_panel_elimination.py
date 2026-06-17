"""Greedy backward elimination of the 25-gene panel using equally-weighted zero-shot AUC
across CNS validation cohorts.

Cohorts (blood excluded):
  GPL16791   n=636  w=0.25  CTD regression, Ensembl base ID
  GSE76220   n=20   w=0.25  plain log1p, HGNC symbol
  GSE122649  n=38   w=0.25  plain log1p, HGNC symbol
  SRP064478  n=15   w=0.25  plain log1p, Ensembl base ID

Algorithm: greedy backward elimination
  At each step, for each remaining gene g, compute
    D_ZS(g) = W.mean_AUC(panel − {g}) − W.mean_AUC(panel)
  Drop the gene with the highest D_ZS (most improvement or least harm).
  Continue until only 1 gene remains (full ranking).

Weights: equal (0.25 per cohort) — each independent dataset has equal vote.

Outputs:
  iterative_panel_elimination_statistics.txt
  iterative_panel_elimination.png
"""
from __future__ import annotations

import gzip
import json
import re
import sys
import tarfile
import warnings
from pathlib import Path

import numpy as np

ALS_DIR = Path(__file__).parents[1]
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset  # noqa: E402

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"
_SRP_QUANT_DIR = ALS_DIR / "resources" / "SRP064478" / "quant"
_SRP_META = ALS_DIR / "resources" / "SRP064478" / "srr_metadata.tsv"
_GSE122649_DIR = ALS_DIR / "resources" / "GSE122649"
_GSE122649_RAW_TAR = _GSE122649_DIR / "GSE122649_RAW.tar"
_GSE122649_SOFT_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE122nnn/GSE122649/"
    "soft/GSE122649_family.soft.gz"
)

RANDOM_STATE = 42
N_BOOTSTRAP = 2_000

_COMPARTMENTS: dict[str, tuple[str, ...]] = {
    "erythrocyte": (
        "ENSG00000206172", "ENSG00000188536", "ENSG00000244734",
        "ENSG00000158578", "ENSG00000170180", "ENSG00000133742",
        "ENSG00000159111", "ENSG00000105610", "ENSG00000179364",
        "ENSG00000223609",
    ),
    "platelet": (
        "ENSG00000163736", "ENSG00000163737", "ENSG00000185245",
    ),
    "endothelial": (
        "ENSG00000261371", "ENSG00000179776", "ENSG00000110799",
    ),
}

_N_COHORT = {"GPL16791": 636, "GSE76220": 20, "GSE122649": 38, "SRP064478": 15}
_TOTAL_N = sum(_N_COHORT.values())  # 709
# Equal weight per cohort: each independent dataset has equal vote on generalisability
_W = {k: 1.0 / len(_N_COHORT) for k in _N_COHORT}


# ---------------------------------------------------------------------------
# Fast helpers
# ---------------------------------------------------------------------------

def _zero_shot_fast(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    params: dict,
) -> float:
    """Train LightGBM on X_tr, score X_te, return AUC (no bootstrap)."""
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LGBMClassifier(**params).fit(X_tr, y_tr)
    return float(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))


def _bootstrap_auc(
    y: np.ndarray,
    scores: np.ndarray,
    n: int = N_BOOTSTRAP,
    seed: int = RANDOM_STATE,
) -> tuple[float, float]:
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], scores[idx]))
    if not aucs:
        raise RuntimeError(
            f"Bootstrap produced 0 valid resamples from {n} draws — "
            f"every draw had a single class. "
            f"y unique={np.unique(y).tolist()}, "
            f"counts={np.bincount(y.astype(int)).tolist()}"
        )
    a = np.array(aucs)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def _zero_shot_ci(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    params: dict,
) -> tuple[float, float, float]:
    """Zero-shot AUC with 95% bootstrap CI."""
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LGBMClassifier(**params).fit(X_tr, y_tr)
    scores = clf.predict_proba(X_te)[:, 1]
    auc = float(roc_auc_score(y_te, scores))
    lo, hi = _bootstrap_auc(y_te, scores)
    return auc, lo, hi


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_gpl24676_ctd() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load GPL24676 with log1p + CTD compartment regression.

    Returns:
        X_ctd: (874, 58929) CTD-corrected log1p, float32
        y: (874,) integer labels
        X_log: (874, 58929) plain log1p, float32  (for symbol-matched cohorts)
        feat: versioned Ensembl feature names
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression

    (ds,) = load_dataset("GSE153960", platform="GPL24676", resources_dir=ALS_DIR / "resources")
    X_raw = ds.X.values.astype(np.float32)
    y = ds.y.values.astype(int)
    feat = list(ds.X.columns)
    X_log = np.log1p(X_raw)

    base_ids = [n.split(".")[0] for n in feat]
    pcs = []
    for comp_bases in _COMPARTMENTS.values():
        base_set = set(comp_bases)
        idx = [i for i, b in enumerate(base_ids) if b in base_set]
        if not idx:
            continue
        pca = PCA(n_components=1, random_state=0)
        pcs.append(pca.fit_transform(X_log[:, idx]))
    Z = np.hstack(pcs)
    reg = LinearRegression().fit(Z, X_log)
    X_ctd = (X_log - reg.predict(Z)).astype(np.float32)
    return X_ctd, y, X_log, feat


def _load_gpl16791_ctd(
    X_train_log: np.ndarray,
    feat_train: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Load GPL16791 and apply CTD correction using PCA fitted on GPL24676 training.

    Returns:
        X16_ctd: (636, 58929) CTD-corrected log1p, float32
        y16: (636,) integer labels
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression

    (ds16,) = load_dataset("GSE153960", platform="GPL16791", resources_dir=ALS_DIR / "resources")
    X16_raw = ds16.X.values.astype(np.float32)
    y16 = ds16.y.values.astype(int)
    feat16 = list(ds16.X.columns)
    assert feat16 == feat_train, "GPL16791/GPL24676 feature lists differ"

    X16_log = np.log1p(X16_raw)
    base_ids = [n.split(".")[0] for n in feat16]
    pcs_tr, pcs_te = [], []
    for comp_bases in _COMPARTMENTS.values():
        base_set = set(comp_bases)
        idx = [i for i, b in enumerate(base_ids) if b in base_set]
        if not idx:
            continue
        pca = PCA(n_components=1, random_state=0)
        pcs_tr.append(pca.fit_transform(X_train_log[:, idx]))
        pcs_te.append(pca.transform(X16_log[:, idx]))
    Z_tr = np.hstack(pcs_tr)
    Z_te = np.hstack(pcs_te)
    reg = LinearRegression().fit(Z_tr, X_train_log)
    X16_ctd = (X16_log - reg.predict(Z_te)).astype(np.float32)
    return X16_ctd, y16


def _load_gse76220() -> tuple[np.ndarray, list[str], np.ndarray]:
    (ds,) = load_dataset("GSE76220", resources_dir=ALS_DIR / "resources")
    X_log = np.log1p(ds.X.values.astype(np.float32))
    return X_log, list(ds.X.columns), ds.y.values.astype(int)


def _parse_gse122649_soft() -> dict[str, str]:
    import requests
    r = requests.get(_GSE122649_SOFT_URL, timeout=60, stream=True)
    content = gzip.decompress(r.content).decode("utf-8", errors="ignore")
    meta: dict[str, str] = {}
    for s in re.split(r"\^SAMPLE", content)[1:]:
        acc_m = re.search(r"!Sample_geo_accession = (GSM\d+)", s)
        diag_m = re.search(r"!Sample_characteristics_ch1 = diagnosis: (.+)", s)
        if acc_m and diag_m:
            meta[acc_m.group(1)] = diag_m.group(1).strip()
    return meta


def _load_gse122649() -> tuple[np.ndarray, list[str], np.ndarray]:
    if not _GSE122649_RAW_TAR.exists():
        raise FileNotFoundError(f"GSE122649_RAW.tar not found at {_GSE122649_RAW_TAR}")
    gsm_to_diag = _parse_gse122649_soft()
    diag_map = {"sALS": 1, "sALS/FTD": 1, "Non-neurological control": 0}
    gene_symbols: list[str] = []
    sample_counts: dict[str, np.ndarray] = {}
    with tarfile.open(_GSE122649_RAW_TAR, "r") as tf:
        for member in tf.getmembers():
            gsm_m = re.search(r"(GSM\d+)", member.name)
            if not gsm_m or gsm_m.group(1) not in gsm_to_diag:
                continue
            gsm = gsm_m.group(1)
            f_obj = tf.extractfile(member)
            if f_obj is None:
                continue
            raw = f_obj.read()
            if member.name.endswith(".gz"):
                raw = gzip.decompress(raw)
            rows = []
            for line in raw.decode("utf-8", errors="ignore").splitlines()[1:]:
                line = line.strip()
                if not line or line.startswith("__"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                try:
                    rows.append((parts[0].strip('"').strip("'"), float(parts[1])))
                except ValueError:
                    continue
            if not rows:
                continue
            syms_here = [r[0] for r in rows]
            if not gene_symbols:
                gene_symbols = syms_here
            elif len(syms_here) != len(gene_symbols):
                continue
            sample_counts[gsm] = np.array([r[1] for r in rows], dtype=np.float32)
    gsm_ids = list(sample_counts.keys())
    X_raw = np.stack([sample_counts[g] for g in gsm_ids])
    y_full = np.array([diag_map.get(gsm_to_diag[g], -1) for g in gsm_ids])
    valid = y_full >= 0
    return np.log1p(X_raw[valid]), gene_symbols, y_full[valid]


def _load_srp064478() -> tuple[np.ndarray, list[str], np.ndarray]:
    import pandas as pd

    meta = pd.read_csv(_SRP_META, sep="\t").set_index("SRR")
    quants: list[pd.Series] = []
    tx_ids: list[str] | None = None
    y_list: list[int] = []
    for srr in meta.index:
        sf = _SRP_QUANT_DIR / srr / "quant.sf"
        if not sf.exists():
            raise FileNotFoundError(f"Missing Salmon output: {sf}")
        df = pd.read_csv(sf, sep="\t")
        df["tx_base"] = df["Name"].str.split(".").str[0]
        tx_counts = df.set_index("tx_base")["NumReads"]
        if tx_ids is None:
            tx_ids = list(tx_counts.index)
        quants.append(tx_counts)
        y_list.append(1 if meta.loc[srr, "condition"] == "ALS" else 0)

    ensg_direct = [t for t in (tx_ids or []) if t.startswith("ENSG")]
    if ensg_direct:
        X_raw = np.array(
            [q.reindex(tx_ids).fillna(0).values for q in quants], dtype=np.float32
        )
        return np.log1p(X_raw), tx_ids or [], np.array(y_list, dtype=int)

    fasta = ALS_DIR / "resources" / "SRP064478" / "index" / "hg38_cdna.fa.gz"
    tx2gene: dict[str, str] = {}
    with gzip.open(str(fasta), "rt") as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            m_enst = re.search(r"(ENST\d+)\.\d+", line)
            m_ensg = re.search(r"gene:(ENSG\d+)\.\d+", line)
            if m_enst and m_ensg:
                tx2gene[m_enst.group(1)] = m_ensg.group(1)

    tx_mat = pd.DataFrame(
        {srr: q.reindex(tx_ids).fillna(0) for srr, q in zip(meta.index, quants)}
    )
    tx_mat.index = tx_ids
    tx_mat["ensg"] = pd.Series(tx_ids).map(tx2gene).values
    gene_mat = tx_mat.dropna(subset=["ensg"]).groupby("ensg").sum()
    ensg_ids = list(gene_mat.index)
    X_raw = gene_mat.values.T.astype(np.float32)
    return np.log1p(X_raw), ensg_ids, np.array(y_list, dtype=int)


# ---------------------------------------------------------------------------
# Weighted mean AUC for a panel subset
# ---------------------------------------------------------------------------

def _wmean_auc(
    panel_idx: list[int],
    # GPL16791 (CTD) pre-extracted 25-column matrices
    Xtr25_ctd: np.ndarray,
    y_train: np.ndarray,
    X16_25_ctd: np.ndarray,
    y16: np.ndarray,
    # log1p training 25-column matrix (for symbol/Ensembl matched cohorts)
    Xtr25_log: np.ndarray,
    # GSE76220
    X76_25: np.ndarray,
    y76: np.ndarray,
    avail76: list[int],
    # GSE122649
    X122_25: np.ndarray,
    y122: np.ndarray,
    avail122: list[int],
    # SRP064478
    Xsrp_25: np.ndarray,
    y_srp: np.ndarray,
    avail_srp: list[int],
    params: dict,
) -> tuple[float, dict[str, float]]:
    """Return (weighted_mean_auc, per_cohort_aucs) for the given panel subset.

    panel_idx: indices into the 25-column pre-extracted matrices.
    avail*: sorted list of panel gene indices available in that cohort;
            X*_25[:, k] = test log1p for panel gene avail*[k].
    """
    from sklearn.preprocessing import StandardScaler

    panel_set = set(panel_idx)
    wmean = 0.0
    cohort_aucs: dict[str, float] = {}

    # ---- GPL16791 (all 25 genes available, CTD-corrected) ----
    Xtr = Xtr25_ctd[:, panel_idx]
    Xte = X16_25_ctd[:, panel_idx]
    sc = StandardScaler().fit(Xtr)
    auc16 = _zero_shot_fast(sc.transform(Xtr), y_train, sc.transform(Xte), y16, params)
    wmean += _W["GPL16791"] * auc16
    cohort_aucs["GPL16791"] = auc16

    # ---- GSE76220 (symbol-matched, log1p) ----
    avail76_set = set(avail76)
    cols76_tr = [i for i in panel_idx if i in avail76_set]
    cols76_te = [k for k, i in enumerate(avail76) if i in panel_set]
    if cols76_tr:
        Xtr = Xtr25_log[:, cols76_tr]
        Xte = X76_25[:, cols76_te]
        sc = StandardScaler().fit(Xtr)
        auc76 = _zero_shot_fast(sc.transform(Xtr), y_train, sc.transform(Xte), y76, params)
    else:
        auc76 = 0.5
    wmean += _W["GSE76220"] * auc76
    cohort_aucs["GSE76220"] = auc76

    # ---- GSE122649 (symbol-matched, log1p) ----
    avail122_set = set(avail122)
    cols122_tr = [i for i in panel_idx if i in avail122_set]
    cols122_te = [k for k, i in enumerate(avail122) if i in panel_set]
    if cols122_tr:
        Xtr = Xtr25_log[:, cols122_tr]
        Xte = X122_25[:, cols122_te]
        sc = StandardScaler().fit(Xtr)
        auc122 = _zero_shot_fast(sc.transform(Xtr), y_train, sc.transform(Xte), y122, params)
    else:
        auc122 = 0.5
    wmean += _W["GSE122649"] * auc122
    cohort_aucs["GSE122649"] = auc122

    # ---- SRP064478 (Ensembl base ID, log1p) ----
    avail_srp_set = set(avail_srp)
    cols_srp_tr = [i for i in panel_idx if i in avail_srp_set]
    cols_srp_te = [k for k, i in enumerate(avail_srp) if i in panel_set]
    if cols_srp_tr:
        Xtr = Xtr25_log[:, cols_srp_tr]
        Xte = Xsrp_25[:, cols_srp_te]
        sc = StandardScaler().fit(Xtr)
        auc_srp = _zero_shot_fast(sc.transform(Xtr), y_train, sc.transform(Xte), y_srp, params)
    else:
        auc_srp = 0.5
    wmean += _W["SRP064478"] * auc_srp
    cohort_aucs["SRP064478"] = auc_srp

    return wmean, cohort_aucs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import warnings as _w
    _w.filterwarnings("ignore", category=UserWarning)

    print("=" * 70)
    print("Iterative Panel Elimination — greedy backward, weighted zero-shot AUC")
    print(f"Weights: GPL16791={_W['GPL16791']:.3f}  GSE76220={_W['GSE76220']:.3f}  "
          f"GSE122649={_W['GSE122649']:.3f}  SRP064478={_W['SRP064478']:.3f}")
    print("=" * 70)

    params_raw = json.loads(_PARAMS_PATH.read_text())
    params = dict(params_raw, colsample_bytree=1.0, n_jobs=-1, verbose=-1)

    # -----------------------------------------------------------------------
    # Load panel
    # -----------------------------------------------------------------------
    df_panel = pd.read_csv(_PANEL_CSV)
    feat_col = next(c for c in df_panel.columns if "feature" in c.lower() or "ensg" in c.lower())
    sym_col = next(c for c in df_panel.columns if "symbol" in c.lower())
    feat25: list[str] = df_panel[feat_col].tolist()
    sym25: list[str] = df_panel[sym_col].tolist()
    feat25_bases: list[str] = [f.split(".")[0] for f in feat25]
    print(f"\n25-gene panel: {sym25}")

    # -----------------------------------------------------------------------
    # Load GPL24676 (once; provides both CTD and plain log1p variants)
    # -----------------------------------------------------------------------
    print("\nLoading GPL24676 (CTD + log1p) ...")
    X_train_ctd, y_train, X_train_log, feat_train = _load_gpl24676_ctd()
    feat_train_base_map: dict[str, int] = {f.split(".")[0]: j for j, f in enumerate(feat_train)}

    # Pre-extract 25-column matrices (CTD and log1p)
    panel_train_cols: list[int] = [feat_train_base_map[b] for b in feat25_bases]
    Xtr25_ctd: np.ndarray = X_train_ctd[:, panel_train_cols].astype(np.float32)
    Xtr25_log: np.ndarray = X_train_log[:, panel_train_cols].astype(np.float32)

    # -----------------------------------------------------------------------
    # Load GPL16791 with CTD
    # -----------------------------------------------------------------------
    print("Loading GPL16791 (CTD) ...")
    X16_ctd, y16 = _load_gpl16791_ctd(X_train_log, feat_train)
    X16_25_ctd: np.ndarray = X16_ctd[:, panel_train_cols].astype(np.float32)
    print(f"  GPL16791: n={len(y16)}, ALS={y16.sum()}, Ctrl={(y16==0).sum()}")

    # -----------------------------------------------------------------------
    # Load GSE76220 (symbol)
    # -----------------------------------------------------------------------
    print("Loading GSE76220 ...")
    X76_log, vocab76, y76 = _load_gse76220()
    vocab76_map = {s: i for i, s in enumerate(vocab76)}
    avail76: list[int] = [i for i, s in enumerate(sym25) if s in vocab76_map]
    if avail76:
        X76_25 = np.column_stack(
            [X76_log[:, vocab76_map[sym25[i]]] for i in avail76]
        ).astype(np.float32)
    else:
        X76_25 = np.empty((len(y76), 0), dtype=np.float32)
    print(f"  GSE76220: n={len(y76)}, {len(avail76)}/25 panel genes matched "
          f"({[sym25[i] for i in avail76]})")

    # -----------------------------------------------------------------------
    # Load GSE122649 (symbol)
    # -----------------------------------------------------------------------
    print("Loading GSE122649 ...")
    X122_log, vocab122, y122 = _load_gse122649()
    vocab122_map = {s: i for i, s in enumerate(vocab122)}
    avail122: list[int] = [i for i, s in enumerate(sym25) if s in vocab122_map]
    if avail122:
        X122_25 = np.column_stack(
            [X122_log[:, vocab122_map[sym25[i]]] for i in avail122]
        ).astype(np.float32)
    else:
        X122_25 = np.empty((len(y122), 0), dtype=np.float32)
    print(f"  GSE122649: n={len(y122)}, {len(avail122)}/25 panel genes matched "
          f"({[sym25[i] for i in avail122]})")

    # -----------------------------------------------------------------------
    # Load SRP064478 (Ensembl base ID)
    # -----------------------------------------------------------------------
    print("Loading SRP064478 ...")
    Xsrp_log, vocab_srp, y_srp = _load_srp064478()
    srp_vocab_map = {v: i for i, v in enumerate(vocab_srp)}
    avail_srp: list[int] = [i for i, b in enumerate(feat25_bases) if b in srp_vocab_map]
    if avail_srp:
        Xsrp_25 = np.column_stack(
            [Xsrp_log[:, srp_vocab_map[feat25_bases[i]]] for i in avail_srp]
        ).astype(np.float32)
    else:
        Xsrp_25 = np.empty((len(y_srp), 0), dtype=np.float32)
    print(f"  SRP064478: n={len(y_srp)}, {len(avail_srp)}/25 panel genes matched "
          f"({[sym25[i] for i in avail_srp]})")

    # -----------------------------------------------------------------------
    # Shared kwargs for _wmean_auc
    # -----------------------------------------------------------------------
    _shared = dict(
        Xtr25_ctd=Xtr25_ctd, y_train=y_train,
        X16_25_ctd=X16_25_ctd, y16=y16,
        Xtr25_log=Xtr25_log,
        X76_25=X76_25, y76=y76, avail76=avail76,
        X122_25=X122_25, y122=y122, avail122=avail122,
        Xsrp_25=Xsrp_25, y_srp=y_srp, avail_srp=avail_srp,
        params=params,
    )

    # -----------------------------------------------------------------------
    # Baseline: all 25 genes
    # -----------------------------------------------------------------------
    print("\nComputing baseline (25-gene panel) ...")
    active: list[int] = list(range(25))
    baseline_wmean, baseline_cohort = _wmean_auc(active, **_shared)
    print(f"  Baseline W.mean AUC = {baseline_wmean:.4f}")
    for cname, auc in baseline_cohort.items():
        print(f"    {cname}: {auc:.4f}  (w={_W[cname]:.3f})")

    # -----------------------------------------------------------------------
    # Greedy backward elimination
    # -----------------------------------------------------------------------
    history: list[dict] = []  # one entry per dropped gene

    # Record the starting point
    history.append({
        "step": 0,
        "panel_size": 25,
        "dropped": None,
        "D_ZS": None,
        "wmean": baseline_wmean,
        "cohort": dict(baseline_cohort),
    })

    current_wmean = baseline_wmean
    step = 0

    while len(active) > 1:
        step += 1
        n_remaining = len(active)
        print(f"\nStep {step}: evaluating {n_remaining} candidates for elimination ...")

        best_d = -np.inf
        best_g = -1
        best_wmean = current_wmean
        best_cohort: dict[str, float] = {}

        for g in active:
            trial = [i for i in active if i != g]
            wmean_trial, cohort_trial = _wmean_auc(trial, **_shared)
            d = wmean_trial - current_wmean  # positive = improvement from dropping g
            print(f"    drop {sym25[g]:>14s}: trial W.mean={wmean_trial:.4f}  "
                  f"D_ZS={d:+.4f}")
            if d > best_d:
                best_d = d
                best_g = g
                best_wmean = wmean_trial
                best_cohort = cohort_trial

        print(f"  Best candidate: {sym25[best_g]}  D_ZS={best_d:+.4f}")

        if best_d <= 0.0:
            print(f"  D_ZS ≤ 0 (critical threshold crossed at k={len(active)})")

        # Drop the best candidate (always continue to exhaustion)
        active.remove(best_g)
        current_wmean = best_wmean
        history.append({
            "step": step,
            "panel_size": len(active),
            "dropped": sym25[best_g],
            "D_ZS": best_d,
            "wmean": current_wmean,
            "cohort": dict(best_cohort),
        })
        print(f"  Dropped: {sym25[best_g]}  → panel now {len(active)} genes  "
              f"W.mean={current_wmean:.4f}")
        for cname, auc in best_cohort.items():
            print(f"    {cname}: {auc:.4f}")

    # -----------------------------------------------------------------------
    # Final panel
    # -----------------------------------------------------------------------
    final_genes = [sym25[i] for i in active]
    final_feats = [feat25[i] for i in active]
    print(f"\n{'='*70}")
    print(f"Final minimal panel: {len(active)} genes")
    print(f"  Symbols: {', '.join(final_genes)}")
    print(f"  ENSGs:   {', '.join([feat25[i] for i in active])}")
    print(f"  W.mean zero-shot AUC = {current_wmean:.4f}")

    # Full evaluation with CI for final panel
    print("\nComputing per-cohort zero-shot AUC with 95% CI for final panel ...")
    from sklearn.preprocessing import StandardScaler

    final_results: dict[str, tuple[float, float, float]] = {}
    panel_set = set(active)

    # GPL16791
    Xtr = Xtr25_ctd[:, active]
    Xte = X16_25_ctd[:, active]
    sc = StandardScaler().fit(Xtr)
    auc16, lo16, hi16 = _zero_shot_ci(sc.transform(Xtr), y_train, sc.transform(Xte), y16, params)
    final_results["GPL16791"] = (auc16, lo16, hi16)

    # GSE76220
    cols76_tr = [i for i in active if i in set(avail76)]
    cols76_te = [k for k, i in enumerate(avail76) if i in panel_set]
    if cols76_tr:
        Xtr = Xtr25_log[:, cols76_tr]
        Xte = X76_25[:, cols76_te]
        sc = StandardScaler().fit(Xtr)
        auc76, lo76, hi76 = _zero_shot_ci(sc.transform(Xtr), y_train, sc.transform(Xte), y76, params)
    else:
        auc76, lo76, hi76 = 0.5, 0.5, 0.5
    final_results["GSE76220"] = (auc76, lo76, hi76)

    # GSE122649
    cols122_tr = [i for i in active if i in set(avail122)]
    cols122_te = [k for k, i in enumerate(avail122) if i in panel_set]
    if cols122_tr:
        Xtr = Xtr25_log[:, cols122_tr]
        Xte = X122_25[:, cols122_te]
        sc = StandardScaler().fit(Xtr)
        auc122, lo122, hi122 = _zero_shot_ci(sc.transform(Xtr), y_train, sc.transform(Xte), y122, params)
    else:
        auc122, lo122, hi122 = 0.5, 0.5, 0.5
    final_results["GSE122649"] = (auc122, lo122, hi122)

    # SRP064478
    cols_srp_tr = [i for i in active if i in set(avail_srp)]
    cols_srp_te = [k for k, i in enumerate(avail_srp) if i in panel_set]
    if cols_srp_tr:
        Xtr = Xtr25_log[:, cols_srp_tr]
        Xte = Xsrp_25[:, cols_srp_te]
        sc = StandardScaler().fit(Xtr)
        auc_srp, lo_srp, hi_srp = _zero_shot_ci(sc.transform(Xtr), y_train, sc.transform(Xte), y_srp, params)
    else:
        auc_srp, lo_srp, hi_srp = 0.5, 0.5, 0.5
    final_results["SRP064478"] = (auc_srp, lo_srp, hi_srp)

    # -----------------------------------------------------------------------
    # Text output
    # -----------------------------------------------------------------------
    lines = [
        "Iterative Panel Elimination — greedy backward, weighted zero-shot AUC",
        "=" * 70,
        "",
        "Cohort weights:",
        f"  GPL16791  n={_N_COHORT['GPL16791']}  w={_W['GPL16791']:.4f}",
        f"  GSE76220  n={_N_COHORT['GSE76220']}   w={_W['GSE76220']:.4f}",
        f"  GSE122649 n={_N_COHORT['GSE122649']}   w={_W['GSE122649']:.4f}",
        f"  SRP064478 n={_N_COHORT['SRP064478']}   w={_W['SRP064478']:.4f}",
        f"  Total N   = {_TOTAL_N}",
        "",
        "Panel gene availability per cohort:",
        f"  GPL16791  : 25/25 (Ensembl base ID, all panel genes)",
        f"  GSE76220  : {len(avail76)}/25 symbol-matched: {[sym25[i] for i in avail76]}",
        f"  GSE122649 : {len(avail122)}/25 symbol-matched: {[sym25[i] for i in avail122]}",
        f"  SRP064478 : {len(avail_srp)}/25 Ensembl-matched: {[sym25[i] for i in avail_srp]}",
        "",
        "Baseline (25-gene panel):",
        f"  W.mean AUC = {baseline_wmean:.4f}",
        *[f"  {k}: {v:.4f}" for k, v in baseline_cohort.items()],
        "",
        "Elimination log:",
        f"  {'Step':>4}  {'Dropped':>14}  {'D_ZS':>8}  {'Panel_size':>10}  {'W.mean':>8}  "
        + "  ".join(f"{k:>10}" for k in _N_COHORT),
        "-" * 100,
    ]
    for h in history[1:]:  # skip step 0 (baseline)
        cohort_str = "  ".join(f"{h['cohort'].get(k, float('nan')):>10.4f}" for k in _N_COHORT)
        lines.append(
            f"  {h['step']:>4}  {h['dropped']:>14}  {h['D_ZS']:>+8.4f}  "
            f"{h['panel_size']:>10}  {h['wmean']:>8.4f}  {cohort_str}"
        )

    lines += [
        "",
        f"Final minimal panel: {len(active)} genes",
        f"  Symbols: {', '.join(final_genes)}",
        f"  ENSGs  : {', '.join(final_feats)}",
        f"  W.mean zero-shot AUC = {current_wmean:.4f}",
        "",
        "Per-cohort zero-shot AUC for final panel (95% CI):",
        f"  {'Cohort':<12} {'n':>5}  {'Genes':>6}  {'AUC':>6}  95% CI",
    ]
    n_final_76 = len(cols76_tr)
    n_final_122 = len(cols122_tr)
    n_final_srp = len(cols_srp_tr)
    lines += [
        f"  {'GPL16791':<12} {len(y16):>5}  {len(active):>6}  {auc16:.4f}  [{lo16:.3f}, {hi16:.3f}]",
        f"  {'GSE76220':<12} {len(y76):>5}  {n_final_76:>6}  {auc76:.4f}  [{lo76:.3f}, {hi76:.3f}]",
        f"  {'GSE122649':<12} {len(y122):>5}  {n_final_122:>6}  {auc122:.4f}  [{lo122:.3f}, {hi122:.3f}]",
        f"  {'SRP064478':<12} {len(y_srp):>5}  {n_final_srp:>6}  {auc_srp:.4f}  [{lo_srp:.3f}, {hi_srp:.3f}]",
    ]

    stat_out = SCRIPT_DIR / "iterative_panel_elimination_statistics.txt"
    stat_out.write_text("\n".join(lines) + "\n")
    print(f"\nSaved: {stat_out}")
    print("\n" + "\n".join(lines))

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    steps_recorded = [h for h in history if h["dropped"] is not None]
    panel_sizes = [h["panel_size"] for h in history]
    wmeans = [h["wmean"] for h in history]
    dropped_names = [h["dropped"] for h in steps_recorded]
    d_values = [h["D_ZS"] for h in steps_recorded]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: W.mean AUC vs panel size
    ax = axes[0]
    ax.plot(panel_sizes, wmeans, "o-", color="#1565C0", lw=2, ms=6)
    ax.axvline(len(active), color="red", ls="--", lw=1.2, label=f"Final panel (k={len(active)})")
    ax.set_xlabel("Panel size (k genes)")
    ax.set_ylabel("Weighted mean zero-shot AUC")
    ax.set_title("Panel size vs. weighted zero-shot AUC")
    ax.set_xticks(range(len(active) - 1, 26))
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # Right: D_ZS per dropped gene (in order of elimination)
    ax = axes[1]
    colors_bar = ["#388E3C" if d > 0 else "#C62828" for d in d_values]
    bars = ax.barh(range(len(dropped_names)), d_values, color=colors_bar, alpha=0.85)
    ax.set_yticks(range(len(dropped_names)))
    ax.set_yticklabels(dropped_names, fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("D_ZS (W.mean AUC improvement from dropping gene)")
    ax.set_title("Genes dropped (order of elimination)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    plt.suptitle(
        f"Greedy backward elimination — 25→{len(active)}-gene minimal panel\n"
        f"W.mean AUC: {baseline_wmean:.4f} → {current_wmean:.4f}  "
        f"(GPL16791×{_W['GPL16791']:.2f} + GSE76220×{_W['GSE76220']:.2f} + "
        f"GSE122649×{_W['GSE122649']:.2f} + SRP064478×{_W['SRP064478']:.2f})",
        fontsize=9,
    )
    plt.tight_layout()
    png_out = SCRIPT_DIR / "iterative_panel_elimination.png"
    plt.savefig(png_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_out}")


if __name__ == "__main__":
    main()
