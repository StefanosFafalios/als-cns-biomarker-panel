# ruff: noqa: E402
"""LR-L2 vs LGBM zero-shot comparison — 25-gene and 15-gene critical panels.

Both models are trained on GPL24676 (n=874) and evaluated zero-shot on all
external CNS cohorts plus the blood negative control (GSE234297).

Identical pipeline for LGBM and LR-L2:
  1. Fit StandardScaler on GPL24676 training columns (matched per cohort).
  2. Apply _safe_scale (NaN → 0) to train and test.
  3. Fit LGBM / LR-L2 on scaled training data.
  4. Predict on scaled test data; compute AUC + bootstrap 95% CI.

Per-cohort preprocessing:
  GPL16791  : CTD compartment regression (GPL24676-fitted) then StandardScaler
  GSE76220  : log1p only, HGNC symbol matching
  GSE122649 : log1p only, HGNC symbol matching
  SRP064478 : log1p only, Ensembl base-ID matching (Salmon)
  GSE234297 : log1p only, Entrez-ID matching (blood negative control)

Panels evaluated:
  25-gene   : full core panel (lgbm_core25_panel.csv)
  15-crit   : 15 cross-cohort critical genes (greedy backward elimination peak,
              iterative_panel_elimination.py; W.mean = 0.8921 at k=15)
              indices in 25-gene panel: [5,6,7,8,9,10,11,12,15,17,18,19,20,22,23,24]

Run from the coffeeBreak project root:
    conda run -n coffeeBreak python -u als_analysis/GSE153960/lr_vs_lgbm_zeroshot.py
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

import pandas as pd  # noqa: E402

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"
_SRP_QUANT_DIR = ALS_DIR / "resources" / "SRP064478" / "quant"
_SRP_META = ALS_DIR / "resources" / "SRP064478" / "srr_metadata.tsv"
_SRP_FASTA = ALS_DIR / "resources" / "SRP064478" / "index" / "hg38_cdna.fa.gz"
_GSE122649_RAW_TAR = ALS_DIR / "resources" / "GSE122649" / "GSE122649_RAW.tar"
_GSE122649_SOFT_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE122nnn/GSE122649/"
    "soft/GSE122649_family.soft.gz"
)
_GSE234297_COUNTS = (
    ALS_DIR / "resources" / "GSE234297" / "suppl" / "GSE234297_gene_raw_counts.txt.gz"
)

RANDOM_STATE = 42
N_BOOTSTRAP = 2_000

# 15-gene critical panel: indices in the 25-gene panel
# (greedy backward elimination peak at k=15, W.mean = 0.8921; equal cohort weights)
# HERC2P8 SMG1P5 SERTAD1 FCN3 PROS1 ANGPT2 SNORD97 TINAGL1 CKMT2 CLDN5
# NR4A1 SOHLH2 HEXB MCEE SLC37A2
CRITICAL_IDX: list[int] = [1, 2, 3, 5, 6, 7, 10, 11, 12, 15, 16, 18, 20, 23, 24]

# Entrez IDs for protein-coding panel genes — used for GSE234297 (Entrez-keyed counts)
_PC_ENTREZ: dict[str, int] = {
    "MECOM": 2122, "SERTAD1": 29950, "FCN3": 8547, "PROS1": 5627,
    "ANGPT2": 285, "EMP1": 2012, "TINAGL1": 64129, "CKMT2": 1160,
    "VWF": 7450, "CLDN5": 7122, "NR4A1": 3164, "SOHLH2": 54937,
    "HEXB": 3074, "MCEE": 84693, "SLC37A2": 219855,
}

_COMPARTMENTS: dict[str, tuple[str, ...]] = {
    "erythrocyte": (
        "ENSG00000206172", "ENSG00000188536", "ENSG00000244734",
        "ENSG00000158578", "ENSG00000170180", "ENSG00000133742",
        "ENSG00000159111", "ENSG00000105610", "ENSG00000179364",
        "ENSG00000223609",
    ),
    "platelet": ("ENSG00000163736", "ENSG00000163737", "ENSG00000185245"),
    "endothelial": ("ENSG00000261371", "ENSG00000179776", "ENSG00000110799"),
}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_gpl24676_ctd() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """GPL24676: CTD regression + log1p. Returns (X_ctd, y, X_log, feat)."""
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
    X_train_log: np.ndarray, feat_train: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """GPL16791 with CTD regression fitted on GPL24676."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    (ds,) = load_dataset("GSE153960", platform="GPL16791", resources_dir=ALS_DIR / "resources")
    X16_raw = ds.X.values.astype(np.float32)
    y16 = ds.y.values.astype(int)
    feat16 = list(ds.X.columns)
    assert feat16 == feat_train
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
    return np.log1p(ds.X.values.astype(np.float32)), list(ds.X.columns), ds.y.values.astype(int)


def _load_gse122649() -> tuple[np.ndarray, list[str], np.ndarray]:
    import requests
    r = requests.get(_GSE122649_SOFT_URL, timeout=60, stream=True)
    content = gzip.decompress(r.content).decode("utf-8", errors="ignore")
    meta: dict[str, str] = {}
    for s in re.split(r"\^SAMPLE", content)[1:]:
        acc_m = re.search(r"!Sample_geo_accession = (GSM\d+)", s)
        diag_m = re.search(r"!Sample_characteristics_ch1 = diagnosis: (.+)", s)
        if acc_m and diag_m:
            meta[acc_m.group(1)] = diag_m.group(1).strip()
    diag_map = {"sALS": 1, "sALS/FTD": 1, "Non-neurological control": 0}
    gene_symbols: list[str] = []
    sample_counts: dict[str, np.ndarray] = {}
    with tarfile.open(_GSE122649_RAW_TAR, "r") as tf:
        for member in tf.getmembers():
            gsm_m = re.search(r"(GSM\d+)", member.name)
            if not gsm_m or gsm_m.group(1) not in meta:
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
    y_full = np.array([diag_map.get(meta[g], -1) for g in gsm_ids])
    valid = y_full >= 0
    return np.log1p(X_raw[valid]), gene_symbols, y_full[valid]


def _load_srp064478() -> tuple[np.ndarray, list[str], np.ndarray]:
    meta_df = pd.read_csv(_SRP_META, sep="\t").set_index("SRR")
    tx2gene: dict[str, str] = {}
    with gzip.open(str(_SRP_FASTA), "rt") as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            m_enst = re.search(r"(ENST\d+)", line)
            m_ensg = re.search(r"gene:(ENSG\d+)", line)
            if m_enst and m_ensg:
                tx2gene[m_enst.group(1)] = m_ensg.group(1)
    gene_counts: dict[str, dict[str, float]] = {}
    y_list: list[int] = []
    for srr in meta_df.index:
        sf = _SRP_QUANT_DIR / srr / "quant.sf"
        df = pd.read_csv(sf, sep="\t")
        df["gene_base"] = df["Name"].str.split(".").str[0].map(tx2gene)
        df = df.dropna(subset=["gene_base"])
        sums = df.groupby("gene_base")["NumReads"].sum()
        gene_counts[srr] = sums.to_dict()
        y_list.append(1 if meta_df.loc[srr, "condition"] == "ALS" else 0)
    all_genes = sorted({g for d in gene_counts.values() for g in d})
    X_raw = np.array(
        [[gene_counts[srr].get(g, 0.0) for g in all_genes] for srr in meta_df.index],
        dtype=np.float32,
    )
    return np.log1p(X_raw), all_genes, np.array(y_list, dtype=int)


def _load_gse234297() -> tuple[np.ndarray, list[str], np.ndarray]:
    df = pd.read_csv(_GSE234297_COUNTS, sep="\t", index_col=0)
    y = np.array([1 if c.startswith("Case") else 0 for c in df.columns], dtype=int)
    entrez_ids = [str(int(x)) for x in df.index]
    return np.log1p(df.values.T.astype(np.float32)), entrez_ids, y


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _bootstrap_auc(
    y: np.ndarray, scores: np.ndarray, n: int = N_BOOTSTRAP, seed: int = RANDOM_STATE,
) -> tuple[float, float]:
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], scores[idx]))
    a = np.array(aucs)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def _safe_scale(sc, X: np.ndarray) -> np.ndarray:
    """Scale and replace NaN (zero-variance features) with 0."""
    return np.nan_to_num(sc.transform(X), nan=0.0)


def _score_both(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    params: dict,
) -> dict[str, tuple[float, float, float]]:
    """Fit StandardScaler on X_tr; train LGBM and LR-L2; return AUC+CI for each."""
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(X_tr)
    X_tr_sc = _safe_scale(sc, X_tr)
    X_te_sc = _safe_scale(sc, X_te)

    results: dict[str, tuple[float, float, float]] = {}
    for name, clf in [
        ("LGBM", LGBMClassifier(**params)),
        ("LR_L2", LogisticRegressionCV(
            cv=5, penalty="l2", solver="liblinear",
            scoring="roc_auc", max_iter=5000, random_state=RANDOM_STATE,
        )),
    ]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr_sc, y_tr)
        scores = clf.predict_proba(X_te_sc)[:, 1]
        auc = float(roc_auc_score(y_te, scores))
        lo, hi = _bootstrap_auc(y_te, scores)
        results[name] = (auc, lo, hi)

    return results


def _eval_cohort(
    cohort: str, tissue: str, n_total: int, panel_label: str, n_genes: int,
    Xtr: np.ndarray, y_tr: np.ndarray, Xte: np.ndarray, y_te: np.ndarray,
    params: dict, rows: list[dict],
) -> None:
    """Evaluate both models on one cohort + panel; append rows and print."""
    res = _score_both(Xtr, y_tr, Xte, y_te, params)
    for model, (auc, lo, hi) in res.items():
        print(f"    {model:6s}  AUC={auc:.4f}  95%CI [{lo:.4f}, {hi:.4f}]")
        rows.append({
            "cohort": cohort, "tissue": tissue, "n": n_total,
            "panel": panel_label, "n_genes": n_genes,
            "model": model, "AUC": round(auc, 4),
            "CI_lo": round(lo, 4), "CI_hi": round(hi, 4),
        })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    warnings.filterwarnings("ignore")

    print("=" * 72)
    print("LR-L2 vs LGBM — zero-shot, 25-gene and 15-gene critical panels")
    print("Train: GPL24676 (n=874, 684 ALS, 190 Control)")
    print("Identical preprocessing + StandardScaler for both models")
    print("=" * 72)

    params_raw = json.loads(_PARAMS_PATH.read_text())
    params = dict(params_raw, colsample_bytree=1.0, n_jobs=-1, verbose=-1)

    df_panel = pd.read_csv(_PANEL_CSV)
    feat_col = next(c for c in df_panel.columns if "feature" in c.lower() or "ensg" in c.lower())
    sym_col = next(c for c in df_panel.columns if "symbol" in c.lower())
    feat25: list[str] = df_panel[feat_col].tolist()
    sym25: list[str] = df_panel[sym_col].tolist()
    feat25_bases: list[str] = [f.split(".")[0] for f in feat25]
    sym_crit: list[str] = [sym25[i] for i in CRITICAL_IDX]
    base_crit: list[str] = [feat25_bases[i] for i in CRITICAL_IDX]

    # ---- Load GPL24676 once ----
    print("\nLoading GPL24676 ...")
    X_train_ctd, y_train, X_train_log, feat_train = _load_gpl24676_ctd()
    feat_train_base_map = {f.split(".")[0]: j for j, f in enumerate(feat_train)}
    panel_cols = [feat_train_base_map[b] for b in feat25_bases]
    crit_cols = [panel_cols[i] for i in CRITICAL_IDX]
    Xtr_ctd_25 = X_train_ctd[:, panel_cols].astype(np.float32)
    Xtr_ctd_16 = X_train_ctd[:, crit_cols].astype(np.float32)
    Xtr_log_25 = X_train_log[:, panel_cols].astype(np.float32)
    Xtr_log_16 = X_train_log[:, crit_cols].astype(np.float32)
    print(f"  n={len(y_train)}  ALS={y_train.sum()}  Ctrl={(y_train==0).sum()}")

    rows: list[dict] = []

    # ================================================================
    # GPL16791 — CTD preprocessing
    # ================================================================
    print("\nGPL16791 (CTD, n=636) ...")
    X16_ctd, y16 = _load_gpl16791_ctd(X_train_log, feat_train)
    Xte_16_25 = X16_ctd[:, panel_cols].astype(np.float32)
    Xte_16_16 = X16_ctd[:, crit_cols].astype(np.float32)
    print(f"  25-gene (25/25):")
    _eval_cohort("GPL16791", "CNS multi-region", len(y16), "25-gene", 25,
                 Xtr_ctd_25, y_train, Xte_16_25, y16, params, rows)
    print(f"  15-crit (15/15):")
    _eval_cohort("GPL16791", "CNS multi-region", len(y16), "15-crit", 15,
                 Xtr_ctd_16, y_train, Xte_16_16, y16, params, rows)

    # ================================================================
    # GSE76220 — log1p, HGNC symbol
    # ================================================================
    print("\nGSE76220 (log1p, n=20) ...")
    X76_log, vocab76, y76 = _load_gse76220()
    v76 = {s: i for i, s in enumerate(vocab76)}

    m76_25 = [s for s in sym25 if s in v76]
    Xtr76_25 = Xtr_log_25[:, [i for i, s in enumerate(sym25) if s in v76]]
    Xte76_25 = X76_log[:, [v76[s] for s in m76_25]]
    print(f"  25-gene ({len(m76_25)}/25):")
    _eval_cohort("GSE76220", "Lumbar SC (LCM)", len(y76), "25-gene", len(m76_25),
                 Xtr76_25, y_train, Xte76_25, y76, params, rows)

    m76_16 = [s for s in sym_crit if s in v76]
    Xtr76_16 = Xtr_log_16[:, [i for i, s in enumerate(sym_crit) if s in v76]]
    Xte76_16 = X76_log[:, [v76[s] for s in m76_16]]
    print(f"  15-crit ({len(m76_16)}/15):")
    _eval_cohort("GSE76220", "Lumbar SC (LCM)", len(y76), "15-crit", len(m76_16),
                 Xtr76_16, y_train, Xte76_16, y76, params, rows)

    # ================================================================
    # GSE122649 — log1p, HGNC symbol
    # ================================================================
    print("\nGSE122649 (log1p, n=38) ...")
    X122_log, vocab122, y122 = _load_gse122649()
    v122 = {s: i for i, s in enumerate(vocab122)}

    m122_25 = [s for s in sym25 if s in v122]
    Xtr122_25 = Xtr_log_25[:, [i for i, s in enumerate(sym25) if s in v122]]
    Xte122_25 = X122_log[:, [v122[s] for s in m122_25]]
    print(f"  25-gene ({len(m122_25)}/25):")
    _eval_cohort("GSE122649", "Motor cortex", len(y122), "25-gene", len(m122_25),
                 Xtr122_25, y_train, Xte122_25, y122, params, rows)

    m122_16 = [s for s in sym_crit if s in v122]
    Xtr122_16 = Xtr_log_16[:, [i for i, s in enumerate(sym_crit) if s in v122]]
    Xte122_16 = X122_log[:, [v122[s] for s in m122_16]]
    print(f"  15-crit ({len(m122_16)}/15):")
    _eval_cohort("GSE122649", "Motor cortex", len(y122), "15-crit", len(m122_16),
                 Xtr122_16, y_train, Xte122_16, y122, params, rows)

    # ================================================================
    # SRP064478 — log1p, Ensembl base ID (Salmon)
    # ================================================================
    print("\nSRP064478 (log1p Salmon, n=15) ...")
    Xsrp_log, vocab_srp, y_srp = _load_srp064478()
    v_srp = {v: i for i, v in enumerate(vocab_srp)}

    m_srp_25_idx = [i for i, b in enumerate(feat25_bases) if b in v_srp]
    Xtr_srp_25 = Xtr_log_25[:, m_srp_25_idx]
    Xte_srp_25 = Xsrp_log[:, [v_srp[feat25_bases[i]] for i in m_srp_25_idx]]
    print(f"  25-gene ({len(m_srp_25_idx)}/25):")
    _eval_cohort("SRP064478", "Cervical SC", len(y_srp), "25-gene", len(m_srp_25_idx),
                 Xtr_srp_25, y_train, Xte_srp_25, y_srp, params, rows)

    m_srp_16_idx = [i for i, b in enumerate(base_crit) if b in v_srp]
    Xtr_srp_16 = Xtr_log_16[:, m_srp_16_idx]
    Xte_srp_16 = Xsrp_log[:, [v_srp[base_crit[i]] for i in m_srp_16_idx]]
    print(f"  15-crit ({len(m_srp_16_idx)}/15):")
    _eval_cohort("SRP064478", "Cervical SC", len(y_srp), "15-crit", len(m_srp_16_idx),
                 Xtr_srp_16, y_train, Xte_srp_16, y_srp, params, rows)

    # ================================================================
    # GSE234297 — log1p, Entrez ID (blood negative control)
    # ================================================================
    print("\nGSE234297 (blood, Entrez ID, n=144) ...")
    Xblood_log, entrez_ids, y_blood = _load_gse234297()
    entrez_to_sym = {str(v): k for k, v in _PC_ENTREZ.items()}
    v_blood = {e: i for i, e in enumerate(entrez_ids)}
    sym25_map = {s: i for i, s in enumerate(sym25)}

    matched_e = [e for e in entrez_to_sym if e in v_blood and entrez_to_sym[e] in sym25_map]
    Xtr_blood_25 = Xtr_log_25[:, [sym25_map[entrez_to_sym[e]] for e in matched_e]]
    Xte_blood_25 = Xblood_log[:, [v_blood[e] for e in matched_e]]
    print(f"  25-gene ({len(matched_e)}/25 protein-coding):")
    _eval_cohort("GSE234297", "Blood", len(y_blood), "25-gene", len(matched_e),
                 Xtr_blood_25, y_train, Xte_blood_25, y_blood, params, rows)

    sym_crit_map = {s: i for i, s in enumerate(sym_crit)}
    matched_e16 = [e for e in entrez_to_sym
                   if e in v_blood and entrez_to_sym[e] in sym_crit_map]
    Xtr_blood_16 = Xtr_log_16[:, [sym_crit_map[entrez_to_sym[e]] for e in matched_e16]]
    Xte_blood_16 = Xblood_log[:, [v_blood[e] for e in matched_e16]]
    print(f"  15-crit ({len(matched_e16)}/15 protein-coding):")
    _eval_cohort("GSE234297", "Blood", len(y_blood), "15-crit", len(matched_e16),
                 Xtr_blood_16, y_train, Xte_blood_16, y_blood, params, rows)

    # ================================================================
    # Output
    # ================================================================
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    df_out = pd.DataFrame(rows)
    print(df_out.to_string(index=False))

    out_csv = SCRIPT_DIR / "lr_vs_lgbm_zeroshot.csv"
    out_txt = SCRIPT_DIR / "lr_vs_lgbm_zeroshot.txt"
    df_out.to_csv(out_csv, index=False)

    header = [
        "LR-L2 vs LGBM — zero-shot, 25-gene and 15-gene critical panels",
        "=" * 72,
        "Train  : GPL24676 (n=874, 684 ALS, 190 Control)",
        "LGBM   : lgbm_top500_best_params.json, colsample_bytree=1.0",
        "LR-L2  : LogisticRegressionCV(cv=5, penalty='l2', solver='liblinear')",
        "Scaler : StandardScaler fit on GPL24676 training columns (same for both models)",
        "Prepro : CTD regression for GPL16791; log1p for GSE76220/GSE122649/SRP064478/GSE234297",
        "",
    ]
    lines: list[str] = header[:]
    for cohort in ["GPL16791", "GSE76220", "GSE122649", "SRP064478", "GSE234297"]:
        cohort_rows = [r for r in rows if r["cohort"] == cohort]
        if not cohort_rows:
            continue
        tissue = cohort_rows[0]["tissue"]
        n = cohort_rows[0]["n"]
        lines.append(f"\n{cohort} ({tissue}, n={n})")
        for panel in ["25-gene", "15-crit"]:
            panel_rows = [r for r in cohort_rows if r["panel"] == panel]
            if not panel_rows:
                continue
            ng = panel_rows[0]["n_genes"]
            lines.append(f"  {panel} ({ng} genes matched):")
            for r in panel_rows:
                lines.append(
                    f"    {r['model']:6s}  AUC={r['AUC']:.4f}"
                    f"  95%CI [{r['CI_lo']:.4f}, {r['CI_hi']:.4f}]"
                )

    out_txt.write_text("\n".join(lines))
    print(f"\nSaved → {out_csv.name}, {out_txt.name}")


if __name__ == "__main__":
    main()
