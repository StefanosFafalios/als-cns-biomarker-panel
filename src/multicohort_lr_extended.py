# ruff: noqa: E402
"""Extend LR-L2 vs LGBM baseline to GSE122649, GSE234297, and SRP064478.

Uses the same loading infrastructure as the existing validation scripts.
Appends results to multicohort_baselines.{csv,txt}.

Run from the repository root:
    conda run -n als-cns-panel python -u src/multicohort_lr_extended.py
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

from utils import load_dataset

import pandas as pd

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"

RANDOM_STATE = 42

# Entrez IDs for the 15 protein-coding panel genes (from blood_validation_gse234297.py)
_PC_ENTREZ: dict[str, int] = {
    "MECOM": 2122, "SERTAD1": 29950, "FCN3": 8547, "PROS1": 5627,
    "ANGPT2": 285, "EMP1": 2012, "TINAGL1": 64129, "CKMT2": 1160,
    "VWF": 7450, "CLDN5": 7122, "NR4A1": 3164, "SOHLH2": 54937,
    "HEXB": 3074, "MCEE": 84693, "SLC37A2": 219855,
}

_GSE122649_TAR = ALS_DIR / "resources" / "GSE122649" / "GSE122649_RAW.tar"
_GSE122649_SOFT_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE122nnn/GSE122649/soft/"
    "GSE122649_family.soft.gz"
)
_GSE122649_RAW_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE122nnn/GSE122649/suppl/"
    "GSE122649_RAW.tar"
)
_GSE234297_COUNTS = ALS_DIR / "resources" / "GSE234297" / "suppl" / "GSE234297_gene_raw_counts.txt.gz"
_SRP_QUANT_DIR = ALS_DIR / "resources" / "SRP064478" / "quant"
_SRP_META = ALS_DIR / "resources" / "SRP064478" / "srr_metadata.tsv"
_SRP_FASTA = ALS_DIR / "resources" / "SRP064478" / "index" / "hg38_cdna.fa.gz"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _fit_models(X_tr: np.ndarray, y_tr: np.ndarray):
    """Fit LGBM and L2-LR on training data. Return (lgbm, lr, lr_scaler)."""
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    params = json.loads(_PARAMS_PATH.read_text())
    params["colsample_bytree"] = 1.0
    lgbm = LGBMClassifier(**params)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        lgbm.fit(X_tr, y_tr)

    scaler = StandardScaler().fit(X_tr)
    lr = LogisticRegressionCV(
        cv=5, solver="liblinear", penalty="l2", scoring="roc_auc",
        max_iter=2000, random_state=RANDOM_STATE, refit=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        lr.fit(scaler.transform(X_tr), y_tr)

    return lgbm, lr, scaler


def _score(name: str, X_te: np.ndarray, y_te: np.ndarray, lgbm, lr, scaler) -> list[dict]:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    rows = []
    for model_name, scores in [
        ("LGBM", lgbm.predict_proba(X_te)[:, 1]),
        ("LR_L2", lr.predict_proba(scaler.transform(X_te))[:, 1]),
    ]:
        rows.append({
            "cohort": name, "model": model_name,
            "n": len(y_te), "n_pos": int(y_te.sum()),
            "AUC": round(float(roc_auc_score(y_te, scores)), 4),
            "brier": round(float(brier_score_loss(y_te, scores)), 4),
            "log_loss": round(float(log_loss(y_te, np.clip(scores, 1e-7, 1 - 1e-7))), 4),
            "mean_pos": round(float(scores[y_te == 1].mean()), 4),
            "mean_neg": round(float(scores[y_te == 0].mean()), 4) if (y_te == 0).any() else float("nan"),
        })
    return rows


def _load_gpl24676_by_symbol(panel_symbols: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load GPL24676; match panel genes by HGNC symbol; return (X_log, y, matched_syms)."""
    (ds,) = load_dataset("GSE153960", platform="GPL24676", resources_dir=ALS_DIR / "resources")
    X_log = np.log1p(ds.X.values.astype(np.float32))
    # Columns are versioned ENSG IDs; map via panel CSV
    panel_df = pd.read_csv(_PANEL_CSV)
    feat_col = next(c for c in panel_df.columns if "feature" in c.lower())
    sym_col = next(c for c in panel_df.columns if "symbol" in c.lower())
    sym_to_feat = dict(zip(panel_df[sym_col], panel_df[feat_col]))
    feat_map = {n: i for i, n in enumerate(ds.X.columns)}
    cols, matched = [], []
    for sym in panel_symbols:
        feat = sym_to_feat.get(sym)
        if feat and feat in feat_map:
            cols.append(feat_map[feat])
            matched.append(sym)
    return X_log[:, cols], ds.y.values.astype(int), matched


def _load_gpl24676_by_ensg(ensg_bases: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load GPL24676; match panel genes by Ensembl base ID; return (X_log, y, matched_bases)."""
    (ds,) = load_dataset("GSE153960", platform="GPL24676", resources_dir=ALS_DIR / "resources")
    X_log = np.log1p(ds.X.values.astype(np.float32))
    base_map = {n.split(".")[0]: i for i, n in enumerate(ds.X.columns)}
    cols, matched = [], []
    for base in ensg_bases:
        if base in base_map:
            cols.append(base_map[base])
            matched.append(base)
    return X_log[:, cols], ds.y.values.astype(int), matched


# ---------------------------------------------------------------------------
# GSE122649 loader (symbol-keyed raw counts)
# ---------------------------------------------------------------------------


def _load_gse122649() -> tuple[np.ndarray, list[str], np.ndarray]:
    """Load GSE122649 raw count tar + SOFT metadata. Return (X_log, gene_syms, y)."""
    import requests

    if not _GSE122649_TAR.exists():
        print(f"  Downloading GSE122649_RAW.tar ...")
        _GSE122649_TAR.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(_GSE122649_RAW_URL, timeout=300, stream=True)
        with open(_GSE122649_TAR, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)

    print("  Fetching GSE122649 SOFT metadata ...")
    r = requests.get(_GSE122649_SOFT_URL, timeout=60, stream=True)
    content = gzip.decompress(r.content).decode("utf-8", errors="ignore")
    samples = re.split(r"\^SAMPLE", content)[1:]
    gsm_to_diag: dict[str, str] = {}
    for s in samples:
        acc = re.search(r"!Sample_geo_accession = (GSM\d+)", s)
        diag = re.search(r"!Sample_characteristics_ch1 = diagnosis: (.+)", s)
        if acc and diag:
            gsm_to_diag[acc.group(1)] = diag.group(1).strip()

    gene_symbols: list[str] = []
    sample_counts: dict[str, np.ndarray] = {}

    with tarfile.open(_GSE122649_TAR, "r") as tf:
        for member in tf.getmembers():
            gsm_m = re.search(r"(GSM\d+)", member.name)
            if not gsm_m or gsm_m.group(1) not in gsm_to_diag:
                continue
            gsm = gsm_m.group(1)
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            raw = fobj.read()
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
            syms = [r[0] for r in rows]
            counts = np.array([r[1] for r in rows], dtype=np.float32)
            if not gene_symbols:
                gene_symbols = syms
            elif len(syms) != len(gene_symbols):
                continue
            sample_counts[gsm] = counts

    gsm_ids = list(sample_counts.keys())
    X_raw = np.stack([sample_counts[g] for g in gsm_ids])
    diag_map = {"sALS": 1, "sALS/FTD": 1, "Non-neurological control": 0}
    y = np.array([diag_map.get(gsm_to_diag[g], -1) for g in gsm_ids])
    valid = y >= 0
    return np.log1p(X_raw[valid].astype(np.float32)), gene_symbols, y[valid]


# ---------------------------------------------------------------------------
# GSE234297 loader (Entrez-keyed raw counts)
# ---------------------------------------------------------------------------


def _load_gse234297_matched(panel_symbols: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load GSE234297 blood counts; match by Entrez ID. Return (X_log, y, matched_syms)."""
    with gzip.open(_GSE234297_COUNTS, "rt") as fh:
        counts = pd.read_csv(fh, sep="\t", index_col=0)
    sample_cols = list(counts.columns)
    y = np.array([1 if c.startswith("Case") else 0 for c in sample_cols], dtype=int)

    matched_syms, cols = [], []
    for sym in panel_symbols:
        eid = _PC_ENTREZ.get(sym)
        if eid and eid in counts.index:
            matched_syms.append(sym)
            cols.append(counts.index.get_loc(eid))

    X = np.log1p(counts.iloc[cols].values.T.astype(np.float32))
    return X, y, matched_syms


# ---------------------------------------------------------------------------
# SRP064478 loader (Salmon ENST → ENSG aggregation)
# ---------------------------------------------------------------------------


def _load_srp064478_matched(panel_bases: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load SRP064478 Salmon quant; aggregate ENST → ENSG; match panel genes."""
    meta = pd.read_csv(_SRP_META, sep="\t").set_index("SRR")

    # Build tx2gene from cDNA FASTA
    tx2gene: dict[str, str] = {}
    print("  Building tx2gene from Salmon FASTA ...")
    with gzip.open(_SRP_FASTA, "rt") as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            m_enst = re.search(r"(ENST\d+)", line)
            m_ensg = re.search(r"gene:(ENSG\d+)", line)
            if m_enst and m_ensg:
                tx2gene[m_enst.group(1)] = m_ensg.group(1)

    panel_base_set = set(panel_bases)
    rows, y_list = [], []

    for srr in meta.index:
        sf = _SRP_QUANT_DIR / srr / "quant.sf"
        df = pd.read_csv(sf, sep="\t")
        df["gene_base"] = df["Name"].str.split(".").str[0].map(tx2gene)
        df = df.dropna(subset=["gene_base"])
        gene_sums = df[df["gene_base"].isin(panel_base_set)].groupby("gene_base")["NumReads"].sum()
        rows.append(gene_sums.to_dict())
        y_list.append(1 if meta.loc[srr, "condition"] == "ALS" else 0)

    matched = [g for g in panel_bases if any(g in r for r in rows)]
    X = np.zeros((len(rows), len(matched)), dtype=np.float32)
    for i, r in enumerate(rows):
        for j, g in enumerate(matched):
            X[i, j] = r.get(g, 0.0)
    return np.log1p(X), np.array(y_list, dtype=int), matched


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    warnings.filterwarnings("ignore")

    panel_df = pd.read_csv(_PANEL_CSV)
    feat_col = next(c for c in panel_df.columns if "feature" in c.lower())
    sym_col = next(c for c in panel_df.columns if "symbol" in c.lower())
    panel_symbols: list[str] = panel_df[sym_col].tolist()
    panel_feats: list[str] = panel_df[feat_col].tolist()
    panel_bases = [f.split(".")[0] for f in panel_feats]
    pc_syms = list(_PC_ENTREZ.keys())

    all_results: list[dict] = []

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("GSE122649 (motor cortex, n=38)")
    print("=" * 60)

    X_122, syms_122, y_122 = _load_gse122649()
    print(f"  Loaded: {X_122.shape}  ALS={int(y_122.sum())}  Ctrl={int((y_122==0).sum())}")

    sym_map_122 = {s: i for i, s in enumerate(syms_122)}
    matched_122 = [s for s in panel_symbols if s in sym_map_122]
    missing_122 = [s for s in panel_symbols if s not in sym_map_122]
    print(f"  Panel matched: {len(matched_122)}/25  Missing: {missing_122[:5]}{'...' if len(missing_122) > 5 else ''}")

    X_te_122 = X_122[:, [sym_map_122[s] for s in matched_122]]
    X_tr_122, y_tr_122, _ = _load_gpl24676_by_symbol(matched_122)
    print(f"  Training: {X_tr_122.shape}")

    lgbm_122, lr_122, sc_122 = _fit_models(X_tr_122, y_tr_122)
    all_results.extend(_score("GSE122649", X_te_122, y_122, lgbm_122, lr_122, sc_122))
    for row in all_results[-2:]:
        print(f"  {row['model']:6s}  AUC={row['AUC']:.4f}  Brier={row['brier']:.4f}  LogLoss={row['log_loss']:.4f}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("GSE234297 (blood, n=144)")
    print("=" * 60)

    X_te_234, y_234, matched_234 = _load_gse234297_matched(pc_syms)
    print(f"  Loaded: {X_te_234.shape}  ALS={int(y_234.sum())}  Ctrl={int((y_234==0).sum())}")
    print(f"  Panel matched: {len(matched_234)}/15")

    X_tr_234, y_tr_234, _ = _load_gpl24676_by_symbol(matched_234)
    print(f"  Training: {X_tr_234.shape}")

    lgbm_234, lr_234, sc_234 = _fit_models(X_tr_234, y_tr_234)
    all_results.extend(_score("GSE234297", X_te_234, y_234, lgbm_234, lr_234, sc_234))
    for row in all_results[-2:]:
        print(f"  {row['model']:6s}  AUC={row['AUC']:.4f}  Brier={row['brier']:.4f}  LogLoss={row['log_loss']:.4f}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SRP064478 (cervical SC, n=15)")
    print("=" * 60)

    X_te_srp, y_srp, matched_srp = _load_srp064478_matched(panel_bases)
    print(f"  Loaded: {X_te_srp.shape}  ALS={int(y_srp.sum())}  Ctrl={int((y_srp==0).sum())}")
    print(f"  Panel matched: {len(matched_srp)}/25 (base IDs)")

    X_tr_srp, y_tr_srp, _ = _load_gpl24676_by_ensg(matched_srp)
    print(f"  Training: {X_tr_srp.shape}")

    lgbm_srp, lr_srp, sc_srp = _fit_models(X_tr_srp, y_tr_srp)
    all_results.extend(_score("SRP064478", X_te_srp, y_srp, lgbm_srp, lr_srp, sc_srp))
    for row in all_results[-2:]:
        print(f"  {row['model']:6s}  AUC={row['AUC']:.4f}  Brier={row['brier']:.4f}  LogLoss={row['log_loss']:.4f}")

    # ------------------------------------------------------------------
    # Merge with existing multicohort_baselines results
    existing_csv = SCRIPT_DIR / "multicohort_baselines.csv"
    existing_df = pd.read_csv(existing_csv) if existing_csv.exists() else pd.DataFrame()
    new_df = pd.DataFrame(all_results)
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined.to_csv(existing_csv, index=False)

    txt_path = SCRIPT_DIR / "multicohort_baselines.txt"
    lines = [
        "Multi-cohort LR-L2 vs LGBM baseline + calibration",
        "=" * 60,
        combined.to_string(index=False),
        "",
        "Interpretation:",
        "  - AUC parity (within 0.02) between LR and LGBM cross-cohort indicates",
        "    that most of the panel's discriminative signal is linearly extractable.",
        "  - Brier and log-loss directly compare calibration: lower is better.",
        "  - Mean-neg: average ALS probability assigned to control samples.",
    ]
    txt_path.write_text("\n".join(lines))

    print("\n" + "=" * 60)
    print("COMBINED RESULTS:")
    print(combined.to_string(index=False))
    print(f"\nSaved → {existing_csv.name}, {txt_path.name}")


if __name__ == "__main__":
    main()
