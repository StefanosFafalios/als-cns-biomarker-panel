"""Zero-shot evaluation of the 15-gene critical panel on 4 CNS cohorts (equal weights).

Critical panel = greedy backward elimination peak (W.mean AUC = 0.8921 at k=15)
under equal cohort weighting (0.25 per cohort). Source: iterative_panel_elimination.py.
This replaces the prior one-pass-LOO 16-gene definition, which the v2 bootstrap-CI
analysis (panel_loo_zeroshot.py) showed was inflated by combinatorial redundancy
(e.g. EMP1: largest individual negative D_ZS but redundant in a 16-gene context).

Members (15):
  12 protein-coding: FCN3, PROS1, ANGPT2, TINAGL1, CKMT2, CLDN5, NR4A1, SOHLH2,
                     HEXB, MCEE, SLC37A2, SERTAD1
  2 pseudogenes:     HERC2P8, SMG1P5
  1 snoRNA:          SNORD97

Panel indices in the 25-gene ordered panel:
  [1, 2, 3, 5, 6, 7, 10, 11, 12, 15, 16, 18, 20, 23, 24]

Outputs:
  panel_critical_eval_statistics.txt
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

# 15-gene critical panel (greedy backward elimination peak, equal weights, W.mean=0.8921)
# Indices into the 25-gene ordered panel (lgbm_core25_panel.csv).
_CRITICAL_IDX = [1, 2, 3, 5, 6, 7, 10, 11, 12, 15, 16, 18, 20, 23, 24]

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
# Helpers (identical to panel_loo_zeroshot.py)
# ---------------------------------------------------------------------------

def _bootstrap_auc(
    y: np.ndarray, scores: np.ndarray,
    n: int = N_BOOTSTRAP, seed: int = RANDOM_STATE,
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


def _zero_shot_ci(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    params: dict,
) -> tuple[float, float, float]:
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
# Data loaders (identical to panel_loo_zeroshot.py)
# ---------------------------------------------------------------------------

def _load_gpl24676_ctd() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
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
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    (ds16,) = load_dataset("GSE153960", platform="GPL16791", resources_dir=ALS_DIR / "resources")
    X16_raw = ds16.X.values.astype(np.float32)
    y16 = ds16.y.values.astype(int)
    feat16 = list(ds16.X.columns)
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
    X_raw = gene_mat.values.T.astype(np.float32)
    return np.log1p(X_raw), list(gene_mat.index), np.array(y_list, dtype=int)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import pandas as pd
    warnings.filterwarnings("ignore", category=UserWarning)

    print("=" * 70)
    print("15-gene critical panel — zero-shot AUC with 95% CI (equal cohort weights)")
    print("Greedy backward elimination peak (W.mean = 0.8921 at k=15)")
    print("Weights: GPL16791=0.250  GSE76220=0.250  GSE122649=0.250  SRP064478=0.250")
    print("=" * 70)

    params_raw = json.loads(_PARAMS_PATH.read_text())
    params = dict(params_raw, colsample_bytree=1.0, n_jobs=-1, verbose=-1)

    # ---- Panel ----
    df_panel = pd.read_csv(_PANEL_CSV)
    feat_col = next(c for c in df_panel.columns if "feature" in c.lower() or "ensg" in c.lower())
    sym_col = next(c for c in df_panel.columns if "symbol" in c.lower())
    feat25: list[str] = df_panel[feat_col].tolist()
    sym25: list[str] = df_panel[sym_col].tolist()
    feat25_bases: list[str] = [f.split(".")[0] for f in feat25]

    panel_syms = [sym25[i] for i in _CRITICAL_IDX]
    panel_bases = [feat25_bases[i] for i in _CRITICAL_IDX]
    print(f"\n15-gene critical panel: {', '.join(panel_syms)}\n")

    # ---- Load GPL24676 ----
    print("Loading GPL24676 (CTD + log1p) ...")
    X_train_ctd, y_train, X_train_log, feat_train = _load_gpl24676_ctd()
    feat_train_base_map: dict[str, int] = {f.split(".")[0]: j for j, f in enumerate(feat_train)}
    panel_train_cols: list[int] = [feat_train_base_map[b] for b in feat25_bases]
    Xtr25_ctd = X_train_ctd[:, panel_train_cols].astype(np.float32)
    Xtr25_log = X_train_log[:, panel_train_cols].astype(np.float32)

    # ---- Load GPL16791 ----
    print("Loading GPL16791 (CTD) ...")
    X16_ctd, y16 = _load_gpl16791_ctd(X_train_log, feat_train)
    X16_25_ctd = X16_ctd[:, panel_train_cols].astype(np.float32)

    # ---- Load GSE76220 ----
    print("Loading GSE76220 ...")
    X76_log, vocab76, y76 = _load_gse76220()
    vocab76_map = {s: i for i, s in enumerate(vocab76)}
    avail76 = [i for i, s in enumerate(sym25) if s in vocab76_map]
    X76_25 = (np.column_stack([X76_log[:, vocab76_map[sym25[i]]] for i in avail76])
               .astype(np.float32)) if avail76 else np.empty((len(y76), 0), dtype=np.float32)

    # ---- Load GSE122649 ----
    print("Loading GSE122649 ...")
    X122_log, vocab122, y122 = _load_gse122649()
    vocab122_map = {s: i for i, s in enumerate(vocab122)}
    avail122 = [i for i, s in enumerate(sym25) if s in vocab122_map]
    X122_25 = (np.column_stack([X122_log[:, vocab122_map[sym25[i]]] for i in avail122])
                .astype(np.float32)) if avail122 else np.empty((len(y122), 0), dtype=np.float32)

    # ---- Load SRP064478 ----
    print("Loading SRP064478 ...")
    Xsrp_log, vocab_srp, y_srp = _load_srp064478()
    srp_vocab_map = {v: i for i, v in enumerate(vocab_srp)}
    avail_srp = [i for i, b in enumerate(feat25_bases) if b in srp_vocab_map]
    Xsrp_25 = (np.column_stack([Xsrp_log[:, srp_vocab_map[feat25_bases[i]]] for i in avail_srp])
                .astype(np.float32)) if avail_srp else np.empty((len(y_srp), 0), dtype=np.float32)

    # ---- Evaluate 15-gene panel ----
    from sklearn.preprocessing import StandardScaler

    panel_set = set(_CRITICAL_IDX)
    print("\nEvaluating 15-gene critical panel with 95% CI ...\n")
    results: dict[str, tuple[float, float, float]] = {}

    # GPL16791 — CTD, all 15 available
    Xtr = Xtr25_ctd[:, _CRITICAL_IDX]
    Xte = X16_25_ctd[:, _CRITICAL_IDX]
    sc = StandardScaler().fit(Xtr)
    auc16, lo16, hi16 = _zero_shot_ci(sc.transform(Xtr), y_train, sc.transform(Xte), y16, params)
    results["GPL16791"] = (auc16, lo16, hi16)
    n_gpl16791 = len(_CRITICAL_IDX)

    # GSE76220 — symbol, log1p
    avail76_set = set(avail76)
    cols76_tr = [i for i in _CRITICAL_IDX if i in avail76_set]
    cols76_te = [k for k, i in enumerate(avail76) if i in panel_set]
    n_gse76220 = len(cols76_tr)
    if cols76_tr:
        Xtr = Xtr25_log[:, cols76_tr]
        Xte = X76_25[:, cols76_te]
        sc = StandardScaler().fit(Xtr)
        auc76, lo76, hi76 = _zero_shot_ci(
            sc.transform(Xtr), y_train, sc.transform(Xte), y76, params
        )
        results["GSE76220"] = (auc76, lo76, hi76)

    # GSE122649 — symbol, log1p
    avail122_set = set(avail122)
    cols122_tr = [i for i in _CRITICAL_IDX if i in avail122_set]
    cols122_te = [k for k, i in enumerate(avail122) if i in panel_set]
    n_gse122649 = len(cols122_tr)
    if cols122_tr:
        Xtr = Xtr25_log[:, cols122_tr]
        Xte = X122_25[:, cols122_te]
        sc = StandardScaler().fit(Xtr)
        auc122, lo122, hi122 = _zero_shot_ci(
            sc.transform(Xtr), y_train, sc.transform(Xte), y122, params
        )
        results["GSE122649"] = (auc122, lo122, hi122)

    # SRP064478 — Ensembl base ID, log1p
    avail_srp_set = set(avail_srp)
    cols_srp_tr = [i for i in _CRITICAL_IDX if i in avail_srp_set]
    cols_srp_te = [k for k, i in enumerate(avail_srp) if i in panel_set]
    n_srp = len(cols_srp_tr)
    if cols_srp_tr:
        Xtr = Xtr25_log[:, cols_srp_tr]
        Xte = Xsrp_25[:, cols_srp_te]
        sc = StandardScaler().fit(Xtr)
        auc_srp, lo_srp, hi_srp = _zero_shot_ci(
            sc.transform(Xtr), y_train, sc.transform(Xte), y_srp, params
        )
        results["SRP064478"] = (auc_srp, lo_srp, hi_srp)

    # Weighted mean
    wmean = sum(_W[k] * results[k][0] for k in _N_COHORT if k in results)

    gene_counts = {
        "GPL16791": n_gpl16791,
        "GSE76220": n_gse76220,
        "GSE122649": n_gse122649,
        "SRP064478": n_srp,
    }

    # ---- Print results ----
    header = (
        f"\n{'Cohort':>12s}  {'N':>5s}  {'Genes':>6s}  "
        f"{'AUC':>7s}  {'95% CI':>16s}\n"
        + "-" * 58
    )
    print(header)
    for k in _N_COHORT:
        if k not in results:
            continue
        auc, lo, hi = results[k]
        print(f"  {k:>12s}  {_N_COHORT[k]:>5d}  {gene_counts[k]:>6d}  "
              f"{auc:>7.4f}  [{lo:.3f}, {hi:.3f}]")
    print(f"\n  Weighted mean AUC = {wmean:.4f}")

    # ---- Save statistics ----
    lines = [
        "15-gene critical panel — zero-shot AUC with 95% CI (equal cohort weights)",
        "=" * 70,
        "",
        "Source: greedy backward elimination peak (iterative_panel_elimination.py),",
        "  W.mean AUC = 0.8921 at k=15 (peak of the elimination trajectory).",
        "  Supersedes the prior one-pass-LOO 16-gene definition: the v2 bootstrap-CI",
        "  analysis (panel_loo_zeroshot.py) showed individual-LOO D_ZS values inflated",
        "  by combinatorial redundancy (e.g. EMP1 had the largest individual",
        "  negative D_ZS but is dropped at greedy step 10 with positive D_ZS).",
        "",
        "Panel members:",
        "  Protein-coding (12): FCN3, PROS1, ANGPT2, TINAGL1, CKMT2, CLDN5,",
        "                       NR4A1, SOHLH2, HEXB, MCEE, SLC37A2, SERTAD1",
        "  Pseudogenes (2):     HERC2P8, SMG1P5",
        "  snoRNA (1):          SNORD97",
        "",
        "Panel indices in 25-gene panel: " + str(_CRITICAL_IDX),
        "",
        "Cohort weights (equal, 0.25 each):",
        *[f"  {k:12s} n={_N_COHORT[k]}  w={_W[k]:.4f}" for k in _N_COHORT],
        f"  Total N = {_TOTAL_N}",
        "",
        "Gene availability:",
        f"  GPL16791  = {n_gpl16791}/{len(_CRITICAL_IDX)}  (Ensembl base ID, all critical genes present)",
        f"  GSE76220  = {n_gse76220}/{len(_CRITICAL_IDX)}  (HGNC symbol; pseudogenes + snoRNA absent)",
        f"  GSE122649 = {n_gse122649}/{len(_CRITICAL_IDX)}  (HGNC symbol; pseudogenes + snoRNA absent)",
        f"  SRP064478 = {n_srp}/{len(_CRITICAL_IDX)}  (Ensembl base ID; pseudogenes present)",
        "",
        "Results (zero-shot, N_bootstrap=2000):",
        "",
        f"{'Cohort':>12s}  {'N':>5s}  {'Genes':>6s}  {'AUC':>7s}  {'95% CI':>16s}",
        "-" * 58,
    ]
    for k in _N_COHORT:
        if k not in results:
            continue
        auc, lo, hi = results[k]
        lines.append(
            f"  {k:>12s}  {_N_COHORT[k]:>5d}  {gene_counts[k]:>6d}  "
            f"{auc:>7.4f}  [{lo:.3f}, {hi:.3f}]"
        )
    lines += [
        "",
        f"  Weighted mean AUC = {wmean:.4f}",
        "",
        "Note: 'Genes' = number of panel genes matched in that cohort's vocabulary.",
        "      GPL16791 uses CTD compartment regression (same as training).",
        "      GSE76220, GSE122649 matched by HGNC symbol.",
        "      SRP064478 matched by Ensembl base ID (captures pseudogenes).",
    ]

    stat_out = SCRIPT_DIR / "panel_critical_eval_statistics.txt"
    stat_out.write_text("\n".join(lines) + "\n")
    print(f"\nSaved: {stat_out}")


if __name__ == "__main__":
    main()
