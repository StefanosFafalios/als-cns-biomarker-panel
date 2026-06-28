"""Per-gene LOO zero-shot contribution to the 25-gene panel (v2 — seed-averaged + bootstrap CI).

For each of the 25 panel genes:
  1. Train an ensemble of N_SEEDS LightGBM models (varying random_state) on GPL24676
     with that gene held out → averaged predicted probabilities per cohort.
  2. Same for the baseline 25-gene panel.
  3. Paired bootstrap (B=2000) the cohort test sets with stratified resampling
     to obtain a 95% CI on each per-gene D_ZS.

D_ZS(g) = W.mean_AUC(panel − {g}) − W.mean_AUC(full panel)
  Positive = removing g improves generalisation (gene is redundant/harmful)
  Negative = removing g hurts generalisation (gene is critical)

Two weighting schemes computed in parallel:
  Equal weights (primary)        : 0.25 per cohort — each independent dataset votes equally
  Sample-size weights (sensitivity): w_c = n_c / sum(n_c)

Critical gene set (primary, equal-weighted):
  gene g is "critical" iff upper 95% bootstrap CI of D_ZS_eq(g) < 0
  i.e. removing g credibly degrades generalisation under cohort-level resampling.

Cohorts (blood excluded):
  GPL16791   n=636  CTD regression, Ensembl base ID
  GSE76220   n=20   plain log1p, HGNC symbol
  GSE122649  n=38   plain log1p, HGNC symbol
  SRP064478  n=15   plain log1p, Ensembl base ID

Outputs:
  panel_loo_zeroshot_statistics.txt
  panel_loo_zeroshot.png
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
_ITER_STAT = SCRIPT_DIR / "iterative_panel_elimination_statistics.txt"


def _parse_drop_steps(text: str) -> dict[str, int]:
    """Parse iterative_panel_elimination_statistics.txt for per-gene drop step.

    Returns {gene_symbol: step_index}. The single surviving gene is assigned
    step max+1 so it sorts to the top (most-critical) end of the panel.
    """
    import re
    drops: dict[str, int] = {}
    in_log = False
    final_gene: str | None = None
    for line in text.splitlines():
        if line.strip().startswith("Step") and "Dropped" in line:
            in_log = True
            continue
        if not in_log:
            continue
        if line.strip().startswith("Final minimal panel"):
            in_log = False
            continue
        if not line.strip() or line.startswith("-"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            step = int(parts[0])
        except ValueError:
            continue
        drops[parts[1]] = step
    for line in text.splitlines():
        m = re.match(r"\s*Symbols:\s*(\S+)", line)
        if m:
            final_gene = m.group(1).rstrip(",")
            break
    if final_gene:
        drops[final_gene] = max(drops.values(), default=0) + 1
    return drops
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
N_SEEDS = 20  # LightGBM ensemble seeds for D_ZS stabilisation
_ENSEMBLE_SEEDS = tuple(range(N_SEEDS))

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
# Helpers
# ---------------------------------------------------------------------------

def _ensemble_scores(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray,
    params: dict,
    seeds: tuple[int, ...] = _ENSEMBLE_SEEDS,
) -> np.ndarray:
    """Train N_SEEDS LGBM models (varying random_state) and return averaged probabilities."""
    from lightgbm import LGBMClassifier
    scores = np.zeros(len(X_te), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in seeds:
            params_s = dict(params, random_state=s)
            clf = LGBMClassifier(**params_s).fit(X_tr, y_tr)
            scores += clf.predict_proba(X_te)[:, 1]
    return (scores / len(seeds)).astype(np.float32)


def _stratified_boot_indices(
    y: np.ndarray, n_boot: int, rng: np.random.Generator,
) -> np.ndarray:
    """Return (n_boot, n) array of stratified bootstrap indices preserving class counts."""
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n_pos = len(pos)
    n_neg = len(neg)
    out = np.empty((n_boot, n_pos + n_neg), dtype=np.int64)
    for b in range(n_boot):
        out[b, :n_pos] = rng.choice(pos, n_pos, replace=True)
        out[b, n_pos:] = rng.choice(neg, n_neg, replace=True)
    return out


# ---------------------------------------------------------------------------
# Data loaders (identical to validation_panel_loo.py)
# ---------------------------------------------------------------------------

def _load_gpl24676_ctd() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    # CTD compartment regression is discovery-side only; cross-cohort transfer now
    # uses raw log1p for every cohort, so the memory-heavy CTD-residualised matrix
    # is no longer built. Returns (X_log, y, X_log, feat) for signature compatibility.
    (ds,) = load_dataset("GSE153960", platform="GPL24676", resources_dir=ALS_DIR / "resources")
    X_log = np.log1p(ds.X.values.astype(np.float32))
    y = ds.y.values.astype(int)
    feat = list(ds.X.columns)
    return X_log, y, X_log, feat


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


def _load_gpl16791_raw(feat_train: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load GPL16791 as plain log1p (no CTD), Ensembl-ID matched to training.

    Compartment regression is retained only for discovery-side feature selection;
    cross-cohort transfer (including GPL16791) is evaluated on raw log1p, uniform
    with the symbol-matched cohorts (the configuration that transfers best; see
    the compartment-regression sensitivity analysis).
    """
    (ds16,) = load_dataset(
        "GSE153960", platform="GPL16791", resources_dir=ALS_DIR / "resources"
    )
    assert list(ds16.X.columns) == feat_train
    X16_log = np.log1p(ds16.X.values.astype(np.float32))
    return X16_log, ds16.y.values.astype(int)


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
# Core evaluation — seed-averaged ensemble scoring per cohort
# ---------------------------------------------------------------------------

def _panel_ensemble_scores(
    panel_idx: list[int],
    Xtr25_ctd: np.ndarray, y_train: np.ndarray,
    X16_25_raw: np.ndarray,
    Xtr25_log: np.ndarray,
    X76_25: np.ndarray, avail76: list[int],
    X122_25: np.ndarray, avail122: list[int],
    Xsrp_25: np.ndarray, avail_srp: list[int],
    params: dict,
    n_te: dict[str, int],
) -> dict[str, np.ndarray]:
    """Compute seed-averaged predicted probabilities per cohort for a panel subset.

    Returns dict mapping cohort name → ensemble score vector (len = n_test).
    If <2 genes are available for a cohort, score = 0.5 vector.
    """
    from sklearn.preprocessing import StandardScaler

    panel_set = set(panel_idx)
    out: dict[str, np.ndarray] = {}

    # GPL16791 — raw log1p (CTD removed; uniform with the symbol-matched cohorts)
    if len(panel_idx) >= 1:
        Xtr = Xtr25_log[:, panel_idx]
        Xte = X16_25_raw[:, panel_idx]
        sc = StandardScaler().fit(Xtr)
        out["GPL16791"] = _ensemble_scores(
            sc.transform(Xtr), y_train, sc.transform(Xte), params
        )
    else:
        out["GPL16791"] = np.full(n_te["GPL16791"], 0.5, dtype=np.float32)

    # GSE76220 — symbol, log1p
    avail76_set = set(avail76)
    cols76_tr = [i for i in panel_idx if i in avail76_set]
    cols76_te = [k for k, i in enumerate(avail76) if i in panel_set]
    if cols76_tr:
        Xtr = Xtr25_log[:, cols76_tr]
        Xte = X76_25[:, cols76_te]
        sc = StandardScaler().fit(Xtr)
        out["GSE76220"] = _ensemble_scores(
            sc.transform(Xtr), y_train, sc.transform(Xte), params
        )
    else:
        out["GSE76220"] = np.full(n_te["GSE76220"], 0.5, dtype=np.float32)

    # GSE122649 — symbol, log1p
    avail122_set = set(avail122)
    cols122_tr = [i for i in panel_idx if i in avail122_set]
    cols122_te = [k for k, i in enumerate(avail122) if i in panel_set]
    if cols122_tr:
        Xtr = Xtr25_log[:, cols122_tr]
        Xte = X122_25[:, cols122_te]
        sc = StandardScaler().fit(Xtr)
        out["GSE122649"] = _ensemble_scores(
            sc.transform(Xtr), y_train, sc.transform(Xte), params
        )
    else:
        out["GSE122649"] = np.full(n_te["GSE122649"], 0.5, dtype=np.float32)

    # SRP064478 — Ensembl base ID, log1p
    avail_srp_set = set(avail_srp)
    cols_srp_tr = [i for i in panel_idx if i in avail_srp_set]
    cols_srp_te = [k for k, i in enumerate(avail_srp) if i in panel_set]
    if cols_srp_tr:
        Xtr = Xtr25_log[:, cols_srp_tr]
        Xte = Xsrp_25[:, cols_srp_te]
        sc = StandardScaler().fit(Xtr)
        out["SRP064478"] = _ensemble_scores(
            sc.transform(Xtr), y_train, sc.transform(Xte), params
        )
    else:
        out["SRP064478"] = np.full(n_te["SRP064478"], 0.5, dtype=np.float32)

    return out


def _wmean_aucs(
    cohort_aucs: dict[str, float], weights: dict[str, float],
) -> float:
    return sum(weights[c] * cohort_aucs[c] for c in weights)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    warnings.filterwarnings("ignore", category=UserWarning)

    print("=" * 70)
    print(f"Panel LOO zero-shot v2 — seed-averaged ({N_SEEDS} seeds) + paired bootstrap CI")
    print("Primary weights (equal): 0.25 per cohort")
    print("Sensitivity weights (sample-size): w_c = n_c / 709")
    print("=" * 70)

    params_raw = json.loads(_PARAMS_PATH.read_text())
    # colsample_bytree=1.0: with 24-25 features, fractional column sampling would
    # subsample to ~1.6 features per tree. n_jobs handled by LGBM internal threading.
    params = dict(params_raw, colsample_bytree=1.0, n_jobs=-1, verbose=-1)
    # random_state in params_raw is overridden per-seed inside _ensemble_scores

    # Sample-size weighted scheme (sensitivity)
    _W_N = {k: _N_COHORT[k] / _TOTAL_N for k in _N_COHORT}

    # ---- Panel ----
    df_panel = pd.read_csv(_PANEL_CSV)
    feat_col = next(c for c in df_panel.columns if "feature" in c.lower() or "ensg" in c.lower())
    sym_col = next(c for c in df_panel.columns if "symbol" in c.lower())
    feat25: list[str] = df_panel[feat_col].tolist()
    sym25: list[str] = df_panel[sym_col].tolist()
    feat25_bases: list[str] = [f.split(".")[0] for f in feat25]

    # ---- Load GPL24676 (once) ----
    print("\nLoading GPL24676 (CTD + log1p) ...")
    X_train_ctd, y_train, X_train_log, feat_train = _load_gpl24676_ctd()
    feat_train_base_map: dict[str, int] = {f.split(".")[0]: j for j, f in enumerate(feat_train)}
    panel_train_cols: list[int] = [feat_train_base_map[b] for b in feat25_bases]
    Xtr25_ctd = X_train_ctd[:, panel_train_cols].astype(np.float32)
    Xtr25_log = X_train_log[:, panel_train_cols].astype(np.float32)

    # ---- Load GPL16791 (raw log1p — uniform with the other zero-shot cohorts) ----
    print("Loading GPL16791 (raw log1p; CTD is discovery-side only) ...")
    X16_log_full, y16 = _load_gpl16791_raw(feat_train)
    X16_25_raw = X16_log_full[:, panel_train_cols].astype(np.float32)

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

    y_per_cohort = {"GPL16791": y16, "GSE76220": y76, "GSE122649": y122, "SRP064478": y_srp}
    n_per_cohort = {c: len(y_per_cohort[c]) for c in _N_COHORT}
    avail_per_cohort = {"GPL16791": list(range(25)), "GSE76220": avail76,
                        "GSE122649": avail122, "SRP064478": avail_srp}

    score_kwargs = dict(
        Xtr25_ctd=Xtr25_ctd, y_train=y_train,
        X16_25_raw=X16_25_raw,
        Xtr25_log=Xtr25_log,
        X76_25=X76_25, avail76=avail76,
        X122_25=X122_25, avail122=avail122,
        Xsrp_25=Xsrp_25, avail_srp=avail_srp,
        params=params,
        n_te=n_per_cohort,
    )

    # ---- Baseline: all 25 genes (seed-averaged ensemble scores per cohort) ----
    print(f"\nComputing baseline ensemble scores ({N_SEEDS} seeds × 4 cohorts) ...")
    full_idx = list(range(25))
    baseline_scores = _panel_ensemble_scores(full_idx, **score_kwargs)
    baseline_aucs = {c: float(roc_auc_score(y_per_cohort[c], baseline_scores[c]))
                     for c in _N_COHORT}
    baseline_wmean_eq = _wmean_aucs(baseline_aucs, _W)
    baseline_wmean_n = _wmean_aucs(baseline_aucs, _W_N)
    print(f"  Baseline W.mean (equal)  = {baseline_wmean_eq:.4f}")
    print(f"  Baseline W.mean (size-w) = {baseline_wmean_n:.4f}")
    for c in _N_COHORT:
        print(f"    {c}: {baseline_aucs[c]:.4f}")

    # ---- LOO ensemble scores per gene per cohort ----
    print(f"\nComputing LOO ensemble scores for {len(sym25)} genes ...")
    loo_scores: dict[int, dict[str, np.ndarray]] = {}
    avail_in_cohort: dict[int, dict[str, bool]] = {}
    for g in range(25):
        trial_idx = [i for i in full_idx if i != g]
        # Decide which cohorts need retraining: only those where g is in avail_idx.
        needs_retrain = {c: (g in avail_per_cohort[c]) for c in _N_COHORT}
        # GPL16791 always retrains (all 25 panel cols present).
        needs_retrain["GPL16791"] = True
        if all(needs_retrain.values()):
            scores_g = _panel_ensemble_scores(trial_idx, **score_kwargs)
        else:
            # Compute only for cohorts that need it; copy baseline for the rest.
            scores_g = {c: baseline_scores[c].copy() for c in _N_COHORT}
            partial = _panel_ensemble_scores(trial_idx, **score_kwargs)
            for c in _N_COHORT:
                if needs_retrain[c]:
                    scores_g[c] = partial[c]
        loo_scores[g] = scores_g
        avail_in_cohort[g] = {c: (g in avail_per_cohort[c]) for c in _N_COHORT}
        # Point estimates for log
        aucs_g = {c: float(roc_auc_score(y_per_cohort[c], scores_g[c])) for c in _N_COHORT}
        wmean_g_eq = _wmean_aucs(aucs_g, _W)
        d_zs_g_eq = wmean_g_eq - baseline_wmean_eq
        print(f"  {sym25[g]:>20s}: W.mean(eq)={wmean_g_eq:.4f}  D_ZS(eq)={d_zs_g_eq:+.4f}  "
              + "  ".join(f"{aucs_g[c]:.4f}" for c in _N_COHORT))

    # ---- Paired bootstrap on test indices (stratified) ----
    print(f"\nRunning paired bootstrap ({N_BOOTSTRAP} resamples) ...")
    rng = np.random.default_rng(RANDOM_STATE)
    boot_idx = {c: _stratified_boot_indices(y_per_cohort[c], N_BOOTSTRAP, rng)
                for c in _N_COHORT}

    # Per-bootstrap baseline cohort AUCs and W.mean
    base_aucs_b = {c: np.empty(N_BOOTSTRAP, dtype=np.float32) for c in _N_COHORT}
    for c in _N_COHORT:
        y_c = y_per_cohort[c]
        sc_c = baseline_scores[c]
        for b in range(N_BOOTSTRAP):
            idx = boot_idx[c][b]
            base_aucs_b[c][b] = roc_auc_score(y_c[idx], sc_c[idx])
    base_wmean_eq_b = np.zeros(N_BOOTSTRAP, dtype=np.float32)
    base_wmean_n_b = np.zeros(N_BOOTSTRAP, dtype=np.float32)
    for c in _N_COHORT:
        base_wmean_eq_b += _W[c] * base_aucs_b[c]
        base_wmean_n_b += _W_N[c] * base_aucs_b[c]

    # Per-gene per-bootstrap D_ZS
    d_zs_eq_boot = np.zeros((25, N_BOOTSTRAP), dtype=np.float32)
    d_zs_n_boot = np.zeros((25, N_BOOTSTRAP), dtype=np.float32)
    for g in range(25):
        loo_wmean_eq_b = np.zeros(N_BOOTSTRAP, dtype=np.float32)
        loo_wmean_n_b = np.zeros(N_BOOTSTRAP, dtype=np.float32)
        for c in _N_COHORT:
            y_c = y_per_cohort[c]
            sc_c = loo_scores[g][c]
            loo_auc_c_b = np.empty(N_BOOTSTRAP, dtype=np.float32)
            for b in range(N_BOOTSTRAP):
                idx = boot_idx[c][b]
                loo_auc_c_b[b] = roc_auc_score(y_c[idx], sc_c[idx])
            loo_wmean_eq_b += _W[c] * loo_auc_c_b
            loo_wmean_n_b += _W_N[c] * loo_auc_c_b
        d_zs_eq_boot[g] = loo_wmean_eq_b - base_wmean_eq_b
        d_zs_n_boot[g] = loo_wmean_n_b - base_wmean_n_b

    # ---- Per-gene point estimate (from non-bootstrap scores) and CI ----
    point_aucs = {g: {c: float(roc_auc_score(y_per_cohort[c], loo_scores[g][c]))
                       for c in _N_COHORT} for g in range(25)}
    point_d_zs_eq = {g: _wmean_aucs(point_aucs[g], _W) - baseline_wmean_eq for g in range(25)}
    point_d_zs_n = {g: _wmean_aucs(point_aucs[g], _W_N) - baseline_wmean_n for g in range(25)}

    ci_eq = {g: (float(np.percentile(d_zs_eq_boot[g], 2.5)),
                  float(np.percentile(d_zs_eq_boot[g], 97.5)))
              for g in range(25)}
    ci_n = {g: (float(np.percentile(d_zs_n_boot[g], 2.5)),
                 float(np.percentile(d_zs_n_boot[g], 97.5)))
             for g in range(25)}

    # Critical set: equal-weight CI upper bound < 0
    critical_eq = [g for g in range(25) if ci_eq[g][1] < 0]
    critical_n = [g for g in range(25) if ci_n[g][1] < 0]
    # Intersection / union with sensitivity scheme
    critical_intersect = sorted(set(critical_eq) & set(critical_n))
    critical_union = sorted(set(critical_eq) | set(critical_n))

    # ---- Results table sorted by point D_ZS (equal) ascending ----
    results = []
    for g in range(25):
        results.append({
            "g": g,
            "gene": sym25[g],
            "feat": feat25[g],
            "D_ZS_eq": point_d_zs_eq[g],
            "D_ZS_eq_lo": ci_eq[g][0],
            "D_ZS_eq_hi": ci_eq[g][1],
            "D_ZS_n": point_d_zs_n[g],
            "D_ZS_n_lo": ci_n[g][0],
            "D_ZS_n_hi": ci_n[g][1],
            "critical_eq": g in critical_eq,
            "critical_n": g in critical_n,
            **{f"auc_{c}": point_aucs[g][c] for c in _N_COHORT},
            **{f"auc_{c}_base": baseline_aucs[c] for c in _N_COHORT},
            "avail_76": avail_in_cohort[g]["GSE76220"],
            "avail_122": avail_in_cohort[g]["GSE122649"],
            "avail_srp": avail_in_cohort[g]["SRP064478"],
        })
    # Sort by greedy backward-elimination drop step DESCENDING — survivors
    # (15-gene critical panel) at top, first-dropped at bottom. Sort secondary
    # by D_ZS_eq ascending so within each block the most-critical comes first.
    drop_steps = _parse_drop_steps(_ITER_STAT.read_text()) if _ITER_STAT.exists() else {}
    results.sort(key=lambda r: (-drop_steps.get(r["gene"], 0), r["D_ZS_eq"]))

    # ---- Text output ----
    lines = [
        "Panel LOO Zero-Shot v2 — seed-averaged + paired bootstrap CI",
        "=" * 70,
        "",
        f"Method: ensemble of {N_SEEDS} LightGBM seeds → averaged probabilities;",
        f"        paired stratified bootstrap (B={N_BOOTSTRAP}) on cohort test sets",
        "        for 95% CI on per-gene D_ZS.",
        "",
        "Cohorts and weights:",
        *[f"  {c:<12s} n={_N_COHORT[c]:>3d}  w_eq={_W[c]:.4f}  w_n={_W_N[c]:.4f}"
          for c in _N_COHORT],
        f"  Total N = {_TOTAL_N}",
        "",
        "Gene availability (matched in cohort vocabulary):",
        f"  GSE76220 = {len(avail76)}/25  GSE122649 = {len(avail122)}/25  "
        f"SRP064478 = {len(avail_srp)}/25  (GPL16791 = 25/25)",
        "",
        f"Baseline W.mean AUC (all 25 genes):",
        f"  equal-weighted    : {baseline_wmean_eq:.4f}",
        f"  size-weighted     : {baseline_wmean_n:.4f}",
        *[f"  {c}: {baseline_aucs[c]:.4f}" for c in _N_COHORT],
        "",
        "Per-gene D_ZS with 95% paired-bootstrap CI",
        "(sorted by greedy backward-elimination drop step DESC — survivors first;",
        " secondary key: D_ZS_eq ascending):",
        "",
        f"{'Gene':>20s}  {'D_ZS_eq':>8s}  {'CI_eq':>22s}  {'D_ZS_n':>8s}  "
        f"{'CI_n':>22s}  {'eq?':>4s}  {'n?':>4s}",
        "-" * 100,
    ]
    for r in results:
        ci_eq_str = f"[{r['D_ZS_eq_lo']:+.4f}, {r['D_ZS_eq_hi']:+.4f}]"
        ci_n_str = f"[{r['D_ZS_n_lo']:+.4f}, {r['D_ZS_n_hi']:+.4f}]"
        flag_eq = " *" if r["critical_eq"] else "  "
        flag_n = " *" if r["critical_n"] else "  "
        lines.append(
            f"{r['gene']:>20s}  {r['D_ZS_eq']:>+8.4f}  {ci_eq_str:>22s}  "
            f"{r['D_ZS_n']:>+8.4f}  {ci_n_str:>22s}  {flag_eq:>4s}  {flag_n:>4s}"
        )

    lines += [
        "",
        "Critical set rule: CI_upper < 0 (removing gene credibly degrades W.mean AUC).",
        "  * = gene flagged critical under that weighting scheme.",
        "",
        f"Critical under equal weights  (n={len(critical_eq)}): "
        + ", ".join(sym25[g] for g in critical_eq) if critical_eq else
        "Critical under equal weights (n=0): (none)",
        f"Critical under sample-size weights (n={len(critical_n)}): "
        + ", ".join(sym25[g] for g in critical_n) if critical_n else
        "Critical under sample-size weights (n=0): (none)",
        f"Intersection (both schemes) (n={len(critical_intersect)}): "
        + ", ".join(sym25[g] for g in critical_intersect) if critical_intersect else
        "Intersection (both schemes) (n=0): (none)",
        f"Union (either scheme) (n={len(critical_union)}): "
        + ", ".join(sym25[g] for g in critical_union) if critical_union else
        "Union (either scheme) (n=0): (none)",
        "",
        "Per-cohort point AUC for each LOO panel (sorted by D_ZS_eq):",
        f"{'Gene':>20s}  " + "  ".join(f"{'AUC_'+c:>12s}" for c in _N_COHORT),
        "-" * 80,
    ]
    for r in results:
        lines.append(
            f"{r['gene']:>20s}  "
            + "  ".join(f"{r['auc_'+c]:>12.4f}" for c in _N_COHORT)
        )

    stat_out = SCRIPT_DIR / "panel_loo_zeroshot_statistics.txt"
    stat_out.write_text("\n".join(lines) + "\n")
    print(f"\nSaved: {stat_out}")
    print(f"\nCritical (equal weights, CI<0): n={len(critical_eq)}: "
          + ", ".join(sym25[g] for g in critical_eq))
    print(f"Critical (size weights, CI<0):  n={len(critical_n)}: "
          + ", ".join(sym25[g] for g in critical_n))

    # ---- CSV of full per-gene table ----
    csv_rows = []
    for r in results:
        csv_rows.append({
            "gene": r["gene"],
            "feature": r["feat"],
            "D_ZS_equal": r["D_ZS_eq"],
            "D_ZS_equal_ci_lo": r["D_ZS_eq_lo"],
            "D_ZS_equal_ci_hi": r["D_ZS_eq_hi"],
            "D_ZS_sizew": r["D_ZS_n"],
            "D_ZS_sizew_ci_lo": r["D_ZS_n_lo"],
            "D_ZS_sizew_ci_hi": r["D_ZS_n_hi"],
            "critical_equal": r["critical_eq"],
            "critical_sizew": r["critical_n"],
            **{f"AUC_{c}_LOO": r[f"auc_{c}"] for c in _N_COHORT},
            **{f"AUC_{c}_baseline": r[f"auc_{c}_base"] for c in _N_COHORT},
            "avail_GSE76220": r["avail_76"],
            "avail_GSE122649": r["avail_122"],
            "avail_SRP064478": r["avail_srp"],
        })
    csv_out = SCRIPT_DIR / "panel_loo_zeroshot.csv"
    pd.DataFrame(csv_rows).to_csv(csv_out, index=False, float_format="%.5f")
    print(f"Saved: {csv_out}")

    # ---- Plot: D_ZS with bootstrap CI bars, equal-weight primary ----
    genes = [r["gene"] for r in results]
    d_eq = [r["D_ZS_eq"] for r in results]
    d_eq_lo = [r["D_ZS_eq_lo"] for r in results]
    d_eq_hi = [r["D_ZS_eq_hi"] for r in results]
    d_n = [r["D_ZS_n"] for r in results]
    d_n_lo = [r["D_ZS_n_lo"] for r in results]
    d_n_hi = [r["D_ZS_n_hi"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(17, 9))
    y_pos = np.arange(len(genes))

    # Index of the first gene NOT in the greedy-tail (k=15 peak). Used to draw
    # the visual separator between the critical-panel block (top) and the
    # greedy-dropouts block (bottom). Genes are sorted by drop step DESC, so
    # the first 15 are critical; the remaining 10 are dropouts.
    n_critical_panel = 15
    sep_y = n_critical_panel - 0.5  # between rows 14 and 15 in 0-indexed y_pos

    _C_RED   = "#C62828"  # strict CI<0 under this scheme
    _C_AMBER = "#FBC02D"  # point D_ZS<0 but CI overlaps 0 (suggestive)
    _C_GREEN = "#388E3C"  # D_ZS>=0 (no evidence of criticality)

    def _color_for(r, key_d, key_crit):
        if r[key_crit]:
            return _C_RED
        if r[key_d] < 0:
            return _C_AMBER
        return _C_GREEN

    # Left panel: equal-weighted D_ZS with CI
    ax = axes[0]
    colors = [_color_for(r, "D_ZS_eq", "critical_eq") for r in results]
    err_lo = [max(0.0, d - lo) for d, lo in zip(d_eq, d_eq_lo)]
    err_hi = [max(0.0, hi - d) for d, hi in zip(d_eq, d_eq_hi)]
    ax.barh(y_pos, d_eq, color=colors, alpha=0.85, edgecolor="white",
            xerr=[err_lo, err_hi], capsize=2,
            error_kw=dict(ecolor="black", lw=0.7))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(genes, fontsize=8)
    ax.axvline(0, color="black", lw=0.9)
    ax.axhline(sep_y, color="black", ls=":", lw=1.2,
                label="Greedy-elim k=15 peak (15-crit panel above)")
    ax.set_xlabel("D_ZS (W.mean AUC change from removing gene, 95% CI)")
    ax.set_title(
        f"Equal-weight LOO D_ZS — {N_SEEDS}-seed ensemble + paired bootstrap CI\n"
        f"Baseline W.mean = {baseline_wmean_eq:.4f}  |  "
        f"Critical (CI<0): n={len(critical_eq)}"
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    for i, r in enumerate(results):
        if r["critical_eq"]:
            ax.text(d_eq_hi[i] + 0.001, y_pos[i], "*", va="center", fontsize=10,
                    color=_C_RED, fontweight="bold")

    # Right panel: sample-size weighted D_ZS with CI (sensitivity)
    ax = axes[1]
    colors2 = [_color_for(r, "D_ZS_n", "critical_n") for r in results]
    err_lo2 = [max(0.0, d - lo) for d, lo in zip(d_n, d_n_lo)]
    err_hi2 = [max(0.0, hi - d) for d, hi in zip(d_n, d_n_hi)]
    ax.barh(y_pos, d_n, color=colors2, alpha=0.85, edgecolor="white",
            xerr=[err_lo2, err_hi2], capsize=2,
            error_kw=dict(ecolor="black", lw=0.7))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(genes, fontsize=8)
    ax.axvline(0, color="black", lw=0.9)
    ax.axhline(sep_y, color="black", ls=":", lw=1.2)
    ax.set_xlabel("D_ZS (sample-size-weighted, 95% CI)")
    ax.set_title(
        f"Sample-size-weighted LOO D_ZS (sensitivity)\n"
        f"Baseline W.mean = {baseline_wmean_n:.4f}  |  "
        f"Critical (CI<0): n={len(critical_n)}"
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    for i, r in enumerate(results):
        if r["critical_n"]:
            ax.text(d_n_hi[i] + 0.001, y_pos[i], "*", va="center", fontsize=10,
                    color=_C_RED, fontweight="bold")

    # Single shared legend below the two panels
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=_C_RED,   edgecolor="white",
              label="Strict critical: 95% CI of D_ZS < 0 (red bars marked *)"),
        Patch(facecolor=_C_AMBER, edgecolor="white",
              label="Suggestive: point D_ZS < 0 but 95% CI overlaps 0"),
        Patch(facecolor=_C_GREEN, edgecolor="white",
              label="No evidence of criticality: D_ZS ≥ 0"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
                bbox_to_anchor=(0.5, -0.02), fontsize=9, frameon=False)

    plt.suptitle(
        f"Per-gene cross-cohort LOO D_ZS — 25-gene panel ({N_SEEDS}-seed ensemble, "
        f"paired bootstrap B={N_BOOTSTRAP})\n"
        f"GPL24676 (n=874) training → 4 CNS validation cohorts (N={_TOTAL_N})  |  "
        f"Genes sorted by greedy backward-elimination drop step (survivors top, "
        f"dropouts bottom); dotted line = 15-crit panel boundary",
        fontsize=10,
    )
    plt.tight_layout()
    png_out = SCRIPT_DIR / "panel_loo_zeroshot.png"
    plt.savefig(png_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_out}")


if __name__ == "__main__":
    main()
