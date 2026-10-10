"""Greedy backward elimination of the 25-gene panel, scored by pooled cross-validation
across every CNS cohort except the discovery cohort (blood excluded).

Units (equal weight each):
  GPL16791   donors absent from GPL24676 (518 samples), raw log1p, Ensembl base ID
  GSE76220   lumbar spinal cord LCM, log2 RPKM, HGNC symbol
  GSE122649  motor cortex, log1p counts, HGNC symbol
  SRP064478  spinal cord, log1p Salmon counts, Ensembl base ID
  GSE255683  lumbar spinal cord, log1p counts, Ensembl base ID
  GSE287256  cervical spinal cord, log1p normalised counts, Ensembl base ID
  GSE26927   ALS arm, cervical spinal cord array, HGNC symbol
  E-MTAB     E-MTAB-8635 (spinal cord) + E-MTAB-2325 (motor cortex) arrays, HGNC symbol;
             one unit (half weight each) because the two share 35 donors

Score (user decision 2026-10-04, findings.md Step 105d; it replaced the zero-shot score,
whose 12-gene optimum did not hold up under pooled CV): the nine datasets are merged
into one sample set (merge_cohorts) and scored by grouped 10-fold CV repeated 5 times
(folds by donor, stratified by cohort x class, out-of-fold scores averaged over the repeats;
the same splits for every candidate so comparisons are paired). One split proved too noisy
for the stopping rule to choose a size (Step 105d), hence the repeats.
Inside each fold every cohort is z-scored against its training controls (ctrl_z), genes a
platform lacks stay NaN, and sample weights give each unit equal total weight. The score
is the _W-weighted mean of the per-cohort AUCs of the out-of-fold predictions.

Algorithm: greedy backward elimination
  At each step, for each remaining gene g, compute
    D_CV(g) = W.mean_AUC(panel − {g}) − W.mean_AUC(panel)
  Drop the gene with the highest D_CV (most improvement or least harm).
  Continue until only 1 gene remains (full ranking).

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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ALS_DIR = Path(__file__).parents[1]
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(ALS_DIR))

from sklearn.metrics import roc_auc_score  # noqa: E402
from utils import PanelLGBMClassifier, load_dataset  # noqa: E402
from utils import selected_lgbm_params  # noqa: E402

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PROGRESS_LOG = SCRIPT_DIR / "iterative_panel_elimination_progress.log"
_CKPT = SCRIPT_DIR / "iterative_panel_elimination_ckpt.json"
# A checkpoint written under another score is never resumed.
_CRITERION = "pooled grouped 5x10-fold CV (averaged OOF), ctrl_z, unit weights"
_SRP_QUANT_DIR = ALS_DIR / "resources" / "SRP064478" / "quant"
_SRP_META = ALS_DIR / "resources" / "SRP064478" / "srr_metadata.tsv"
_GSE122649_DIR = ALS_DIR / "resources" / "GSE122649"
_GSE122649_RAW_TAR = _GSE122649_DIR / "GSE122649_RAW.tar"
_GSE122649_SOFT_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE122nnn/GSE122649/"
    "soft/GSE122649_family.soft.gz"
)

RANDOM_STATE = 42
N_REPEATS = 5  # repeats of the grouped 10-fold split; OOF scores averaged (Step 105e)
N_BOOTSTRAP = 2_000

_COMPARTMENTS: dict[str, tuple[str, ...]] = {
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

# Trimming units (user design, 2026-10-03): the discovery cohort proposes the panel;
# every other CNS cohort votes on it with equal weight. Cohort weights within a unit
# split the unit's weight.
_UNITS: dict[str, list[str]] = {
    "GPL16791": ["GPL16791"],
    "GSE76220": ["GSE76220"],
    "GSE122649": ["GSE122649"],
    "SRP064478": ["SRP064478"],
    "GSE255683": ["GSE255683"],
    "GSE287256": ["GSE287256"],
    "GSE26927": ["GSE26927"],
    "E-MTAB": ["E-MTAB-8635", "E-MTAB-2325"],
}
_W = {c: 1.0 / len(_UNITS) / len(cs) for cs in _UNITS.values() for c in cs}

# ---------------------------------------------------------------------------
# Progress logging
# ---------------------------------------------------------------------------
# Greedy backward elimination evaluates sum(25..2) = 324 candidate panels, each
# a zero-shot fit across 4 cohorts. Under the corrected colsample_bytree every
# fit is materially more expensive than before (all panel genes per tree rather
# than one), so the run is long and needs observable progress. stdout alone is
# not enough: `conda run` buffers it unless --no-capture-output is passed, which
# is why earlier run logs stayed empty. This writes a separate line-flushed file
# that is readable while the job is in flight regardless of how it was launched.

_PROGRESS_STATE: dict[str, object] = {"done": 0, "total": 0, "t0": 0.0, "session": 0}


def _progress_init(total_evals: int, done: int = 0, resumed: bool = False) -> None:
    """Set up progress accounting; append rather than truncate when resuming.

    Args:
        total_evals: Total candidate evaluations expected for a full run.
        done: Evaluations already completed (non-zero when resuming).
        resumed: If ``True``, append to the existing progress file instead of
            starting a new one, so a multi-day run keeps one continuous record.
    """
    import time

    _PROGRESS_STATE["done"] = done
    _PROGRESS_STATE["total"] = total_evals
    _PROGRESS_STATE["t0"] = time.time()
    _PROGRESS_STATE["session"] = 0
    header = (
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
        f"{'RESUME' if resumed else 'START'} "
        f"total_candidate_evaluations={total_evals} already_done={done}\n"
    )
    if resumed and _PROGRESS_LOG.exists():
        with open(_PROGRESS_LOG, "a") as fh:
            fh.write(header)
            fh.flush()
    else:
        _PROGRESS_LOG.write_text(header)


# ---------------------------------------------------------------------------
# Resumable checkpointing
# ---------------------------------------------------------------------------
# The greedy search is sum(2..25) = 324 candidate evaluations, each fitting four
# LightGBM models. Under the corrected colsample_bytree every fit searches all
# panel features rather than one, so a full run is a multi-day job. State is
# therefore checkpointed after EVERY candidate evaluation: an interruption loses
# at most one evaluation, and the run picks up mid-step exactly where it stopped.
#
# Writes are atomic (temp file + replace) so a kill during a write cannot leave
# a truncated checkpoint behind.


def _save_ckpt(
    active: list[int],
    current_wmean: float,
    step: int,
    history: list[dict],
    partial: dict[int, dict],
    done: int,
) -> None:
    """Atomically persist greedy state, including partial within-step results."""
    payload = {
        "active": [int(i) for i in active],
        "current_wmean": float(current_wmean),
        "step": int(step),
        "history": history,
        "partial": {str(k): v for k, v in partial.items()},
        "done": int(done),
        "criterion": _CRITERION,
    }
    tmp = _CKPT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(_CKPT)


def _load_ckpt() -> dict | None:
    """Return the saved greedy state, or ``None`` if there is no valid checkpoint."""
    if not _CKPT.exists():
        return None
    try:
        d = json.loads(_CKPT.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [warn] unreadable checkpoint ({exc}); starting fresh", flush=True)
        return None
    if d.get("criterion") != _CRITERION:
        print(
            f"  [warn] checkpoint scored by {d.get('criterion', 'zero-shot')!r}; starting fresh",
            flush=True,
        )
        return None
    d["partial"] = {int(k): v for k, v in d.get("partial", {}).items()}
    d["active"] = [int(i) for i in d["active"]]
    return d


def _progress(msg: str, tick: bool = False) -> None:
    """Append a timestamped progress line, with ETA once ticking has started.

    Args:
        msg: Message to record.
        tick: If ``True``, count this call as one completed candidate
            evaluation and append percentage complete plus an ETA.
    """
    import time

    now = time.time()
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    if tick:
        _PROGRESS_STATE["done"] = int(_PROGRESS_STATE["done"]) + 1
        _PROGRESS_STATE["session"] = int(_PROGRESS_STATE.get("session", 0)) + 1
        done = int(_PROGRESS_STATE["done"])
        total = int(_PROGRESS_STATE["total"]) or 1
        elapsed = now - float(_PROGRESS_STATE["t0"])
        # Rate must come from evaluations completed *this session*: on resume,
        # `done` includes prior work while `elapsed` does not, which would
        # otherwise make the ETA far too optimistic.
        rate = elapsed / max(int(_PROGRESS_STATE["session"]), 1)
        eta_s = rate * (total - done)
        line += (
            f"  [{done}/{total} = {100.0 * done / total:5.1f}%]"
            f"  elapsed={elapsed / 60:.1f}m"
            f"  eta={eta_s / 60:.1f}m"
        )
    with open(_PROGRESS_LOG, "a") as fh:
        fh.write(line + "\n")
        fh.flush()
    print(line, flush=True)


# ---------------------------------------------------------------------------
# Fast helpers
# ---------------------------------------------------------------------------


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
    # CTD compartment regression is discovery-side only; cross-cohort transfer now
    # uses raw log1p for every cohort, so the memory-heavy CTD-residualised matrix
    # is no longer built. Returns (X_log, y, X_log, feat) for signature compatibility.
    (ds,) = load_dataset(
        "GSE153960", platform="GPL24676", resources_dir=ALS_DIR / "resources"
    )
    X_log = np.log1p(ds.X.values.astype(np.float32))
    y = ds.y.values.astype(int)
    feat = list(ds.X.columns)
    return X_log, y, X_log, feat


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

    (ds16,) = load_dataset(
        "GSE153960", platform="GPL16791", resources_dir=ALS_DIR / "resources"
    )
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
    # GSE76220 is ALREADY log-scale (log2 RPKM: range -6.2..10.6, 63% negative
    # despite the 'RPKM' filename). A further log1p would null every value
    # <= -1 (77% of panel genes). Per-gene z-scoring downstream absorbs the
    # log-base and length-normalisation differences vs log1p(counts) training.
    X_log = ds.X.values.astype(np.float32)
    return X_log, list(ds.X.columns), ds.y.values.astype(int)


def _parse_gse122649_soft() -> dict[str, str]:
    from utils import geo_soft_text

    content = geo_soft_text("GSE122649")
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


def _subject_ids(platform: str) -> np.ndarray:
    (ds,) = load_dataset(
        "GSE153960", platform=platform, resources_dir=ALS_DIR / "resources"
    )
    return ds.meta.loc[ds.y.index, "subject id"].astype(str).to_numpy()


def _trim_cohorts(
    sym25: list[str], feat25_bases: list[str], panel_train_cols: list[int]
) -> dict[str, tuple[np.ndarray, list[int], np.ndarray, np.ndarray]]:
    """Every trimming cohort as {name: (matrix over available panel genes, panel indices, labels, donor ids)}.

    Names double as utils.zero_shot_scale keys. Shared with greedy_significance_stop.py.
    """
    from new_cohort_validation import _emtab, _gse26927_arms, _gse255683, _gse287256

    def by_key(
        X: np.ndarray, vocab: list[str], y: np.ndarray, keys: list[str]
    ) -> tuple:
        vm = {k: j for j, k in enumerate(vocab)}
        avail = [i for i, k in enumerate(keys) if k in vm]
        M = X[:, [vm[keys[i]] for i in avail]] if avail else np.empty((len(y), 0))
        return M.astype(np.float32), avail, np.asarray(y).astype(int)

    (ds16,) = load_dataset(
        "GSE153960", platform="GPL16791", resources_dir=ALS_DIR / "resources"
    )
    sid16 = ds16.meta.loc[ds16.y.index, "subject id"].astype(str).to_numpy()
    keep = ~np.isin(sid16, _subject_ids("GPL24676"))
    X16 = np.log1p(ds16.X.values[keep][:, panel_train_cols].astype(np.float32))
    cohorts = {
        "GPL16791": (
            X16,
            list(range(len(sym25))),
            ds16.y.values[keep].astype(int),
            sid16[keep],
        )
    }
    cohorts["GSE76220"] = by_key(*_load_gse76220(), sym25)
    cohorts["GSE122649"] = by_key(*_load_gse122649(), sym25)
    cohorts["SRP064478"] = by_key(*_load_srp064478(), feat25_bases)
    for c in (
        _gse255683(),
        _gse287256(),
        _gse26927_arms()["ALS"],
        _emtab("E-MTAB-8635"),
        _emtab("E-MTAB-2325"),
    ):
        v = by_key(c.X, c.genes, c.y, sym25 if c.by_symbol else feat25_bases)
        d = _emtab_donors(c)
        cohorts[c.scale_id] = v if d is None else (*v, d)
    assert set(cohorts) == set(_W), (set(cohorts), set(_W))
    for c, v in cohorts.items():
        if len(v) == 3:
            cohorts[c] = (*v, np.array([f"{c}_{i}" for i in range(len(v[2]))]))
        assert len(cohorts[c][3]) == len(v[2]), c
    for c, (M, av, y, _) in cohorts.items():
        print(
            f"  {c}: n={len(y)} ({int(y.sum())} ALS)  {len(av)}/{len(sym25)} panel genes  w={_W[c]:.4f}"
        )
    return cohorts


def _emtab_donors(c) -> np.ndarray | None:
    """Donor id per row of an E-MTAB cohort, looked up by array name (shared patients get one id).

    Returns None for any other cohort. Rows follow c.samples, so arrays dropped by the
    array QC (new_cohort_validation.QC_MIN_R) cannot misalign donors.
    """
    if not c.scale_id.startswith("E-MTAB"):
        return None
    s = pd.read_csv(ALS_DIR / "resources" / c.scale_id / f"{c.scale_id}.sdrf.txt", sep="	")
    if c.scale_id == "E-MTAB-2325":
        ids = ("EM" + s["Characteristics[Patient code]"].astype(str)).to_numpy()
        names = s["Scan Name"].str.strip()
    else:
        code = s["Characteristics[patient code (e-mtab-2325)]"]
        ids = np.where(code.notna(), "EM" + code.fillna(0).astype(int).astype(str),
                       "EM8635_" + s["Characteristics[individual]"])
        names = s["Array Data File"].str.strip()
    return pd.Series(ids, index=names.to_numpy())[c.samples].to_numpy()


@dataclass
class Merged:
    """The trimming cohorts stacked into one sample set over the 25 panel columns."""

    X: np.ndarray  # samples x 25; NaN where the platform lacks the gene
    y: np.ndarray
    cid: np.ndarray  # cohort name per sample
    donor: np.ndarray  # donor id per sample (shared E-MTAB patients share one)
    w: np.ndarray  # equal total weight per unit (_W), mean 1


def merge_cohorts(cohorts: dict[str, tuple], n_genes: int) -> Merged:
    """Stack the _trim_cohorts output into one sample set."""
    vals = list(cohorts.values())
    X = np.full((sum(len(v[2]) for v in vals), n_genes), np.nan, np.float32)
    i = 0
    for M, av, y, _ in vals:
        X[i : i + len(y), av] = M
        i += len(y)
    cid = np.concatenate([[c] * len(v[2]) for c, v in cohorts.items()])
    w = np.array([_W[c] / len(cohorts[c][2]) for c in cid])
    return Merged(
        X,
        np.concatenate([v[2] for v in vals]),
        cid,
        np.concatenate([v[3] for v in vals]),
        w / w.mean(),
    )


def cv_folds(m: Merged, n_splits: int = 10, n_repeats: int = N_REPEATS) -> np.ndarray:
    """Fold ids, shape (n_repeats, samples): grouped by donor, stratified by cohort x class."""
    from sklearn.model_selection import StratifiedGroupKFold

    strata = np.char.add(m.cid.astype(str), m.y.astype(str))
    fold = np.empty((n_repeats, len(m.y)), int)
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits, shuffle=True, random_state=RANDOM_STATE + r)
        for f, (_, te) in enumerate(cv.split(m.X, strata, m.donor)):
            fold[r, te] = f
    return fold


def ctrl_z(
    X: np.ndarray, y: np.ndarray, cid: np.ndarray, train: np.ndarray
) -> np.ndarray:
    """Z-score each cohort against its training controls, scaled by the pooled within-class SD."""
    Z = np.empty_like(X)
    for c in np.unique(cid):
        r, t = cid == c, (cid == c) & train
        mu0, mu1 = (
            np.nanmean(X[t & (y == 0)], axis=0),
            np.nanmean(X[t & (y == 1)], axis=0),
        )
        sd = np.nanstd(X[t] - np.where(y[t, None] == 1, mu1, mu0), axis=0, ddof=2)
        Z[r] = (X[r] - mu0) / np.where(sd > 0, sd, 1.0)
    return Z


def pooled_oof(
    m: Merged, cols: list[int], fold: np.ndarray, params: dict
) -> np.ndarray:
    """Out-of-fold scores on the merged cohorts, averaged over the repeats (rows) of *fold*.

    The scaling is refitted on each fold's training rows. Pass ``fold=m.donor[None]``
    for leave-one-donor-out.
    """
    X, s = m.X[:, cols], np.zeros(len(m.y))
    for rep in np.atleast_2d(fold):
        for f in np.unique(rep):
            te = rep == f
            Z = ctrl_z(X, m.y, m.cid, ~te)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = PanelLGBMClassifier(**params).fit(
                    Z[~te], m.y[~te], sample_weight=m.w[~te]
                )
            s[te] += clf.predict_proba(Z[te])[:, 1]
    return s / len(np.atleast_2d(fold))


def _wmean_auc(
    panel_idx: list[int], m: Merged, fold: np.ndarray, params: dict
) -> tuple[float, dict[str, float]]:
    """Return (weighted mean pooled-CV AUC, per-cohort AUC) for the panel subset *panel_idx*."""
    s = pooled_oof(m, sorted(panel_idx), fold, params)
    aucs = {c: float(roc_auc_score(m.y[m.cid == c], s[m.cid == c])) for c in _W}
    return sum(_W[c] * a for c, a in aucs.items()), aucs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import warnings as _w

    import matplotlib.pyplot as plt
    import pandas as pd

    _w.filterwarnings("ignore", category=UserWarning)

    print("=" * 70)
    print("Iterative Panel Elimination — greedy backward, weighted pooled-CV AUC")
    print("Units: " + ", ".join(f"{u}={'+'.join(cs)}" for u, cs in _UNITS.items()))
    print("=" * 70)

    params_raw = selected_lgbm_params()
    params = dict(params_raw, colsample_bytree=1.0, n_jobs=-1, verbose=-1)

    # -----------------------------------------------------------------------
    # Load panel
    # -----------------------------------------------------------------------
    df_panel = pd.read_csv(_PANEL_CSV)
    feat_col = next(
        c for c in df_panel.columns if "feature" in c.lower() or "ensg" in c.lower()
    )
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
    feat_train_base_map: dict[str, int] = {
        f.split(".")[0]: j for j, f in enumerate(feat_train)
    }

    panel_train_cols: list[int] = [feat_train_base_map[b] for b in feat25_bases]
    del X_train_ctd, X_train_log

    print("Loading trimming cohorts ...")
    cohorts = _trim_cohorts(sym25, feat25_bases, panel_train_cols)
    merged = merge_cohorts(cohorts, len(sym25))
    fold = cv_folds(merged)
    _shared = dict(m=merged, fold=fold, params=params)

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
    history.append(
        {
            "step": 0,
            "panel_size": 25,
            "dropped": None,
            "D_CV": None,
            "wmean": baseline_wmean,
            "cohort": dict(baseline_cohort),
        }
    )

    current_wmean = baseline_wmean
    step = 0

    # Greedy drops one gene per step, so total work is sum(k) for k = len(active)
    # down to 2 candidate evaluations.
    _n0 = len(active)
    _total_evals = sum(range(2, _n0 + 1))

    # ---- resume from checkpoint if one exists ----
    partial: dict[int, dict] = {}
    _ck = _load_ckpt()
    if _ck is not None:
        active = _ck["active"]
        current_wmean = _ck["current_wmean"]
        step = _ck["step"]
        history = _ck["history"]
        partial = _ck["partial"]
        _progress_init(_total_evals, done=_ck.get("done", 0), resumed=True)
        _progress(
            f"RESUMED at step {step}: panel k={len(active)}, "
            f"W.mean={current_wmean:.4f}, "
            f"{len(partial)} candidate(s) already evaluated in this step"
        )
    else:
        _progress_init(_total_evals)
        _progress(f"baseline W.mean={baseline_wmean:.4f} over {_n0} genes")

    while len(active) > 1:
        # On resume mid-step, `step` and `partial` are already set for the step
        # in progress, so only advance when starting a genuinely new step.
        if not partial:
            step += 1
        n_remaining = len(active)
        _progress(
            f"Step {step}: evaluating {n_remaining} candidates (panel k={n_remaining})"
            + (f" [{len(partial)} cached]" if partial else "")
        )

        best_d = -np.inf
        best_g = -1
        best_wmean = current_wmean
        best_cohort: dict[str, float] = {}

        for g in active:
            if g in partial:
                # already evaluated before the interruption
                wmean_trial = partial[g]["wmean"]
                cohort_trial = partial[g]["cohort"]
            else:
                trial = [i for i in active if i != g]
                wmean_trial, cohort_trial = _wmean_auc(trial, **_shared)
                partial[g] = {"wmean": wmean_trial, "cohort": dict(cohort_trial)}
                _progress(
                    f"    drop {sym25[g]:>14s}: trial W.mean={wmean_trial:.4f}  "
                    f"D_CV={wmean_trial - current_wmean:+.4f}",
                    tick=True,
                )
                # Checkpoint after EVERY evaluation: losing one fit is acceptable,
                # losing a whole step is not.
                _save_ckpt(
                    active,
                    current_wmean,
                    step,
                    history,
                    partial,
                    int(_PROGRESS_STATE["done"]),
                )
            d = wmean_trial - current_wmean  # positive = improvement from dropping g
            if d > best_d:
                best_d = d
                best_g = g
                best_wmean = wmean_trial
                best_cohort = cohort_trial

        _progress(
            f"  Step {step} DONE -> drop {sym25[best_g]}  D_CV={best_d:+.4f}  "
            f"new W.mean={best_wmean:.4f}  panel k={len(active) - 1}"
        )

        if best_d <= 0.0:
            _progress(f"  D_CV <= 0 (critical threshold crossed at k={len(active)})")

        # Drop the best candidate (always continue to exhaustion)
        active.remove(best_g)
        current_wmean = best_wmean
        history.append(
            {
                "step": step,
                "panel_size": len(active),
                "dropped": sym25[best_g],
                "D_CV": best_d,
                "wmean": current_wmean,
                "cohort": dict(best_cohort),
            }
        )
        # Step complete: clear within-step results and checkpoint the new panel.
        partial = {}
        _save_ckpt(
            active,
            current_wmean,
            step,
            history,
            partial,
            int(_PROGRESS_STATE["done"]),
        )
        print(
            f"  Dropped: {sym25[best_g]}  → panel now {len(active)} genes  "
            f"W.mean={current_wmean:.4f}"
        )
        for cname, auc in best_cohort.items():
            print(f"    {cname}: {auc:.4f}")

    # -----------------------------------------------------------------------
    # Final panel
    # -----------------------------------------------------------------------
    final_genes = [sym25[i] for i in active]
    final_feats = [feat25[i] for i in active]
    print(f"\n{'=' * 70}")
    print(f"Final minimal panel: {len(active)} genes")
    print(f"  Symbols: {', '.join(final_genes)}")
    print(f"  ENSGs:   {', '.join([feat25[i] for i in active])}")
    print(f"  W.mean pooled-CV AUC = {current_wmean:.4f}")

    # Full evaluation with CI for final panel
    print("\nComputing per-cohort pooled-CV AUC with 95% CI for final panel ...")
    s_final = pooled_oof(merged, sorted(active), fold, params)
    final_results: dict[str, tuple[float, float, float, int]] = {}
    for c in cohorts:
        r = merged.cid == c
        lo, hi = _bootstrap_auc(merged.y[r], s_final[r])
        n_genes = len([pi for pi in cohorts[c][1] if pi in set(active)])
        final_results[c] = (
            float(roc_auc_score(merged.y[r], s_final[r])),
            lo,
            hi,
            n_genes,
        )

    # -----------------------------------------------------------------------
    # Text output
    # -----------------------------------------------------------------------
    names = list(cohorts)
    lines = [
        "Iterative Panel Elimination — greedy backward, weighted pooled-CV AUC",
        "=" * 70,
        "",
        "Cohort weights (equal per unit; units: "
        + ", ".join(f"{u}={'+'.join(cs)}" for u, cs in _UNITS.items())
        + "):",
        *[
            f"  {c:<12} n={len(cohorts[c][2]):>4}  w={_W[c]:.4f}  genes "
            f"{len(cohorts[c][1])}/{len(sym25)}: {[sym25[i] for i in cohorts[c][1]]}"
            for c in names
        ],
        "",
        "Baseline (25-gene panel):",
        f"  W.mean AUC = {baseline_wmean:.4f}",
        *[f"  {k}: {v:.4f}" for k, v in baseline_cohort.items()],
        "",
        "Elimination log:",
        f"  {'Step':>4}  {'Dropped':>14}  {'D_CV':>8}  {'Panel_size':>10}  {'W.mean':>8}  "
        + "  ".join(f"{k:>11}" for k in names),
        "-" * 150,
    ]
    for h in history[1:]:  # skip step 0 (baseline)
        cohort_str = "  ".join(
            f"{h['cohort'].get(k, float('nan')):>11.4f}" for k in names
        )
        lines.append(
            f"  {h['step']:>4}  {h['dropped']:>14}  {h['D_CV']:>+8.4f}  "
            f"{h['panel_size']:>10}  {h['wmean']:>8.4f}  {cohort_str}"
        )
    lines += [
        "",
        f"Final minimal panel: {len(active)} genes",
        f"  Symbols: {', '.join(final_genes)}",
        f"  ENSGs  : {', '.join(final_feats)}",
        f"  W.mean pooled-CV AUC = {current_wmean:.4f}",
        "",
        "Per-cohort pooled-CV AUC for final panel (95% CI):",
        f"  {'Cohort':<12} {'n':>5}  {'Genes':>6}  {'AUC':>6}  95% CI",
        *[
            f"  {c:<12} {len(cohorts[c][2]):>5}  {r[3]:>6}  {r[0]:.4f}  [{r[1]:.3f}, {r[2]:.3f}]"
            for c, r in final_results.items()
        ],
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
    d_values = [h["D_CV"] for h in steps_recorded]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: W.mean AUC vs panel size
    ax = axes[0]
    ax.plot(panel_sizes, wmeans, "o-", color="#1565C0", lw=2, ms=6)
    ax.axvline(
        len(active),
        color="red",
        ls="--",
        lw=1.2,
        label=f"Final panel (k={len(active)})",
    )
    ax.set_xlabel("Panel size (k genes)")
    ax.set_ylabel("Weighted mean pooled-CV AUC")
    ax.set_title("Panel size vs. weighted pooled-CV AUC")
    ax.set_xticks(range(len(active) - 1, 26))
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # Right: D_CV per dropped gene (in order of elimination)
    ax = axes[1]
    colors_bar = ["#388E3C" if d > 0 else "#C62828" for d in d_values]
    ax.barh(range(len(dropped_names)), d_values, color=colors_bar, alpha=0.85)
    ax.set_yticks(range(len(dropped_names)))
    ax.set_yticklabels(dropped_names, fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("D_CV (W.mean AUC improvement from dropping gene)")
    ax.set_title("Genes dropped (order of elimination)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    plt.suptitle(
        f"Greedy backward elimination — 25→{len(active)}-gene minimal panel\n"
        f"W.mean AUC: {baseline_wmean:.4f} → {current_wmean:.4f}  "
        f"({len(_UNITS)} cohort units, equal weight)",
        fontsize=9,
    )
    plt.tight_layout()
    png_out = SCRIPT_DIR / "iterative_panel_elimination.png"
    plt.savefig(png_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_out}")


if __name__ == "__main__":
    main()
