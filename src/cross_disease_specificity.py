# ruff: noqa: E402
"""Cross-disease specificity of the ALS panel: AD, FTLD-TDP, HD and PD.

Motivated by Zhou et al. 2026 (Cell 189:1-17, doi:10.1016/j.cell.2026.06.013),
which reports a TOP1-mediated somatic-damage mechanism shared by ALS, FTD and
Alzheimer's disease.  If an upstream mechanism is shared, a transcriptomic ALS
signature invites the objection that it reports pan-neurodegenerative change.
Step 10 (`disease_specificity.py`) only tested groups internal to GSE153960
(ALS-FTD and a mixed, un-subtyped FTLD-TDP/PSP group); Alzheimer's, Parkinson's
and Huntington's disease had never been tested.

Design
------
A raw cross-platform *score* comparison is not a valid specificity test: the
panel already attenuates 0.96 -> 0.78 from GPL24676 to GPL16791 on platform
grounds alone.  Each disease is therefore scored **against its own in-cohort
controls**, and the reported quantity is the within-cohort AUC.  AUC ~ 0.5 means
the panel is blind to that disease (specificity confirmed); a high AUC means the
panel also detects it.  An ALS cohort (GSE153960 GPL16791, n=636) is processed
through the identical pipeline as a positive control.

Two harmonisation modes are reported for every cohort:
  canonical — StandardScaler fit on the matched GPL24676 training columns and
              applied to the target (the recipe used throughout this study,
              `lr_vs_lgbm_zeroshot.py`).  Scale-valid only when both sides are
              log1p counts.
  zscore    — per-gene standardisation fit *within* each cohort separately.
              Scale-invariant, so it is the primary mode for the microarray
              cohort where `canonical` is not scale-valid.

Model: **LightGBM is the primary model**, consistent with the rest of this study.
LR-L2 is reported alongside it purely as a sensitivity baseline, in the same role
it plays in `lr_baseline.py` / supplementary `sec:lr_baseline`.  The two disagree
materially here (LR-L2 gives higher non-ALS AUCs in every cohort), so both are
always printed and plotted and the disagreement is treated as a finding rather
than resolved by picking one.

ALS vs other neurological disease
---------------------------------
Disease-vs-own-controls does not answer "can the panel tell ALS apart from
another neurodegenerative disease?".  A raw cross-cohort comparison of ALS cases
against, say, HD cases would confound disease with study and platform.  This
script therefore also computes a **control-referenced z-score** per cohort,
z = (score - mean(score | that cohort's controls)) / sd(score | those controls),
which expresses every case relative to its own study's baseline, and then
compares the ALS z-distribution against each other disease's z-distribution
(AUC + Mann-Whitney U).  This is batch-robust in a way the raw scores are not,
though it still assumes the control groups are comparable across studies.
The within-cohort equivalent already exists as Step 10 (ALS vs the held-out
FTLD/PSP group inside GSE153960, AUC = 0.9067), which remains the cleanest
ALS-vs-other-disease estimate because it is free of any cross-study component.

Cohorts
-------
  GSE125583  Alzheimer's disease   AD 219 / ctrl 70   fusiform gyrus  GPL16791
  GSE13162   FTLD-TDP (FTLD-U)     39 / ctrl 17       frontal/hippo/cereb GPL571
  GSE64810   Huntington's disease  HD 20 / ctrl 49    BA9             GPL11154
  GSE68719   Parkinson's disease   PD 29 / ctrl 44    BA9             GPL11154
  GSE153960  ALS (positive ctrl)   GPL16791 subset    mixed CNS       GPL16791

No open post-mortem CNS PSP series exists, so the tau-only negative control
remains outstanding (see findings.md Step 33).

Caveat recorded in the report: GSE64810 and GSE68719 share their control
samples (both re-analyse the same neurologically-normal GSMs), so the HD and PD
results are not statistically independent of one another.

Outputs
-------
  cross_disease_specificity.csv             — sensitivity matrix
  cross_disease_specificity_statistics.txt  — full text report
  cross_disease_specificity.png             — violins + ROC + Braak + region
"""

from __future__ import annotations

import gzip
import re
import sys
import tarfile
import urllib.request
import warnings
from pathlib import Path

import numpy as np

ALS_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ALS_DIR))

from utils import critical_panel_label, critical_panel_symbols, load_dataset

# Panel size is derived, so every reader-facing label must be too.
_CRIT_LABEL = critical_panel_label()


SCRIPT_DIR = Path(__file__).parent
RESOURCES_DIR = ALS_DIR / "resources"

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"

PREFIX = "cross_disease_specificity"
RANDOM_STATE = 42
N_BOOTSTRAP = 2_000
N_SPLITS = 5  # matches disease_specificity.py / all CV in this study

# ---------------------------------------------------------------------------
# colsample_bytree feature-space correction
# ---------------------------------------------------------------------------
# lgbm_top500_best_params.json was optimised by BayesianOptimizer on the
# top-500 SHAP feature space (biomarker_discovery_top500.py slices to TOP_N=500
# before optimising), where colsample_bytree = 0.0667 selects ~33 features per
# tree.  colsample_bytree is a *fraction*, so reusing it verbatim on a small
# panel silently changes what the model does: on 25 or 15 genes it floors to
# ONE feature per tree, removing every feature interaction and turning the
# ensemble into an additive model.
#
# The fix needs no re-optimisation: convert the tuned fraction back to the
# absolute number of features it encodes, then re-express that for the actual
# feature space.  For any panel of <= 33 genes this saturates at 1.0.
TUNED_FEATURE_SPACE = 500


def _adjust_colsample(params: dict, n_features: int) -> dict:
    """Re-express ``colsample_bytree`` for a feature space of *n_features*.

    Args:
        params: Hyperparameters as tuned on ``TUNED_FEATURE_SPACE`` features.
        n_features: Number of features actually being modelled.

    Returns:
        A copy of *params* with ``colsample_bytree`` set to
        ``min(1.0, k / n_features)``, where ``k`` is the absolute
        features-per-tree count the tuned fraction encodes.
    """
    out = dict(params)
    k = params["colsample_bytree"] * TUNED_FEATURE_SPACE
    out["colsample_bytree"] = float(min(1.0, k / max(n_features, 1)))
    return out


# Critical panel (greedy backward elimination peak, Steps 24-26)
CRITICAL_PANEL = critical_panel_symbols()

# Entrez IDs for the protein-coding panel genes. Copied verbatim from
# lr_vs_lgbm_zeroshot.py (_PC_ENTREZ) so Entrez-keyed cohorts are matched
# identically to the established GSE234297 handling.
_PC_ENTREZ: dict[str, int] = {
    "MECOM": 2122,
    "SERTAD1": 29950,
    "FCN3": 8547,
    "PROS1": 5627,
    "ANGPT2": 285,
    "EMP1": 2012,
    "TINAGL1": 64129,
    "CKMT2": 1160,
    "VWF": 7450,
    "CLDN5": 7122,
    "NR4A1": 3164,
    "SOHLH2": 54937,
    "HEXB": 3074,
    "MCEE": 84693,
    "SLC37A2": 219855,
}

# Symbol aliases needed for microarray platforms annotated to older names
SYMBOL_ALIASES: dict[str, list[str]] = {
    "MECOM": ["MECOM", "EVI1"],
    "TINAGL1": ["TINAGL1", "LCN7", "TINAGL"],
    "SLC37A2": ["SLC37A2"],
    "SERTAD1": ["SERTAD1", "TRIP-Br1", "SEI1"],
}

_GPL571_ANNOT_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
    "?acc=GPL571&targ=self&form=text&view=data"
)

COHORT_META: dict[str, dict[str, str]] = {
    "GSE125583": {
        "disease": "Alzheimer's disease",
        "tissue": "fusiform gyrus",
        "platform": "GPL16791",
        "assay": "bulk RNA-seq",
        "id_type": "entrez",
    },
    "GSE13162": {
        "disease": "FTLD-TDP (FTLD-U)",
        "tissue": "frontal / hippocampus / cerebellum",
        "platform": "GPL571",
        "assay": "microarray (HG-U133A_2)",
        "id_type": "symbol",
    },
    "GSE64810": {
        "disease": "Huntington's disease",
        "tissue": "BA9 prefrontal cortex",
        "platform": "GPL11154",
        "assay": "bulk RNA-seq",
        "id_type": "ensembl",
    },
    "GSE68719": {
        "disease": "Parkinson's disease",
        "tissue": "BA9 prefrontal cortex",
        "platform": "GPL11154",
        "assay": "bulk RNA-seq",
        "id_type": "ensembl",
    },
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _download(url: str, dest: Path, desc: str = "") -> Path:
    """Download *url* to *dest* if not already present, with retries."""
    import time

    if dest.exists() and dest.stat().st_size > 1_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(4):
        try:
            print(f"  downloading {desc or dest.name} (attempt {attempt + 1}) ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(req, timeout=900) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.replace(dest)
            return dest
        except Exception as exc:  # pragma: no cover - network dependent
            last = exc
            time.sleep(6)
    raise RuntimeError(f"download failed for {url}: {last}")


def _series_matrix_meta(path: Path) -> dict[str, list[str]]:
    """Return every ``!Sample_*`` header row of a series matrix as a value list."""
    out: dict[str, list[str]] = {}
    with gzip.open(path, "rt", errors="ignore") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!Sample_"):
                continue
            key = line.split("\t", 1)[0].lstrip("!")
            vals = re.findall(r'"([^"]*)"', line)
            if not vals:
                continue
            out.setdefault(key, [])
            if key in out and out[key]:
                out[f"{key}__{len(out)}"] = vals
            else:
                out[key] = vals
    return out


def _sample_char_rows(path: Path) -> list[list[str]]:
    """Return all ``!Sample_characteristics_ch1`` rows in file order."""
    rows: list[list[str]] = []
    with gzip.open(path, "rt", errors="ignore") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                break
            if line.startswith("!Sample_characteristics_ch1"):
                rows.append(re.findall(r'"([^"]*)"', line))
    return rows


def _bootstrap_auc(
    y: np.ndarray, scores: np.ndarray, n: int, seed: int = RANDOM_STATE
) -> tuple[float, float, float]:
    """Return (AUC, lower 95% CI, upper 95% CI) by stratified bootstrap."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    point = float(roc_auc_score(y, scores))
    aucs: list[float] = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], scores[idx]))
    if not aucs:
        return point, float("nan"), float("nan")
    a = np.asarray(aucs)
    return point, float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


# ---------------------------------------------------------------------------
# Cohort loaders — each returns (expr DataFrame [samples x gene_key], y, meta)
# ---------------------------------------------------------------------------


def _load_gse125583() -> tuple["object", np.ndarray, "object"]:
    """Alzheimer's disease, fusiform gyrus (AD 219 / control 70)."""
    import pandas as pd

    d = RESOURCES_DIR / "GSE125583"
    mat = _download(
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE125nnn/GSE125583/matrix/"
        "GSE125583_series_matrix.txt.gz",
        d / "matrix" / "GSE125583_series_matrix.txt.gz",
        "GSE125583 series matrix",
    )
    tar_path = _download(
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE125nnn/GSE125583/suppl/"
        "GSE125583_RAW.tar",
        d / "suppl" / "GSE125583_RAW.tar",
        "GSE125583_RAW.tar (92 MB)",
    )

    meta_rows = _sample_char_rows(mat)
    gsms = _series_matrix_meta(mat)["Sample_geo_accession"]
    diag = next(r for r in meta_rows if r and r[0].startswith("diagnosis"))
    braak = next((r for r in meta_rows if r and r[0].startswith("braak")), None)

    label = {
        g: (1 if "alzheimer" in v.lower() else 0)
        for g, v in zip(gsms, [x.split(":", 1)[1].strip() for x in diag])
    }
    braak_map: dict[str, float] = {}
    if braak is not None:
        roman = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}
        for g, v in zip(gsms, [x.split(":", 1)[1].strip() for x in braak]):
            braak_map[g] = float(roman.get(v, np.nan))

    # Per-GSM count files inside the tar
    series: dict[str, "pd.Series"] = {}
    with tarfile.open(tar_path) as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        print(f"  GSE125583_RAW.tar: {len(members)} member files")
        for m in members:
            gsm = m.name.split("_")[0]
            fh = tf.extractfile(m)
            if fh is None:
                continue
            raw = fh.read()
            if m.name.endswith(".gz"):
                raw = gzip.decompress(raw)
            txt = raw.decode("utf8", "ignore")
            # Per-sample TSV: comment lines, then ID_REF / VALUE / count.
            # ID_REF is an Entrez Gene ID; VALUE is size-factor-adjusted nRPKM.
            idx: list[str] = []
            val: list[float] = []
            for line in txt.splitlines():
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                try:
                    v = float(parts[1])
                except ValueError:
                    continue  # the ID_REF/VALUE/count header row
                idx.append(parts[0].strip())
                val.append(v)
            if idx:
                series[gsm] = pd.Series(val, index=idx, dtype=np.float32)

    expr = pd.DataFrame(series).T  # samples x genes
    expr = expr.loc[[g for g in expr.index if g in label]]
    y = np.array([label[g] for g in expr.index], dtype=int)
    meta = pd.DataFrame(
        {"braak": [braak_map.get(g, np.nan) for g in expr.index]}, index=expr.index
    )
    print(f"  GSE125583: expr={expr.shape}  AD={y.sum()}  ctrl={(y == 0).sum()}")
    return expr, y, meta


def _load_gse13162() -> tuple["object", np.ndarray, "object"]:
    """FTLD-U (FTLD-TDP) microarray cohort — labels from sample titles."""
    import pandas as pd

    d = RESOURCES_DIR / "GSE13162"
    mat = _download(
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE13nnn/GSE13162/matrix/"
        "GSE13162_series_matrix.txt.gz",
        d / "matrix" / "GSE13162_series_matrix.txt.gz",
        "GSE13162 series matrix",
    )
    annot = d / "GPL571_annot.txt"
    if not annot.exists():
        _download(_GPL571_ANNOT_URL, annot, "GPL571 platform annotation")

    hdr = _series_matrix_meta(mat)
    titles = hdr["Sample_title"]
    gsms = hdr["Sample_geo_accession"]

    label: dict[str, int] = {}
    group: dict[str, str] = {}
    region: dict[str, str] = {}
    for g, t in zip(gsms, titles):
        sid = t.split(" ")[0]
        tl = t.lower()
        if sid.startswith("N"):
            label[g], group[g] = 0, "Control"
        elif sid.startswith("P"):
            label[g], group[g] = 1, "FTLD-U (progranulin)"
        elif sid.startswith("S"):
            label[g], group[g] = 1, "FTLD-U (sporadic)"
        else:
            continue
        if "frontal" in tl:
            region[g] = "frontal"
        elif "hippocamp" in tl:
            region[g] = "hippocampus"
        elif "cerebel" in tl:  # covers the 'cerebelum' typo in N5B
            region[g] = "cerebellum"
        else:
            region[g] = "unknown"

    # Expression table (probe x sample, log2 intensities)
    probe_df = pd.read_csv(
        mat,
        sep="\t",
        comment="!",
        index_col=0,
        compression="gzip",
        low_memory=False,
    )
    probe_df = probe_df.apply(pd.to_numeric, errors="coerce").dropna(how="all")

    # Probe -> symbol, collapsing to the highest-mean probe per symbol
    probe_to_sym: dict[str, str] = {}
    txt = annot.read_text(errors="ignore").split("\n")
    hdr_i = next(i for i, ln in enumerate(txt) if ln.startswith("ID\t"))
    cols = txt[hdr_i].split("\t")
    sym_i = cols.index("Gene Symbol")
    for ln in txt[hdr_i + 1 :]:
        f = ln.split("\t")
        if len(f) <= sym_i:
            continue
        syms = [s.strip() for s in f[sym_i].split("///") if s.strip()]
        if syms:
            probe_to_sym[f[0]] = syms[0]

    probe_df = probe_df.loc[probe_df.index.isin(probe_to_sym)]
    syms_series = pd.Series(
        [probe_to_sym[p] for p in probe_df.index], index=probe_df.index
    )
    means = probe_df.mean(axis=1)
    best = means.groupby(syms_series).idxmax()
    collapsed = probe_df.loc[best.values]
    collapsed.index = best.index

    expr = collapsed.T  # samples x symbols
    expr = expr.loc[[g for g in expr.index if g in label]]
    y = np.array([label[g] for g in expr.index], dtype=int)
    meta = pd.DataFrame(
        {
            "group": [group[g] for g in expr.index],
            "region": [region[g] for g in expr.index],
        },
        index=expr.index,
    )
    print(
        f"  GSE13162: expr={expr.shape}  FTLD-U={y.sum()}  ctrl={(y == 0).sum()}  "
        f"regions={dict(meta['region'].value_counts())}"
    )
    return expr, y, meta


def _load_deseq_cohort(
    gse: str, url: str, fname: str, disease_prefix: str, has_symbol_col: bool
) -> tuple["object", np.ndarray, "object"]:
    """Load a DESeq2-normalised counts cohort keyed by sample-name prefix."""
    import pandas as pd

    d = RESOURCES_DIR / gse
    path = _download(url, d / "suppl" / fname, f"{gse} normalised counts")

    df = pd.read_csv(path, sep="\t", index_col=0, compression="gzip", low_memory=False)
    if has_symbol_col:
        df = df.drop(columns=[df.columns[0]])
    df.index = [str(i).strip('"').split(".")[0] for i in df.index]
    df.columns = [str(c).strip('"') for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")

    keep = [c for c in df.columns if c.startswith(("C_", disease_prefix))]
    df = df[keep]
    expr = df.T
    y = np.array([0 if c.startswith("C_") else 1 for c in expr.index], dtype=int)
    meta = pd.DataFrame({"sample": list(expr.index)}, index=expr.index)
    print(f"  {gse}: expr={expr.shape}  disease={y.sum()}  ctrl={(y == 0).sum()}")
    return expr, y, meta


def _load_als_positive_control() -> tuple["object", np.ndarray, "object"]:
    """GSE153960 GPL16791 subset (n=636) — ALS positive control."""
    import pandas as pd

    (ds,) = load_dataset("GSE153960", platform="GPL16791", resources_dir=RESOURCES_DIR)
    expr = ds.X.copy()
    expr.columns = [str(c).split(".")[0] for c in expr.columns]
    y = ds.y.values.astype(int)
    print(f"  GPL16791: expr={expr.shape}  ALS={y.sum()}  ctrl={(y == 0).sum()}")
    return expr, y, pd.DataFrame(index=expr.index)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _match_panel(
    panel_feats: list[str],
    panel_syms: list[str],
    target_cols: set[str],
    id_type: str,
) -> tuple[list[str], list[str], list[str]]:
    """Return (train_keys, target_keys, matched_symbols) for a target cohort."""
    train_keys: list[str] = []
    target_keys: list[str] = []
    matched: list[str] = []
    for feat, sym in zip(panel_feats, panel_syms):
        if id_type == "ensembl":
            base = feat.split(".")[0]
            if base in target_cols:
                train_keys.append(feat)
                target_keys.append(base)
                matched.append(sym)
        elif id_type == "entrez":
            eid = _PC_ENTREZ.get(sym)
            if eid is not None and str(eid) in target_cols:
                train_keys.append(feat)
                target_keys.append(str(eid))
                matched.append(sym)
        else:
            for cand in SYMBOL_ALIASES.get(sym, [sym]):
                if cand in target_cols:
                    train_keys.append(feat)
                    target_keys.append(cand)
                    matched.append(sym)
                    break
    return train_keys, target_keys, matched


def _score(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    params: dict,
    mode: str,
    y_te: np.ndarray | None = None,
    adjust_colsample: bool = True,
) -> dict[str, dict]:
    """Fold-averaged cross-validated scoring of a target cohort.

    Follows the convention established in ``disease_specificity.py`` (Step 10):
    a 5-fold stratified CV is run on the GPL24676 training set; the training-set
    estimate is out-of-fold, and the held-out target cohort is scored by every
    fold model with the per-sample scores averaged across folds.  This removes
    the dependence on a single full-data fit and exposes fold-to-fold
    variability on the model side, which a single fit hides.

    Args:
        X_tr: Training matrix (matched panel columns, log1p).
        y_tr: Training labels.
        X_te: Target matrix (same columns, same transform).
        params: LightGBM hyperparameters (carries random_state and verbose).
        mode: ``"canonical"`` (scaler fit on training) or ``"zscore"``
            (per-gene standardisation fit within each cohort).
        y_te: Optional target labels, used only to report per-fold target AUCs.

    Returns:
        ``{model: {"scores": mean target score per sample,
                   "oof_auc": training out-of-fold AUC,
                   "fold_aucs": per-fold target AUCs (empty if y_te is None)}}``
    """
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    if mode not in ("canonical", "zscore"):  # pragma: no cover
        raise ValueError(mode)

    # Re-express the tuned colsample_bytree for this feature space (see
    # _adjust_colsample). colsample_sensitivity.py passes adjust_colsample=False
    # because it sets the value explicitly.
    if adjust_colsample:
        params = _adjust_colsample(params, X_tr.shape[1])

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    models = ("LGBM", "LR-L2")
    fold_scores: dict[str, list[np.ndarray]] = {m: [] for m in models}
    oof: dict[str, np.ndarray] = {m: np.zeros(len(y_tr)) for m in models}
    fold_aucs: dict[str, list[float]] = {m: [] for m in models}

    for tr_idx, val_idx in skf.split(X_tr, y_tr):
        sc = StandardScaler().fit(X_tr[tr_idx])
        Xf = sc.transform(X_tr[tr_idx])
        Xv = sc.transform(X_tr[val_idx])
        # The target is harmonised with the same rule as the full-cohort case:
        # canonical borrows the fold's training scaler, zscore standardises the
        # target within itself (so it never sees training-set location/scale).
        Xt = (
            sc.transform(X_te)
            if mode == "canonical"
            else (StandardScaler().fit_transform(X_te))
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            fitted = {
                "LGBM": LGBMClassifier(**params).fit(Xf, y_tr[tr_idx]),
                "LR-L2": LogisticRegression(
                    penalty="l2", max_iter=5_000, random_state=RANDOM_STATE
                ).fit(Xf, y_tr[tr_idx]),
            }
        for m, clf in fitted.items():
            oof[m][val_idx] = clf.predict_proba(Xv)[:, 1]
            s = clf.predict_proba(Xt)[:, 1]
            fold_scores[m].append(s)
            if y_te is not None and len(np.unique(y_te)) > 1:
                fold_aucs[m].append(float(roc_auc_score(y_te, s)))

    return {
        m: {
            "scores": np.mean(fold_scores[m], axis=0),
            "oof_auc": float(roc_auc_score(y_tr, oof[m])),
            "fold_aucs": fold_aucs[m],
        }
        for m in models
    }


def _loo_deltas(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    params: dict,
    mode: str,
    symbols: list[str],
    baseline: dict[str, dict],
) -> dict[str, dict[str, float]]:
    """Leave-one-gene-out Delta AUC on a target cohort.

    Uses the same sign convention as the rest of the study:
    ``Delta = baseline AUC - leave-one-out AUC``, so a **positive** delta means
    the gene is irreplaceable in that cohort (removing it degrades transfer) and
    a negative delta means it is redundant there.  Every refit is fold-averaged
    exactly as in :func:`_score`, so the deltas are cross-validated on the model
    side rather than derived from a single fit.

    Args:
        X_tr: Training matrix restricted to the matched panel columns.
        y_tr: Training labels.
        X_te: Target matrix, same column order.
        y_te: Target labels.
        params: LightGBM hyperparameters.
        mode: Harmonisation mode passed through to :func:`_score`.
        symbols: Gene symbols in the column order of *X_tr* / *X_te*.
        baseline: Result of :func:`_score` on the full matched panel.

    Returns:
        ``{model: {symbol: delta_auc}}``.
    """
    from sklearn.metrics import roc_auc_score

    base_auc = {m: float(roc_auc_score(y_te, r["scores"])) for m, r in baseline.items()}
    out: dict[str, dict[str, float]] = {m: {} for m in baseline}
    n = X_tr.shape[1]
    for j, sym in enumerate(symbols):
        keep = [k for k in range(n) if k != j]
        res = _score(X_tr[:, keep], y_tr, X_te[:, keep], params, mode)
        for m, r in res.items():
            loo_auc = float(roc_auc_score(y_te, r["scores"]))
            out[m][sym] = base_auc[m] - loo_auc
        print(
            f"      LOO -{sym:<10} "
            + "  ".join(f"{m} d={out[m][sym]:+.4f}" for m in out)
        )
    return out


def _als_vs_disease(
    score_store: dict, cohorts: dict, panel: str, mode: str
) -> list[dict]:
    """Compare ALS cases against each other disease using control-referenced z.

    Each cohort's scores are converted to
    ``z = (score - mean(control scores)) / sd(control scores)`` so that every
    case is expressed relative to its own study's baseline.  ALS z-values are
    then compared against each other disease's z-values.  This is batch-robust
    in a way raw cross-cohort scores are not, but it still assumes the control
    groups are broadly comparable across studies.
    """
    from scipy.stats import mannwhitneyu

    als_key = "GSE153960-GPL16791"
    rows: list[dict] = []
    for model in ("LGBM", "LR-L2"):
        zs: dict[str, np.ndarray] = {}
        for gse in cohorts:
            k = (gse, panel, mode, model)
            if k not in score_store:
                continue
            s = score_store[k]
            y = cohorts[gse][1]
            ctl = s[y == 0]
            sd = ctl.std(ddof=1)
            if not np.isfinite(sd) or sd == 0:
                continue
            zs[gse] = (s[y == 1] - ctl.mean()) / sd
        if als_key not in zs:
            continue
        z_als = zs[als_key]
        for gse, z_dis in zs.items():
            if gse == als_key:
                continue
            y_pair = np.r_[np.ones(len(z_als), int), np.zeros(len(z_dis), int)]
            s_pair = np.r_[z_als, z_dis]
            auc, lo, hi = _bootstrap_auc(y_pair, s_pair, N_BOOTSTRAP)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                p = float(mannwhitneyu(z_als, z_dis, alternative="two-sided")[1])
            rows.append(
                {
                    "model": model,
                    "comparison": f"ALS vs {COHORT_META[gse]['disease']}",
                    "cohort_other": gse,
                    "n_als": len(z_als),
                    "n_other": len(z_dis),
                    "median_z_als": float(np.median(z_als)),
                    "median_z_other": float(np.median(z_dis)),
                    "auc_als_vs_other": auc,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "mwu_p": p,
                }
            )
            print(
                f"  [{model}] ALS vs {COHORT_META[gse]['disease']:<38}"
                f"AUC={auc:.4f} [{lo:.3f}, {hi:.3f}]  p={p:.2e}"
            )
        # sanity check that the reference is well-behaved
        rows.append(
            {
                "model": model,
                "comparison": "ALS median control-referenced z (reference)",
                "cohort_other": als_key,
                "n_als": len(z_als),
                "n_other": 0,
                "median_z_als": float(np.median(z_als)),
                "median_z_other": float("nan"),
                "auc_als_vs_other": float("nan"),
                "ci_lo": float("nan"),
                "ci_hi": float("nan"),
                "mwu_p": float("nan"),
            }
        )
    return rows


def main() -> None:
    """Run the cross-disease specificity analysis and write all outputs."""
    import json

    import pandas as pd
    from scipy.stats import mannwhitneyu, spearmanr

    print("=" * 74)
    print("Cross-disease specificity — ALS panel vs AD / FTLD-TDP / HD / PD")
    print("=" * 74)

    params = json.loads(_PARAMS_PATH.read_text())
    panel = pd.read_csv(_PANEL_CSV)
    panel_feats: list[str] = panel["feature"].tolist()
    panel_syms: list[str] = panel["symbol"].tolist()

    print("\n[1] Loading GPL24676 training cohort ...")
    (ds_tr,) = load_dataset(
        "GSE153960", platform="GPL24676", resources_dir=RESOURCES_DIR
    )
    y_tr = ds_tr.y.values.astype(int)
    X_tr_all = ds_tr.X
    print(f"  training: n={len(y_tr)}  ALS={y_tr.sum()}  ctrl={(y_tr == 0).sum()}")

    panel_defs = {
        "25-gene": list(zip(panel_feats, panel_syms)),
        _CRIT_LABEL: [
            (f, s) for f, s in zip(panel_feats, panel_syms) if s in CRITICAL_PANEL
        ],
    }

    print("\n[2] Loading target cohorts ...")
    cohorts: dict[str, tuple] = {}
    cohorts["GSE125583"] = _load_gse125583()
    cohorts["GSE13162"] = _load_gse13162()
    cohorts["GSE64810"] = _load_deseq_cohort(
        "GSE64810",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE64nnn/GSE64810/suppl/"
        "GSE64810_mlhd_DESeq2_norm_counts_adjust.txt.gz",
        "GSE64810_mlhd_DESeq2_norm_counts_adjust.txt.gz",
        "H_",
        has_symbol_col=False,
    )
    cohorts["GSE68719"] = _load_deseq_cohort(
        "GSE68719",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE68nnn/GSE68719/suppl/"
        "GSE68719_mlpd_PCG_DESeq2_norm_counts.txt.gz",
        "GSE68719_mlpd_PCG_DESeq2_norm_counts.txt.gz",
        "P_",
        has_symbol_col=True,
    )
    cohorts["GSE153960-GPL16791"] = _load_als_positive_control()
    COHORT_META["GSE153960-GPL16791"] = {
        "disease": "ALS Spectrum MND (positive control)",
        "tissue": "mixed CNS",
        "platform": "GPL16791",
        "assay": "bulk RNA-seq",
        "id_type": "ensembl",
    }

    print("\n[3] Scoring (5-fold CV on training; target scores fold-averaged) ...")
    rows: list[dict] = []
    score_store: dict[tuple[str, str, str, str], np.ndarray] = {}
    loo_store: dict[str, dict[str, dict[str, float]]] = {}
    for gse, (expr, y_te, meta) in cohorts.items():
        info = COHORT_META[gse]
        tgt_cols = set(expr.columns)
        # log1p only for count-based cohorts; microarray is already log2
        log1p = info["assay"] != "microarray (HG-U133A_2)"
        for pname, pdef in panel_defs.items():
            feats = [f for f, _ in pdef]
            syms = [s for _, s in pdef]
            tr_keys, te_keys, matched = _match_panel(
                feats, syms, tgt_cols, info["id_type"]
            )
            if len(tr_keys) < 3:
                print(f"  {gse} {pname}: only {len(tr_keys)} genes matched — skip")
                continue
            Xtr = np.log1p(X_tr_all[tr_keys].values.astype(np.float64))
            Xte_raw = expr[te_keys].values.astype(np.float64)
            Xte = np.log1p(np.clip(Xte_raw, 0, None)) if log1p else Xte_raw

            modes = ["zscore"] if not log1p else ["canonical", "zscore"]
            for mode in modes:
                preds = _score(Xtr, y_tr, Xte, params, mode, y_te=y_te)
                # Primary configuration only: LOO is expensive and the critical
                # z-score result is what the manuscript quotes.
                if pname == _CRIT_LABEL and mode == "zscore":
                    loo_store[gse] = _loo_deltas(
                        Xtr, y_tr, Xte, y_te, params, mode, matched, preds
                    )
                for model, res in preds.items():
                    s = res["scores"]
                    auc, lo, hi = _bootstrap_auc(y_te, s, N_BOOTSTRAP)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        p = float(
                            mannwhitneyu(
                                s[y_te == 1], s[y_te == 0], alternative="two-sided"
                            )[1]
                        )
                    rows.append(
                        {
                            "cohort": gse,
                            "disease": info["disease"],
                            "tissue": info["tissue"],
                            "platform": info["platform"],
                            "assay": info["assay"],
                            "panel": pname,
                            "genes_matched": f"{len(tr_keys)}/{len(feats)}",
                            "mode": mode,
                            "model": model,
                            "n_disease": int(y_te.sum()),
                            "n_control": int((y_te == 0).sum()),
                            "auc": auc,
                            "ci_lo": lo,
                            "ci_hi": hi,
                            "median_score_disease": float(np.median(s[y_te == 1])),
                            "median_score_control": float(np.median(s[y_te == 0])),
                            "mwu_p": p,
                            "train_oof_auc": res["oof_auc"],
                            "fold_auc_min": (
                                float(np.min(res["fold_aucs"]))
                                if res["fold_aucs"]
                                else float("nan")
                            ),
                            "fold_auc_max": (
                                float(np.max(res["fold_aucs"]))
                                if res["fold_aucs"]
                                else float("nan")
                            ),
                            "fold_auc_sd": (
                                float(np.std(res["fold_aucs"], ddof=1))
                                if len(res["fold_aucs"]) > 1
                                else float("nan")
                            ),
                        }
                    )
                    score_store[(gse, pname, mode, model)] = s
                    print(
                        f"  {gse:<20}{pname:<9}{mode:<10}{model:<7}"
                        f"AUC={auc:.4f} [{lo:.3f}, {hi:.3f}]  "
                        f"folds[{np.min(res['fold_aucs']):.3f}-"
                        f"{np.max(res['fold_aucs']):.3f}]  "
                        f"OOF={res['oof_auc']:.4f}  "
                        f"matched {len(tr_keys)}/{len(feats)}"
                    )

    df = pd.DataFrame(rows)
    df.to_csv(SCRIPT_DIR / f"{PREFIX}.csv", index=False)

    # Persist per-sample scores and labels so figures can be regenerated
    # without repeating the ~30 min CV + LOO computation.
    # Braak stage travels with the scores: the dose-response figure needs the
    # per-sample pairing, and re-parsing the GSE125583 SOFT record just to plot
    # it would be a second, independently drifting source for the same numbers.
    _extra = {}
    if "braak" in cohorts["GSE125583"][2].columns:
        _extra["__braak__|GSE125583"] = (
            cohorts["GSE125583"][2]["braak"].values.astype(float)
        )
    np.savez_compressed(
        SCRIPT_DIR / f"{PREFIX}_scores.npz",
        **{"|".join(k): v for k, v in score_store.items()},
        **{f"__y__|{g}": cohorts[g][1] for g in cohorts},
        **_extra,
    )

    # --- ALS vs each other disease (control-referenced z) ----------------
    print("\n[4] ALS vs other neurological disease (control-referenced z) ...")
    avd = pd.DataFrame(_als_vs_disease(score_store, cohorts, _CRIT_LABEL, "zscore"))
    if not avd.empty:
        avd.to_csv(SCRIPT_DIR / f"{PREFIX}_als_vs_disease.csv", index=False)

    # --- LOO table ------------------------------------------------------
    loo_rows: list[dict] = []
    for gse, per_model in loo_store.items():
        for model, deltas in per_model.items():
            for sym, d in deltas.items():
                loo_rows.append(
                    {
                        "cohort": gse,
                        "disease": COHORT_META[gse]["disease"],
                        "model": model,
                        "symbol": sym,
                        "delta_auc": d,
                    }
                )
    loo_df = pd.DataFrame(loo_rows)
    if not loo_df.empty:
        loo_df.to_csv(SCRIPT_DIR / f"{PREFIX}_loo.csv", index=False)

    # --- Braak dose-response (GSE125583) --------------------------------
    # Reported for BOTH models: LR-L2 gives systematically higher non-ALS AUCs
    # than LGBM, so quoting only one would bias the specificity claim.
    braak_res: dict[str, dict[str, float]] = {}
    expr_ad, y_ad, meta_ad = cohorts["GSE125583"]
    b = meta_ad["braak"].values.astype(float)
    ok = ~np.isnan(b)
    ad_ok = ok & (y_ad == 1)
    for model in ("LGBM", "LR-L2"):
        key = ("GSE125583", _CRIT_LABEL, "zscore", model)
        if key not in score_store:
            continue
        s = score_store[key]
        rho, p = spearmanr(b[ok], s[ok])
        rho2, p2 = spearmanr(b[ad_ok], s[ad_ok])
        braak_res[model] = {
            "all_rho": float(rho),
            "all_p": float(p),
            "ad_only_rho": float(rho2),
            "ad_only_p": float(p2),
            "n_all": int(ok.sum()),
            "n_ad": int(ad_ok.sum()),
        }

    # --- region contrast (GSE13162) -------------------------------------
    region_res: dict[str, dict[str, float]] = {}
    expr_f, y_f, meta_f = cohorts["GSE13162"]
    reg = meta_f["region"].values
    cort = np.isin(reg, ["frontal", "hippocampus"]) & (y_f == 1)
    cere = (reg == "cerebellum") & (y_f == 1)
    for model in ("LGBM", "LR-L2"):
        key_f = ("GSE13162", _CRIT_LABEL, "zscore", model)
        if key_f not in score_store or cort.sum() < 3 or cere.sum() < 3:
            continue
        s = score_store[key_f]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            p = float(mannwhitneyu(s[cort], s[cere], alternative="two-sided")[1])
        region_res[model] = {
            "n_cortical": int(cort.sum()),
            "n_cerebellar": int(cere.sum()),
            "median_cortical": float(np.median(s[cort])),
            "median_cerebellar": float(np.median(s[cere])),
            "mwu_p": p,
        }

    _write_report(df, braak_res, region_res, cohorts, y_tr, avd, loo_df)
    _plot(df, score_store, cohorts, braak_res, avd)
    if not loo_df.empty:
        _plot_loo(loo_df)
    print(f"\nWrote {PREFIX}.csv, {PREFIX}_statistics.txt, {PREFIX}.png")


def _write_report(
    df,
    braak_res: dict,
    region_res: dict,
    cohorts: dict,
    y_tr: np.ndarray,
    avd=None,
    loo_df=None,
) -> None:
    """Write the plain-text statistics report."""
    lines: list[str] = []
    add = lines.append
    add("Cross-disease specificity of the ALS panel")
    add("=" * 74)
    add("Training : GSE153960 GPL24676, ALS Spectrum MND vs Non-Neurological")
    add(
        f"           Control (n={len(y_tr)}, ALS={int(y_tr.sum())}, "
        f"control={int((y_tr == 0).sum())})"
    )
    add(f"Bootstrap: N={N_BOOTSTRAP}, random_state={RANDOM_STATE}")
    add("")
    add("CROSS-VALIDATION PROVENANCE")
    add("  Scoring follows disease_specificity.py (Step 10): 5-fold stratified")
    add(f"  CV on the GPL24676 training set (random_state={RANDOM_STATE}); the")
    add("  training estimate is out-of-fold, and each target cohort is scored by")
    add("  all 5 fold models with per-sample scores AVERAGED across folds. The")
    add("  fold_auc_min/max columns show model-side fold-to-fold spread; the")
    add("  bootstrap CI covers target-sample variability. Target AUCs are")
    add("  zero-shot: no target sample enters training or thresholding.")
    add("  Caveat that CV does NOT remove: the panel and hyperparameters were")
    add("  selected on this same training cohort, so the training-side estimate")
    add("  is post-selection and optimistic (as noted in the main text).")
    add("")
    add("MODEL: LightGBM is primary, consistent with the rest of the study.")
    add("  LR-L2 is a sensitivity baseline only (cf. lr_baseline.py). The two")
    add("  disagree materially here and both are always reported.")
    add("")
    add("Reading the table: each disease is scored against ITS OWN in-cohort")
    add("controls. AUC ~ 0.5 => the panel is blind to that disease (specificity")
    add("confirmed). High AUC => the panel also detects it.")
    add("")
    add("Modes: canonical = StandardScaler fit on matched GPL24676 training")
    add("columns, applied to target (study-wide recipe; valid when both sides")
    add("are log1p counts). zscore = per-gene standardisation within each")
    add("cohort separately (scale-invariant; the only valid mode for the")
    add("microarray cohort).")
    add("")
    add("-" * 74)
    add("SENSITIVITY MATRIX")
    add("-" * 74)
    hdr = (
        f"{'cohort':<20}{'panel':<9}{'mode':<10}{'model':<7}"
        f"{'AUC':>7}{'95% CI':>18}{'MWU p':>11}  genes"
    )
    add(hdr)
    for _, r in df.iterrows():
        add(
            f"{r['cohort']:<20}{r['panel']:<9}{r['mode']:<10}{r['model']:<7}"
            f"{r['auc']:>7.4f}"
            f"{f'[{r.ci_lo:.3f}, {r.ci_hi:.3f}]':>18}"
            f"{r['mwu_p']:>11.2e}  {r['genes_matched']}"
        )
    add("")

    add("-" * 74)
    add("MODEL GAP BY COHORT  (LR-L2 minus LGBM, averaged over panel x mode)")
    add("-" * 74)
    add("The linear baseline's advantage should be read against the positive")
    add("control: a model that merely transferred better across platforms would")
    add("gain on ALS too. If the ALS gap is ~0 while the non-ALS gaps are large,")
    add("LR-L2 is behaving as a composition index, not as a better classifier.")
    add("These are the numbers the main text quotes -- derived here, never typed.")
    add(f"{'cohort':<22}{'cells':>6}{'mean LR-L2 - LGBM':>20}")
    wide = df.pivot_table(
        index=["cohort", "panel", "mode"], columns="model", values="auc"
    )
    lgbm_col = next((c for c in wide.columns if "LGBM" in str(c).upper()), None)
    lr_col = next((c for c in wide.columns if str(c).upper().startswith("LR")), None)
    if lgbm_col is not None and lr_col is not None:
        gaps = (wide[lr_col] - wide[lgbm_col]).dropna()
        for cohort, g in sorted(
            gaps.groupby(level=0), key=lambda kv: kv[1].mean()
        ):
            add(f"{cohort:<22}{len(g):>6}{g.mean():>+20.4f}")
    else:
        add("  (both models not present; gap undefined)")
    add("")

    add("-" * 74)
    add("PER-COHORT DETAIL")
    add("-" * 74)
    for gse in df["cohort"].unique():
        sub = df[df["cohort"] == gse]
        r0 = sub.iloc[0]
        add(f"{gse} — {r0['disease']}")
        add(f"  tissue={r0['tissue']}  platform={r0['platform']}  assay={r0['assay']}")
        add(f"  n disease={r0['n_disease']}  n control={r0['n_control']}")
        prim = sub[sub["mode"] == "zscore"]
        best = prim.loc[prim["auc"].idxmax()] if len(prim) else sub.iloc[0]
        add(
            f"  best zscore AUC = {best['auc']:.4f} "
            f"[{best['ci_lo']:.3f}, {best['ci_hi']:.3f}] "
            f"({best['panel']}, {best['model']})"
        )
        add(
            f"  median score disease={best['median_score_disease']:.4f}  "
            f"control={best['median_score_control']:.4f}"
        )
        add("")

    if braak_res:
        add("-" * 74)
        add(f"BRAAK DOSE-RESPONSE (GSE125583, {_CRIT_LABEL} / zscore)")
        add("-" * 74)
        add("If the panel were reading Alzheimer's pathology, its score would")
        add("rise with Braak stage. Reported for both models.")
        for model, r in braak_res.items():
            add(f"  [{model}]")
            add(
                f"    all samples : Spearman rho={r['all_rho']:+.4f}  "
                f"p={r['all_p']:.3e}  (n={r['n_all']})"
            )
            add(
                f"    AD only     : Spearman rho={r['ad_only_rho']:+.4f}  "
                f"p={r['ad_only_p']:.3e}  (n={r['n_ad']})"
            )
        add("")

    if region_res:
        add("-" * 74)
        add(f"REGION CONTRAST IN FTLD-U (GSE13162, {_CRIT_LABEL} / zscore)")
        add("-" * 74)
        add("Zhou et al. report TOP1 damage in cortical but not cerebellar")
        add("neurons. This tests whether the panel score follows that pattern.")
        add("Underpowered on the cerebellar side; read as directional only.")
        for model, r in region_res.items():
            add(f"  [{model}]")
            add(
                f"    cortical (frontal+hippocampus) n={r['n_cortical']}  "
                f"median={r['median_cortical']:.4f}"
            )
            add(
                f"    cerebellar                     n={r['n_cerebellar']}  "
                f"median={r['median_cerebellar']:.4f}"
            )
            add(f"    Mann-Whitney U p={r['mwu_p']:.3e}")
        add("")

    if avd is not None and not avd.empty:
        add("-" * 74)
        add(f"ALS vs OTHER NEUROLOGICAL DISEASE (control-referenced z, {_CRIT_LABEL})")
        add("-" * 74)
        add("Each cohort's scores are converted to")
        add("  z = (score - mean(that cohort's controls)) / sd(those controls)")
        add("so every case is expressed relative to its own study's baseline;")
        add("ALS z-values are then compared against each other disease's.")
        add("AUC ~ 0.5 => the panel cannot tell ALS from that disease.")
        add("Assumes control groups are broadly comparable across studies.")
        add("The cleanest ALS-vs-other-disease estimate remains Step 10")
        add("(ALS vs held-out FTLD/PSP INSIDE GSE153960, AUC = 0.9067), which")
        add("has no cross-study component at all.")
        add("")
        cmp_rows = avd[avd["auc_als_vs_other"].notna()]
        add(f"{'model':<7}{'comparison':<44}{'AUC':>7}{'95% CI':>18}{'MWU p':>11}")
        for _, r in cmp_rows.iterrows():
            add(
                f"{r['model']:<7}{r['comparison']:<44}{r['auc_als_vs_other']:>7.4f}"
                f"{f'[{r.ci_lo:.3f}, {r.ci_hi:.3f}]':>18}{r['mwu_p']:>11.2e}"
            )
        add("")
        add("Median control-referenced z by group:")
        for _, r in cmp_rows.iterrows():
            add(
                f"  [{r['model']}] ALS={r['median_z_als']:+.3f}  "
                f"{r['comparison'].replace('ALS vs ', ''):<38}"
                f"={r['median_z_other']:+.3f}"
            )
        add("")

    if loo_df is not None and not loo_df.empty:
        add("-" * 74)
        add(f"LEAVE-ONE-GENE-OUT PER COHORT ({_CRIT_LABEL}, zscore, fold-averaged)")
        add("-" * 74)
        add("Delta = baseline AUC - leave-one-out AUC, the same sign convention")
        add("used throughout this study: POSITIVE = irreplaceable in that cohort,")
        add("NEGATIVE = redundant there. Every refit is fold-averaged, so these")
        add("deltas are cross-validated on the model side.")
        add("")
        for model in sorted(loo_df["model"].unique()):
            sub = loo_df[loo_df["model"] == model]
            piv = sub.pivot_table(index="symbol", columns="cohort", values="delta_auc")
            cols = [
                c
                for c in [
                    "GSE153960-GPL16791",
                    "GSE13162",
                    "GSE64810",
                    "GSE68719",
                    "GSE125583",
                ]
                if c in piv.columns
            ]
            piv = piv[cols]
            add(f"  [{model}]")
            add("    " + f"{'gene':<12}" + "".join(f"{c[-8:]:>11}" for c in cols))
            ref = cols[0] if cols else None
            piv = piv.sort_values(ref, ascending=False) if ref else piv
            for sym, r in piv.iterrows():
                add(
                    "    "
                    + f"{sym:<12}"
                    + "".join(
                        f"{r[c]:>+11.4f}" if np.isfinite(r[c]) else f"{'n/a':>11}"
                        for c in cols
                    )
                )
            add("")
            # which genes drive each non-ALS cohort
            for c in cols[1:]:
                top = piv[c].dropna().sort_values(ascending=False).head(4)
                add(
                    f"    drivers in {c}: "
                    + ", ".join(f"{s} ({v:+.4f})" for s, v in top.items())
                )
            add("")

    add("-" * 74)
    add("HEADLINE READING")
    add("-" * 74)
    prim = df[(df["panel"] == _CRIT_LABEL) & (df["mode"] == "zscore")]
    for model in ("LGBM", "LR-L2"):
        sub = prim[prim["model"] == model].set_index("cohort")
        if sub.empty:
            continue
        add(f"  [{model}, {_CRIT_LABEL}, zscore]")
        for gse in [
            "GSE153960-GPL16791",
            "GSE13162",
            "GSE64810",
            "GSE68719",
            "GSE125583",
        ]:
            if gse not in sub.index:
                continue
            r = sub.loc[gse]
            add(
                f"    {r['disease']:<38}AUC={r['auc']:.4f} "
                f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
            )
    add("")
    add("  LightGBM (primary, canonical model of this study) and LR-L2")
    add("  (sensitivity baseline) disagree materially, and that disagreement is")
    add("  itself a result: LR-L2 gives higher non-ALS discrimination in EVERY")
    add("  cohort. Any conclusion that holds under one model but not the other")
    add("  must be reported as model-dependent rather than as established.")
    add("  In particular, whether the panel detects FTLD-TDP above chance is")
    add("  model-dependent, so the 'TDP-43 proteinopathy detector' reading is")
    add("  NOT confirmed by these data and is only a hypothesis.")
    add("  What is robust across both models: the panel does not track")
    add("  Alzheimer's disease (near-chance AUC under both, and no Braak")
    add("  dose-response at n=287), and Huntington's disease scores higher than")
    add("  Alzheimer's or Parkinson's under both.")
    add("")

    add("-" * 74)
    add("CAVEATS")
    add("-" * 74)
    add("1. GSE64810 (HD) and GSE68719 (PD) re-analyse the SAME neurologically-")
    add("   normal control GSMs, so the HD and PD results are not statistically")
    add("   independent of each other.")
    add("2. GSE68719 is an all-male cohort; sex is not balanced against the")
    add("   training data.")
    add("3. GSE13162 is a microarray cohort (HG-U133A_2). Non-coding panel")
    add("   members are unmeasurable and some protein-coding members have no")
    add("   probe, so the matched gene count is reduced (see table).")
    add("4. No open post-mortem CNS PSP series exists, so the tau-only negative")
    add("   control is still outstanding (findings.md Step 33).")
    add("5. All AUCs are zero-shot: no target-cohort samples were used in")
    add("   training or in threshold selection.")
    add("")

    (SCRIPT_DIR / f"{PREFIX}_statistics.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _plot_loo(loo_df) -> None:
    """Heatmap of leave-one-gene-out Delta AUC, genes x cohorts, per model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted(loo_df["model"].unique())
    order = [
        "GSE153960-GPL16791",
        "GSE13162",
        "GSE64810",
        "GSE68719",
        "GSE125583",
    ]
    short = {
        "GSE153960-GPL16791": "ALS\n(pos ctrl)",
        "GSE13162": "FTLD-TDP",
        "GSE64810": "HD",
        "GSE68719": "PD",
        "GSE125583": "AD",
    }
    fig, axes = plt.subplots(
        1, len(models), figsize=(7.2 * len(models), 7.4), squeeze=False
    )
    for i, model in enumerate(models):
        ax = axes[0][i]
        sub = loo_df[loo_df["model"] == model]
        piv = sub.pivot_table(index="symbol", columns="cohort", values="delta_auc")
        cols = [c for c in order if c in piv.columns]
        piv = piv[cols]
        if cols:
            piv = piv.sort_values(cols[0], ascending=False)
        vmax = float(np.nanmax(np.abs(piv.values))) or 0.01
        im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([short.get(c, c) for c in cols], fontsize=9)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index, fontsize=9)
        for r in range(piv.shape[0]):
            for c in range(piv.shape[1]):
                v = piv.values[r, c]
                if np.isfinite(v):
                    ax.text(
                        c,
                        r,
                        f"{v:+.3f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if abs(v) > 0.6 * vmax else "black",
                    )
        ax.set_title(
            f"{model} — leave-one-gene-out $\\Delta$AUC\n"
            "red = irreplaceable ($\\Delta$>0), blue = redundant ($\\Delta$<0)",
            fontsize=11,
        )
        fig.colorbar(
            im, ax=ax, fraction=0.035, pad=0.02, label="$\\Delta$AUC (baseline $-$ LOO)"
        )
    fig.suptitle(
        "Which panel genes carry the signal in each disease "
        f"({_CRIT_LABEL}, z-score mode, fold-averaged)",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / f"{PREFIX}_loo.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot(df, score_store: dict, cohorts: dict, braak_res: dict, avd=None) -> None:
    """Four-panel summary figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.26)

    order = [
        "GSE153960-GPL16791",
        "GSE125583",
        "GSE13162",
        "GSE64810",
        "GSE68719",
    ]
    labels = {
        "GSE153960-GPL16791": "ALS\n(pos. ctrl)",
        "GSE125583": "Alzheimer's",
        "GSE13162": "FTLD-TDP",
        "GSE64810": "Huntington's",
        "GSE68719": "Parkinson's",
    }
    # LightGBM is the primary model throughout this study; LR-L2 appears only as
    # a sensitivity baseline (cf. lr_baseline.py). Both are plotted in panel (b).
    PRIMARY_MODEL = "LGBM"
    key_of = {g: (g, _CRIT_LABEL, "zscore", PRIMARY_MODEL) for g in order}

    # --- (a) score distributions, disease vs own control -----------------
    ax = fig.add_subplot(gs[0, :2])
    pos = 0
    ticks: list[float] = []
    for g in order:
        k = key_of[g]
        if k not in score_store:
            continue
        s = score_store[k]
        y = cohorts[g][1]
        for cls, col in ((0, "#4878d0"), (1, "#d65f5f")):
            vals = s[y == cls]
            if len(vals) < 2:
                continue
            parts = ax.violinplot(
                [vals], positions=[pos], showmedians=True, widths=0.75
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(col)
                pc.set_alpha(0.72)
            for kk in ("cmedians", "cmaxes", "cmins", "cbars"):
                if kk in parts:
                    parts[kk].set_color("black")
                    parts[kk].set_linewidth(0.9)
            pos += 1
        ticks.append(pos - 1.5)
        pos += 0.9
    ax.set_xticks(ticks)
    ax.set_xticklabels([labels[g] for g in order if key_of[g] in score_store])
    ax.set_ylabel(f"ALS panel score ({_CRIT_LABEL}, z-score mode)")
    ax.set_title(
        "(a) Panel score by cohort — blue = that cohort's own controls, red = disease"
    )
    ax.grid(axis="y", alpha=0.25)

    # --- (b) AUC forest, BOTH models -------------------------------------
    # Both models are plotted so the figure cannot be read as having picked
    # whichever model flatters the specificity claim.
    ax = fig.add_subplot(gs[0, 2])
    base = df[(df["panel"] == _CRIT_LABEL) & (df["mode"] == "zscore")]
    present = [g for g in order if g in set(base["cohort"])]
    offs = {"LGBM": -0.16, "LR-L2": +0.16}
    mcol = {"LGBM": "#4878d0", "LR-L2": "#d65f5f"}
    for model, off in offs.items():
        sub = base[base["model"] == model].set_index("cohort").reindex(present)
        if sub["auc"].isna().all():
            continue
        ypos = np.arange(len(present)) + off
        ax.errorbar(
            sub["auc"],
            ypos,
            xerr=[sub["auc"] - sub["ci_lo"], sub["ci_hi"] - sub["auc"]],
            fmt="o",
            color=mcol[model],
            capsize=3,
            markersize=5.5,
            label=model,
        )
    ax.axvline(0.5, ls="--", c="grey", lw=1.1)
    ax.text(
        0.49,
        -0.42,
        "chance (panel blind)",
        fontsize=7,
        color="grey",
        ha="right",
        va="center",
    )
    ax.set_yticks(np.arange(len(present)))
    ax.set_yticklabels([labels[g] for g in present], fontsize=9)
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(-0.75, len(present) - 0.25)
    ax.set_xlabel("AUC vs own controls")
    ax.set_title("(b) Zero-shot AUC, both models\nwith 95% bootstrap CI")
    ax.legend(fontsize=7.5, loc="center left")
    ax.grid(axis="x", alpha=0.25)

    # --- (c) ROC overlay -------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    cmap = {
        "GSE153960-GPL16791": "#333333",
        "GSE125583": "#d65f5f",
        "GSE13162": "#ee854a",
        "GSE64810": "#6acc65",
        "GSE68719": "#4878d0",
    }
    for g in order:
        k = key_of[g]
        if k not in score_store:
            continue
        y = cohorts[g][1]
        fpr, tpr, _ = roc_curve(y, score_store[k])
        row = df[
            (df["cohort"] == g)
            & (df["panel"] == _CRIT_LABEL)
            & (df["mode"] == "zscore")
            & (df["model"] == PRIMARY_MODEL)
        ]
        auc = row["auc"].iloc[0] if len(row) else float("nan")
        ax.plot(
            fpr,
            tpr,
            color=cmap[g],
            lw=1.9,
            label=f"{labels[g].replace(chr(10), ' ')} ({auc:.3f})",
        )
    ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("(c) ROC, disease vs own controls")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.25)

    # --- (d) Braak dose-response ----------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    k = ("GSE125583", _CRIT_LABEL, "zscore", PRIMARY_MODEL)
    if k in score_store and PRIMARY_MODEL in braak_res:
        br = braak_res[PRIMARY_MODEL]
        s = score_store[k]
        b = cohorts["GSE125583"][2]["braak"].values.astype(float)
        ok = ~np.isnan(b)
        data = [s[ok & (b == stage)] for stage in range(1, 7)]
        keep = [(i + 1, d) for i, d in enumerate(data) if len(d) >= 3]
        if keep:
            ax.boxplot([d for _, d in keep], positions=[i for i, _ in keep], widths=0.6)
        ax.set_xlabel("Braak stage")
        ax.set_ylabel("ALS panel score")
        ax.set_title(
            f"(d) Alzheimer's: score vs Braak stage\n"
            f"Spearman $\\rho$={br['all_rho']:+.3f}, "
            f"p={br['all_p']:.2f} (n={br['n_all']}) — no dose-response"
        )
        ax.grid(axis="y", alpha=0.25)

    # --- (e) ALS vs each other disease, control-referenced z --------------
    ax = fig.add_subplot(gs[1, 2])
    if avd is not None and not avd.empty:
        sub = avd[(avd["model"] == PRIMARY_MODEL) & avd["auc_als_vs_other"].notna()]
        sub = sub.iloc[::-1]
        if not sub.empty:
            ypos = np.arange(len(sub))
            ax.errorbar(
                sub["auc_als_vs_other"],
                ypos,
                xerr=[
                    sub["auc_als_vs_other"] - sub["ci_lo"],
                    sub["ci_hi"] - sub["auc_als_vs_other"],
                ],
                fmt="o",
                color="black",
                capsize=3,
                markersize=6,
            )
            ax.set_yticks(ypos)
            ax.set_yticklabels(
                [
                    c.replace("ALS vs ", "")
                    .replace(" disease", "")
                    .replace(" (FTLD-U)", "")
                    for c in sub["comparison"]
                ],
                fontsize=9,
            )
            ax.axvline(0.5, ls="--", c="grey", lw=1.1)
            ax.text(
                0.49,
                -0.42,
                "chance = cannot separate",
                fontsize=7,
                color="grey",
                ha="right",
                va="center",
            )
            ax.set_xlim(0.0, 1.05)
            ax.set_ylim(-0.7, len(sub) - 0.3)
    ax.set_xlabel("AUC, ALS vs that disease")
    ax.set_title(
        f"(e) ALS vs other disease ({PRIMARY_MODEL})\n"
        "control-referenced z, batch-robust"
    )
    ax.grid(axis="x", alpha=0.25)

    fig.suptitle(
        "Cross-disease specificity of the ALS panel — each disease scored "
        "against its own in-cohort controls",
        fontsize=13,
        y=0.98,
    )
    fig.savefig(SCRIPT_DIR / f"{PREFIX}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
