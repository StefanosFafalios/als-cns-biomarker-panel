"""Panel validation in independent spinal-cord and motor-cortex cohorts (Step 101).

None of these cohorts was used to discover the 25-gene panel or to select the
critical panel, so they give the critical panel its first external test that the
selection never saw.

  GSE255683    lumbar spinal cord, RNA-seq exonic counts          ALS 10 / ctrl 10
  GSE287256    cervical spinal cord, RNA-seq DESeq2-normalised     ALS 8 / ctrl 4
  GSE26927     cervical spinal cord, Illumina HumanRef-8 v2        ALS arm, replicates averaged
  E-MTAB-8635  spinal cord, Agilent 014850 (raw GenePix files)     ALS 30 / ctrl 10
  E-MTAB-2325  motor cortex, Agilent 014850                        ALS 31 / ctrl 10

E-MTAB-2325 shares 35 donors with E-MTAB-8635 (SDRF patient codes), so it is
reported but left out of the pooled estimate.

Analysis fixed before the run:
  * Model: the manuscript's zero-shot LightGBM (random_panel_null_cohorts._auc),
    trained on all GPL24676 samples.
  * Scaling: count cohorts log1p with the training scaler (as GSE122649); arrays
    log2 and standardised within cohort (utils.LOG_SCALE_COHORTS, as GSE76220).
    Sensitivity: the RNA-seq cohorts standardised within cohort.
  * Per cohort and panel (25-gene; critical): AUC, 95% stratified bootstrap CI
    over donors (B=2000), and the empirical p against B=1000 random gene sets of
    the same size (all candidate genes; SFM pool), as random_panel_null_cohorts.py.
  * Pooled: inverse-variance weighted AUC over the four independent cohorts, SE
    from the bootstrap; its null pools the i-th random set of every cohort with
    the same weights.
  * GSE26927 disease arms (AD, PD, HD, MS, schizophrenia): AUC of cases against
    their own tissue-matched controls, each arm standardised within itself.

Outputs: new_cohort_validation_statistics.txt, new_cohort_validation.png
"""

from __future__ import annotations

import gzip
import io
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
sys.path.insert(0, str(SCRIPT_DIR))

from cross_disease_specificity import (  # noqa: E402
    _download,
    _sample_char_rows,
    _series_matrix_meta,
)
from random_panel_null_cohorts import _ZS_PARAMS, _null, _symbols  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from utils import (  # noqa: E402
    collapse_duplicate_rows,
    critical_panel_features,
    load_dataset,
    normalise_symbols,
    zero_shot_scale,
)

RES = ALS_DIR / "resources"
B_NULL = 1000
B_BOOT = 2000
RANDOM_STATE = 42
POOLED = ("GSE255683", "GSE287256", "GSE26927 ALS", "E-MTAB-8635")
OUT_TXT = SCRIPT_DIR / "new_cohort_validation_statistics.txt"
OUT_PNG = SCRIPT_DIR / "new_cohort_validation.png"
_ANNOT_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={}&targ=self&form=text&view=data"


@dataclass
class Cohort:
    """One test cohort, one row per donor."""

    name: str
    tissue: str
    X: np.ndarray  # donors x genes
    genes: list[str]  # HGNC symbols or Ensembl base ids
    by_symbol: bool
    y: np.ndarray
    scale_id: str  # zero_shot_scale key: GEO/AE accession


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _platform_symbols(gpl: str) -> dict[str, str]:
    """Probe -> first HGNC symbol from the GEO platform table (cached)."""
    path = _download(_ANNOT_URL.format(gpl), RES / "platforms" / f"{gpl}_annot.txt", f"{gpl} annotation")
    txt = path.read_text(errors="ignore").splitlines()
    head = next(i for i, ln in enumerate(txt) if ln.startswith("ID\t"))
    cols = txt[head].split("\t")
    col = next(cols.index(c) for c in ("GENE_SYMBOL", "Gene Symbol", "SYMBOL", "Symbol", "ILMN_Gene") if c in cols)
    out = {}
    for ln in txt[head + 1:]:
        f = ln.split("\t")
        if len(f) > col and f[col].strip():
            out[f[0]] = re.split(r"\s*///\s*", f[col].strip())[0]
    return out


def _to_genes(probes: pd.DataFrame, probe_sym: dict[str, str], cohort: str) -> pd.DataFrame:
    """probes x samples -> symbols x samples, keeping the most expressed probe per symbol."""
    probes = probes.loc[probes.index.isin(probe_sym)]
    probes.index = [probe_sym[p] for p in probes.index]
    return collapse_duplicate_rows(normalise_symbols(probes, cohort))


def _quantile_normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Columns (arrays) mapped onto the mean sorted distribution."""
    ref = np.sort(df.to_numpy(), axis=0).mean(axis=1)
    ranks = df.rank(method="first").to_numpy().astype(int) - 1
    return pd.DataFrame(ref[ranks], index=df.index, columns=df.columns)


def _gse255683() -> Cohort:
    d = RES / "GSE255683"
    meta = _series_matrix_meta(d / "matrix" / "GSE255683_series_matrix.txt.gz")
    diag = next(r for r in _sample_char_rows(d / "matrix" / "GSE255683_series_matrix.txt.gz")
                if r[0].startswith("diagnosis"))
    label = {g: int(v.endswith("ALS")) for g, v in zip(meta["Sample_geo_accession"], diag)}
    cols = {}
    with tarfile.open(d / "suppl" / "GSE255683_RAW.tar") as tf:
        for m in tf.getmembers():
            gsm = re.match(r"(GSM\d+)", Path(m.name).name).group(1)
            t = pd.read_csv(io.BytesIO(gzip.decompress(tf.extractfile(m).read())), sep="\t")
            cols[gsm] = pd.Series(t["count"].to_numpy(), index=t["name"].str.split(".").str[0])
    df = pd.DataFrame(cols).fillna(0)
    df = df.groupby(level=0).sum()
    gsms = list(df.columns)
    return Cohort("GSE255683", "lumbar spinal cord", np.log1p(df.T.to_numpy(np.float32)), list(df.index),
                  False, np.array([label[g] for g in gsms]), "GSE255683")


def _gse287256() -> Cohort:
    d = RES / "GSE287256"
    meta = _series_matrix_meta(d / "matrix" / "GSE287256_series_matrix.txt.gz")
    lib = next(v for k, v in meta.items() if k.startswith("Sample_description") and v[0].startswith("Library"))
    title = {x.split(": ")[1]: t for x, t in zip(lib, meta["Sample_title"])}
    df = pd.read_csv(d / "suppl" / "GSE287256_ALS_bulkRNAseq_results_bulk_counts_normalized.csv.gz")
    df = df.set_index(df["gene_ID"].str.split(".").str[0])
    libs = [c for c in df.columns if c in title]
    X = df[libs].groupby(level=0).sum()
    y = np.array([int("_ALS" in title[c]) for c in libs])
    return Cohort("GSE287256", "cervical spinal cord", np.log1p(X.T.to_numpy(np.float32)), list(X.index),
                  False, y, "GSE287256")


def _gse26927_arms() -> dict[str, Cohort]:
    """One cohort per disease arm; technical replicates (a/b suffix) averaged per donor."""
    path = RES / "GSE26927" / "matrix" / "GSE26927_series_matrix.txt.gz"
    meta = _series_matrix_meta(path)
    probes = pd.read_csv(path, sep="\t", comment="!", index_col=0).apply(pd.to_numeric, errors="coerce")
    probes = np.log2(probes.clip(lower=1.0))  # Rosetta-normalised linear intensities
    genes = _to_genes(probes, _platform_symbols("GPL6255"), "GSE26927")
    tissue = dict(zip(meta["Sample_geo_accession"], meta["Sample_source_name_ch1"]))
    arms: dict[str, Cohort] = {}
    parsed = [re.match(r"BNE_([A-Z]+)_(Control|Disease)_(.+?)[ab]?$", t) for t in meta["Sample_title"]]
    for arm in sorted({m.group(1) for m in parsed}):
        rows = [(g, m.group(2), m.group(3)) for g, m in zip(meta["Sample_geo_accession"], parsed) if m.group(1) == arm]
        donor = {g: f"{grp}_{sid}" for g, grp, sid in rows}
        sub = genes[[g for g, _, _ in rows]].T.groupby(lambda g: donor[g]).mean()
        y = np.array([int(k.startswith("Disease")) for k in sub.index])
        print(f"  GSE26927 {arm}: {len(rows)} arrays -> {len(sub)} donors ({int(y.sum())} cases)")
        arms[arm] = Cohort(f"GSE26927 {arm}", tissue[rows[0][0]], sub.to_numpy(np.float32),
                           list(sub.columns), True, y, "GSE26927")
    return arms


def _emtab(acc: str) -> Cohort:
    d = RES / acc
    sdrf = pd.read_csv(d / f"{acc}.sdrf.txt", sep="\t")
    y_of = sdrf["Factor Value[disease]"].str.contains("sclerosis", case=False).astype(int)
    if acc == "E-MTAB-2325":
        raw = pd.read_csv(d / "raw_data.txt", sep="\t", skiprows=[1], index_col=0)
        raw.columns = [c.strip() for c in raw.columns]
        key = dict(zip(sdrf["Scan Name"].str.strip(), y_of))
        tissue = "motor cortex"
    else:
        arrays = {}
        for f in sdrf["Array Data File"].str.strip():
            lines = (d / "gpr" / f).read_text(errors="ignore").splitlines()
            head = next(i for i, ln in enumerate(lines) if ln.lstrip('"').startswith("Block"))
            g = pd.read_csv(io.StringIO("\n".join(lines[head:])), sep="\t")
            sig = (g["F532 Median"] - g["B532 Median"]).where(g["Flags"] >= 0)
            arrays[f] = sig.groupby(g["ID"]).median()
        raw = pd.DataFrame(arrays)
        raw = raw.loc[raw.index.str.startswith("A_")]
        raw = raw.apply(lambda r: r.fillna(r.median()), axis=1).dropna()
        key = dict(zip(sdrf["Array Data File"].str.strip(), y_of))
        tissue = "spinal cord"
    qn = _quantile_normalise(np.log2(raw.clip(lower=1.0)))
    genes = _to_genes(qn, _platform_symbols("GPL6480"), acc)[list(key)]
    return Cohort(acc, tissue, genes.T.to_numpy(np.float32), list(genes.index), True,
                  np.array([key[c] for c in genes.columns]), acc)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _scores(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, scale_id: str, within: bool = False) -> np.ndarray:
    """Zero-shot probabilities; the same model and scaling as random_panel_null_cohorts._auc."""
    from lightgbm import LGBMClassifier
    from sklearn.preprocessing import StandardScaler

    if within:
        ztr, zte = StandardScaler().fit_transform(xtr), StandardScaler().fit_transform(xte)
    else:
        ztr, zte = zero_shot_scale(xtr, xte, scale_id)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        clf = LGBMClassifier(**_ZS_PARAMS).fit(ztr, ytr)
    return clf.predict_proba(zte)[:, 1]


def _boot(y: np.ndarray, s: np.ndarray) -> tuple[float, float, float]:
    """Stratified bootstrap over donors: (2.5%, 97.5%, SD) of the AUC."""
    rng = np.random.default_rng(RANDOM_STATE)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    a = [roc_auc_score(y[i], s[i]) for i in (
        np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(B_BOOT))]
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), float(np.std(a))


def main() -> None:
    (tr,) = load_dataset("GSE153960", platform="GPL24676", resources_dir=RES)
    ytr = tr.y.to_numpy().astype(int)
    xtr_all = np.log1p(tr.X.to_numpy(dtype=np.float32))
    tr_col = {str(c).split(".")[0]: j for j, c in enumerate(tr.X.columns)}
    clean = [n.split(".")[0] for n in (SCRIPT_DIR / "lgbm_prefilter_names.txt").read_text().splitlines()]
    pool = [f.split(".")[0] for f in pd.read_csv(SCRIPT_DIR / "lgbm_iter_final_shap_ranking.csv").feature]
    p25 = pd.read_csv(SCRIPT_DIR / "lgbm_core25_panel.csv")
    panels = {
        "25-gene": [f.split(".")[0] for f in p25.feature],
        "critical": [f.split(".")[0] for f in critical_panel_features()],
    }
    sym = _symbols(clean)
    sym.update(dict(zip([f.split(".")[0] for f in p25.feature], p25.symbol, strict=True)))

    print("Loading cohorts ...")
    arms = _gse26927_arms()
    cohorts = [_gse255683(), _gse287256(), arms.pop("ALS"), _emtab("E-MTAB-8635"), _emtab("E-MTAB-2325")]
    cohorts[2].name = "GSE26927 ALS"
    for c in cohorts:
        print(f"  {c.name}: {len(c.y)} donors ({int(c.y.sum())} ALS), {len(c.genes)} genes")

    def vmap_of(c: Cohort) -> dict[str, int]:
        col = {g: j for j, g in enumerate(c.genes)}
        if c.by_symbol:
            return {b: col[s] for b, s in sym.items() if s in col and b in tr_col}
        return {b: j for b, j in col.items() if b in tr_col}

    rng = np.random.default_rng(RANDOM_STATE)
    rows, nulls = [], {}
    for c in cohorts:
        vmap = vmap_of(c)
        for pname, genes in panels.items():
            avail = [g for g in genes if g in vmap]
            xtr, xte = xtr_all[:, [tr_col[g] for g in avail]], c.X[:, [vmap[g] for g in avail]]
            s = _scores(xtr, ytr, xte, c.scale_id)
            auc = roc_auc_score(c.y, s)
            lo, hi, sd = _boot(c.y, s)
            row = dict(cohort=c.name, tissue=c.tissue, panel=pname, n_als=int(c.y.sum()),
                       n_ctl=int((c.y == 0).sum()), genes=len(avail), of=len(genes), auc=auc, lo=lo, hi=hi, sd=sd)
            if c.scale_id not in ("GSE26927", "E-MTAB-8635", "E-MTAB-2325"):
                row["auc_within"] = roc_auc_score(c.y, _scores(xtr, ytr, xte, c.scale_id, within=True))
            for label, src in (("broad", clean), ("SFM", pool)):
                cands = [g for g in src if g in vmap]
                draws = [rng.choice(len(cands), len(avail), replace=False) for _ in range(B_NULL)]
                null = _null(xtr_all, tr_col, c.X, vmap, ytr, c.y, c.scale_id, cands, draws)
                nulls[(c.name, pname, label)] = null
                row[f"{label}_mean"] = null.mean()
                row[f"{label}_p"] = (int((null >= auc).sum()) + 1) / (B_NULL + 1)
            rows.append(row)
            print(f"  {c.name:<13} {pname:<9} {len(avail):>2}/{len(genes)} AUC {auc:.4f} [{lo:.3f}, {hi:.3f}] "
                  f"p_broad {row['broad_p']:.4f} p_SFM {row['SFM_p']:.4f}", flush=True)
    res = pd.DataFrame(rows)

    pooled = {}
    for pname in panels:
        sub = res[(res.panel == pname) & res.cohort.isin(POOLED)].set_index("cohort").loc[list(POOLED)]
        w = 1 / sub.sd**2
        est = float((w * sub.auc).sum() / w.sum())
        se = float(1 / np.sqrt(w.sum()))
        null = sum(w[c] * nulls[(c, pname, "broad")] for c in POOLED) / w.sum()
        pooled[pname] = dict(auc=est, lo=est - 1.96 * se, hi=est + 1.96 * se, null_mean=float(null.mean()),
                             p=(int((null >= est).sum()) + 1) / (B_NULL + 1))

    arm_rows = []
    for arm, c in arms.items():
        vmap = vmap_of(c)
        for pname, genes in panels.items():
            avail = [g for g in genes if g in vmap]
            s = _scores(xtr_all[:, [tr_col[g] for g in avail]], ytr, c.X[:, [vmap[g] for g in avail]], c.scale_id)
            lo, hi, _ = _boot(c.y, s)
            arm_rows.append(dict(arm=arm, tissue=c.tissue, panel=pname, n_case=int(c.y.sum()),
                                 n_ctl=int((c.y == 0).sum()), genes=len(avail), auc=roc_auc_score(c.y, s), lo=lo, hi=hi))
    arm_df = pd.DataFrame(arm_rows)

    _report(res, pooled, arm_df)
    _plot(res, pooled, arm_df)


def _report(res: pd.DataFrame, pooled: dict, arm_df: pd.DataFrame) -> None:
    lines = [
        "Panel validation in independent spinal-cord and motor-cortex cohorts",
        "=" * 78,
        "Zero-shot LightGBM (top-500 params, colsample 1.0) trained on all GPL24676; counts log1p +",
        "training scaler, arrays log2 + within-cohort standardisation. CI: stratified donor bootstrap",
        f"B={B_BOOT}. p: empirical, {B_NULL} random gene sets of equal size (broad = all candidates,",
        "SFM = pre-filtered pool). E-MTAB-2325 shares 35 donors with E-MTAB-8635 and is not pooled.",
        "",
        f"{'cohort':<14}{'tissue':<22}{'panel':<10}{'ALS/ctl':>8}{'genes':>7}{'AUC':>8}{'95% CI':>17}"
        f"{'broad':>8}{'p':>8}{'SFM':>8}{'p':>8}{'within':>8}",
        "-" * 126,
    ]
    for r in res.itertuples():
        within = f"{r.auc_within:>8.4f}" if not pd.isna(getattr(r, "auc_within", np.nan)) else f"{'':>8}"
        lines.append(
            f"{r.cohort:<14}{r.tissue:<22}{r.panel:<10}{f'{r.n_als}/{r.n_ctl}':>8}{f'{r.genes}/{r.of}':>7}"
            f"{r.auc:>8.4f}  [{r.lo:.3f}, {r.hi:.3f}]{r.broad_mean:>8.4f}{r.broad_p:>8.4f}"
            f"{r.SFM_mean:>8.4f}{r.SFM_p:>8.4f}{within}"
        )
    lines += ["", f"Pooled over {', '.join(POOLED)} (inverse-variance; null pools the i-th random set):"]
    for pname, p in pooled.items():
        lines.append(f"  {pname:<9} AUC {p['auc']:.4f} [{p['lo']:.3f}, {p['hi']:.3f}]  "
                     f"random-set mean {p['null_mean']:.4f}  p {p['p']:.4f}")
    lines += ["", "GSE26927 disease arms (cases vs own controls, arm standardised within itself):",
              f"{'arm':<7}{'tissue':<52}{'panel':<10}{'case/ctl':>9}{'genes':>7}{'AUC':>8}{'95% CI':>17}"]
    for r in arm_df.itertuples():
        lines.append(f"{r.arm:<7}{r.tissue:<52}{r.panel:<10}{f'{r.n_case}/{r.n_ctl}':>9}{r.genes:>7}"
                     f"{r.auc:>8.4f}  [{r.lo:.3f}, {r.hi:.3f}]")
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _plot(res: pd.DataFrame, pooled: dict, arm_df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.4, 1]})
    names = list(dict.fromkeys(res.cohort)) + ["Pooled (4 independent)"]
    for k, (pname, colour) in enumerate((("25-gene", "#d62728"), ("critical", "#1f77b4"))):
        sub = res[res.panel == pname].set_index("cohort")
        auc = [*sub.auc, pooled[pname]["auc"]]
        lo = [*sub.lo, pooled[pname]["lo"]]
        hi = [*sub.hi, pooled[pname]["hi"]]
        yy = np.arange(len(names)) + (k - 0.5) * 0.3
        ax.errorbar(auc, yy, xerr=[np.subtract(auc, lo), np.subtract(hi, auc)], fmt="o", color=colour,
                    capsize=3, label=f"{pname} panel")
        ax.scatter([*sub.broad_mean, pooled[pname]["null_mean"]], yy, marker="|", s=120, color="grey")
    ax.axvline(0.5, color="grey", ls=":", lw=1)
    ax.set_yticks(np.arange(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlabel("Zero-shot AUC (95% CI); grey tick = random-gene-set mean")
    ax.set_title("Independent cohorts: ALS vs control")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    sub = arm_df[arm_df.panel == "critical"]
    bx.barh(sub.arm, sub.auc, xerr=[sub.auc - sub.lo, sub.hi - sub.auc], color="#1f77b4", alpha=0.8, capsize=3)
    als = res[(res.cohort == "GSE26927 ALS") & (res.panel == "critical")].iloc[0]
    bx.barh(["ALS"], [als.auc], xerr=[[als.auc - als.lo], [als.hi - als.auc]], color="#d62728", alpha=0.8, capsize=3)
    bx.axvline(0.5, color="grey", ls=":", lw=1)
    bx.set_xlabel("Zero-shot AUC, cases vs own controls (critical panel)")
    bx.set_title("GSE26927: specificity across diseases")
    bx.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
