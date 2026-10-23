"""All-cohort gene discovery by random-effects meta-analysis (Step 102).

The goal here is identifying genes, not estimating classifier performance, so every
CNS cohort is used for discovery. A gene that moves the same way in independent
cohorts, tissues and platforms is stronger evidence than a gene chosen in one cohort.

Design, fixed before the run:
  * Units (independent donor sets, ALS vs non-neurological control, donor level):
      NYGC GPL24676; NYGC GPL16791 without the 38 donors shared with GPL24676
      (both: z-scored within tissue, then averaged per donor); GSE76220; GSE122649;
      SRP064478; GSE255683; GSE287256; GSE26927 (ALS arm); E-MTAB-8635.
    E-MTAB-2325 shares 35 donors with E-MTAB-8635 and enters only the cortex analysis.
  * Scale: counts as log(CPM + 1) keeping genes with CPM >= 1 in >= 25% of samples;
    DESeq2-normalised counts likewise; GSE76220 as loaded; arrays log2, dropping the
    25% lowest-mean genes. Genes keyed by HGNC symbol (most expressed row per symbol).
  * Effect: Hedges' g per gene and unit. Meta: DerSimonian-Laird random effects, BH
    FDR, I^2, direction consistency. A gene needs >= 4 units (>= 3 in subsets).
  * Robust: q < 0.05 and every leave-one-unit-out estimate keeps its sign with p < 0.05.
  * Subsets: spinal cord only, cortex only (NYGC tissue-specific donor means), and
    external only (no NYGC unit). Sensitivity: genes residualised within each unit on
    five marker-based cell-type scores (composition-adjusted).
  * Genes the discovery pipeline excluded upfront (known ALS genes and the
    contamination pool, which includes sex-chromosome genes) are flagged, not dropped.

Outputs: multicohort_meta_analysis.csv, multicohort_meta_analysis_statistics.txt,
multicohort_meta_analysis.png
"""

from __future__ import annotations

import functools
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ALS_DIR = Path(__file__).parents[1]
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(ALS_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from lgbm_iterative_pipeline import KNOWN_ALS_BASES  # noqa: E402
from new_cohort_validation import (  # noqa: E402
    _emtab,
    _gse26927_arms,
    _gse255683,
    _gse287256,
)
from nygc_clinical_association import _donor_means  # noqa: E402
from panel_critical_eval import (  # noqa: E402
    _load_gse76220,
    _load_gse122649,
    _load_srp064478,
)
from random_panel_null_cohorts import _symbols  # noqa: E402
from scipy.stats import norm  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402
from utils import critical_panel_symbols, load_dataset  # noqa: E402
from utils import CORE_PANEL_SIZE  # noqa: E402

RES = ALS_DIR / "resources"
OUT = SCRIPT_DIR / "multicohort_meta_analysis"
MIN_K, MIN_K_SUBSET = 4, 3
_MARKERS = {  # as snrna_spinal_cord.py; scores for the composition-adjusted sensitivity
    "neuron": ["RBFOX3", "SNAP25", "SYT1", "RBFOX1"],
    "astrocyte": ["AQP4", "GFAP", "SLC1A2", "GJA1", "ALDH1L1"],
    "microglia": ["CSF1R", "P2RY12", "CX3CR1", "C1QB", "AIF1"],
    "oligodendrocyte": ["PLP1", "MBP", "MOBP", "MOG", "CNP"],
    "endothelial": ["FLT1", "PECAM1", "ABCB1", "SLC2A1"],
}


@dataclass
class Unit:
    """Donors x symbols matrix for one independent donor set."""

    name: str
    tissue: str  # "spinal cord", "cortex" or "mixed"
    X: pd.DataFrame
    y: np.ndarray
    nygc: bool = False


def _to_symbols(df: pd.DataFrame, sym: dict[str, str] | None) -> pd.DataFrame:
    """samples x genes -> samples x symbols (most expressed column per symbol)."""
    if sym is not None:
        df = df.loc[:, [c for c in df.columns if c in sym]]
        df.columns = [sym[c] for c in df.columns]
    if df.columns.duplicated().any():
        order = np.argsort(-df.mean().to_numpy(), kind="stable")
        first = ~pd.Index(df.columns[order]).duplicated()
        df = df.iloc[:, np.sort(order[first])]
    return df


_ALIAS_CACHE = RES / "platforms" / "symbol_to_current_hgnc.json"


def _current_symbols(names: list[str], current: set[str]) -> dict[str, str]:
    """Legacy symbol -> current HGNC symbol via MyGeneInfo (alias/retired scopes; cached).

    Array annotations (GPL6255, GPL6480) and older count tables use symbols that HGNC
    has since renamed, so without this a renamed gene splits into two rows and loses
    the units that use the old name. Only unambiguous hits onto a current symbol count.
    """
    import mygene

    cache = json.loads(_ALIAS_CACHE.read_text()) if _ALIAS_CACHE.exists() else {}
    todo = sorted({n for n in names if n not in current and n not in cache})
    if todo:
        hits: dict[str, set[str]] = {}
        for r in mygene.MyGeneInfo().querymany(todo, scopes="symbol,alias,retired", fields="symbol",
                                               species="human", verbose=False):
            if not r.get("notfound") and r.get("symbol"):
                hits.setdefault(r["query"], set()).add(r["symbol"])
        cache.update({n: (next(iter(hits[n])) if len(hits.get(n, ())) == 1 else "") for n in todo})
        _ALIAS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _ALIAS_CACHE.write_text(json.dumps(cache))
    return {n: cache[n] for n in names if cache.get(n) and cache[n] in current and cache[n] != n}


def _log_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    """log(CPM + 1), keeping genes with CPM >= 1 in at least 25% of samples."""
    cpm = counts.div(counts.sum(axis=1), axis=0) * 1e6
    return np.log1p(cpm.loc[:, (cpm >= 1).mean() >= 0.25])


def _array_filter(x: pd.DataFrame) -> pd.DataFrame:
    return x.loc[:, x.mean() >= x.mean().quantile(0.25)]


@functools.cache
def _nygc_ds(platform: str):
    (ds,) = load_dataset("GSE153960", platform=platform, resources_dir=RES)
    t = ds.meta.loc[ds.y.index, "tissue"].astype(str)
    print(f"  {platform} tissues: {t.value_counts().to_dict()}")
    return ds


def _nygc(platform: str, sym: dict[str, str], drop_donors: frozenset, tissue_class: str | None) -> tuple[Unit, set[str]]:
    ds = _nygc_ds(platform)
    meta = ds.meta.loc[ds.y.index]
    donor = meta["subject id"].astype(str)
    tissue = meta["tissue"].astype(str)
    cls = np.where(tissue.str.contains("Spinal", case=False), "spinal cord",
                   np.where(tissue.str.contains("Cortex", case=False), "cortex", "other"))
    keep = ~donor.isin(drop_donors).to_numpy()
    if tissue_class:
        keep &= cls == tissue_class
    counts = pd.DataFrame(ds.X.to_numpy(np.float32)[keep], columns=[c.split(".")[0] for c in ds.X.columns])
    x = _to_symbols(_log_cpm(counts), sym)
    x = _donor_means(x, tissue[keep].reset_index(drop=True), donor[keep].reset_index(drop=True))
    y = pd.Series(ds.y.to_numpy()[keep], index=donor[keep].to_numpy()).groupby(level=0).first()
    name = f"NYGC {platform}" + (" (disjoint)" if drop_donors else "")
    return Unit(name, tissue_class or "mixed", x, y.loc[x.index].to_numpy().astype(int), True), set(donor)


def _from_loader(name: str, tissue: str, x: np.ndarray, genes: list, y: np.ndarray, kind: str,
                 sym: dict[str, str] | None) -> Unit:
    df = pd.DataFrame(np.asarray(x, dtype=np.float64), columns=[str(g).split(".")[0] if sym else str(g) for g in genes])
    if kind == "counts":
        df = _log_cpm(np.expm1(df))
    elif kind == "array":
        df = _array_filter(df)
    df = _to_symbols(df.loc[:, df.std() > 0], sym)
    return Unit(name, tissue, df, np.asarray(y).astype(int))


def _harmonise(units: list[Unit], current: set[str]) -> None:
    """Rename legacy symbols to current HGNC symbols in place, then re-collapse."""
    for u in units:
        rename = _current_symbols(list(u.X.columns), current)
        if rename:
            u.X = _to_symbols(u.X.rename(columns=rename), None)
            print(f"  {u.name}: {len(rename)} legacy symbols renamed")


def _hedges(u: Unit) -> tuple[pd.Series, pd.Series]:
    a, c = u.X[u.y == 1], u.X[u.y == 0]
    n1, n0 = len(a), len(c)
    sp = np.sqrt(((n1 - 1) * a.var() + (n0 - 1) * c.var()) / (n1 + n0 - 2))
    j = 1 - 3 / (4 * (n1 + n0) - 9)
    g = j * (a.mean() - c.mean()) / sp.replace(0, np.nan)
    v = (n1 + n0) / (n1 * n0) + g**2 / (2 * (n1 + n0))
    return g, v


def _meta(G: pd.DataFrame, V: pd.DataFrame, min_k: int) -> pd.DataFrame:
    """DerSimonian-Laird random effects per gene (rows) across units (columns)."""
    w = 1 / V
    k = G.notna().sum(axis=1)
    fe = (w * G).sum(axis=1) / w.sum(axis=1)
    q = (w * G.sub(fe, axis=0) ** 2).sum(axis=1)
    c = w.sum(axis=1) - (w**2).sum(axis=1) / w.sum(axis=1)
    tau2 = ((q - (k - 1)) / c).clip(lower=0)
    ws = 1 / V.add(tau2, axis=0)
    est = (ws * G).sum(axis=1) / ws.sum(axis=1)
    se = 1 / np.sqrt(ws.sum(axis=1))
    p = 2 * norm.sf(np.abs(est / se))
    out = pd.DataFrame(dict(g=est, lo=est - 1.96 * se, hi=est + 1.96 * se, p=p, k=k,
                            i2=((q - (k - 1)) / q).clip(lower=0).fillna(0),
                            same_sign=(np.sign(G).eq(np.sign(est), axis=0)).sum(axis=1)))
    out = out[(out.k >= min_k) & out.p.notna()].copy()
    out["q"] = multipletests(out.p, method="fdr_bh")[1]
    return out


def _residualise(u: Unit) -> Unit:
    """Genes regressed on marker-based cell-type scores within the unit."""
    z = (u.X - u.X.mean()) / u.X.std().replace(0, np.nan)
    scores = [z[[g for g in mk if g in z]].mean(axis=1) for mk in _MARKERS.values() if any(g in z for g in mk)]
    d = np.column_stack([np.ones(len(z)), *scores])
    xv = u.X.to_numpy()
    beta = np.linalg.lstsq(d, np.nan_to_num(xv), rcond=None)[0]
    return Unit(u.name, u.tissue, pd.DataFrame(xv - d @ beta, index=u.X.index, columns=u.X.columns), u.y, u.nygc)


def _effects(units: list[Unit]) -> tuple[pd.DataFrame, pd.DataFrame]:
    gs, vs = {}, {}
    for u in units:
        gs[u.name], vs[u.name] = _hedges(u)
    return pd.DataFrame(gs), pd.DataFrame(vs)


def build_units() -> dict:
    """Load, scale and harmonise every unit; shared with minimal_panel_stability.py."""
    clean = [n.split(".")[0] for n in (SCRIPT_DIR / "lgbm_prefilter_names.txt").read_text().splitlines()]
    pool = json.loads((SCRIPT_DIR / "lgbm_iter_contamination_pool.json").read_text())
    sym = _symbols(clean)
    sym.update({str(k).split(".")[0]: v for k, v in pool.get("symbol_map", {}).items() if v})
    excluded = {sym.get(str(b).split(".")[0]) for b in list(pool["pool"]) + list(KNOWN_ALS_BASES)} - {None}

    print("Loading units ...")
    u24, donors24 = _nygc("GPL24676", sym, frozenset(), None)
    donors24 = frozenset(donors24)
    u16, _ = _nygc("GPL16791", sym, donors24, None)
    x76, v76, y76 = _load_gse76220()
    x122, v122, y122 = _load_gse122649()
    xsrp, vsrp, ysrp = _load_srp064478()
    c255, c287, e8635, e2325 = _gse255683(), _gse287256(), _emtab("E-MTAB-8635"), _emtab("E-MTAB-2325")
    a26 = _gse26927_arms()["ALS"]
    units = [
        u24, u16,
        _from_loader("GSE76220", "spinal cord", x76, v76, y76, "asis", None),
        _from_loader("GSE122649", "cortex", x122, v122, y122, "counts", None),
        _from_loader("SRP064478", "spinal cord", xsrp, vsrp, ysrp, "counts", sym),
        _from_loader("GSE255683", "spinal cord", c255.X, c255.genes, c255.y, "counts", sym),
        _from_loader("GSE287256", "spinal cord", c287.X, c287.genes, c287.y, "counts", sym),
        _from_loader("GSE26927", "spinal cord", a26.X, a26.genes, a26.y, "array", None),
        _from_loader("E-MTAB-8635", "spinal cord", e8635.X, e8635.genes, e8635.y, "array", None),
    ]
    mc = _from_loader("E-MTAB-2325", "cortex", e2325.X, e2325.genes, e2325.y, "array", None)
    _harmonise(units + [mc], set(sym.values()))
    sc24, _ = _nygc("GPL24676", sym, frozenset(), "spinal cord")
    sc16, _ = _nygc("GPL16791", sym, donors24, "spinal cord")
    cx24, _ = _nygc("GPL24676", sym, frozenset(), "cortex")
    cx16, _ = _nygc("GPL16791", sym, donors24, "cortex")
    for u in units + [mc]:
        print(f"  {u.name:<22} {u.tissue:<12} donors {len(u.y):>4} ({int(u.y.sum())} ALS)  genes {u.X.shape[1]}")
    return dict(units=units, mc=mc, sc=[sc24, sc16], cx=[cx24, cx16], excluded=excluded)


def main() -> None:
    b = build_units()
    units, mc, excluded = b["units"], b["mc"], b["excluded"]
    (sc24, sc16), (cx24, cx16) = b["sc"], b["cx"]
    G, V = _effects(units)
    res = _meta(G, V, MIN_K)
    # leave-one-unit-out robustness
    loo_ok = pd.Series(True, index=res.index)
    for col in G.columns:
        m = _meta(G.drop(columns=col).loc[res.index], V.drop(columns=col).loc[res.index], MIN_K - 1)
        m = m.reindex(res.index)
        loo_ok &= (np.sign(m.g) == np.sign(res.g)) & (m.p < 0.05)
    res["robust"] = (res.q < 0.05) & loo_ok
    subsets = {
        "sc": [sc24, sc16] + [u for u in units if u.tissue == "spinal cord" and not u.nygc],
        "cortex": [cx24, cx16, units[3], mc],
        "ext": [u for u in units if not u.nygc],
    }
    for tag, us in subsets.items():
        m = _meta(*_effects(us), MIN_K_SUBSET)
        res = res.join(m[["g", "q"]].add_prefix(f"{tag}_"), how="left")
    ma = _meta(*_effects([_residualise(u) for u in units]), MIN_K)
    res = res.join(ma[["g", "q"]].add_prefix("adj_"), how="left")
    res["excluded_upfront"] = res.index.isin(excluded)
    panel25 = pd.read_csv(SCRIPT_DIR / "lgbm_core25_panel.csv").symbol.tolist()
    crit = set(critical_panel_symbols())
    res["panel"] = np.where(res.index.isin(crit), "critical", np.where(res.index.isin(panel25), f"{CORE_PANEL_SIZE}-gene", ""))
    res = res.join(G.add_prefix("g_"), how="left").sort_values("p")
    res.to_csv(OUT.with_suffix(".csv"), index_label="gene")
    _report(res, units, mc, panel25)
    _plot(res, G)


def _report(res: pd.DataFrame, units: list[Unit], mc: Unit, panel25: list[str]) -> None:
    novel = res[res.robust & ~res.excluded_upfront]
    lines = [
        "All-cohort gene discovery: random-effects meta-analysis of Hedges' g (donor level)",
        "=" * 88,
        f"{'unit':<22}{'tissue':<13}{'donors':>7}{'ALS':>5}{'genes':>7}",
        *[f"{u.name:<22}{u.tissue:<13}{len(u.y):>7}{int(u.y.sum()):>5}{u.X.shape[1]:>7}" for u in units],
        f"{mc.name:<22}{mc.tissue:<13}{len(mc.y):>7}{int(mc.y.sum()):>5}{mc.X.shape[1]:>7}  (cortex analysis only)",
        "",
        f"Genes tested (>= {MIN_K} units): {len(res)}",
        f"  q < 0.05: {int((res.q < 0.05).sum())}   robust (q < 0.05 and every leave-one-unit-out p < 0.05, same sign): "
        f"{int(res.robust.sum())}   of which not excluded upfront: {len(novel)}",
        f"  robust and q < 0.05 without NYGC (external only): {int((res.robust & (res.ext_q < 0.05)).sum())}",
        f"  robust and q < 0.05 after composition adjustment: {int((res.robust & (res.adj_q < 0.05)).sum())}",
        "",
        "Top 40 robust genes not excluded upfront, by |g| (g > 0: higher in ALS):",
        f"{'gene':<12}{'g':>7}{'95% CI':>17}{'q':>10}{'k':>3}{'sign':>6}{'I2':>6}{'SC g':>7}{'ctx g':>7}{'ext q':>9}{'adj q':>9}  panel",
    ]
    top = novel.reindex(novel.g.abs().sort_values(ascending=False).index).head(40)
    for gname, r in top.iterrows():
        lines.append(f"{gname:<12}{r.g:>7.2f}  [{r.lo:+.2f}, {r.hi:+.2f}]{r.q:>10.2e}{int(r.k):>3}{int(r.same_sign):>4}/{int(r.k)}"
                     f"{r.i2:>6.2f}{r.sc_g:>7.2f}{r.cortex_g:>7.2f}{r.ext_q:>9.2e}{r.adj_q:>9.2e}  {r.panel}")
    lines += ["", f"{CORE_PANEL_SIZE}-gene panel members:",
              f"{'gene':<18}{'g':>7}{'q':>10}{'k':>3}{'robust':>8}{'ext q':>10}{'adj q':>10}{'SC g':>7}{'ctx g':>7}"]
    for gname in panel25:
        if gname in res.index:
            r = res.loc[gname]
            lines.append(f"{gname:<18}{r.g:>7.2f}{r.q:>10.2e}{int(r.k):>3}{str(bool(r.robust)):>8}{r.ext_q:>10.2e}"
                         f"{r.adj_q:>10.2e}{r.sc_g:>7.2f}{r.cortex_g:>7.2f}")
        else:
            lines.append(f"{gname:<18}  measured in fewer than {MIN_K} units")
    n_rob = int(res.loc[res.index.isin(panel25)].robust.sum())
    lines += ["", f"Panel members robust: {n_rob} of {sum(g in res.index for g in panel25)} testable."]
    OUT.with_name(OUT.name + "_statistics.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _plot(res: pd.DataFrame, G: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1, 1.2]})
    lq = -np.log10(res.q.clip(lower=1e-300))
    ax.scatter(res.g, lq, s=3, c="#bbbbbb", rasterized=True)
    rob = res.robust & ~res.excluded_upfront
    ax.scatter(res.g[rob], lq[rob], s=5, c="#1f77b4", label="robust")
    pan = res.panel != ""
    ax.scatter(res.g[pan], lq[pan], s=30, c="#d62728", label=f"{CORE_PANEL_SIZE}-gene panel")
    for gname in res.index[pan]:
        ax.annotate(gname, (res.g[gname], lq[gname]), fontsize=6)
    ax.axhline(-np.log10(0.05), color="grey", ls=":", lw=1)
    ax.set_xlabel("meta-analytic Hedges' g (ALS vs control)")
    ax.set_ylabel("-log10 q")
    ax.legend(fontsize=8)
    ax.set_title("All-cohort random-effects meta-analysis")
    top = res[rob].reindex(res[rob].g.abs().sort_values(ascending=False).index).head(30).index
    m = G.loc[top]
    im = bx.imshow(m.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    bx.set_yticks(range(len(top)), top, fontsize=7)
    bx.set_xticks(range(m.shape[1]), m.columns, rotation=40, ha="right", fontsize=7)
    bx.set_title("Per-unit Hedges' g, top 30 robust genes")
    fig.colorbar(im, ax=bx, fraction=0.03)
    plt.tight_layout()
    fig.savefig(OUT.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        main()
