# ruff: noqa: E402
"""TOP1 genomic-damage axis check against the ALS panel (Zhou et al. 2026).

Zhou Z., Luquette L.J., Dong G. et al. (2026), "Recurrent patterns of
TOP1-mediated neuronal genomic damage shared by major neurodegenerative
disorders", Cell 189:1-17, doi:10.1016/j.cell.2026.06.013.

That study reports no per-gene damage score and deposits no open expression
data (all accessions are dbGaP-controlled), so its damage map cannot be joined
to our feature space directly.  What *can* be tested is whether the
transcriptional programmes it implicates are visible in GSE153960, and whether
our panel sits where a transcription-coupled damage process would predict.

Three checks
------------
A. DNA-repair / TOP1-axis gene set (their Fig S8/S10 pathway blocks) —
   differential expression ALS vs control, plus membership in the 3,704-feature
   SelectFromModel space, the top-500 SHAP ranking, and the core 25-gene panel.
B. Cryptic-exon gene set (their TDP-43 loss-of-function validation targets) —
   differential expression, plus Spearman correlation of each gene against the
   out-of-fold panel score, testing whether the panel tracks a TDP-43
   loss-of-function programme.
C. Expression-decile placement of the 25 panel genes — their damage burden
   scales with transcriptional activity genome-wide, so a panel reading that
   axis should be concentrated in the top expression deciles.

Outputs
-------
  top1_axis_check_statistics.txt  — full text report
  top1_axis_check.csv             — per-gene table for sets A and B
  top1_axis_check.png             — 3-panel figure (A volcano, B correlations,
                                    C decile placement)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ALS_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset
from utils import selected_lgbm_params  # noqa: E402

SCRIPT_DIR = Path(__file__).parent
PLATFORM = "GPL24676"

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PREFILTER_X_NPY = SCRIPT_DIR / "lgbm_prefilter_X.npy"
_PREFILTER_NAMES_TXT = SCRIPT_DIR / "lgbm_prefilter_names.txt"
_SFM_RANKING_CSV = SCRIPT_DIR / "lgbm_iter_final_shap_ranking.csv"
_TOP500_RANKING_CSV = SCRIPT_DIR / "lgbm_top500_final_shap_ranking.csv"

PREFIX = "top1_axis_check"
RANDOM_STATE = 42
N_SPLITS = 5

# ---------------------------------------------------------------------------
# Gene sets transcribed from Zhou et al. 2026 figure labels
# (the paper provides no machine-readable gene table; these are the curated
#  sets named in Fig S8/S10 pathway blocks and the cryptic-exon validation)
# ---------------------------------------------------------------------------

DNA_REPAIR_SET: dict[str, list[str]] = {
    "TOP1 / SSB repair": [
        "TOP1",
        "TDP1",
        "PARP1",
        "PARP2",
        "XRCC1",
        "POLB",
        "LIG3",
        "PNKP",
    ],
    "Base excision repair": [
        "NTHL1",
        "OGG1",
        "MUTYH",
        "NEIL1",
        "FEN1",
        "LIG1",
        "PCNA",
        "HUS1",
    ],
    "DSB repair / checkpoint": ["MRE11", "MUS81", "ATM", "ATR"],
    "Ribonucleotide metabolism": [
        "RNASEH2A",
        "RNASEH2B",
        "RNASEH2C",
        "RRM1",
        "RRM2B",
    ],
    "Proteostasis / autophagy": [
        "PIAS4",
        "RNF4",
        "USP13",
        "BECN1",
        "CHMP7",
        "ATG7",
        "SNAP29",
    ],
}

CRYPTIC_EXON_SET: list[str] = [
    "RAP1GAP",
    "STMN2",
    "ATP8A2",
    "KALRN",
    "NCAM1",
    "SHISA9",
    "GAD1",
    "GAD2",
]


# ---------------------------------------------------------------------------
# Symbol resolution
# ---------------------------------------------------------------------------


def _symbols_to_ensembl(symbols: list[str]) -> dict[str, list[str]]:
    """Map HGNC symbols to Ensembl gene IDs via MyGeneInfo.

    Args:
        symbols: HGNC gene symbols.

    Returns:
        Mapping from symbol to the list of Ensembl gene IDs reported for it.
    """
    import mygene

    mg = mygene.MyGeneInfo()
    out: dict[str, list[str]] = {}
    try:
        results = mg.querymany(
            symbols,
            scopes="symbol,alias",
            fields="ensembl.gene",
            species="human",
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"  [warn] mygene failed: {exc}")
        return out

    for r in results:
        ens = r.get("ensembl")
        if ens is None:
            continue
        ids = [ens["gene"]] if isinstance(ens, dict) else [e["gene"] for e in ens]
        out.setdefault(r["query"], []).extend(ids)
    return out


def _match_columns(
    ensembl_ids: list[str], col_base_to_full: dict[str, str]
) -> str | None:
    """Return the first versioned column matching any of *ensembl_ids*."""
    for eid in ensembl_ids:
        if eid in col_base_to_full:
            return col_base_to_full[eid]
    return None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _gene_stats(x_raw: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (univariate AUC >= 0.5, log2 fold-change, Mann-Whitney p).

    Args:
        x_raw: Raw (untransformed) RSEM counts for one gene. ``ds.X`` from
            :func:`utils.load_dataset` is raw, not log1p — verified directly
            (max 1.46e7). AUC and Mann-Whitney U are rank-based so they are
            unaffected by the transform; the fold-change is computed on the raw
            linear scale, which is why no ``expm1`` back-transform is applied.
        y: Binary labels (1 = ALS Spectrum MND, 0 = control).
    """
    from scipy.stats import mannwhitneyu
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(y, x_raw)
    auc_sym = max(auc, 1.0 - auc)

    mean_als = float(x_raw[y == 1].mean())
    mean_ctl = float(x_raw[y == 0].mean())
    log2fc = float(np.log2((mean_als + 1.0) / (mean_ctl + 1.0)))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        p = float(
            mannwhitneyu(x_raw[y == 1], x_raw[y == 0], alternative="two-sided")[1]
        )
    return float(auc_sym), log2fc, p


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        val = pvals[idx] * n / (n - rank + 1)
        prev = min(prev, val)
        adj[idx] = prev
    return np.clip(adj, 0.0, 1.0)


def _oof_panel_scores(X_panel: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
    """5-fold stratified out-of-fold panel probabilities.

    ``colsample_bytree`` is re-expressed for the panel feature space: the value
    of the selected classifier was tuned on 500 features (~25 per tree)
    and would otherwise floor to 1 feature per tree on a 25-gene panel. See
    ``cross_disease_specificity._adjust_colsample``.
    """
    from cross_disease_specificity import _adjust_colsample
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from utils import TunedLGBMClassifier as LGBMClassifier

    params = _adjust_colsample(params, X_panel.shape[1])
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y), dtype=float)
    for tr, te in skf.split(X_panel, y):
        sc = StandardScaler().fit(X_panel[tr])
        # params carries random_state and verbose (as in disease_specificity.py)
        clf = LGBMClassifier(**params)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            clf.fit(sc.transform(X_panel[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X_panel[te]))[:, 1]
    return oof


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all three TOP1-axis checks and write outputs."""

    import pandas as pd
    from scipy.stats import spearmanr

    print("=" * 72)
    print("TOP1-axis check vs Zhou et al. 2026 — GSE153960 GPL24676")
    print("=" * 72)

    (ds,) = load_dataset(
        "GSE153960", platform=PLATFORM, resources_dir=ALS_DIR / "resources"
    )
    y = ds.y.values.astype(int)
    print(f"  Samples: n={len(y)}  ALS={y.sum()}  Control={(y == 0).sum()}")

    col_base_to_full = {c.split(".")[0]: c for c in ds.X.columns}

    # --- resolve all requested symbols ---------------------------------
    all_symbols = [g for genes in DNA_REPAIR_SET.values() for g in genes]
    all_symbols += CRYPTIC_EXON_SET
    print(f"\n[1] Resolving {len(all_symbols)} symbols via MyGeneInfo ...")
    sym_to_ens = _symbols_to_ensembl(sorted(set(all_symbols)))
    print(f"  Resolved: {len(sym_to_ens)}/{len(set(all_symbols))}")

    # --- panel OOF score (needed for check B) ---------------------------
    panel = pd.read_csv(_PANEL_CSV)
    panel_features: list[str] = panel["feature"].tolist()
    panel_symbols: list[str] = panel["symbol"].tolist()

    all_names = _PREFILTER_NAMES_TXT.read_text().splitlines()
    name_to_col = {n: i for i, n in enumerate(all_names)}
    X_full = np.load(_PREFILTER_X_NPY, mmap_mode="r")
    panel_cols = [name_to_col[f] for f in panel_features]
    X_panel = np.asarray(X_full[:, panel_cols], dtype=np.float32)

    params = selected_lgbm_params()
    print("\n[2] Computing 5-fold OOF panel scores ...")
    oof = _oof_panel_scores(X_panel, y, params)
    from sklearn.metrics import roc_auc_score

    print(f"  OOF panel AUC = {roc_auc_score(y, oof):.4f}")

    # --- membership sets ------------------------------------------------
    sfm_feats = set(pd.read_csv(_SFM_RANKING_CSV)["feature"])
    top500_feats = set(pd.read_csv(_TOP500_RANKING_CSV)["feature"])
    panel_feats = set(panel_features)

    # --- checks A and B -------------------------------------------------
    rows: list[dict] = []
    for set_name, genes in list(DNA_REPAIR_SET.items()) + [
        ("Cryptic exon (TDP-43 LOF)", CRYPTIC_EXON_SET)
    ]:
        for sym in genes:
            col = _match_columns(sym_to_ens.get(sym, []), col_base_to_full)
            if col is None:
                rows.append(
                    {"set": set_name, "symbol": sym, "feature": None, "measured": False}
                )
                continue
            x = ds.X[col].values.astype(np.float64)
            auc, log2fc, p = _gene_stats(x, y)
            rho, prho = spearmanr(x, oof)
            rows.append(
                {
                    "set": set_name,
                    "symbol": sym,
                    "feature": col,
                    "measured": True,
                    "auc": auc,
                    "log2fc": log2fc,
                    "direction": "up_in_ALS" if log2fc > 0 else "down_in_ALS",
                    "p_mwu": p,
                    "spearman_r_vs_panel_score": float(rho),
                    "spearman_p": float(prho),
                    "in_sfm_3704": col in sfm_feats,
                    "in_top500_shap": col in top500_feats,
                    "in_core25_panel": col in panel_feats,
                }
            )

    df = pd.DataFrame(rows)
    meas = df["measured"].values
    padj = np.full(len(df), np.nan)
    padj[meas] = _bh_fdr(df.loc[meas, "p_mwu"].values)
    df["p_adj_bh"] = padj
    df.to_csv(SCRIPT_DIR / f"{PREFIX}.csv", index=False)

    # --- check C: expression deciles ------------------------------------
    print("\n[3] Expression-decile placement of the 25 panel genes ...")
    # Streamed over the memmap — avoids materialising the 874x47,822 matrix
    mean_expr = X_full.mean(axis=0)
    deciles = (
        np.digitize(mean_expr, np.percentile(mean_expr, np.arange(10, 100, 10))) + 1
    )
    panel_dec = {sym: int(deciles[c]) for sym, c in zip(panel_symbols, panel_cols)}

    _write_report(df, panel_dec, deciles, y, oof)
    _plot(df, panel_dec)
    print(f"\nWrote {PREFIX}.csv, {PREFIX}_statistics.txt, {PREFIX}.png")


def _write_report(
    df,
    panel_dec: dict[str, int],
    deciles: np.ndarray,
    y: np.ndarray,
    oof: np.ndarray,
) -> None:
    """Write the plain-text statistics report."""
    import collections

    from sklearn.metrics import roc_auc_score

    lines: list[str] = []
    add = lines.append
    add("TOP1-axis check vs Zhou et al. 2026 (Cell 189:1-17)")
    add("=" * 69)
    add("Dataset : GSE153960 GPL24676 post-mortem CNS, raw RSEM counts")
    add("          (AUC / MWU / Spearman are rank-based; log2FC on linear scale)")
    add(
        f"Samples : n={len(y)}  ALS Spectrum MND={int(y.sum())}  "
        f"Control={int((y == 0).sum())}"
    )
    add(
        f"OOF 25-gene panel AUC : {roc_auc_score(y, oof):.4f} "
        f"(5-fold stratified CV, random_state={RANDOM_STATE})"
    )
    add("")
    add("NOTE: Zhou et al. report no per-gene damage score and deposit no open")
    add("expression data (dbGaP phs003530.v2.p1 / phs001485.v3.p1 controlled;")
    add("code only at Zenodo 10.5281/zenodo.20174164). Their damage map cannot")
    add("be joined to our feature space. The gene sets below are transcribed")
    add("from their figure labels and tested in OUR cohort.")
    add("")

    n_meas = int(df["measured"].sum())
    add(f"Genes measured in GSE153960 : {n_meas}/{len(df)}")
    unmeasured = df.loc[~df["measured"], "symbol"].tolist()
    if unmeasured:
        add(f"Not measured : {', '.join(unmeasured)}")
    add("")

    for set_name in df["set"].unique():
        sub = df[(df["set"] == set_name) & df["measured"]].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("p_adj_bh")
        add("-" * 69)
        add(f"{set_name}  (n={len(sub)} measured)")
        add("-" * 69)
        add(
            f"{'symbol':<12}{'AUC':>7}{'log2FC':>9}{'BH-q':>11}"
            f"{'rho(score)':>12}  flags"
        )
        for _, r in sub.iterrows():
            flags = []
            if r["in_sfm_3704"]:
                flags.append("SFM")
            if r["in_top500_shap"]:
                flags.append("top500")
            if r["in_core25_panel"]:
                flags.append("PANEL")
            add(
                f"{r['symbol']:<12}{r['auc']:>7.4f}{r['log2fc']:>9.2f}"
                f"{r['p_adj_bh']:>11.3e}{r['spearman_r_vs_panel_score']:>12.3f}"
                f"  {','.join(flags) if flags else '-'}"
            )
        n_sig = int((sub["p_adj_bh"] < 0.05).sum())
        add(f"  BH-q < 0.05 : {n_sig}/{len(sub)}")
        add(f"  in SFM 3,704 space : {int(sub['in_sfm_3704'].sum())}/{len(sub)}")
        add(f"  in top-500 SHAP    : {int(sub['in_top500_shap'].sum())}/{len(sub)}")
        add(f"  in core-25 panel   : {int(sub['in_core25_panel'].sum())}/{len(sub)}")
        add("")

    add("-" * 69)
    add("Expression-decile placement of the 25-gene panel")
    add("-" * 69)
    add("Zhou et al. report damage burden rising with transcriptional activity")
    add("genome-wide. A panel reading that axis should concentrate in decile 10.")
    add("Deciles computed over all 47,822 pre-filtered features (mean log1p).")
    add("")
    counts = collections.Counter(panel_dec.values())
    for d in range(1, 11):
        bar = "#" * counts.get(d, 0)
        add(f"  decile {d:>2} : {counts.get(d, 0):>2}  {bar}")
    vals = np.array(list(panel_dec.values()))
    add("")
    add(f"  median decile = {np.median(vals):.1f}  mean = {vals.mean():.2f}")
    add(
        f"  panel genes in deciles 9-10 : {int((vals >= 9).sum())}/25 "
        f"(expected 5/25 if placement were uniform)"
    )
    add("")
    add(
        "  per-gene: "
        + ", ".join(
            f"{s}={d}" for s, d in sorted(panel_dec.items(), key=lambda kv: -kv[1])
        )
    )
    add("")

    (SCRIPT_DIR / f"{PREFIX}_statistics.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _plot(df, panel_dec: dict[str, int]) -> None:
    """Three-panel summary figure."""
    import collections

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # --- (a) volcano of the DNA-repair sets -----------------------------
    ax = axes[0]
    rep = df[df["measured"] & (df["set"] != "Cryptic exon (TDP-43 LOF)")]
    q = rep["p_adj_bh"].values
    ax.scatter(
        rep["log2fc"],
        -np.log10(np.clip(q, 1e-300, None)),
        c=np.where(q < 0.05, "#d65f5f", "#9aa0a6"),
        s=42,
        edgecolor="black",
        linewidth=0.4,
        zorder=3,
    )
    for _, r in rep.iterrows():
        if r["p_adj_bh"] < 0.05 and abs(r["log2fc"]) > 0.15:
            ax.annotate(
                r["symbol"],
                (r["log2fc"], -np.log10(max(r["p_adj_bh"], 1e-300))),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )
    ax.axhline(-np.log10(0.05), ls="--", c="black", lw=0.8, zorder=1)
    ax.axvline(0, ls=":", c="grey", lw=0.8, zorder=1)
    ax.set_xlabel("log$_2$ fold-change (ALS vs control)")
    ax.set_ylabel("$-$log$_{10}$ BH-$q$")
    ax.set_title("(a) DNA-repair / TOP1-axis genes\n(Zhou et al. Fig S8/S10 sets)")
    ax.grid(alpha=0.25, zorder=0)

    # --- (b) cryptic-exon correlation with panel score ------------------
    ax = axes[1]
    cry = df[df["measured"] & (df["set"] == "Cryptic exon (TDP-43 LOF)")]
    cry = cry.sort_values("spearman_r_vs_panel_score")
    cols = ["#4878d0" if v < 0 else "#d65f5f" for v in cry["spearman_r_vs_panel_score"]]
    ax.barh(
        cry["symbol"],
        cry["spearman_r_vs_panel_score"],
        color=cols,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axvline(0, c="black", lw=0.9)
    ax.set_xlabel(r"Spearman $\rho$ vs OOF panel score")
    ax.set_title("(b) TDP-43 cryptic-exon targets\nvs panel score")
    ax.grid(axis="x", alpha=0.25)

    # --- (c) expression-decile placement --------------------------------
    ax = axes[2]
    counts = collections.Counter(panel_dec.values())
    xs = list(range(1, 11))
    ys = [counts.get(d, 0) for d in xs]
    ax.bar(xs, ys, color="#6acc65", edgecolor="black", linewidth=0.6)
    ax.axhline(
        2.5, ls="--", c="black", lw=0.9, label="uniform expectation (2.5/decile)"
    )
    ax.set_xticks(xs)
    ax.set_xlabel("Mean-expression decile (all 47,822 features)")
    ax.set_ylabel("Panel genes")
    ax.set_title("(c) Where the 25-gene panel sits\non the expression axis")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / f"{PREFIX}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
