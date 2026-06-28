# ruff: noqa: E402
"""Panel Gene Markov Blanket Network for Drug Target Discovery.

For each of the 11 protein-coding critical-panel genes, discovers its sparse
transcriptional neighbourhood within the 3,704 SFM gene space using LASSO
neighbourhood selection (Meinshausen & Bühlmann 2006).

Method: LassoCV (5-fold CV, coordinate descent).  For gene X as target,
non-zero LASSO coefficients identify genes whose expression is conditionally
related to X after controlling for all others — the sparse Gaussian graphical
model neighbourhood, which approximates the Markov Blanket.

Cohorts
-------
  Primary  : GSE153960 GPL24676  (n = 874,  3,704 SFM features)
  Replicate: GSE153960 GPL16791  (n = 636,  same feature space)

An edge X → neighbour Y is *replicated* if Pearson |r| > 0.10 with the same
sign in GPL16791 (lenient threshold given n=636 and shared composition).

Outputs
-------
  panel_mb_results.csv         — full neighbourhood per panel gene
  panel_mb_replicated.csv      — replicated edges only
  panel_mb_network.png         — network visualisation
  panel_mb_statistics.txt      — human-readable summary + ORA
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ALS_DIR = Path(__file__).parents[1]
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PREFILTER_X = SCRIPT_DIR / "lgbm_prefilter_X.npy"
_PREFILTER_NAMES = SCRIPT_DIR / "lgbm_prefilter_names.txt"
_SFM_RANKING = SCRIPT_DIR / "lgbm_iter_final_shap_ranking.csv"
_PROFILING_CSV = SCRIPT_DIR / "gene_profiling_summary.csv"

PLATFORM_PRIMARY = "GPL24676"
PLATFORM_REPLICATE = "GPL16791"
LASSO_CV = 5           # folds for LassoCV alpha tuning
LASSO_JOBS = -1        # parallel jobs (-1 = all cores)
REPL_R_THRESHOLD = 0.10  # minimum |Pearson r| in GPL16791 for replication

# 12 protein-coding members of the 15-gene greedy-tail critical panel
# (greedy backward elimination peak at k=15, W.mean = 0.9029; equal cohort weights)
_CRITICAL_CODING_GENES: frozenset[str] = frozenset({
    "FCN3", "PROS1", "ANGPT2", "TINAGL1", "CKMT2", "CLDN5",
    "NR4A1", "SOHLH2", "HEXB", "MCEE", "SLC37A2", "SERTAD1",
})

# ORA libraries
_ORA_LIBRARIES: list[str] = [
    "GO_Biological_Process_2023",
    "KEGG_2021_Human",
    "Reactome_2022",
]
_ORA_PADJ = 0.05
_ORA_DELAY = 1.2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_sfm_matrix() -> tuple[np.ndarray, list[str]]:
    """Slice the pre-filter cache to the 3,704 SFM features (GPL24676)."""
    sfm_df = pd.read_csv(_SFM_RANKING)
    sfm_features: list[str] = sfm_df["feature"].tolist()

    all_names = _PREFILTER_NAMES.read_text().splitlines()
    name_to_col = {n: i for i, n in enumerate(all_names)}

    kept = [f for f in sfm_features if f in name_to_col]
    col_idx = [name_to_col[f] for f in kept]

    X_full = np.load(_PREFILTER_X)
    return X_full[:, col_idx].astype(np.float32), kept


def _load_gpl16791_sfm(sfm_features: list[str]) -> np.ndarray | None:
    """Load GPL16791 and subset to SFM features (log1p)."""
    try:
        (ds,) = load_dataset(
            "GSE153960",
            platform=PLATFORM_REPLICATE,
            resources_dir=ALS_DIR / "resources",
        )
    except Exception as exc:
        print(f"  [warn] Could not load GPL16791: {exc}")
        return None

    feat_to_col = {f: i for i, f in enumerate(ds.X.columns)}
    col_idx: list[int] = []
    kept_mask: list[bool] = []
    for f in sfm_features:
        if f in feat_to_col:
            col_idx.append(feat_to_col[f])
            kept_mask.append(True)
        else:
            kept_mask.append(False)

    n_kept = sum(kept_mask)
    if n_kept < len(sfm_features) * 0.5:
        print(f"  [warn] GPL16791 covers only {n_kept}/{len(sfm_features)} SFM features")
        return None

    X_raw = ds.X.values.astype(np.float32)
    X_sub = np.full((len(X_raw), len(sfm_features)), np.nan, dtype=np.float32)
    local_j = 0
    for sfm_i, present in enumerate(kept_mask):
        if present:
            X_sub[:, sfm_i] = X_raw[:, col_idx[local_j]]
            local_j += 1
    return np.log1p(np.nan_to_num(X_sub, nan=0.0))


# ---------------------------------------------------------------------------
# Panel gene column mapping
# ---------------------------------------------------------------------------


def _resolve_panel_cols(
    sfm_features: list[str],
    prof_df: pd.DataFrame,
) -> dict[str, int]:
    """Map critical-panel gene symbols → SFM column index."""
    base_to_col = {f.split(".")[0]: i for i, f in enumerate(sfm_features)}
    sym_to_col: dict[str, int] = {}
    for _, row in prof_df.iterrows():
        sym = str(row.get("symbol", ""))
        if sym not in _CRITICAL_CODING_GENES:
            continue
        base = str(row.get("feature", "")).split(".")[0]
        if base in base_to_col:
            sym_to_col[sym] = base_to_col[base]
    return sym_to_col


# ---------------------------------------------------------------------------
# LASSO neighbourhood selection
# ---------------------------------------------------------------------------


def _lasso_neighbours(
    X: np.ndarray,
    target_col: int,
) -> tuple[list[int], float]:
    """Run LassoCV with gene at target_col as target; return neighbour cols."""
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    y = X[:, target_col].astype(np.float64)
    cand_cols = [i for i in range(X.shape[1]) if i != target_col]
    X_cand = X[:, cand_cols].astype(np.float64)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_cand)
    y_sc = (y - y.mean()) / (y.std() + 1e-12)

    lasso = LassoCV(
        cv=LASSO_CV,
        n_jobs=LASSO_JOBS,
        max_iter=3000,
        random_state=42,
    ).fit(X_sc, y_sc)

    nz = np.where(lasso.coef_ != 0)[0]
    nb_cols = [cand_cols[i] for i in nz]
    return nb_cols, float(lasso.alpha_)


# ---------------------------------------------------------------------------
# Replication in GPL16791
# ---------------------------------------------------------------------------


def _replicate_neighbours(
    X16: np.ndarray,
    target_col: int,
    nb_cols: list[int],
    primary_X: np.ndarray,
    primary_r_threshold: float = REPL_R_THRESHOLD,
) -> list[int]:
    """Return nb_cols whose |Pearson r| > threshold and same sign in GPL16791."""
    if X16 is None or not nb_cols:
        return []

    y16 = X16[:, target_col].astype(np.float64)
    y_pri = primary_X[:, target_col].astype(np.float64)

    replicated: list[int] = []
    for col in nb_cols:
        x16 = X16[:, col].astype(np.float64)
        x_pri = primary_X[:, col].astype(np.float64)
        r16 = float(np.corrcoef(x16, y16)[0, 1])
        r_pri = float(np.corrcoef(x_pri, y_pri)[0, 1])
        if abs(r16) >= primary_r_threshold and np.sign(r16) == np.sign(r_pri):
            replicated.append(col)
    return replicated


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------


def _annotate_symbols(ensg_bases: list[str]) -> dict[str, str]:
    """Batch mygene query for HGNC symbols."""
    import mygene

    mg = mygene.MyGeneInfo()
    out: dict[str, str] = {}
    if not ensg_bases:
        return out
    try:
        results = mg.querymany(
            ensg_bases,
            scopes="ensembl.gene",
            fields="symbol",
            species="human",
            verbose=False,
        )
        for r in results:
            if "symbol" in r:
                out[r["query"]] = r["symbol"]
    except Exception as exc:
        print(f"  [warn] mygene failed: {exc}")
    return out


# ---------------------------------------------------------------------------
# ORA
# ---------------------------------------------------------------------------


def _run_ora(symbols: list[str]) -> pd.DataFrame:
    """Run Enrichr ORA across _ORA_LIBRARIES."""
    import gseapy

    rows: list[pd.DataFrame] = []
    for lib in _ORA_LIBRARIES:
        time.sleep(_ORA_DELAY)
        try:
            enr = gseapy.enrichr(
                gene_list=symbols,
                gene_sets=lib,
                outdir=None,
                no_plot=True,
                verbose=False,
            )
            df = enr.results
            if df is not None and not df.empty:
                df = df[df["Adjusted P-value"] < _ORA_PADJ].copy()
                df["Library"] = lib
                rows.append(df)
        except Exception as exc:
            print(f"    [warn] ORA {lib}: {exc}")
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values("Combined Score", ascending=False)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _plot_network(
    mb_records: list[dict],
    sym_to_col: dict[str, int],
    sfm_features: list[str],
    col_to_sym: dict[int, str],
    out_path: Path,
) -> None:
    """Bipartite network: panel genes (left) ↔ top-30 MB candidates (right)."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    panel_genes = sorted(sym_to_col.keys())
    mb_count: dict[int, set[str]] = {}
    for rec in mb_records:
        for col in rec["nb_cols"]:
            mb_count.setdefault(col, set()).add(rec["symbol"])

    mb_sorted = sorted(mb_count.items(), key=lambda x: -len(x[1]))[:30]
    mb_col_set = {col for col, _ in mb_sorted}

    fig, ax = plt.subplots(figsize=(14, max(8, len(panel_genes) * 0.7 + 2)))
    ax.axis("off")

    y_panel = {g: i for i, g in enumerate(reversed(panel_genes))}
    y_mb = {col: i for i, (col, _) in enumerate(mb_sorted)}

    for rec in mb_records:
        pg = rec["symbol"]
        py = y_panel.get(pg, 0)
        rep_set = set(rec.get("replicated_cols", []))
        for col in rec["nb_cols"]:
            if col in mb_col_set:
                my = y_mb[col]
                rep = col in rep_set
                lc = "#2ca02c" if rep else "#aec7e8"
                lw = 1.8 if rep else 0.8
                ax.plot([0, 1], [py, my], color=lc, lw=lw, alpha=0.6, zorder=1)

    for g, py in y_panel.items():
        ax.scatter(0, py, s=200, color="#1f77b4", zorder=3)
        ax.text(-0.03, py, g, ha="right", va="center", fontsize=9, fontweight="bold")

    for col, panel_set in mb_sorted:
        my = y_mb[col]
        sym = col_to_sym.get(col, sfm_features[col].split(".")[0][:14])
        n_pg = len(panel_set)
        ax.scatter(1, my, s=80 + 70 * n_pg, color="#ff7f0e", zorder=3, alpha=0.85)
        ax.text(1.03, my, sym, ha="left", va="center", fontsize=7.5)

    ax.set_xlim(-0.35, 1.55)
    ax.set_ylim(-1, max(len(panel_genes), len(mb_sorted)) + 0.5)
    ax.set_title(
        "Panel gene LASSO neighbourhood network\n"
        "Left: 11 protein-coding critical-panel genes  |  "
        "Right: top-30 candidates (size ∝ panel genes connected)\n"
        "Green: replicated in GPL16791  |  Blue: GPL24676 only",
        fontsize=9,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + "=" * 65)
    print("Panel Gene LASSO Neighbourhood Network")
    print("=" * 65)

    print("\n[1] Loading SFM matrix (GPL24676) ...")
    X_primary, sfm_features = _load_sfm_matrix()
    n_prim, p_sfm = X_primary.shape
    print(f"  n={n_prim},  p_SFM={p_sfm}")

    print("\n[2] Loading GPL16791 for replication ...")
    X_replicate = _load_gpl16791_sfm(sfm_features)
    if X_replicate is not None:
        print(f"  n={X_replicate.shape[0]},  p={X_replicate.shape[1]}")
    else:
        print("  Not available — replication step skipped")

    prof_df = pd.read_csv(_PROFILING_CSV)
    sym_to_col = _resolve_panel_cols(sfm_features, prof_df)
    print(f"\n[3] Resolved {len(sym_to_col)}/{len(_CRITICAL_CODING_GENES)} panel genes")
    for sym, col in sorted(sym_to_col.items()):
        print(f"  {sym:<12} → col {col}  ({sfm_features[col]})")

    print(f"\n[4] LASSO neighbourhood (CV={LASSO_CV}, jobs={LASSO_JOBS}) per gene ...")
    mb_records: list[dict] = []
    all_nb_cols: set[int] = set()

    for sym, target_col in sorted(sym_to_col.items()):
        print(f"\n  [{sym}] ...", flush=True)
        t0 = time.time()
        nb_cols, alpha = _lasso_neighbours(X_primary, target_col)
        elapsed = time.time() - t0
        print(f"    neighbours={len(nb_cols)},  alpha={alpha:.5f},  {elapsed:.1f}s", flush=True)

        rep_cols: list[int] = []
        if X_replicate is not None:
            rep_cols = _replicate_neighbours(X_replicate, target_col, nb_cols, X_primary)
            print(f"    replicated={len(rep_cols)}/{len(nb_cols)}", flush=True)

        mb_records.append({
            "symbol": sym,
            "target_col": target_col,
            "nb_cols": nb_cols,
            "replicated_cols": rep_cols,
            "lasso_alpha": alpha,
        })
        all_nb_cols.update(nb_cols)

    print("\n[5] Annotating candidate symbols via mygene ...", flush=True)
    nb_bases = list({sfm_features[c].split(".")[0] for c in all_nb_cols})
    base_to_sym = _annotate_symbols(nb_bases)
    col_to_sym: dict[int, str] = {
        col: base_to_sym.get(sfm_features[col].split(".")[0], sfm_features[col].split(".")[0])
        for col in all_nb_cols
    }

    rows: list[dict] = []
    for rec in mb_records:
        sym = rec["symbol"]
        rep_set = set(rec["replicated_cols"])
        for col in rec["nb_cols"]:
            nb_sym = col_to_sym.get(col, sfm_features[col].split(".")[0])
            x = X_primary[:, col].astype(np.float64)
            y = X_primary[:, rec["target_col"]].astype(np.float64)
            r = float(np.corrcoef(x, y)[0, 1])
            rows.append({
                "panel_gene": sym,
                "nb_gene": nb_sym,
                "nb_feature": sfm_features[col],
                "pearson_r": round(r, 4),
                "replicated": col in rep_set,
            })

    results_df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["panel_gene", "nb_gene", "nb_feature", "pearson_r", "replicated"]
    )
    out_csv = SCRIPT_DIR / "panel_mb_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"  Saved → {out_csv.name}  ({len(results_df)} total edges)")

    rep_df = results_df[results_df["replicated"]] if not results_df.empty else results_df
    out_rep = SCRIPT_DIR / "panel_mb_replicated.csv"
    rep_df.to_csv(out_rep, index=False)
    print(f"  Saved → {out_rep.name}  ({len(rep_df)} replicated edges)")

    print("\n[6] ORA on replicated neighbours ...", flush=True)
    rep_symbols = sorted({
        r for r in rep_df["nb_gene"].tolist()
        if not str(r).startswith("ENSG") and not str(r).startswith("LOC")
    }) if not rep_df.empty else []
    print(f"  {len(rep_symbols)} candidate symbols")
    ora_df = _run_ora(rep_symbols) if rep_symbols else pd.DataFrame()
    if not ora_df.empty:
        out_ora = SCRIPT_DIR / "panel_mb_ora_results.csv"
        ora_df.to_csv(out_ora, index=False)
        print(f"  Saved → {out_ora.name}")

    print("\n[7] Plotting ...", flush=True)
    _plot_network(mb_records, sym_to_col, sfm_features, col_to_sym,
                  SCRIPT_DIR / "panel_mb_network.png")

    print("\n[8] Writing statistics ...", flush=True)
    lines: list[str] = [
        "Panel Gene LASSO Neighbourhood Network",
        "=" * 65,
        f"Method: LASSO neighbourhood selection (Meinshausen & Bühlmann 2006)",
        f"Primary cohort  : {PLATFORM_PRIMARY}  n={n_prim}",
        f"Replicate cohort: {PLATFORM_REPLICATE}  "
        + (f"n={X_replicate.shape[0]}" if X_replicate is not None else "NOT LOADED"),
        f"SFM search space: {p_sfm} genes",
        f"Replication threshold: |r| ≥ {REPL_R_THRESHOLD}, same sign",
        "",
    ]

    for rec in mb_records:
        sym = rec["symbol"]
        nb_syms = [col_to_sym.get(c, sfm_features[c].split(".")[0]) for c in rec["nb_cols"]]
        rep_syms = [col_to_sym.get(c, sfm_features[c].split(".")[0]) for c in rec["replicated_cols"]]
        sub = results_df[results_df["panel_gene"] == sym].sort_values("pearson_r", key=abs, ascending=False)
        lines += [
            "─" * 55,
            f"Panel gene: {sym}   (LASSO α={rec['lasso_alpha']:.5f})",
            f"  Neighbours (GPL24676)  : {len(rec['nb_cols'])}",
            f"  Replicated (GPL16791)  : {len(rec['replicated_cols'])}",
        ]
        if not sub.empty:
            lines.append("  Top 10 by |Pearson r|:")
            for _, row in sub.head(10).iterrows():
                mark = "✓" if row["replicated"] else " "
                lines.append(
                    f"    [{mark}] {str(row['nb_gene']):<18} r={row['pearson_r']:+.3f}"
                )
        lines.append("")

    if not ora_df.empty:
        lines += ["─" * 55, "ORA — Replicated neighbours:", ""]
        for _, row in ora_df.head(20).iterrows():
            lib = str(row.get("Library", "")).split("_")[0]
            lines.append(
                f"  [{lib:<10}] {str(row['Term'])[:55]:<55} "
                f"AdjP={row['Adjusted P-value']:.2e}  Genes={row['Genes']}"
            )

    (SCRIPT_DIR / "panel_mb_statistics.txt").write_text("\n".join(lines))
    print(f"  Saved → panel_mb_statistics.txt")

    print("\n" + "=" * 65)
    print("DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
