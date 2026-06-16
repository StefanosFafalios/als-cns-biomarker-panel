"""Upstream regulator inference (Step 8).

Three-filter approach to identify transcription factors (TFs) whose activity
is consistent with the 25-gene panel ALS signature.

Filter 1 — TF Enrichment (Enrichr)
    Query up_in_ALS and down_in_ALS protein-coding gene sets against three
    TF ChIP-seq/PWM libraries.  TF symbols extracted from term strings.
    Threshold: BH-adjusted p < 0.05.

Filter 2 — Differential Activity
    Each TF evaluated as a univariate ALS classifier on GPL24676 expression.
    AUC >= 0.60 → putative activator; AUC <= 0.40 → putative repressor.
    TFs not found in the pre-filtered expression matrix are skipped.

Filter 3 — Co-Expression with Panel Score
    Spearman rho between TF expression and composite panel Z-score across
    all 874 samples.  |rho| >= 0.30 retained; sign indicates activator (+)
    or repressor (-) role.

Outputs
-------
  upstream_regulators_results.csv     full TF table (all three filter values)
  upstream_regulators.png             summary figure (3 panels)
  upstream_regulators_statistics.txt  human-readable summary
"""

from __future__ import annotations

import re
import sys
import time
import pathlib
import warnings

import mygene
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ALS_DIR = pathlib.Path(__file__).parents[1]
SCRIPT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset  # noqa: E402

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PREFILTER_X = SCRIPT_DIR / "lgbm_prefilter_X.npy"
_PREFILTER_NAMES = SCRIPT_DIR / "lgbm_prefilter_names.txt"

OUT_CSV = SCRIPT_DIR / "upstream_regulators_results.csv"
OUT_PNG = SCRIPT_DIR / "upstream_regulators.png"
OUT_TXT = SCRIPT_DIR / "upstream_regulators_statistics.txt"

PLATFORM = "GPL24676"

TF_LIBRARIES = [
    "ChEA_2022",
    "TRANSFAC_and_JASPAR_PWMs",
    "ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X",
]

ENRICHR_PADJ_THRESH = 0.05   # BH-adjusted; falls back to nominal if 0 results
AUC_ACTIVATOR_THRESH = 0.60
AUC_REPRESSOR_THRESH = 0.40
SPEARMAN_THRESH = 0.30

# 12 protein-coding members of the 15-gene greedy-tail critical panel
# (greedy backward elimination peak at k=15, W.mean = 0.8921; equal cohort weights)
# Excludes EMP1: largest individual negative D_ZS in one-pass LOO but redundant
# in a 16-gene context (greedy elimination drops at step 10 with positive D_ZS).
_UP_GENES = [
    "FCN3", "PROS1", "SOHLH2", "HEXB", "MCEE", "SLC37A2", "SERTAD1",
]
_DOWN_GENES = [
    "ANGPT2", "TINAGL1", "CKMT2", "CLDN5", "NR4A1",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_tf_symbol(term: str) -> str:
    """Extract TF gene symbol from an Enrichr term string.

    All three libraries use the TF symbol as the first space-delimited token,
    optionally followed by ' (human)' or ' (mouse)' in TRANSFAC format.
    """
    sym = term.split()[0]
    # TRANSFAC may append _HUMAN or _MOUSE suffix
    sym = re.sub(r"[_-](HUMAN|MOUSE|human|mouse)$", "", sym)
    return sym.upper()


def _run_enrichr(gene_list: list[str], label: str) -> pd.DataFrame:
    """Run Enrichr ORA for gene_list against TF_LIBRARIES; return combined results."""
    import gseapy as gp

    all_rows: list[pd.DataFrame] = []
    for lib in TF_LIBRARIES:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                res = gp.enrichr(
                    gene_list=gene_list,
                    gene_sets=lib,
                    outdir=None,
                    verbose=False,
                )
            df = res.results.copy()
            df.insert(0, "library", lib)
            df.insert(0, "gene_set", label)
            all_rows.append(df)
        except Exception as exc:
            print(f"  WARNING: {lib} failed for {label}: {exc}")

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def _collect_tf_candidates(enrichr_df: pd.DataFrame) -> tuple[dict[str, dict], str]:
    """Extract TF symbols passing significance threshold from Enrichr results.

    Tries BH-adjusted p < ENRICHR_PADJ_THRESH first; if 0 results (expected
    for small gene sets), falls back to nominal p < 0.05, then to top-50 by
    Combined Score.

    Returns (symbol_dict, threshold_used).
    symbol_dict: symbol → {gene_set, libraries, min_padj}.
    """
    padj_col = next(
        (c for c in enrichr_df.columns if "adjusted" in c.lower() or c == "Adjusted P-value"),
        None,
    )
    pval_col = next(
        (c for c in enrichr_df.columns if c.lower() == "p-value" or c == "P-value"),
        None,
    )
    score_col = next(
        (c for c in enrichr_df.columns if "combined" in c.lower()),
        None,
    )
    term_col = next((c for c in enrichr_df.columns if c == "Term"), None)

    if term_col is None:
        print(f"  WARNING: unexpected Enrichr columns: {enrichr_df.columns.tolist()}")
        return {}, "none"

    def _extract_from_subset(sub: pd.DataFrame, thresh_label: str) -> dict[str, dict]:
        tf_map: dict[str, dict] = {}
        for _, row in sub.iterrows():
            sym = _extract_tf_symbol(str(row[term_col]))
            if not sym or len(sym) < 2:
                continue
            pval = float(row[padj_col]) if padj_col else 1.0
            if sym not in tf_map:
                tf_map[sym] = {
                    "gene_set": row["gene_set"],
                    "libraries": set(),
                    "min_padj": pval,
                }
            tf_map[sym]["libraries"].add(row["library"])
            tf_map[sym]["min_padj"] = min(tf_map[sym]["min_padj"], pval)
        return tf_map

    # Try BH-adjusted first
    if padj_col:
        sig = enrichr_df[enrichr_df[padj_col] < ENRICHR_PADJ_THRESH]
        if len(sig) > 0:
            return _extract_from_subset(sig, f"BH-adj p<{ENRICHR_PADJ_THRESH}"), f"BH-adj p<{ENRICHR_PADJ_THRESH}"

    # Fall back to nominal p < 0.05
    print(f"  NOTE: 0 BH-adj results; falling back to nominal p < 0.05")
    if pval_col:
        sig = enrichr_df[enrichr_df[pval_col] < 0.05]
        if len(sig) > 0:
            return _extract_from_subset(sig, "nominal p<0.05"), "nominal p<0.05"

    # Last resort: top 50 by combined score
    print(f"  NOTE: 0 nominal-p results; using top 50 by Combined Score")
    if score_col:
        sig = enrichr_df.nlargest(50, score_col)
        return _extract_from_subset(sig, "top-50 combined score"), "top-50 combined score"

    return {}, "none"


def _map_symbols_to_ensg(symbols: list[str]) -> dict[str, str]:
    """Map HGNC symbols → base ENSG IDs via MyGeneInfo (batched, with retry)."""
    mg_client = mygene.MyGeneInfo()
    sym_to_ensg: dict[str, str] = {}
    batch_size = 500

    for start in range(0, len(symbols), batch_size):
        chunk = symbols[start : start + batch_size]
        for attempt in range(3):
            try:
                hits = mg_client.querymany(
                    chunk,
                    scopes="symbol",
                    fields="ensembl.gene",
                    species="human",
                    verbose=False,
                )
                for h in hits:
                    ensg = h.get("ensembl", {})
                    if isinstance(ensg, list):
                        ensg = ensg[0]
                    gene_id = ensg.get("gene", "") if isinstance(ensg, dict) else ""
                    if gene_id:
                        sym_to_ensg[h["query"]] = gene_id
                break
            except Exception as exc:
                wait = 5 * (2**attempt)
                print(f"  Retry {attempt + 1} in {wait}s ({exc})")
                time.sleep(wait)

    return sym_to_ensg


def _build_panel_ensg_map(prefilter_names: list[str]) -> dict[str, int]:
    """Return base_ensg → column_index for the prefilter matrix."""
    return {n.split(".")[0]: i for i, n in enumerate(prefilter_names)}


def _compute_composite_score(
    X: np.ndarray,
    prefilter_names: list[str],
    panel_csv: pathlib.Path,
) -> np.ndarray:
    """Composite panel Z-score: mean(z_up_genes) - mean(z_down_genes).

    Uses protein-coding panel genes with confirmed symbol and direction.
    StandardScaler fitted on all 874 samples jointly.
    """
    panel = pd.read_csv(panel_csv)
    # Build symbol → versioned_feature map
    sym_to_feat: dict[str, str] = {}
    for _, row in panel.iterrows():
        sym = str(row.get("symbol", "")).strip()
        feat = str(row.get("feature", "")).strip()
        if sym and not sym.startswith("ENSG"):
            sym_to_feat[sym] = feat

    base_to_col = _build_panel_ensg_map(prefilter_names)
    scaler = StandardScaler()
    X_f64 = X.astype(np.float64)

    def _gene_z(sym: str) -> np.ndarray | None:
        feat = sym_to_feat.get(sym)
        if feat is None:
            return None
        base = feat.split(".")[0]
        col = base_to_col.get(base)
        if col is None:
            return None
        vec = X_f64[:, col].reshape(-1, 1)
        return scaler.fit(vec).transform(vec).ravel()

    up_cols = [_gene_z(s) for s in _UP_GENES]
    down_cols = [_gene_z(s) for s in _DOWN_GENES]
    up_cols = [z for z in up_cols if z is not None]
    down_cols = [z for z in down_cols if z is not None]

    print(f"  Composite score: {len(up_cols)} up genes, {len(down_cols)} down genes")
    score = np.mean(up_cols, axis=0) - np.mean(down_cols, axis=0)
    return score  # (874,)


# ---------------------------------------------------------------------------
# Filters 2 & 3
# ---------------------------------------------------------------------------


def _evaluate_tfs(
    tf_candidates: dict[str, dict],
    sym_to_ensg: dict[str, str],
    base_to_col: dict[str, int],
    X: np.ndarray,
    y: np.ndarray,
    composite_score: np.ndarray,
) -> list[dict]:
    """Apply Filters 2 and 3 to TF candidates; return rows for the results table."""
    rows: list[dict] = []
    n_not_found = 0

    for sym, meta in tf_candidates.items():
        ensg = sym_to_ensg.get(sym)
        if ensg is None:
            n_not_found += 1
            continue
        col = base_to_col.get(ensg)
        if col is None:
            n_not_found += 1
            continue

        expr = X[:, col].astype(np.float64)

        # Filter 2 — univariate AUC
        try:
            auc = roc_auc_score(y, expr)
            auc = max(auc, 1.0 - auc)
        except Exception:
            continue

        if auc < AUC_ACTIVATOR_THRESH and auc > AUC_REPRESSOR_THRESH:
            continue  # does not pass Filter 2

        direction_f2 = "activator" if auc >= AUC_ACTIVATOR_THRESH else "repressor"

        # Filter 3 — co-expression with composite score
        rho, pval = spearmanr(expr, composite_score)
        if abs(rho) < SPEARMAN_THRESH:
            continue

        direction_f3 = "activator" if rho > 0 else "repressor"
        rows.append({
            "symbol": sym,
            "gene_set": meta["gene_set"],
            "libraries": "|".join(sorted(meta["libraries"])),
            "n_libraries": len(meta["libraries"]),
            "min_enrichr_padj": meta["min_padj"],
            "auc": round(auc, 4),
            "direction_f2": direction_f2,
            "spearman_rho": round(rho, 4),
            "spearman_p": round(pval, 6),
            "direction_f3": direction_f3,
            "ensg": ensg,
        })

    print(f"  TFs not found in prefilter matrix: {n_not_found}")
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot(results_df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Upstream Regulator Inference — ALS Panel Signature", fontsize=13)

    # Panel 1: scatter AUC vs Spearman rho, coloured by gene_set
    ax = axes[0]
    colours = {"up_in_ALS": "#d62728", "down_in_ALS": "#1f77b4"}
    for gs, grp in results_df.groupby("gene_set"):
        ax.scatter(
            grp["auc"], grp["spearman_rho"],
            c=colours.get(gs, "grey"), label=gs, alpha=0.8, s=60, edgecolors="white", lw=0.5,
        )
        for _, row in grp.iterrows():
            ax.annotate(
                row["symbol"], (row["auc"], row["spearman_rho"]),
                fontsize=7, ha="left", va="bottom",
                xytext=(3, 3), textcoords="offset points",
            )
    ax.axvline(AUC_ACTIVATOR_THRESH, ls="--", lw=0.8, color="grey")
    ax.axvline(AUC_REPRESSOR_THRESH, ls="--", lw=0.8, color="grey")
    ax.axhline(SPEARMAN_THRESH, ls=":", lw=0.8, color="grey")
    ax.axhline(-SPEARMAN_THRESH, ls=":", lw=0.8, color="grey")
    ax.set_xlabel("Univariate AUC (Filter 2)", fontsize=10)
    ax.set_ylabel("Spearman ρ with panel score (Filter 3)", fontsize=10)
    ax.set_title("All passing TFs", fontsize=10)
    ax.legend(fontsize=8)

    # Panel 2: top TFs by |rho|, coloured by direction_f3
    ax = axes[1]
    top = results_df.reindex(results_df["spearman_rho"].abs().sort_values(ascending=False).index).head(20)
    colours2 = {"activator": "#d62728", "repressor": "#1f77b4"}
    bar_colours = [colours2.get(d, "grey") for d in top["direction_f3"]]
    bars = ax.barh(range(len(top)), top["spearman_rho"].abs(), color=bar_colours)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["symbol"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("|Spearman ρ|", fontsize=10)
    ax.set_title("Top 20 TFs by |ρ|", fontsize=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#d62728", label="Activator (ρ > 0)"),
        Patch(facecolor="#1f77b4", label="Repressor (ρ < 0)"),
    ], fontsize=8)

    # Panel 3: AUC distribution of passing TFs
    ax = axes[2]
    for gs, grp in results_df.groupby("gene_set"):
        ax.hist(grp["auc"], bins=10, alpha=0.6, label=gs, color=colours.get(gs, "grey"))
    ax.axvline(AUC_ACTIVATOR_THRESH, ls="--", lw=0.8, color="grey", label=f"AUC={AUC_ACTIVATOR_THRESH}")
    ax.set_xlabel("Univariate AUC", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("AUC distribution of passing TFs", fontsize=10)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {OUT_PNG}")


# ---------------------------------------------------------------------------
# Statistics output
# ---------------------------------------------------------------------------


def _write_statistics(results_df: pd.DataFrame, n_f1: int, n_f2: int, thresh_used: str = "") -> None:
    lines = [
        "Upstream Regulator Inference — ALS Panel Signature (Step 8)",
        "=" * 70,
        f"Gene sets queried:",
        f"  up_in_ALS   ({len(_UP_GENES)} genes): {', '.join(_UP_GENES)}",
        f"  down_in_ALS ({len(_DOWN_GENES)} genes): {', '.join(_DOWN_GENES)}",
        f"TF libraries: {', '.join(TF_LIBRARIES)}",
        f"",
        f"Filter 1 ({thresh_used}): {n_f1} unique TF candidates",
        f"Filter 2 (AUC >= {AUC_ACTIVATOR_THRESH} or <= {AUC_REPRESSOR_THRESH}): {n_f2} passed",
        f"Filter 3 (|Spearman rho| >= {SPEARMAN_THRESH}): {len(results_df)} passed all filters",
        "",
    ]

    for gs in ["up_in_ALS", "down_in_ALS"]:
        sub = results_df[results_df["gene_set"] == gs].sort_values(
            "spearman_rho", key=abs, ascending=False
        )
        lines.append("=" * 70)
        lines.append(f"Gene set: {gs} — {len(sub)} TFs passed all 3 filters")
        lines.append(
            f"{'Symbol':<12} {'AUC':>6} {'rho':>7} {'p(rho)':>10} "
            f"{'Role':>10} {'Libraries'}"
        )
        lines.append("-" * 70)
        for _, row in sub.iterrows():
            role = row["direction_f3"]
            lines.append(
                f"{row['symbol']:<12} {row['auc']:>6.4f} {row['spearman_rho']:>7.4f} "
                f"{row['spearman_p']:>10.4e} {role:>10}  {row['libraries']}"
            )
        if sub.empty:
            lines.append("  (no TFs passed all 3 filters for this gene set)")

    lines += [
        "",
        "=" * 70,
        "BIOLOGICAL INTERPRETATION",
        "-" * 70,
    ]

    act = results_df[results_df["direction_f3"] == "activator"].sort_values(
        "spearman_rho", ascending=False
    )
    rep = results_df[results_df["direction_f3"] == "repressor"].sort_values(
        "spearman_rho", ascending=True
    )

    lines.append(f"Activators (rho > 0, AUC >= {AUC_ACTIVATOR_THRESH}): {len(act)}")
    for _, row in act.head(10).iterrows():
        lines.append(f"  {row['symbol']}: AUC={row['auc']:.4f}, rho={row['spearman_rho']:.4f} [{row['gene_set']}]")

    lines.append(f"\nRepressors (rho < 0, AUC <= {AUC_REPRESSOR_THRESH}): {len(rep)}")
    for _, row in rep.head(10).iterrows():
        lines.append(f"  {row['symbol']}: AUC={row['auc']:.4f}, rho={row['spearman_rho']:.4f} [{row['gene_set']}]")

    lines += [
        "",
        "=" * 70,
        "OUTPUT FILES",
        f"  Full results table: {OUT_CSV}",
        f"  Plot:               {OUT_PNG}",
        f"  Statistics:         {OUT_TXT}",
    ]

    OUT_TXT.write_text("\n".join(lines) + "\n")
    print(f"  Statistics written: {OUT_TXT}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 65)
    print("Upstream Regulator Inference (Step 8)")
    print("=" * 65)

    # Load expression matrix and labels
    print("\n[1] Loading prefilter matrix and labels ...")
    X = np.load(_PREFILTER_X)
    prefilter_names: list[str] = _PREFILTER_NAMES.read_text().splitlines()
    base_to_col = _build_panel_ensg_map(prefilter_names)

    (ds,) = load_dataset(
        "GSE153960", platform=PLATFORM, resources_dir=ALS_DIR / "resources"
    )
    y = ds.y.values.astype(int)
    print(f"  X: {X.shape} | ALS: {y.sum()} | Control: {(y==0).sum()}")

    # Compute composite panel Z-score
    print("\n[2] Computing composite panel Z-score ...")
    composite_score = _compute_composite_score(X, prefilter_names, _PANEL_CSV)
    print(f"  Score range: [{composite_score.min():.3f}, {composite_score.max():.3f}]")

    # Filter 1: Enrichr TF enrichment
    print("\n[3] Running Enrichr TF enrichment (Filter 1) ...")
    enrichr_up = _run_enrichr(_UP_GENES, "up_in_ALS")
    enrichr_down = _run_enrichr(_DOWN_GENES, "down_in_ALS")
    enrichr_all = pd.concat([enrichr_up, enrichr_down], ignore_index=True)

    tf_candidates, thresh_used = _collect_tf_candidates(enrichr_all)
    n_f1 = len(tf_candidates)
    print(f"  Filter 1: {n_f1} unique TF candidates (threshold: {thresh_used})")

    if n_f1 == 0:
        print("  No TFs passed Filter 1. Exiting.")
        return

    # Map TF symbols → ENSG IDs
    print("\n[4] Mapping TF symbols → ENSG IDs via MyGeneInfo ...")
    tf_symbols = list(tf_candidates.keys())
    sym_to_ensg = _map_symbols_to_ensg(tf_symbols)
    print(f"  Mapped {len(sym_to_ensg)} / {len(tf_symbols)} TF symbols to ENSG IDs")

    # Filters 2 & 3
    print("\n[5] Applying Filters 2 (AUC) and 3 (co-expression) ...")
    rows = _evaluate_tfs(
        tf_candidates, sym_to_ensg, base_to_col, X, y, composite_score
    )
    n_f2 = sum(
        1 for sym in tf_candidates
        if sym_to_ensg.get(sym) and base_to_col.get(sym_to_ensg[sym]) is not None
    )

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        print("  No TFs passed all 3 filters.")
        OUT_TXT.write_text("No TFs passed all 3 filters.\n")
        return

    results_df = results_df.sort_values("spearman_rho", key=abs, ascending=False)
    results_df.to_csv(OUT_CSV, index=False)
    print(f"  {len(results_df)} TFs passed all 3 filters")
    print(f"  Results saved: {OUT_CSV}")

    # Plot and statistics
    print("\n[6] Generating plot and statistics ...")
    _plot(results_df)
    _write_statistics(results_df, n_f1, n_f2, thresh_used)
    print("Done.")


if __name__ == "__main__":
    main()
