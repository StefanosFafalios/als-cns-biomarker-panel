"""Post-process panel_mb_results.csv with a stricter primary |r| threshold.

Applies PRIMARY_R_THRESHOLD to the existing LASSO results (no re-run),
regenerates ORA, network figure, and statistics file.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
PRIMARY_R_THRESHOLD = 0.65   # |Pearson r| in GPL24676 — only keep edges at/above this

_ORA_LIBRARIES = [
    "GO_Biological_Process_2023",
    "KEGG_2021_Human",
    "Reactome_2022",
]
_ORA_PADJ = 0.05
_ORA_DELAY = 1.2


def _run_ora(symbols: list[str]) -> pd.DataFrame:
    import gseapy

    rows = []
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


def _plot_network(
    panel_genes: list[str],
    nb_syms: dict[str, list[tuple[str, float]]],
    out_path: Path,
) -> None:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    # Count how many panel genes each nb gene connects to
    nb_count: dict[str, set[str]] = {}
    for pg, pairs in nb_syms.items():
        for sym, _ in pairs:
            nb_count.setdefault(sym, set()).add(pg)

    nb_sorted = sorted(nb_count.items(), key=lambda x: -len(x[1]))[:30]
    nb_sym_set = {s for s, _ in nb_sorted}

    fig, ax = plt.subplots(figsize=(14, max(8, len(panel_genes) * 0.7 + 2)))
    ax.axis("off")

    y_panel = {g: i for i, g in enumerate(reversed(sorted(panel_genes)))}
    y_nb = {sym: i for i, (sym, _) in enumerate(nb_sorted)}

    for pg, pairs in nb_syms.items():
        py = y_panel.get(pg, 0)
        for sym, r in pairs:
            if sym in nb_sym_set:
                my = y_nb[sym]
                lc = "#2ca02c"
                ax.plot([0, 1], [py, my], color=lc, lw=max(0.6, abs(r) * 2.5), alpha=0.55, zorder=1)

    for g, py in y_panel.items():
        ax.scatter(0, py, s=200, color="#1f77b4", zorder=3)
        ax.text(-0.03, py, g, ha="right", va="center", fontsize=9, fontweight="bold")

    for sym, pg_set in nb_sorted:
        my = y_nb[sym]
        n_pg = len(pg_set)
        ax.scatter(1, my, s=80 + 70 * n_pg, color="#ff7f0e", zorder=3, alpha=0.85)
        ax.text(1.03, my, sym, ha="left", va="center", fontsize=7.5)

    ax.set_xlim(-0.35, 1.55)
    ax.set_ylim(-1, max(len(panel_genes), len(nb_sorted)) + 0.5)
    ax.set_title(
        f"Panel gene LASSO neighbourhood network (|r| ≥ {PRIMARY_R_THRESHOLD}, 100% replicated)\n"
        "Left: critical-panel genes with high-confidence neighbours  |  "
        "Right: top-30 candidates (size ∝ panel genes connected)\n"
        "All edges replicated in GPL16791 (ρ ≥ 0.10, same sign)",
        fontsize=9,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {out_path.name}")


def main() -> None:
    print(f"\nPost-processing with PRIMARY_R_THRESHOLD = {PRIMARY_R_THRESHOLD}")
    print("=" * 60)

    results_df = pd.read_csv(SCRIPT_DIR / "panel_mb_results.csv")
    print(f"  Loaded {len(results_df)} total LASSO edges")

    # Apply primary r threshold AND require replication
    filtered = results_df[
        (results_df["pearson_r"].abs() >= PRIMARY_R_THRESHOLD) &
        (results_df["replicated"])
    ].copy()
    print(f"  After |r| >= {PRIMARY_R_THRESHOLD} AND replicated: {len(filtered)} edges")

    # Save filtered replicated CSV
    out_rep = SCRIPT_DIR / "panel_mb_replicated.csv"
    filtered.to_csv(out_rep, index=False)
    print(f"  Saved -> {out_rep.name}")

    # Save filtered full CSV (primary threshold, all replication status)
    full_filtered = results_df[results_df["pearson_r"].abs() >= PRIMARY_R_THRESHOLD].copy()
    out_full = SCRIPT_DIR / "panel_mb_results.csv"
    full_filtered.to_csv(out_full, index=False)
    print(f"  Saved -> {out_full.name}  ({len(full_filtered)} primary edges >= {PRIMARY_R_THRESHOLD})")

    panel_genes = sorted(filtered["panel_gene"].unique())
    print(f"\nPanel genes with edges: {panel_genes}")
    for pg, grp in filtered.groupby("panel_gene"):
        top = grp.nlargest(5, "pearson_r")
        tops = ", ".join(f"{r['nb_gene']}({r['pearson_r']:+.3f})" for _, r in top.iterrows())
        print(f"  {pg:<12}: {len(grp)} edges | top: {tops}")

    # ORA per panel gene
    print("\nRunning per-gene ORA on filtered neighbours ...")
    all_ora = []
    for pg in panel_genes:
        sub = filtered[filtered["panel_gene"] == pg]
        syms = sorted({s for s in sub["nb_gene"].tolist()
                       if not str(s).startswith("ENSG") and not str(s).startswith("LOC")})
        if len(syms) < 5:
            print(f"  {pg}: only {len(syms)} symbols, skipping ORA")
            continue
        print(f"  {pg}: {len(syms)} symbols", flush=True)
        ora_df = _run_ora(syms)
        if not ora_df.empty:
            ora_df["panel_gene"] = pg
            all_ora.append(ora_df)

    if all_ora:
        full_ora = pd.concat(all_ora, ignore_index=True).sort_values(
            ["panel_gene", "Combined Score"], ascending=[True, False]
        )
        out_ora = SCRIPT_DIR / "panel_mb_ora_results.csv"
        full_ora.to_csv(out_ora, index=False)
        print(f"\n  Saved -> {out_ora.name}  ({len(full_ora)} significant terms)")
        print("\nTop ORA results per panel gene:")
        for pg, grp in full_ora.groupby("panel_gene"):
            print(f"\n  {pg}:")
            for _, r in grp.head(5).iterrows():
                lib = str(r.get("Library", "")).split("_")[0]
                print(f"    [{lib:<10}] {str(r['Term'])[:60]:<60} AdjP={r['Adjusted P-value']:.2e}")
    else:
        full_ora = pd.DataFrame()
        print("  No significant ORA terms")

    # Network figure
    print("\nRegenerating network figure ...")
    nb_syms: dict[str, list[tuple[str, float]]] = {}
    for pg, grp in filtered.groupby("panel_gene"):
        nb_syms[pg] = [(row["nb_gene"], row["pearson_r"])
                       for _, row in grp.sort_values("pearson_r", key=abs, ascending=False).iterrows()]
    _plot_network(panel_genes, nb_syms, SCRIPT_DIR / "panel_mb_network.png")

    # Statistics file
    print("Writing statistics ...")
    lines: list[str] = [
        "Panel Gene LASSO Neighbourhood Network (filtered)",
        "=" * 60,
        f"Method: LASSO neighbourhood selection (Meinshausen & Bühlmann 2006)",
        f"Primary cohort  : GPL24676  n=874",
        f"Replicate cohort: GPL16791  n=636",
        f"SFM search space: 3704 genes",
        f"Primary threshold: |Pearson r| >= {PRIMARY_R_THRESHOLD} in GPL24676",
        f"Replication threshold: |Pearson r| >= 0.10, same sign in GPL16791",
        f"All {len(filtered)} reported edges satisfy BOTH thresholds (100% replication).",
        "",
    ]
    for pg, grp in filtered.groupby("panel_gene"):
        sub = grp.sort_values("pearson_r", key=abs, ascending=False)
        lines += [
            "─" * 55,
            f"Panel gene: {pg}",
            f"  High-confidence edges: {len(grp)}  (all replicated in GPL16791)",
            "  Top 10 by |Pearson r|:",
        ]
        for _, row in sub.head(10).iterrows():
            lines.append(f"    [✓] {str(row['nb_gene']):<18} r={row['pearson_r']:+.3f}")
        lines.append("")

    if not full_ora.empty:
        lines += ["─" * 55, f"ORA — per panel gene (BH-adj p < {_ORA_PADJ}):", ""]
        for pg, grp in full_ora.groupby("panel_gene"):
            lines.append(f"  [{pg}]")
            for _, r in grp.head(8).iterrows():
                lib = str(r.get("Library", "")).split("_")[0]
                lines.append(
                    f"    [{lib:<10}] {str(r['Term'])[:55]:<55} "
                    f"AdjP={r['Adjusted P-value']:.2e}  Score={r['Combined Score']:.0f}"
                )
            lines.append("")

    (SCRIPT_DIR / "panel_mb_statistics.txt").write_text("\n".join(lines))
    print(f"  Saved -> panel_mb_statistics.txt")
    print("\nDONE")


if __name__ == "__main__":
    main()
