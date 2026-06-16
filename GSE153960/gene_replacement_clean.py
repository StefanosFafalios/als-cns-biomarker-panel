"""Compute artifact-corrected gene replacement statistics.

Reads gene_replacement_results.csv (already computed) and
shap500_artifact_audit.csv (just computed), removes confirmed artifacts
from the candidate pool, and re-derives:

  - Corrected replaceability % per panel gene
  - Best non-artifact replacement per panel gene
  - Delta vs original (how much artifacts inflated / deflated the stats)

Outputs
-------
  gene_replacement_clean_statistics.txt
  gene_replacement_clean.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent

_REPL_CSV  = SCRIPT_DIR / "gene_replacement_results.csv"
_AUDIT_CSV = SCRIPT_DIR / "shap500_artifact_audit.csv"
_ORIG_STATS = SCRIPT_DIR / "gene_replacement_statistics.txt"


def main() -> None:
    print("=" * 72)
    print("Artifact-Corrected Gene Replacement Statistics")
    print("=" * 72)

    repl = pd.read_csv(_REPL_CSV)
    audit = pd.read_csv(_AUDIT_CSV)

    # Build artifact set (confirmed_artifact only — not borderline)
    artifact_bases = set(
        audit.loc[audit["classification"] == "confirmed_artifact", "ensg_base"]
    )
    print(f"\nConfirmed artifact ENSG bases to exclude: {len(artifact_bases)}")

    repl["candidate_base"] = repl["candidate_ensg"].str.split(".").str[0]
    repl["is_artifact"]    = repl["candidate_base"].isin(artifact_bases)

    total_cands = len(repl)
    artifact_rows = repl["is_artifact"].sum()
    print(f"Total replacement rows: {total_cands}")
    print(f"Rows from artifact candidates: {artifact_rows} ({100*artifact_rows/total_cands:.1f}%)")

    # Clean pool: exclude artifacts
    clean = repl[~repl["is_artifact"]].copy()

    # -----------------------------------------------------------------------
    # Re-aggregate per panel gene
    # -----------------------------------------------------------------------
    rows = []
    for sym, grp in clean.groupby("panel_symbol"):
        n_cands   = len(grp)
        n_repl    = int(grp["is_replacement"].sum())
        repl_pct  = 100.0 * n_repl / n_cands if n_cands > 0 else 0.0
        # Best replacement (highest AUC among valid ones)
        valid = grp[grp["is_replacement"]]
        if len(valid) > 0:
            best_idx  = valid["replacement_auc"].idxmax()
            best_ensg = valid.loc[best_idx, "candidate_ensg"]
            best_auc  = float(valid.loc[best_idx, "replacement_auc"])
        else:
            best_ensg = "none"
            best_auc  = float("nan")
        rows.append({
            "symbol":         sym,
            "n_cands_clean":  n_cands,
            "n_repl_clean":   n_repl,
            "repl_pct_clean": repl_pct,
            "best_repl_clean": best_ensg,
            "best_auc_clean": best_auc,
        })

    clean_df = pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # Compare with original statistics
    # -----------------------------------------------------------------------
    # Parse original stats from gene_replacement_statistics.txt
    orig_rows = []
    in_table = False
    for line in _ORIG_STATS.read_text().splitlines():
        if line.startswith("Gene ") and "n_corr" in line:
            in_table = True
            continue
        if in_table and line.startswith("-"):
            continue
        if in_table and line.strip() == "":
            break
        if in_table:
            parts = line.split()
            if len(parts) >= 7:
                orig_rows.append({
                    "symbol":       parts[0],
                    "n_tot_orig":   int(parts[3]),
                    "n_repl_orig":  int(parts[4]),
                    "repl_pct_orig": float(parts[5].rstrip("%")),
                    "best_orig":    parts[6],
                    "best_auc_orig": float(parts[7]),
                })

    orig_df  = pd.DataFrame(orig_rows)
    merged   = orig_df.merge(clean_df, on="symbol", how="outer")
    merged["delta_repl_pct"] = merged["repl_pct_clean"] - merged["repl_pct_orig"]
    merged["best_changed"]   = merged["best_repl_clean"] != merged["best_orig"].apply(
        lambda x: x.split(".")[0] if isinstance(x, str) else x
    )
    merged.sort_values("repl_pct_clean", inplace=True)

    # -----------------------------------------------------------------------
    # Print and save statistics
    # -----------------------------------------------------------------------
    header = (
        f"{'Gene':<18}  {'Orig%':>6}  {'Clean%':>7}  {'Δ':>6}  "
        f"{'CleanRepl':>6}  {'CleanBest':>22}  {'BestAUC':>8}  Changed"
    )
    sep = "-" * len(header)

    lines = [
        "Artifact-Corrected Gene Replacement Statistics",
        "=" * 72,
        f"Artifacts excluded: {len(artifact_bases)}  "
        f"(confirmed max GTEx TPM < 1.0)",
        "",
        header,
        sep,
    ]
    for _, r in merged.iterrows():
        changed = "yes" if r["best_changed"] else "no"
        lines.append(
            f"{r['symbol']:<18}  {r['repl_pct_orig']:>6.1f}%  "
            f"{r['repl_pct_clean']:>6.1f}%  "
            f"{r['delta_repl_pct']:>+6.1f}%  "
            f"{int(r['n_repl_clean']):>6}  "
            f"{str(r['best_repl_clean']).split('.')[0]:>22}  "
            f"{r['best_auc_clean']:>8.4f}  {changed}"
        )
    lines += [
        sep,
        "",
        f"Panel genes whose BEST replacement changed: "
        f"{merged['best_changed'].sum()} / {len(merged)}",
        "",
        "Genes with largest replaceability drop after cleaning:",
    ]
    top_drops = merged.nsmallest(5, "delta_repl_pct")
    for _, r in top_drops.iterrows():
        lines.append(
            f"  {r['symbol']:<18}  {r['repl_pct_orig']:.1f}% → "
            f"{r['repl_pct_clean']:.1f}%  (Δ{r['delta_repl_pct']:+.1f}%)"
        )

    out_txt = SCRIPT_DIR / "gene_replacement_clean_statistics.txt"
    out_txt.write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\nSaved: {out_txt}")

    # -----------------------------------------------------------------------
    # Figure: side-by-side replaceability bars (original vs clean)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    syms  = merged["symbol"].tolist()
    y_pos = np.arange(len(syms))

    for ax, col, label, color in [
        (axes[0], "repl_pct_orig",  "Original (incl. artifacts)", "#AAAAAA"),
        (axes[1], "repl_pct_clean", "Artifact-corrected",         "#4A90D9"),
    ]:
        ax.barh(y_pos, merged[col], color=color, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(syms, fontsize=8)
        ax.set_xlabel("Replaceability (%)", fontsize=10)
        ax.set_xlim(0, 110)
        ax.axvline(50, color="#888888", lw=0.7, ls="--")
        ax.set_title(label, fontsize=10)
        ax.grid(axis="x", color="#EEEEEE", lw=0.6)

    # Annotate changed-best genes on right panel
    for i, (_, r) in enumerate(merged.iterrows()):
        if r["best_changed"]:
            axes[1].text(
                r["repl_pct_clean"] + 1, i,
                "★", va="center", fontsize=8, color="#E05C5C",
            )

    fig.suptitle(
        "Gene Replacement Replaceability — Original vs Artifact-Corrected\n"
        "(★ = best replacement gene changed after removing artifacts)",
        fontsize=10,
    )
    plt.tight_layout()
    out_png = SCRIPT_DIR / "gene_replacement_clean.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out_png}")
    plt.close()


if __name__ == "__main__":
    main()
