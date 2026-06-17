"""Regenerate gene_replacement.png restricted to the 15 protein-coding panel genes.

Reads the pre-computed artifact-corrected statistics from
gene_replacement_clean_statistics.txt, filters to the 15 protein-coding
transferable panel members, and writes gene_replacement.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).parent

_PC15 = {
    "MECOM", "SERTAD1", "FCN3", "PROS1", "ANGPT2", "EMP1", "TINAGL1",
    "CKMT2", "VWF", "CLDN5", "NR4A1", "SOHLH2", "HEXB", "MCEE", "SLC37A2",
}

# Members of the 15-gene greedy-tail critical panel (protein-coding only)
_CRITICAL_PC = {
    "FCN3", "PROS1", "ANGPT2", "TINAGL1", "CKMT2", "CLDN5",
    "NR4A1", "SOHLH2", "HEXB", "MCEE", "SLC37A2", "SERTAD1",
}

# Gene-level drug targets (4 lowest replaceability among critical-panel PCs)
_GENE_LEVEL = {"CKMT2", "NR4A1", "TINAGL1", "SERTAD1"}

_CLEAN_STATS = BASE / "gene_replacement_clean_statistics.txt"


def _parse_clean_stats() -> list[dict]:
    rows = []
    in_table = False
    for line in _CLEAN_STATS.read_text().splitlines():
        if line.startswith("Gene") and "Orig%" in line:
            in_table = True
            continue
        if in_table and line.startswith("-"):
            continue
        if in_table and line.strip() == "":
            break
        if in_table:
            parts = line.split()
            if len(parts) >= 5:
                symbol = parts[0]
                try:
                    repl_pct_clean = float(parts[2].rstrip("%"))
                    n_repl_clean = int(parts[4])
                except (ValueError, IndexError):
                    continue
                rows.append({
                    "symbol": symbol,
                    "repl_pct_clean": repl_pct_clean,
                    "n_repl_clean": n_repl_clean,
                })
    return rows


def main() -> None:
    rows = _parse_clean_stats()
    rows = [r for r in rows if r["symbol"] in _PC15]
    rows.sort(key=lambda r: r["n_repl_clean"], reverse=True)

    symbols = [r["symbol"] for r in rows]
    n_repl = [r["n_repl_clean"] for r in rows]
    repl_pct = [r["repl_pct_clean"] / 100.0 for r in rows]

    # Colour: red = 15-crit member; grey = greedy dropout (MECOM, VWF, EMP1)
    colors = ["#C62828" if s in _CRITICAL_PC else "#888" for s in symbols]

    # Label decoration: ★ for gene-level drug targets (lowest 4 replaceability)
    deco_symbols = [
        f"★ {s}" if s in _GENE_LEVEL else (f"{s}" if s in _CRITICAL_PC else f"{s} (dropped)")
        for s in symbols
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    ax.barh(deco_symbols, n_repl, color=colors, edgecolor="white", height=0.7)
    ax.set_xlabel("Number of valid replacements (artifact-corrected)", fontsize=11)
    ax.set_title(
        "Valid substitute genes per panel gene",
        fontsize=10,
    )
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.8)
    ax.grid(True, axis="x", alpha=0.3)

    ax = axes[1]
    ax.barh(deco_symbols, repl_pct, color=colors, edgecolor="white", height=0.7)
    ax.set_xlabel(
        "Replaceability fraction (valid replacements / candidates tested)",
        fontsize=11,
    )
    ax.set_title("Replaceability score per panel gene", fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.axvline(0, color="k", lw=0.8)
    ax.grid(True, axis="x", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#C62828", label="15-crit panel member (greedy-tail survivor)"),
        Patch(facecolor="#888",    label="Greedy-elimination dropout (MECOM, VWF, EMP1)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
                bbox_to_anchor=(0.5, -0.02), fontsize=9, frameon=False)

    plt.suptitle(
        "Gene Replacement Analysis — 15 Protein-Coding Panel Members  (n=874, 5-fold CV)\n"
        "★ = lowest-replaceability gene-level drug targets (CKMT2, NR4A1, TINAGL1, SERTAD1)",
        fontsize=10,
    )
    plt.tight_layout()
    out = BASE / "gene_replacement.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
