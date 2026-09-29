"""Regenerate gene_replacement.png restricted to the 15 protein-coding panel genes.

Reads the pre-computed artifact-corrected statistics from
gene_replacement_clean_statistics.txt, filters to the 15 protein-coding
transferable panel members, and writes gene_replacement.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE.parent))

from utils import critical_panel_symbols  # noqa: E402

# The 15 protein-coding members of the *25-gene* panel. This is the plot's x-axis
# universe and is deliberately NOT the critical panel: the grey bars are exactly
# the genes greedy elimination dropped. The 25-gene panel is the greedy's input
# and does not change when the critical panel does.
_PC15 = {
    "MECOM",
    "SERTAD1",
    "FCN3",
    "PROS1",
    "ANGPT2",
    "EMP1",
    "TINAGL1",
    "CKMT2",
    "VWF",
    "CLDN5",
    "NR4A1",
    "SOHLH2",
    "HEXB",
    "MCEE",
    "SLC37A2",
}

# Protein-coding members of the canonical critical panel (panel_critical.csv)
_CRITICAL_PC = set(critical_panel_symbols(protein_coding_only=True))

# Gene-level drug targets to star: critical-panel protein-coding members below this
# artifact-corrected replaceability (%), the threshold the main text uses. Membership
# is derived in main(): _CRITICAL_PC changes with the canonical panel definition.
_GENE_LEVEL_MAX_PCT = 25.0

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
                rows.append(
                    {
                        "symbol": symbol,
                        "repl_pct_clean": repl_pct_clean,
                        "n_repl_clean": n_repl_clean,
                    }
                )
    return rows


def main() -> None:
    rows = _parse_clean_stats()
    rows = [r for r in rows if r["symbol"] in _PC15]
    rows.sort(key=lambda r: r["n_repl_clean"], reverse=True)

    symbols = [r["symbol"] for r in rows]
    n_repl = [r["n_repl_clean"] for r in rows]
    repl_pct = [r["repl_pct_clean"] / 100.0 for r in rows]

    # the least replaceable critical members are the genes to drug directly
    gene_level = {
        r["symbol"]
        for r in rows
        if r["symbol"] in _CRITICAL_PC and r["repl_pct_clean"] < _GENE_LEVEL_MAX_PCT
    }

    # Colour: red = critical-panel member; grey = greedy-elimination dropout
    colors = ["#C62828" if s in _CRITICAL_PC else "#888" for s in symbols]

    # Label decoration: ★ for gene-level drug targets (lowest replaceability)
    deco_symbols = [
        f"★ {s}"
        if s in gene_level
        else (f"{s}" if s in _CRITICAL_PC else f"{s} (dropped)")
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
        Patch(facecolor="#C62828", label="Critical-panel member (greedy-tail survivor)"),
        Patch(
            facecolor="#888",
            label="Greedy-elimination dropout ("
            + ", ".join(sorted(s for s in symbols if s not in _CRITICAL_PC))
            + ")",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9,
        frameon=False,
    )

    plt.suptitle(
        "Gene Replacement Analysis — 15 Protein-Coding Panel Members  (n=874, donor-grouped 5-fold CV)\n"
        "★ = lowest-replaceability gene-level drug targets ("
        + ", ".join(sorted(gene_level))
        + ")",
        fontsize=10,
    )
    plt.tight_layout()
    out = BASE / "gene_replacement.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
