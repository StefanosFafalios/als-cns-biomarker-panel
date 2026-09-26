"""Sensitivity analysis of cross-cohort panel LOO D_ZS under alternative weightings.

Compares the headline 4-cohort equal-weight D_ZS against:
  - 3-cohort equal-weight (excludes degenerate SRP064478)
  - 4-cohort sample-size-weighted
  - 3-cohort sample-size-weighted

Inputs: panel_loo_zeroshot.csv (machine-readable per-gene per-cohort AUCs) plus
panel_loo_zeroshot_statistics.txt for the per-cohort sample sizes only.

This script used to parse the whole table out of the .txt. That broke silently
when the producer's layout changed: the weight column became `w_eq=` rather than
`w=`, n was space-padded (`n= 20`), and the per-gene table was replaced by D_ZS
with confidence intervals. The parser then returned zero rows and an empty size
map, and the failure only surfaced as a KeyError several lines later. Reading the
CSV removes the whole class of problem -- prose layout is free to change.
Outputs:
  - cross_cohort_loo_sensitivity.csv  (per-gene D_ZS under all 4 schemes)
  - cross_cohort_loo_sensitivity.txt  (sign-flip and rank-correlation summary)
  - cross_cohort_loo_sensitivity.png  (4-panel comparison)
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).parent
STATS_PATH = SCRIPT_DIR / "panel_loo_zeroshot_statistics.txt"
CSV_PATH = SCRIPT_DIR / "panel_loo_zeroshot.csv"
COHORTS = ("GPL16791", "GSE76220", "GSE122649", "SRP064478")


def _parse_stats() -> tuple[pd.DataFrame, dict[str, float], dict[str, int]]:
    """Load per-gene LOO AUCs, per-cohort baselines, and per-cohort sample sizes.

    Returns:
        ``(per_gene_df, baseline_auc_by_cohort, n_by_cohort)``.

    Raises:
        SystemExit: if the CSV is missing, or if either the gene table or the
            sample-size map comes back empty -- previously both failed silently
            and surfaced as a KeyError much later.
    """
    if not CSV_PATH.exists():
        raise SystemExit(f"{CSV_PATH.name} missing -- run panel_loo_zeroshot.py first.")
    raw_df = pd.read_csv(CSV_PATH)

    df = pd.DataFrame({"Gene": raw_df["gene"]})
    base: dict[str, float] = {}
    for c in COHORTS:
        loo_col, base_col = f"AUC_{c}_LOO", f"AUC_{c}_baseline"
        if loo_col not in raw_df.columns:
            raise SystemExit(f"{CSV_PATH.name} has no column {loo_col}")
        df[f"AUC_{c}"] = raw_df[loo_col]
        base[c] = float(raw_df[base_col].iloc[0])

    # Sample sizes are only in the human-readable file. Tolerant of padding and
    # of whatever the weight columns are called this week.
    n: dict[str, int] = {}
    if STATS_PATH.exists():
        for ln in STATS_PATH.read_text().splitlines():
            m = re.match(r"\s+(\w+)\s+n=\s*(\d+)", ln)
            if m and m.group(1) in COHORTS:
                n[m.group(1)] = int(m.group(2))

    missing = set(COHORTS) - set(n)
    if missing:
        raise SystemExit(
            f"no sample size parsed for {sorted(missing)} from {STATS_PATH.name}; "
            "the size line format has changed again -- fix the regex rather than "
            "letting downstream weighting silently use partial data."
        )
    if df.empty:
        raise SystemExit(f"{CSV_PATH.name} contained no genes")
    return df, base, n


def main() -> None:
    df, base, n = _parse_stats()
    print(f"Parsed {len(df)} genes, cohorts: {list(base)}")
    print(f"  baseline AUCs: {base}")
    print(f"  sample sizes:  {n}")

    cohorts = list(COHORTS)
    cols = [f"AUC_{c}" for c in cohorts]
    N4 = sum(n.values())
    N3 = N4 - n["SRP064478"]

    b4_eq = sum(base.values()) / 4
    b3_eq = sum(base[c] for c in cohorts if c != "SRP064478") / 3
    b4_n = sum(base[c] * n[c] for c in cohorts) / N4
    b3_n = sum(base[c] * n[c] for c in cohorts if c != "SRP064478") / N3

    print("\nBaselines:")
    print(f"  4-cohort equal-weight:    {b4_eq:.4f}")
    print(
        f"  3-cohort equal-weight:    {b3_eq:.4f}   (sensitivity: excludes SRP064478)"
    )
    print(f"  4-cohort sample-weighted: {b4_n:.4f}")
    print(f"  3-cohort sample-weighted: {b3_n:.4f}")

    df["D_4eq"] = df[cols].sum(axis=1) / 4 - b4_eq
    df["D_3eq"] = df[cols[:3]].sum(axis=1) / 3 - b3_eq
    df["D_4n"] = (
        df["AUC_GPL16791"] * n["GPL16791"]
        + df["AUC_GSE76220"] * n["GSE76220"]
        + df["AUC_GSE122649"] * n["GSE122649"]
        + df["AUC_SRP064478"] * n["SRP064478"]
    ) / N4 - b4_n
    df["D_3n"] = (
        df["AUC_GPL16791"] * n["GPL16791"]
        + df["AUC_GSE76220"] * n["GSE76220"]
        + df["AUC_GSE122649"] * n["GSE122649"]
    ) / N3 - b3_n

    # Status by scheme
    for sch in ("D_4eq", "D_3eq", "D_4n", "D_3n"):
        df[f"crit_{sch}"] = df[sch] < 0

    df.sort_values("D_4eq", inplace=True)
    df.to_csv(SCRIPT_DIR / "cross_cohort_loo_sensitivity.csv", index=False)
    print("\nSaved -> cross_cohort_loo_sensitivity.csv")

    # Sign-flip and rank-correlation summary
    lines: list[str] = [
        "Cross-cohort panel LOO — weighting sensitivity",
        "=" * 60,
        "Baseline mean AUC under each scheme:",
        f"  3-cohort equal-weight (PRIMARY, drop SRP): {b3_eq:.4f}",
        f"  4-cohort equal-weight (selection scheme):   {b4_eq:.4f}",
        f"  4-cohort sample-weighted:                   {b4_n:.4f}",
        f"  3-cohort sample-weighted:                   {b3_n:.4f}",
        "",
        "Critical-status flips relative to the 4-cohort equal-weight scheme under",
        "which greedy elimination was run (the reference is the selection scheme,",
        "not the primary reporting scheme; flip counts are symmetric in intent):",
    ]
    for sch, lbl in [
        ("D_3eq", "3-cohort equal-weight"),
        ("D_4n", "4-cohort sample-weighted"),
        ("D_3n", "3-cohort sample-weighted"),
    ]:
        flips = df[(df["D_4eq"].lt(0)) != (df[sch].lt(0))]
        lines.append(f"  vs {lbl}: {len(flips)} flips")
        if not flips.empty:
            for _, row in flips.iterrows():
                lines.append(
                    f"    {row['Gene']:<16}  D_4eq={row['D_4eq']:+.4f}  "
                    f"{sch}={row[sch]:+.4f}"
                )

    lines.append("")
    lines.append("Spearman rank correlation between D_ZS schemes:")
    corr = df[["D_4eq", "D_3eq", "D_4n", "D_3n"]].corr(method="spearman")
    lines.append(corr.round(3).to_string())

    # Genes robustly critical across all 4 schemes
    robust = df[
        (df["D_4eq"] < 0) & (df["D_3eq"] < 0) & (df["D_4n"] < 0) & (df["D_3n"] < 0)
    ]
    lines.append("")
    lines.append(f"Robustly critical (D_ZS<0 under all 4 schemes; n={len(robust)}):")
    lines.append("  " + ", ".join(robust["Gene"].tolist()))

    flippy = df[
        (df["D_4eq"] < 0) & ~((df["D_3eq"] < 0) & (df["D_4n"] < 0) & (df["D_3n"] < 0))
    ]
    lines.append("")
    lines.append(f"4eq-critical but flips under another scheme (n={len(flippy)}):")
    for _, row in flippy.iterrows():
        flag = []
        if not (row["D_3eq"] < 0):
            flag.append("3eq")
        if not (row["D_4n"] < 0):
            flag.append("4n")
        if not (row["D_3n"] < 0):
            flag.append("3n")
        lines.append(f"  {row['Gene']:<16}  flips under: {', '.join(flag)}")

    (SCRIPT_DIR / "cross_cohort_loo_sensitivity.txt").write_text("\n".join(lines))
    print("Saved -> cross_cohort_loo_sensitivity.txt")
    print("\n" + "\n".join(lines))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    df_sorted = df.sort_values("D_4eq")
    y = np.arange(len(df_sorted))
    ax = axes[0]
    ax.barh(
        y - 0.3,
        df_sorted["D_4eq"],
        height=0.3,
        label="4-cohort equal-weight (selection scheme)",
        color="#1f77b4",
    )
    ax.barh(
        y,
        df_sorted["D_3eq"],
        height=0.3,
        label="3-cohort equal-weight (primary; drop SRP)",
        color="#ff7f0e",
    )
    ax.barh(
        y + 0.3,
        df_sorted["D_4n"],
        height=0.3,
        label="4-cohort sample-weighted",
        color="#2ca02c",
    )
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(df_sorted["Gene"], fontsize=8)
    ax.set_xlabel(r"$D_{ZS}$ (mean AUC change when gene removed)")
    ax.set_title("Per-gene $D_{ZS}$ under three weighting schemes")
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()

    ax2 = axes[1]
    schemes = [
        ("D_4eq", "4-cohort\nequal-weight"),
        ("D_3eq", "3-cohort\nequal-weight"),
        ("D_4n", "4-cohort\nsample-weighted"),
        ("D_3n", "3-cohort\nsample-weighted"),
    ]
    rho_grid = np.zeros((4, 4))
    for i, (s1, _) in enumerate(schemes):
        for j, (s2, _) in enumerate(schemes):
            rho_grid[i, j], _ = spearmanr(df[s1], df[s2])
    im = ax2.imshow(rho_grid, vmin=0, vmax=1, cmap="RdYlGn")
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels([s[1] for s in schemes], fontsize=8)
    ax2.set_yticklabels([s[1] for s in schemes], fontsize=8)
    for i in range(4):
        for j in range(4):
            ax2.text(
                j,
                i,
                f"{rho_grid[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color="black",
            )
    ax2.set_title("Spearman rank correlation between weighting schemes")
    plt.colorbar(im, ax=ax2, label=r"Spearman $\rho$")

    plt.tight_layout()
    fig.savefig(
        SCRIPT_DIR / "cross_cohort_loo_sensitivity.png", dpi=150, bbox_inches="tight"
    )
    plt.close("all")
    print("Saved -> cross_cohort_loo_sensitivity.png")


if __name__ == "__main__":
    main()
