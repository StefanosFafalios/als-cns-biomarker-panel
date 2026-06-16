"""Sensitivity analysis of cross-cohort panel LOO D_ZS under alternative weightings.

Compares the headline 4-cohort equal-weight D_ZS against:
  - 3-cohort equal-weight (excludes degenerate SRP064478)
  - 4-cohort sample-size-weighted
  - 3-cohort sample-size-weighted

Inputs: panel_loo_zeroshot_statistics.txt (existing per-gene per-cohort AUC table).
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


def _parse_stats() -> tuple[pd.DataFrame, dict[str, float], dict[str, int]]:
    raw = STATS_PATH.read_text().splitlines()
    base: dict[str, float] = {}
    n: dict[str, int] = {}
    for ln in raw:
        m = re.match(r"\s+(\w+)\s+n=(\d+)\s+w=", ln)
        if m:
            n[m.group(1)] = int(m.group(2))
        m2 = re.match(r"\s+(GPL16791|GSE76220|GSE122649|SRP064478):\s+(\d\.\d+)", ln)
        if m2:
            base[m2.group(1)] = float(m2.group(2))

    rows = []
    started = False
    for ln in raw:
        if ln.startswith("--"):
            started = True
            continue
        if not started or not ln.strip():
            continue
        if ln.startswith("Interpretation"):
            break
        parts = ln.split()
        if len(parts) >= 6 and re.match(r"[-+]?\d+\.\d+", parts[1]):
            try:
                rows.append({
                    "Gene": parts[0],
                    "AUC_GPL16791": float(parts[3]),
                    "AUC_GSE76220": float(parts[4]),
                    "AUC_GSE122649": float(parts[5]),
                    "AUC_SRP064478": float(parts[6]),
                })
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows), base, n


def main() -> None:
    df, base, n = _parse_stats()
    print(f"Parsed {len(df)} genes, cohorts: {list(base)}")
    print(f"  baseline AUCs: {base}")
    print(f"  sample sizes:  {n}")

    cohorts = ["GPL16791", "GSE76220", "GSE122649", "SRP064478"]
    cols = [f"AUC_{c}" for c in cohorts]
    N4 = sum(n.values())
    N3 = N4 - n["SRP064478"]

    b4_eq = sum(base.values()) / 4
    b3_eq = sum(base[c] for c in cohorts if c != "SRP064478") / 3
    b4_n = sum(base[c] * n[c] for c in cohorts) / N4
    b3_n = sum(base[c] * n[c] for c in cohorts if c != "SRP064478") / N3

    print(f"\nBaselines:")
    print(f"  4-cohort equal-weight:    {b4_eq:.4f}")
    print(f"  3-cohort equal-weight:    {b3_eq:.4f}   (sensitivity: excludes SRP064478)")
    print(f"  4-cohort sample-weighted: {b4_n:.4f}")
    print(f"  3-cohort sample-weighted: {b3_n:.4f}")

    df["D_4eq"] = df[cols].sum(axis=1) / 4 - b4_eq
    df["D_3eq"] = df[cols[:3]].sum(axis=1) / 3 - b3_eq
    df["D_4n"] = (
        df["AUC_GPL16791"] * n["GPL16791"] + df["AUC_GSE76220"] * n["GSE76220"]
        + df["AUC_GSE122649"] * n["GSE122649"] + df["AUC_SRP064478"] * n["SRP064478"]
    ) / N4 - b4_n
    df["D_3n"] = (
        df["AUC_GPL16791"] * n["GPL16791"] + df["AUC_GSE76220"] * n["GSE76220"]
        + df["AUC_GSE122649"] * n["GSE122649"]
    ) / N3 - b3_n

    # Status by scheme
    for sch in ("D_4eq", "D_3eq", "D_4n", "D_3n"):
        df[f"crit_{sch}"] = df[sch] < 0

    df.sort_values("D_4eq", inplace=True)
    df.to_csv(SCRIPT_DIR / "cross_cohort_loo_sensitivity.csv", index=False)
    print(f"\nSaved -> cross_cohort_loo_sensitivity.csv")

    # Sign-flip and rank-correlation summary
    lines: list[str] = [
        "Cross-cohort panel LOO — weighting sensitivity",
        "=" * 60,
        f"Baseline mean AUC under each scheme:",
        f"  4-cohort equal-weight (primary):    {b4_eq:.4f}",
        f"  3-cohort equal-weight (drop SRP):   {b3_eq:.4f}",
        f"  4-cohort sample-weighted:           {b4_n:.4f}",
        f"  3-cohort sample-weighted:           {b3_n:.4f}",
        "",
        "Critical-status flips relative to the headline 4-cohort equal-weight scheme:",
    ]
    for sch, lbl in [("D_3eq", "3-cohort equal-weight"),
                     ("D_4n", "4-cohort sample-weighted"),
                     ("D_3n", "3-cohort sample-weighted")]:
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
    robust = df[(df["D_4eq"] < 0) & (df["D_3eq"] < 0) & (df["D_4n"] < 0) & (df["D_3n"] < 0)]
    lines.append("")
    lines.append(f"Robustly critical (D_ZS<0 under all 4 schemes; n={len(robust)}):")
    lines.append("  " + ", ".join(robust["Gene"].tolist()))

    flippy = df[(df["D_4eq"] < 0) & ~((df["D_3eq"] < 0) & (df["D_4n"] < 0) & (df["D_3n"] < 0))]
    lines.append("")
    lines.append(f"4eq-critical but flips under another scheme (n={len(flippy)}):")
    for _, row in flippy.iterrows():
        flag = []
        if not (row["D_3eq"] < 0): flag.append("3eq")
        if not (row["D_4n"] < 0): flag.append("4n")
        if not (row["D_3n"] < 0): flag.append("3n")
        lines.append(f"  {row['Gene']:<16}  flips under: {', '.join(flag)}")

    (SCRIPT_DIR / "cross_cohort_loo_sensitivity.txt").write_text("\n".join(lines))
    print(f"Saved -> cross_cohort_loo_sensitivity.txt")
    print("\n" + "\n".join(lines))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    df_sorted = df.sort_values("D_4eq")
    y = np.arange(len(df_sorted))
    ax = axes[0]
    ax.barh(y - 0.3, df_sorted["D_4eq"], height=0.3, label="4-cohort equal-weight (primary)", color="#1f77b4")
    ax.barh(y, df_sorted["D_3eq"], height=0.3, label="3-cohort equal-weight (drop SRP)", color="#ff7f0e")
    ax.barh(y + 0.3, df_sorted["D_4n"], height=0.3, label="4-cohort sample-weighted", color="#2ca02c")
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(df_sorted["Gene"], fontsize=8)
    ax.set_xlabel(r"$D_{ZS}$ (mean AUC change when gene removed)")
    ax.set_title("Per-gene $D_{ZS}$ under three weighting schemes")
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()

    ax2 = axes[1]
    schemes = [("D_4eq", "4-cohort\nequal-weight"), ("D_3eq", "3-cohort\nequal-weight"),
               ("D_4n", "4-cohort\nsample-weighted"), ("D_3n", "3-cohort\nsample-weighted")]
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
            ax2.text(j, i, f"{rho_grid[i, j]:.2f}", ha="center", va="center",
                     fontsize=10, color="black")
    ax2.set_title("Spearman rank correlation between weighting schemes")
    plt.colorbar(im, ax=ax2, label=r"Spearman $\rho$")

    plt.tight_layout()
    fig.savefig(SCRIPT_DIR / "cross_cohort_loo_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("Saved -> cross_cohort_loo_sensitivity.png")


if __name__ == "__main__":
    main()
