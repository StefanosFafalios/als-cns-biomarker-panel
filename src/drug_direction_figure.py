"""Direction-of-change evidence for the critical panel and the drugs that would reverse it (Step 105g).

A drug candidate has to move its gene back towards the control state. This collects,
for each critical-panel gene, every estimate of the direction it moves in ALS and asks
whether any annotated agent acts in the opposite direction:

  * bulk, all cohorts: random-effects Hedges' g over the nine donor-level units
    (multicohort_meta_analysis.csv), raw and composition-adjusted;
  * per cell, spinal cord: ALS vs control in the gene's top cell type
    (snrna_percell_v2.csv: GSE287257, reference typing, count model);
  * per cell, cortex: the same cell type in two independent snRNA cohorts
    (snrna_cortex_percell.csv);
  * agents: DGIdb interaction types from drug_targets_summary.csv. A gene down in ALS
    needs an activator, a gene up in ALS an inhibitor. Agents annotated with neither
    type are listed as direction-unknown.

Outputs: drug_direction_table.csv, drug_direction_statistics.txt, drug_direction.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from utils import critical_panel_symbols  # noqa: E402

_TIER = {"confirmed": 1, "suggestive": 2}


def _agents(cell: object) -> list[str]:
    return [a for a in str(cell).split("|") if a and a != "nan"]


def build() -> pd.DataFrame:
    """One row per critical-panel gene with every direction estimate and the agent match."""
    genes = critical_panel_symbols(protein_coding_only=True)
    meta = pd.read_csv(SCRIPT_DIR / "multicohort_meta_analysis.csv").set_index("gene")
    # spinal per-cell: the reference-typed count model (step 41c), as drug_targets.py tiers it
    v2 = pd.read_csv(SCRIPT_DIR / "snrna_percell_v2.csv")
    sc = v2[v2.family == "panel"].set_index("gene").rename(columns={"glm_log2fc": "log2fc_v2", "glm_q": "q_v2"})
    sc = sc.assign(log2fc=sc.log2fc_v2, q=sc.q_v2)
    cx = pd.read_csv(SCRIPT_DIR / "snrna_cortex_percell.csv")
    dt = pd.read_csv(SCRIPT_DIR / "drug_targets_summary.csv").set_index("symbol")
    rows = []
    for g in genes:
        d = dt.loc[g]
        m = meta.reindex([g]).iloc[0]  # all NaN when measured in < 4 units (step 42 skips it)
        need = "raise" if d.direction == "down_in_ALS" else "lower"
        inh, act = _agents(d.inhibitors_dgidb), _agents(d.activators_dgidb)
        typed = set(inh) | set(act)
        other = [a for a in _agents(d.approved_drugs_dgidb) + _agents(d.phase3_drugs_ot) if a not in typed]
        matched = act if need == "raise" else inh
        opposed = inh if need == "raise" else act
        row = dict(
            gene=g, disease_direction=d.direction, therapeutic_direction=need,
            tier=_TIER.get(d.snrna_evidence, 3), snrna_evidence=d.snrna_evidence,
            meta_g=m.g, meta_lo=m.lo, meta_hi=m.hi, meta_q=m.q,
            meta_same_sign=f"{int(m.same_sign)}/{int(m.k)}" if pd.notna(m.k) else "n/a",
            adj_g=m.adj_g, adj_q=m.adj_q,
            # NaN for a gene not in the nuclei's gene reference (ENSG00000280893 has no symbol)
            sc_celltype=sc.celltype.get(g, "not in nuclei"), sc_log2fc=sc.log2fc.get(g, np.nan),
            sc_q=sc.q.get(g, np.nan),
            matched_agents="|".join(dict.fromkeys(matched)), opposed_agents="|".join(dict.fromkeys(opposed)),
            direction_unknown_agents="|".join(dict.fromkeys(other)),
        )
        for r in cx[cx.gene == g].itertuples():
            row[f"{r.cohort}_log2fc"], row[f"{r.cohort}_p"] = r.log2fc, r.p
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["tier", "gene"]).reset_index(drop=True)



def _plot(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cohorts = [c[:-7] for c in df.columns if c.endswith("_log2fc") and c != "sc_log2fc"]
    # Table 4 of the manuscript lists the agents, so the figure keeps the two estimate panels at
    # a width that prints legibly when scaled to the text width.
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.55 * len(df) + 2.0), sharey=True)
    y = np.arange(len(df))[::-1]
    ax = axes[0]
    for yi, r in zip(y, df.itertuples()):
        sig = r.meta_q < 0.05
        ax.plot([r.meta_lo, r.meta_hi], [yi, yi], color="#444", lw=1.4)
        ax.plot(r.meta_g, yi, "o", color="#444" if sig else "white", mec="#444", ms=8)
        ax.plot(r.adj_g, yi - 0.22, "D", color="#d95f02" if r.adj_q < 0.05 else "white", mec="#d95f02", ms=6)
        if pd.isna(r.meta_g):
            ax.text(0.05, yi, "too few datasets", va="center", fontsize=8, color="#777", style="italic")
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.gene}  (tier {r.tier})" for r in df.itertuples()])
    ax.set_xlabel("Bulk Hedges' g (ALS minus control)")
    ax.set_title("A. Bulk tissue, nine cohorts\n(circle: raw, diamond: composition-adjusted;\nfilled = q < 0.05)",
                 fontsize=9.5)
    ax = axes[1]
    marks = {cohorts[0]: "^", cohorts[1]: "v"} if len(cohorts) == 2 else {c: "s" for c in cohorts}
    for yi, r in zip(y, df.itertuples()):
        ax.plot(r.sc_log2fc, yi, "o", color="#1b9e77" if r.sc_q < 0.05 else "white", mec="#1b9e77", ms=9)
        for c in cohorts:
            v, p = getattr(r, f"{c}_log2fc"), getattr(r, f"{c}_p")
            ax.plot(v, yi - 0.2, marks[c], color="#7570b3" if p < 0.05 else "white", mec="#7570b3", ms=6)
        if r.sc_celltype == "not in nuclei":
            ax.text(0.05, yi, "not in the nuclei", va="center", fontsize=8, color="#777", style="italic")
    vals = pd.concat([df.sc_log2fc] + [df[f"{c}_log2fc"] for c in cohorts]).dropna()
    ax.set_xlim(vals.min() - 0.25, vals.max() + 0.25)  # keep edge markers inside the frame
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Per-cell log2 fold change, top cell type")
    ax.set_title("B. Single nuclei\n(circle: spinal cord, reference typing;\ntriangles: cortex, "
                 + ", ".join(f"{m} {c}" for c, m in marks.items()) + ";\nfilled = significant)", fontsize=9.5)
    plt.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = build()
    df.to_csv(SCRIPT_DIR / "drug_direction_table.csv", index=False)
    lines = [
        "Direction of change and direction-matched agents, critical panel",
        "=" * 70,
        "need = opposite of the disease direction; matched = DGIdb activators (need raise) or",
        "inhibitors (need lower); opposite = agents acting in the disease direction.",
        "",
        f"{'gene':<9}{'tier':>5} {'dir':<6}{'need':<6}{'bulk g [95% CI]':<24}{'q':>8}{'same':>6}"
        f"{'adj g':>7}{'adj q':>8}  {'per-cell (spinal)':<26} matched | opposite | mode unknown",
    ]
    for r in df.itertuples():
        lines.append(
            f"{r.gene:<9}{r.tier:>5} {('down' if r.disease_direction.startswith('down') else 'up'):<6}{r.therapeutic_direction:<6}"
            f"{f'{r.meta_g:+.2f} [{r.meta_lo:+.2f}, {r.meta_hi:+.2f}]':<24}{r.meta_q:>8.2g}{r.meta_same_sign:>6}"
            f"{r.adj_g:>+7.2f}{r.adj_q:>8.2g}  {f'{r.sc_celltype} {r.sc_log2fc:+.2f} (q {r.sc_q:.2g})':<26} "
            f"{r.matched_agents or '-'} | {r.opposed_agents or '-'} | {r.direction_unknown_agents or '-'}"
        )
    n_t1 = int((df.tier == 1).sum())
    n_t1_matched = int(((df.tier == 1) & (df.matched_agents != "")).sum())
    lines += ["", f"Tier-1 genes (per-cell change confirmed): {n_t1}; with a direction-matched agent: {n_t1_matched}",
              f"Genes with any direction-matched agent: {', '.join(df[df.matched_agents != ''].gene) or 'none'}"]
    (SCRIPT_DIR / "drug_direction_statistics.txt").write_text("\n".join(lines) + "\n")
    _plot(df, SCRIPT_DIR / "drug_direction.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
