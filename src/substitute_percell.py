"""Per-cell test of the druggable substitutes of the critical-panel genes (Step 105j).

Takes every non-inferior substitute that has a restoring DGIdb agent
(critical_replacement_pooled.csv) and applies the single-nucleus test the panel genes
received: ALS vs control donor pseudobulk in the gene's top cell type in cervical spinal
cord (GSE287257, 8 vs 4 donors; MWU, BH across the substitutes), then the same cell type
in two cortex cohorts with independent donors (GSE219280 C9orf72, GSE212630 TDP-43-high).
Graded as for the panel genes (drug_targets._snrna_evidence) against the substitute's own
bulk direction: confirmed (q < 0.05, same direction), suggestive (p < 0.05), none,
opposite, not measured.

Outputs: substitute_percell.csv, substitute_percell_statistics.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

from drug_targets import _snrna_evidence  # noqa: E402
from snrna_cortex_percell import cortex_tests  # noqa: E402
from snrna_spinal_cord import percell_tests, pseudobulk  # noqa: E402

_ORDER = {"confirmed": 0, "suggestive": 1, "none": 2, "opposite": 3, "not_measured": 4}


def main() -> None:
    rep = pd.read_csv(SCRIPT_DIR / "critical_replacement_pooled.csv")
    sub = rep[rep.non_inferior & rep.agents.fillna("").astype(bool)].copy()
    out = SCRIPT_DIR / "substitute_percell_statistics.txt"
    if sub.empty:
        out.write_text("No non-inferior substitute has a restoring DGIdb agent.\n")
        print(out.read_text())
        return
    genes = sorted(set(sub.substitute))
    pc = percell_tests(pseudobulk(genes), genes).set_index("gene")
    top = pc.celltype.to_dict()
    cx = cortex_tests(sorted(top), top) if top else pd.DataFrame(columns=["cohort", "gene", "log2fc", "p"])

    rows = []
    for r in sub.itertuples():
        rec = pc.loc[r.substitute].to_dict() if r.substitute in pc.index else None
        row = dict(panel_gene=r.panel_gene, substitute=r.substitute, wmean=r.wmean, meta_g=r.meta_g,
                   meta_q=r.meta_q, need=r.need, approved=r.approved, agents=r.agents,
                   evidence=_snrna_evidence("down_in_ALS" if r.meta_g < 0 else "up_in_ALS", rec),
                   celltype=rec["celltype"] if rec else None, log2fc=rec["log2fc"] if rec else np.nan,
                   p=rec["p"] if rec else np.nan, q=rec["q"] if rec else np.nan)
        for c in cx.itertuples():
            if c.gene == r.substitute:
                row[f"{c.cohort}_log2fc"], row[f"{c.cohort}_p"] = c.log2fc, c.p
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.sort_values(["evidence", "meta_q"], key=lambda s: s.map(_ORDER) if s.name == "evidence" else s)
    df.to_csv(SCRIPT_DIR / "substitute_percell.csv", index=False)

    cohorts = sorted({c[:-7] for c in df.columns if c.endswith("_log2fc") and c != "log2fc"})
    lines = ["Per-cell test of the druggable non-inferior substitutes", "=" * 60,
             "Spinal cord GSE287257 (8 ALS / 4 control), top cell type, donor pseudobulk, MWU, BH across "
             f"{len(genes)} substitutes;", "cortex: same cell type, independent donors (log2FC, p). Evidence graded "
             "against the substitute's own bulk direction.", "",
             f"{'panel':<8}{'substitute':<11}{'bulk g':>7}{'q':>8}  {'need':<6}{'evidence':<13}{'cell type':<16}"
             f"{'log2FC':>7}{'q':>7}" + "".join(f"{c:>18}" for c in cohorts) + "  approved agents"]
    for r in df.itertuples():
        cx_cells = "".join(
            f"{getattr(r, f'{c}_log2fc'):>+10.2f} ({getattr(r, f'{c}_p'):.2f})" if pd.notna(getattr(r, f"{c}_log2fc", np.nan))
            else f"{'-':>18}" for c in cohorts)
        lines.append(f"{r.panel_gene:<8}{r.substitute:<11}{r.meta_g:>+7.2f}{r.meta_q:>8.2g}  {r.need:<6}{r.evidence:<13}"
                     f"{str(r.celltype):<16}{r.log2fc:>+7.2f}{r.q:>7.3f}{cx_cells}  "
                     f"{r.approved if isinstance(r.approved, str) and r.approved else '-'}")
    n = df.drop_duplicates("substitute").evidence.value_counts().to_dict()
    lines += ["", "Substitutes by per-cell evidence: " + ", ".join(f"{k} {v}" for k, v in n.items())]
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
