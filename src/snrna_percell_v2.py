"""Per-cell test, version 2: reference cell typing and a count model (Step 105k).

Version 1 (snrna_spinal_cord.py) types nuclei by four canonical markers per cell type and
compares donor means of log-normalised expression with a Mann-Whitney test, whose smallest
two-sided p with 8 ALS and 4 control donors is 0.004. Version 2 changes both:

  * typing: best correlation to the subtypes of the healthy spinal-cord atlas GSE190442
    (celltype_reference), which splits capillary/venous from arterial endothelium; genes
    under test and 25-gene panel members are excluded from the typing markers;
  * test: quasi-Poisson GLM of the donor pseudobulk count (summed raw counts, offset log
    library size) on ALS, with a 95% CI; BH within the panel genes and within the substitutes.

Applied to the protein-coding critical genes and the druggable non-inferior substitutes
(substitute_percell.csv), next to the version-1 result for each.

Outputs: snrna_percell_v2.csv, snrna_percell_v2_statistics.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

from snrna_spinal_cord import _mwu, percell_tests, pseudobulk  # noqa: E402
from utils import critical_panel_symbols  # noqa: E402


def main() -> None:
    panel = critical_panel_symbols(protein_coding_only=True)
    v1p = pd.read_csv(SCRIPT_DIR / "snrna_spinal_cord_percell.csv").set_index("gene")
    v1s = pd.read_csv(SCRIPT_DIR / "substitute_percell.csv").drop_duplicates("substitute").set_index("substitute")
    subs = [g for g in v1s.index if g not in panel]
    pb = pseudobulk(panel + subs, typing="reference")

    frames = []
    for fam, genes, v1 in (("panel", panel, v1p), ("substitute", subs, v1s)):
        t = percell_tests(pb, genes).assign(family=fam)
        t["v1_celltype"] = t.gene.map(v1.celltype)
        t["v1_log2fc"] = t.gene.map(v1.log2fc)
        t["v1_q"] = t.gene.map(v1.q)
        frames.append(t)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(SCRIPT_DIR / "snrna_percell_v2.csv", index=False)

    comp = pb.pivot_table(index=["donor", "cond"], columns="celltype", values="frac").reset_index()
    a, c = comp.cond == "ALS", comp.cond == "control"
    lines = ["Per-cell test v2: GSE190442 reference typing + quasi-Poisson pseudobulk (GSE287257 spinal cord)", "=" * 92,
             f"Donors: {int(a.sum())} ALS, {int(c.sum())} control. Composition (nucleus fraction, ALS vs control, MWU):"]
    lines += [f"  {ct:<22}{comp.loc[a, ct].mean():>8.3f}{comp.loc[c, ct].mean():>8.3f}  p {_mwu(comp.loc[a, ct].to_numpy(), comp.loc[c, ct].to_numpy()):.3g}"
              for ct in comp.columns[2:]]
    lines += ["", f"{'gene':<9}{'family':<11}{'v2 cell type':<22}{'GLM log2FC [95% CI]':<26}{'p':>9}{'q':>8}"
              f"{'MWU q':>8}  | v1 {'cell type':<16}{'log2FC':>7}{'q':>7}"]
    for r in df.itertuples():
        glm = (f"{r.glm_log2fc:+.2f} [{r.glm_lo:+.2f}, {r.glm_hi:+.2f}]" if pd.notna(getattr(r, "glm_log2fc", np.nan))
               else "-")
        lines.append(f"{r.gene:<9}{r.family:<11}{r.celltype:<22}{glm:<26}{getattr(r, 'glm_p', np.nan):>9.3g}"
                     f"{getattr(r, 'glm_q', np.nan):>8.3g}{r.q:>8.3g}  | {str(r.v1_celltype):<19}{r.v1_log2fc:>+7.2f}{r.v1_q:>7.3g}")
    (SCRIPT_DIR / "snrna_percell_v2_statistics.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
