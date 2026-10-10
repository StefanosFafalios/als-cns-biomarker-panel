"""The per-cell genes in the Pineda et al. 2024 cortex atlas (Step 105k).

Pineda et al. (Cell 2024) profiled motor (BA4) and prefrontal (BA9) cortex of 17 sporadic
ALS, 16 C9orf72 ALS, 13 sporadic FTLD, 11 C9orf72 FTLD and 16 pathologically normal donors and published
cell-type-specific differential expression (Table S6, limma on pseudobulk; Synapse
syn51105515). This looks up the protein-coding critical genes and the druggable
substitutes in the cell type that matches each gene's top cell type in the spinal-cord
nuclei. The authors' own statistics are reported unchanged; the atlas has no fibroblast
or ependymal class, so genes whose spinal top cell type is fibroblast are not looked up.

Outputs: pineda_cortex_lookup.csv, pineda_cortex_lookup_statistics.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from utils import critical_panel_symbols  # noqa: E402

_S06 = SCRIPT_DIR.parent / "resources" / "Pineda2024" / "supplemental" / "S06 - DEGs cell type-specific"
_FILES = {("BA4", "sALS"): "6_1_deg_res_BA4_SALS.xlsx", ("BA4", "C9ALS"): "6_2_deg_res_BA4_C9ALS.xlsx",
          ("BA9", "sALS"): "6_5_deg_res_BA9_SALS.xlsx", ("BA9", "C9ALS"): "6_6_deg_res_BA9_C9ALS.xlsx",
          # FTLD groups of the same atlas: is a change specific to ALS?
          ("BA4", "sFTLD"): "6_3_deg_res_BA4_SFTLD.xlsx", ("BA4", "C9FTLD"): "6_4_deg_res_BA4_C9FTLD.xlsx",
          ("BA9", "sFTLD"): "6_7_deg_res_BA9_SFTLD.xlsx", ("BA9", "C9FTLD"): "6_8_deg_res_BA9_C9FTLD.xlsx"}
# spinal-cord cell type -> Pineda Table S6 sheets
_SHEETS = {"endothelial": ["Vasc.Endo"], "pericyte": ["Vasc.Mural.Pericyte"], "microglia": ["Glia.Micro"],
           "astrocyte": ["Glia.Astro.GFAP-pos", "Glia.Astro.GFAP-neg"], "oligodendrocyte": ["Glia.Oligo"],
           "OPC": ["Glia.OPC"]}


def targets() -> dict[str, tuple[str, str, float]]:
    """gene -> (family, spinal top cell type, spinal log2FC): critical genes and suggestive substitutes."""
    sc = pd.read_csv(SCRIPT_DIR / "snrna_spinal_cord_percell.csv").set_index("gene")
    subs = pd.read_csv(SCRIPT_DIR / "substitute_percell.csv").drop_duplicates("substitute").set_index("substitute")
    genes = {g: ("panel", sc.loc[g, "celltype"], sc.loc[g, "log2fc"]) for g in critical_panel_symbols(protein_coding_only=True)
             if g in sc.index}  # absent = not in the nuclei's gene reference
    genes.update({g: ("substitute", r.celltype, r.log2fc) for g, r in subs.iterrows()
                  if g not in genes and r.evidence == "suggestive"})
    return genes


def main() -> None:
    genes = targets()
    need = sorted({s for _, ct, _ in genes.values() for s in _SHEETS.get(ct, [])})
    rows = []
    for (region, dx), f in _FILES.items():
        book = pd.read_excel(_S06 / f, sheet_name=need)
        for g, (fam, ct, sc_lfc) in genes.items():
            for sheet in _SHEETS.get(ct, []):
                hit = book[sheet][book[sheet].Gene == g]
                r = hit.iloc[0] if len(hit) else None
                rows.append(dict(gene=g, family=fam, spinal_celltype=ct, spinal_log2fc=sc_lfc, region=region, dx=dx,
                                 sheet=sheet, logFC=r.logFC if r is not None else None,
                                 p=r.pvalue if r is not None else None, padj=r.padj if r is not None else None))
    df = pd.DataFrame(rows)
    df.to_csv(SCRIPT_DIR / "pineda_cortex_lookup.csv", index=False)

    lines = ["Critical genes and suggestive substitutes in the Pineda et al. 2024 cortex atlas (Table S6)", "=" * 88,
             "Authors' limma pseudobulk results; cell type matched to the gene's spinal-cord top cell type.",
             "logFC (padj) per region and group; '-' = not tested (below the authors' expression filter).", "",
             f"{'gene':<9}{'family':<11}{'sheet':<22}{'spinal':>7}" + "".join(f"{f'{r} {d}':>18}" for r, d in _FILES)]
    for (g, sheet), s in df.groupby(["gene", "sheet"], sort=False):
        s = s.set_index(["region", "dx"])
        cells = "".join(f"{f'{s.logFC[k]:+.2f} ({s.padj[k]:.2g})' if pd.notna(s.logFC[k]) else '-':>18}"
                        for k in _FILES)
        lines.append(f"{g:<9}{s.family.iloc[0]:<11}{sheet:<22}{s.spinal_log2fc.iloc[0]:>+7.2f}{cells}")
    (SCRIPT_DIR / "pineda_cortex_lookup_statistics.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
