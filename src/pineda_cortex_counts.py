"""The per-cell genes in the Pineda et al. 2024 cortex atlas, re-tested with our count model (Step 105m).

pineda_cortex_lookup.py reads the authors' limma statistics. This re-tests the same genes
on their processed counts (Synapse syn53421586: per-library counts_fil.mtx with the
authors' cell annotation, reproduce/download_synapse_pineda.py) with the quasi-Poisson
donor pseudobulk model used in the spinal cord, so both tissues share one method.

  * pseudobulk: per donor x region x the authors' DGE_Group, summed counts of each target
    gene and the summed library size (all genes); donors with < 10 nuclei of the type drop out;
  * test: each disease group (sporadic/C9orf72 ALS, sporadic/C9orf72 FTLD) against the
    pathologically normal donors, per region (BA4 motor, BA9 prefrontal), no covariates;
    BH across the critical genes within each comparison.

Outputs: pineda_cortex_counts_pseudobulk.csv (cache), pineda_cortex_counts.csv,
pineda_cortex_counts_statistics.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

from pineda_cortex_lookup import _SHEETS, targets  # noqa: E402
from snrna_spinal_cord import quasi_poisson  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

_PROC = SCRIPT_DIR.parent / "resources" / "Pineda2024" / "processed"
_CACHE = SCRIPT_DIR / "pineda_cortex_counts_pseudobulk.csv"
_DX = {"SALS": "sALS", "C9ALS": "C9ALS", "SFTLD": "sFTLD", "C9FTLD": "C9FTLD"}
MIN_NUCLEI = 10
# the count matrices also label fibroblasts, which Table S6 has no sheet for
_GROUPS = {**_SHEETS, "fibroblast": ["Vasc.Fibro"]}


def library_pseudobulk(lib: Path, genes: list[str], groups: set[str]) -> pd.DataFrame:
    """Summed counts of *genes* and library size per donor x region x DGE_Group of one library.

    Args:
        lib: Folder with counts_fil.mtx (genes x cells, MatrixMarket), row_metadata.tsv (Gene)
            and col_metadata.tsv (Donor, Region, Group, DGE_Group), rows in matrix order.
        genes: Gene symbols to sum (the first row of a duplicated symbol is used).
        groups: DGE_Group values to keep.
    """
    keys = ["Donor", "Region", "Group", "DGE_Group"]
    if "DGE_Group" not in pd.read_csv(lib / "col_metadata.tsv", sep="\t", nrows=0).columns:
        return pd.DataFrame(columns=keys)  # the five mouse M1 libraries (220616_BL6_MCX_*) carry no DGE_Group
    cm = pd.read_csv(lib / "col_metadata.tsv", sep="\t", usecols=keys)
    rm = pd.read_csv(lib / "row_metadata.tsv", sep="\t", usecols=["Gene"])
    with (lib / "counts_fil.mtx").open() as fh:
        line = fh.readline()
        while line.startswith("%"):
            line = fh.readline()
        n_g, n_c, _ = map(int, line.split())
        t = pd.read_csv(fh, sep=" ", header=None, dtype=np.int64).to_numpy()
    if (n_g, n_c) != (len(rm), len(cm)):
        raise ValueError(f"{lib.name}: matrix {n_g}x{n_c} vs metadata {len(rm)}x{len(cm)}")
    gi, ci, v = t[:, 0] - 1, t[:, 1] - 1, t[:, 2]
    pos = np.full(n_g, -1)
    first = rm.reset_index().drop_duplicates("Gene").set_index("Gene")["index"]
    for k, g in enumerate(genes):
        if g in first.index:
            pos[first[g]] = k
    hit = pos[gi] >= 0
    cell_gene = np.zeros((n_c, len(genes)))
    np.add.at(cell_gene, (ci[hit], pos[gi[hit]]), v[hit])
    df = pd.DataFrame(cell_gene, columns=genes).assign(lib=np.bincount(ci, weights=v, minlength=n_c), n=1)
    df = pd.concat([cm, df], axis=1)
    df = df[df.DGE_Group.isin(groups)]
    return df.groupby(["Donor", "Region", "Group", "DGE_Group"], as_index=False).sum()


def pseudobulk(genes: list[str], groups: set[str]) -> pd.DataFrame:
    """Donor x region x cell group pseudobulk over all libraries (cached in _CACHE)."""
    if _CACHE.exists():
        pb = pd.read_csv(_CACHE)
        if set(genes) <= set(pb.columns):
            return pb
    libs = sorted(p for p in _PROC.iterdir() if (p / "counts_fil.mtx").exists())
    if not libs:
        raise FileNotFoundError(f"No libraries in {_PROC}; run reproduce/download_synapse_pineda.py first.")
    parts = []
    for i, lib in enumerate(libs, 1):
        parts.append(library_pseudobulk(lib, genes, groups))
        print(f"  {i}/{len(libs)} {lib.name}", flush=True)
    pb = pd.concat(parts).groupby(["Donor", "Region", "Group", "DGE_Group"], as_index=False).sum()
    pb.to_csv(_CACHE, index=False)
    return pb


def main() -> None:
    tg = targets()
    genes = sorted(g for g, (_, ct, _) in tg.items() if ct in _GROUPS)
    pb = pseudobulk(genes, {s for ss in _GROUPS.values() for s in ss})
    pb = pb[pb.n >= MIN_NUCLEI]
    rows = []
    for region in ("BA4", "BA9"):
        for grp, dx in _DX.items():
            for g in genes:
                fam, ct, sc_lfc = tg[g]
                for sheet in _GROUPS[ct]:
                    s = pb[(pb.Region == region) & (pb.DGE_Group == sheet) & pb.Group.isin([grp, "PN"])]
                    case = (s.Group == grp).to_numpy()
                    rec = dict(gene=g, family=fam, sheet=sheet, spinal_log2fc=sc_lfc, region=region, dx=dx,
                               n_case=int(case.sum()), n_pn=int((~case).sum()))
                    if case.any() and (~case).any() and s.loc[case, g].sum() > 0 and s.loc[~case, g].sum() > 0:
                        lfc, se, p = quasi_poisson(s[g].to_numpy(float), s.lib.to_numpy(float), case)
                        rec.update(log2fc=lfc, lo=lfc - 1.96 * se, hi=lfc + 1.96 * se, p=p)
                    rows.append(rec)
    df = pd.DataFrame(rows)
    df["q"] = np.nan
    for _, idx in df[df.p.notna()].groupby(["region", "dx", "family"]).groups.items():
        df.loc[idx, "q"] = multipletests(df.loc[idx, "p"], method="fdr_bh")[1]
    lk = pd.read_csv(SCRIPT_DIR / "pineda_cortex_lookup.csv").rename(columns={"logFC": "authors_logFC", "padj": "authors_padj"})
    df = df.merge(lk[["gene", "sheet", "region", "dx", "authors_logFC", "authors_padj"]], how="left",
                  on=["gene", "sheet", "region", "dx"])
    df.to_csv(SCRIPT_DIR / "pineda_cortex_counts.csv", index=False)

    lines = ["Critical genes and suggestive substitutes in the Pineda et al. 2024 cortex atlas: count model", "=" * 94,
             f"Quasi-Poisson donor pseudobulk on the authors' processed counts and cell annotation (>= {MIN_NUCLEI} nuclei",
             "per donor); each group vs pathologically normal donors; BH within region x group x family.",
             "Cells: log2FC (q); authors' limma logFC in brackets.", "",
             f"{'gene':<9}{'sheet':<22}{'spinal':>7}" + "".join(f"{f'{r} {d}':>22}" for r in ("BA4", "BA9") for d in _DX.values())]
    for (g, sheet), s in df.groupby(["gene", "sheet"], sort=False):
        s = s.set_index(["region", "dx"])
        cells = ""
        for k in [(r, d) for r in ("BA4", "BA9") for d in _DX.values()]:
            r = s.loc[k]
            ours = f"{r.log2fc:+.2f} ({r.q:.2g})" if pd.notna(r.log2fc) else "-"
            auth = f"[{r.authors_logFC:+.2f}]" if pd.notna(r.authors_logFC) else "[-]"
            cells += f"{ours + ' ' + auth:>22}"
        lines.append(f"{g:<9}{sheet:<22}{s.spinal_log2fc.iloc[0]:>+7.2f}{cells}")
    n = df.groupby(["region", "dx"])[["n_case", "n_pn"]].max()
    lines += ["", "Donors with >= 10 nuclei of the type (max over types): "
              + ", ".join(f"{r} {d} {int(v.n_case)} vs {int(v.n_pn)}" for (r, d), v in n.iterrows())]
    (SCRIPT_DIR / "pineda_cortex_counts_statistics.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
