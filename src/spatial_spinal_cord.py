"""Spatial replication of the per-cell changes in independent ALS spinal cords (Step 105k).

GSE288365 (Nat Neurosci 2026, the "Molecular drivers of neuroinflammation" study): Visium
CytAssist on FFPE cervical spinal cord, one section per donor, 9 non-neurological controls
and 9 ALS (5 sporadic, 4 C9orf72), none of them GSE287256/7 donors. GEO lists no diagnosis;
it comes from the paper's Supplementary Table 1, mapped to the section titles in _DONORS.

A 55-um spot holds several cells, so each gene is tested where its cell type concentrates:
  * spot expression CP10k + log1p; cell-type score = mean z of the panel-free markers of
    snrna_spinal_cord._MARKERS within the section;
  * per donor and cell type: the top decile of spots by that score, raw counts summed
    (pseudobulk) with their summed library size;
  * each gene in its top cell type from the spinal-cord nuclei (snrna_spinal_cord_percell.csv,
    substitute_percell.csv): quasi-Poisson GLM of the donor count on ALS, offset log library
    size, adjusted for the donor's mean cell-type score in those spots (abundance left after
    taking the top decile); BH within the panel genes and within the substitutes.

Outputs: spatial_spinal_cord.csv, spatial_spinal_cord_statistics.txt
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

from scipy.stats import mannwhitneyu  # noqa: E402
from snrna_spinal_cord import _MARKERS, quasi_poisson  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402
from utils import critical_panel_symbols  # noqa: E402

_SUPPL = SCRIPT_DIR.parent / "resources" / "GSE288365" / "suppl"
TOP = 0.10
# section-title fragment -> (Supplementary Table 1 label, diagnosis)
_DONORS = {
    "GBB-17-21": ("H8_GB17", "control"), "GBB-18-01": ("H5_GB18", "control"),
    "GBB19-13": ("H7_GB19", "control"), "GBB20-16": ("H1_GB20", "control"),
    "GBB21-03": ("H4_GB21", "control"), "GWF-17-23": ("H0_GW23", "control"),
    "NEUDW867LF9": ("H2_NLF9", "control"), "NEUHD481VCL": ("H3_NVCL", "control"),
    "NEUUJ189PPH": ("H9_NPPH", "control"),
    "GWF-17-24": ("C5_GW24", "C9"), "MILLERMI-AU-87": ("C3_AU87", "C9"),
    "MILLERMI-N14-280": ("C4_N280", "C9"), "MILLERMINEUWM_344ZLM": ("C2_NZLM", "C9"),
    "GWF-20-50": ("S1_GW50", "sALS"), "GWF-19-37": ("S2_GW37", "sALS"),
    "GWF_19-35": ("S4_GW35", "sALS"), "GWF_19-47": ("S5_GW47", "sALS"),
    "GWF-18-33": ("S3_GW18", "sALS"),  # the one GW label left once the others match
}


def _section(prefix: str, genes: list[str]) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """In-tissue spots of one section: raw counts of *genes*, library sizes, cell-type scores."""
    feat = pd.read_csv(f"{prefix}_features.tsv.gz", sep="\t", header=None)
    keep_f = (feat[2] == "Gene Expression").to_numpy() if feat.shape[1] > 2 else np.ones(len(feat), bool)
    bc = pd.read_csv(f"{prefix}_barcodes.tsv.gz", header=None)[0].to_numpy()
    with gzip.open(f"{prefix}_matrix.mtx.gz") as fh:
        mat = sp.csr_matrix(scipy.io.mmread(fh))[keep_f]  # genes x spots
    names = feat[1].to_numpy()[keep_f]
    pos = pd.read_csv(f"{prefix}_tissue_positions.csv.gz", header=None)
    if not str(pos.iloc[0, 1]).isdigit():  # Space Ranger >= 2 writes a header row
        pos = pos.iloc[1:]
    in_tissue = set(pos[pos[1].astype(int) == 1][0])
    cols = np.array([b in in_tissue for b in bc])
    mat = mat[:, cols]
    lib = np.asarray(mat.sum(axis=0)).ravel()
    row = {}
    for i, n in enumerate(names):
        row.setdefault(n, i)
    want = sorted(set(genes) | {g for mk in _MARKERS.values() for g in mk})
    present = [g for g in want if g in row]
    raw = pd.DataFrame(mat[[row[g] for g in present]].toarray().T, columns=present)
    logn = np.log1p(raw.div(np.where(lib > 0, lib, 1) / 1e4, axis=0))
    scores = pd.DataFrame({ct: ((logn[mk_p] - logn[mk_p].mean()) / (logn[mk_p].std() + 1e-8)).mean(axis=1)
                           for ct, mk in _MARKERS.items() if (mk_p := [g for g in mk if g in logn])})
    return raw, lib, scores


def main() -> None:
    panel = critical_panel_symbols(protein_coding_only=True)
    top = pd.read_csv(SCRIPT_DIR / "snrna_spinal_cord_percell.csv").set_index("gene")
    sub = pd.read_csv(SCRIPT_DIR / "substitute_percell.csv").drop_duplicates("substitute").set_index("substitute")
    sub = sub[sub.celltype.notna()]
    # a panel gene absent from the spinal table has no symbol in the nuclei's reference (no top cell type)
    tests = {g: ("panel", top.loc[g, "celltype"], top.loc[g, "log2fc"]) for g in panel if g in top.index}
    tests.update({g: ("substitute", r.celltype, r.log2fc) for g, r in sub.iterrows() if g not in tests})

    prefixes = sorted({str(p)[: -len("_matrix.mtx.gz")] for p in _SUPPL.glob("GSE288365_*_matrix.mtx.gz")})
    if not prefixes:
        raise SystemExit(f"No GSE288365 matrices in {_SUPPL}; run reproduce/download_geo_data.sh first.")
    rows = []  # one per donor x cell type: summed counts, library size, mean score
    for prefix in prefixes:
        key = next((k for k in _DONORS if k in Path(prefix).name), None)
        if key is None:
            raise SystemExit(f"no diagnosis for section {Path(prefix).name}")
        label, dx = _DONORS[key]
        raw, lib, scores = _section(prefix, list(tests))
        for ct in scores:
            sel = (scores[ct] >= scores[ct].quantile(1 - TOP)).to_numpy()
            rows.append(dict(donor=label, dx=dx, celltype=ct, n_spots=int(sel.sum()), lib=float(lib[sel].sum()),
                             score=float(scores[ct][sel].mean()), **raw[sel].sum().to_dict()))
        print(f"  {label:<8} {dx:<8} {len(lib)} spots", flush=True)
    # sections split into two halves (HCTOP/HCBOT) are one donor: sum counts, weight scores by spots
    pb = pd.DataFrame(rows)
    pb["score"] *= pb.n_spots
    pb = pb.groupby(["donor", "dx", "celltype"], as_index=False).sum(numeric_only=True)
    pb["score"] /= pb.n_spots

    out = []
    for g, (family, ct, sc_lfc) in tests.items():
        s = pb[pb.celltype == ct]
        if g not in s or s[g].sum() == 0:
            out.append(dict(gene=g, family=family, celltype=ct, snrna_log2fc=sc_lfc))
            continue
        als = (s.dx != "control").to_numpy()
        lfc, se, p = quasi_poisson(s[g].to_numpy(float), s.lib.to_numpy(float), als, s.score.to_numpy(float))
        cpm = np.log2(s[g] / s.lib * 1e6 + 1)
        p_mwu = float(mannwhitneyu(cpm[als], cpm[~als], alternative="two-sided").pvalue)
        out.append(dict(gene=g, family=family, celltype=ct, snrna_log2fc=sc_lfc, log2fc=lfc,
                        ci_lo=lfc - 1.96 * se, ci_hi=lfc + 1.96 * se, p=p, p_mwu=p_mwu,
                        n_als=int(als.sum()), n_ctl=int((~als).sum()), same_sign=bool(np.sign(lfc) == np.sign(sc_lfc))))
    df = pd.DataFrame(out)
    df["q"] = np.nan
    for fam in ("panel", "substitute"):
        m = (df.family == fam) & df.p.notna()
        if m.any():
            df.loc[m, "q"] = multipletests(df.loc[m, "p"], method="fdr_bh")[1]
    df.to_csv(SCRIPT_DIR / "spatial_spinal_cord.csv", index=False)

    lines = ["Spatial replication in independent ALS cervical spinal cords (GSE288365, Visium CytAssist)", "=" * 88,
             f"Donors: {pb[pb.dx == 'control'].donor.nunique()} control, {pb[pb.dx != 'control'].donor.nunique()} ALS "
             f"({pb[pb.dx == 'sALS'].donor.nunique()} sporadic, {pb[pb.dx == 'C9'].donor.nunique()} C9orf72). Per donor: "
             f"top {TOP:.0%} of spots by the gene's cell-type score, summed counts.",
             "Quasi-Poisson GLM, offset log library, adjusted for mean cell-type score; BH within family. "
             "snRNA = spinal-cord nuclei log2FC (GSE287257).", "",
             f"{'gene':<9}{'family':<11}{'cell type':<16}{'snRNA':>7}{'spatial':>8}{'95% CI':>17}{'p':>9}{'q':>8}"
             f"{'MWU p':>8}  same sign"]
    for r in df.itertuples():
        if pd.isna(getattr(r, "log2fc", np.nan)):
            lines.append(f"{r.gene:<9}{r.family:<11}{r.celltype:<16}{r.snrna_log2fc:>+7.2f}  not measured")
            continue
        lines.append(f"{r.gene:<9}{r.family:<11}{r.celltype:<16}{r.snrna_log2fc:>+7.2f}{r.log2fc:>+8.2f}"
                     f"{f'[{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]':>17}{r.p:>9.3g}{r.q:>8.3g}{r.p_mwu:>8.3g}  {r.same_sign}")
    (SCRIPT_DIR / "spatial_spinal_cord_statistics.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
