# ruff: noqa: E402
"""Single-nucleus per-cell validation of the panel — experiment #1.

Public snRNA-seq (GSE219280; C9orf72 ALS/FTD frontal+motor cortex, cellBender
filtered). Tests at single-nucleus resolution, on a balanced C9ALS-vs-Control
frontal-cortex subset, whether:

  (a) the composition shift replicates -- microglial expansion + endothelial
      depletion in ALS (validates the bulk deconvolution #3 directly);
  (b) the panel marker genes localise to the expected cell types
      (SLC37A2/HEXB/PROS1 -> microglia; CLDN5/VWF/ANGPT2 -> endothelial);
  (c) FCN3 and SERTAD1 (the bulk sign-reversal candidates) are detectable and
      per-cell altered in ALS vs control, or whether FCN3 is essentially absent
      from nuclei (which would instead support the plasma-infiltration model).

Method: read each filtered .h5 (h5py + scipy; no scanpy), CP10k+log1p normalise,
assign each nucleus to a CNS cell type by canonical marker score (argmax;
FCN3/SERTAD1 are NOT markers), then pseudobulk per (sample, cell type) and test
C9ALS vs Control (Mann-Whitney, n=6 vs 6).

Outputs
-------
  snrna_percell_validation_statistics.txt
  snrna_percell_validation.png
"""

from __future__ import annotations

import glob
import re
import warnings
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

SCRIPT_DIR = Path(__file__).parent
_H5_DIR = SCRIPT_DIR.parent / "resources" / "GSE219280" / "suppl"
MIN_COUNTS = 500  # per-nucleus QC floor

# Canonical CNS cell-type markers (FCN3/SERTAD1 deliberately excluded)
_MARKERS = {
    "neuron": ["RBFOX3", "SYT1", "SNAP25", "SYP", "MEG3"],
    "astrocyte": ["AQP4", "GFAP", "SLC1A2", "GJA1", "ALDH1L1"],
    "microglia": ["AIF1", "CSF1R", "P2RY12", "CX3CR1", "C1QB"],
    "oligodendrocyte": ["PLP1", "MBP", "MOBP", "MOG", "CNP"],
    "OPC": ["PDGFRA", "CSPG4", "OLIG1", "OLIG2"],
    "endothelial": ["CLDN5", "FLT1", "PECAM1", "VWF"],
}
_TEST = ["FCN3", "SERTAD1"]
_PANEL = [
    "SLC37A2",
    "HEXB",
    "PROS1",
    "EMP1",
    "ANGPT2",
    "TINAGL1",
    "MCEE",
    "CKMT2",
    "NR4A1",
    "SOHLH2",
    "MECOM",
]
_ALL_GENES = sorted(set(sum(_MARKERS.values(), []) + _TEST + _PANEL))


def _read_h5_targets(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (lognorm target-gene matrix [cells x genes], total_counts, genes).

    Reads the CellRanger/cellBender sparse matrix (genes x cells), computes
    per-nucleus totals, extracts only the target genes, CP10k+log1p normalises.
    """
    with h5py.File(path, "r") as f:
        g = f["matrix"]
        data, indices, indptr = g["data"][:], g["indices"][:], g["indptr"][:]
        shape = tuple(g["shape"][:])  # (n_genes, n_cells)
        names = np.array([x.decode() for x in g["features"]["name"][:]])
    # CSC: columns are cells
    mat = sp.csc_matrix((data, indices, indptr), shape=shape)
    totals = np.asarray(mat.sum(axis=0)).ravel()  # per-nucleus total counts
    # first column index per target symbol
    name_to_row: dict[str, int] = {}
    for i, nm in enumerate(names):
        name_to_row.setdefault(nm, i)
    rows = [name_to_row.get(gname, -1) for gname in _ALL_GENES]
    present = [gname for gname, r in zip(_ALL_GENES, rows) if r >= 0]
    rows = [r for r in rows if r >= 0]
    sub = mat[rows, :].toarray().T.astype(np.float32)  # (cells, genes)
    # CP10k + log1p
    sf = np.where(totals > 0, totals / 1e4, 1.0)
    sub = np.log1p(sub / sf[:, None])
    return sub, totals, present


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.stats import mannwhitneyu

    files = sorted(glob.glob(str(_H5_DIR / "*cellBender_corrected_filtered.h5")))
    if not files:
        print(f"No .h5 in {_H5_DIR}; run the download first.")
        return
    print(f"Reading {len(files)} snRNA samples ...")

    # accumulate per-sample per-celltype pseudobulk (mean log-norm) + fractions
    pb_rows = []  # one row per (sample, celltype)
    spec_accum: dict[str, list[np.ndarray]] = {ct: [] for ct in _MARKERS}
    genes_ref: list[str] = []
    for path in files:
        fn = Path(path).name
        m = re.search(r"_(C9ALS|Control)_", fn)
        cond = m.group(1) if m else "?"
        tissue = "MotorCortex" if "MotorCortex" in fn else "FrontalCortex"
        samp = fn.split("_cellBender")[0]
        sub, totals, genes = _read_h5_targets(path)
        if not genes_ref:
            genes_ref = genes
        gidx = {gname: j for j, gname in enumerate(genes)}
        keep = totals >= MIN_COUNTS
        sub = sub[keep]
        # marker score per cell type (mean of z-scored markers across nuclei)
        scores = np.zeros((sub.shape[0], len(_MARKERS)), dtype=np.float32)
        for k, (ct, mk) in enumerate(_MARKERS.items()):
            cols = [gidx[g0] for g0 in mk if g0 in gidx]
            if not cols:
                scores[:, k] = -np.inf
                continue
            block = sub[:, cols]
            z = (block - block.mean(0)) / (block.std(0) + 1e-8)
            scores[:, k] = z.mean(1)
        assign = np.array(list(_MARKERS))[scores.argmax(1)]
        n_cells = len(assign)
        for ct in _MARKERS:
            mask = assign == ct
            frac = mask.sum() / n_cells
            row = {
                "sample": samp,
                "cond": cond,
                "tissue": tissue,
                "celltype": ct,
                "frac": frac,
                "n": int(mask.sum()),
            }
            if mask.sum() >= 10:
                means = sub[mask].mean(0)
                for gname in _TEST + _PANEL:
                    if gname in gidx:
                        row[gname] = float(means[gidx[gname]])
                spec_accum[ct].append(sub[mask].mean(0))
            pb_rows.append(row)
        print(f"  {samp[:42]:<44} {cond:<8} {n_cells} nuclei")

    pb = pd.DataFrame(pb_rows)

    lines = [
        "Single-nucleus per-cell validation (GSE219280) — experiment #1",
        "=" * 66,
        f"Samples: {len(files)} (C9ALS vs Control; frontal + motor cortex, "
        "6 subjects x 2 regions per group); cellBender filtered.",
        "Per-nucleus marker-score cell-type assignment (FCN3/SERTAD1 excluded "
        "from markers).",
        "",
        "(a) COMPOSITION: cell-type fraction, C9ALS vs Control (MWU on per-sample "
        "fractions)",
        "-" * 66,
        f"{'cell type':<16}{'C9ALS frac':>12}{'Ctrl frac':>12}{'p (MWU)':>12}",
    ]
    for ct in _MARKERS:
        a = pb[(pb.celltype == ct) & (pb.cond == "C9ALS")]["frac"].values
        c = pb[(pb.celltype == ct) & (pb.cond == "Control")]["frac"].values
        try:
            _, p = mannwhitneyu(a, c, alternative="two-sided")
        except ValueError:
            p = np.nan
        lines.append(f"{ct:<16}{np.mean(a):>12.3f}{np.mean(c):>12.3f}{p:>12.3g}")

    lines += [
        "",
        "(b) MARKER-GENE CELL-TYPE SPECIFICITY (mean log-norm across all nuclei "
        "of each type)",
        "-" * 66,
        f"{'gene':<10}" + "".join(f"{ct[:5]:>9}" for ct in _MARKERS),
    ]
    spec = {
        ct: (np.mean(spec_accum[ct], 0) if spec_accum[ct] else np.zeros(len(genes_ref)))
        for ct in _MARKERS
    }
    gidx_ref = {g0: j for j, g0 in enumerate(genes_ref)}
    for gname in [
        "SLC37A2",
        "HEXB",
        "PROS1",
        "EMP1",
        "CLDN5",
        "VWF",
        "ANGPT2",
        "FCN3",
        "SERTAD1",
    ]:
        if gname in gidx_ref:
            j = gidx_ref[gname]
            lines.append(
                f"{gname:<10}" + "".join(f"{spec[ct][j]:>9.2f}" for ct in _MARKERS)
            )

    lines += [
        "",
        "(c) FCN3 & SERTAD1 -- per-cell-type expression + C9ALS vs Control "
        "(pseudobulk MWU)",
        "-" * 66,
    ]
    for gname in _TEST:
        lines.append(f"{gname}:")
        for ct in _MARKERS:
            sub_pb = (
                pb[(pb.celltype == ct) & pb[gname].notna()] if gname in pb else None
            )
            if sub_pb is None or len(sub_pb) == 0:
                continue
            a = sub_pb[sub_pb.cond == "C9ALS"][gname].values
            c = sub_pb[sub_pb.cond == "Control"][gname].values
            if len(a) < 3 or len(c) < 3:
                continue
            try:
                _, p = mannwhitneyu(a, c, alternative="two-sided")
            except ValueError:
                p = np.nan
            direction = "UP in ALS" if np.mean(a) > np.mean(c) else "down in ALS"
            lines.append(
                f"  {ct:<15} ALS={np.mean(a):.3f} Ctrl={np.mean(c):.3f} "
                f"p={p:.3g}  ({direction})"
            )

    (SCRIPT_DIR / "snrna_percell_validation_statistics.txt").write_text(
        "\n".join(lines) + "\n"
    )
    print("\n" + "\n".join(lines))

    # figure: composition (left) + FCN3/SERTAD1 per-cell-type (right)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cts = list(_MARKERS)
    x = np.arange(len(cts))
    af = [pb[(pb.celltype == ct) & (pb.cond == "C9ALS")]["frac"].mean() for ct in cts]
    cf = [pb[(pb.celltype == ct) & (pb.cond == "Control")]["frac"].mean() for ct in cts]
    axes[0].bar(x - 0.2, af, 0.4, label="C9ALS", color="#d62728", alpha=0.8)
    axes[0].bar(x + 0.2, cf, 0.4, label="Control", color="#1f77b4", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([c[:5] for c in cts], fontsize=8)
    axes[0].set_ylabel("nucleus fraction")
    axes[0].set_title("(a) Cell-type composition (snRNA): C9ALS vs Control")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")

    for gi, gname in enumerate(_TEST):
        if gname not in pb:
            continue
        av = [
            pb[(pb.celltype == ct) & (pb.cond == "C9ALS")][gname].mean() for ct in cts
        ]
        cv = [
            pb[(pb.celltype == ct) & (pb.cond == "Control")][gname].mean() for ct in cts
        ]
        off = -0.2 if gi == 0 else 0.2
        axes[1].bar(x + off, av, 0.18, label=f"{gname} ALS", alpha=0.8)
        axes[1].bar(x + off + 0.18, cv, 0.18, label=f"{gname} Ctrl", alpha=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([c[:5] for c in cts], fontsize=8)
    axes[1].set_ylabel("mean log-norm expression")
    axes[1].set_title("(c) FCN3 / SERTAD1 per cell type")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(
        SCRIPT_DIR / "snrna_percell_validation.png", dpi=150, bbox_inches="tight"
    )
    print("\nSaved -> snrna_percell_validation.{txt,png}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        main()
