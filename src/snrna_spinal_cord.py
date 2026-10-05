"""Single-nucleus localisation of the panel in ALS cervical spinal cord (Step 101).

GSE287257: CellRanger filtered matrices from 8 ALS and 4 control cervical spinal
cords, the donors of the GSE287256 bulk cohort, so this is not a second
validation cohort. It asks where the panel's genes are expressed in spinal cord
and whether their expression changes within a cell type:

  (a) composition: nucleus fraction per cell type, ALS vs control (MWU on donors);
  (b) localisation: mean log-normalised expression of each protein-coding panel
      gene per cell type, averaged over donors, and its top cell type;
  (c) per-cell change: ALS vs control donor pseudobulk in the gene's top cell
      type (MWU, BH across genes). With 8 vs 4 donors the smallest two-sided MWU
      p is 0.004, so (c) can only flag large, consistent shifts.

Cell types are assigned per nucleus by canonical marker score, as in
snrna_percell_validation.py, with pericyte and meningeal-fibroblast types added
for spinal cord. The endothelial markers exclude the panel genes CLDN5 and VWF,
so their localisation is not circular.

Outputs: snrna_spinal_cord_statistics.txt, snrna_spinal_cord.png,
snrna_spinal_cord_percell.csv (table (c); drug_targets.py ranks candidates by it)
"""

from __future__ import annotations

import re
import sys
import tarfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

from scipy.stats import mannwhitneyu  # noqa: E402
from snrna_percell_validation import MIN_COUNTS, _read_h5_targets  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402
from utils import core_panel_symbols  # noqa: E402

_SUPPL = SCRIPT_DIR.parent / "resources" / "GSE287257" / "suppl"
_MARKERS = {
    "neuron": ["RBFOX3", "SNAP25", "SYT1", "RBFOX1"],
    "astrocyte": ["AQP4", "GFAP", "SLC1A2", "GJA1", "ALDH1L1"],
    "microglia": ["CSF1R", "P2RY12", "CX3CR1", "C1QB", "AIF1"],
    "oligodendrocyte": ["PLP1", "MBP", "MOBP", "MOG", "CNP"],
    "OPC": ["PDGFRA", "OLIG1", "OLIG2", "PCDH15"],
    "endothelial": ["FLT1", "PECAM1", "ABCB1", "SLC2A1"],
    "pericyte": ["PDGFRB", "RGS5", "KCNJ8", "ABCC9"],
    "fibroblast": ["DCN", "COL1A2", "LUM", "COL1A1"],
}
# the protein-coding members of the 25-gene panel
_PANEL = sorted(core_panel_symbols(protein_coding_only=True))
_GENES = sorted(set(sum(_MARKERS.values(), [])) | set(_PANEL))
_CT = list(_MARKERS)


def _h5_files() -> list[Path]:
    files = sorted(_SUPPL.glob("*_filtered_feature_bc_matrix.h5"))
    if not files and (_SUPPL / "GSE287257_RAW.tar").exists():
        with tarfile.open(_SUPPL / "GSE287257_RAW.tar") as tf:
            tf.extractall(_SUPPL, filter="data")
        files = sorted(_SUPPL.glob("*_filtered_feature_bc_matrix.h5"))
    return files


def main() -> None:
    files = _h5_files()
    if not files:
        print(f"No .h5 in {_SUPPL}; run reproduce/download_geo_data.sh first.")
        return
    rows = []
    for path in files:
        m = re.search(r"GSM\d+_(\d+)_(ALS|control)_", path.name)
        donor, cond = m.group(1), m.group(2)
        sub, totals, genes = _read_h5_targets(str(path), _GENES)
        sub = sub[totals >= MIN_COUNTS]
        gi = {g: j for j, g in enumerate(genes)}
        scores = np.full((sub.shape[0], len(_CT)), -np.inf, dtype=np.float32)
        for k, mk in enumerate(_MARKERS.values()):
            cols = [gi[g] for g in mk if g in gi]
            if cols:
                block = sub[:, cols]
                scores[:, k] = ((block - block.mean(0)) / (block.std(0) + 1e-8)).mean(1)
        assign = np.array(_CT)[scores.argmax(1)]
        for ct in _CT:
            mask = assign == ct
            row = dict(donor=donor, cond=cond, celltype=ct, n=int(mask.sum()), frac=mask.mean())
            if mask.sum() >= 10:
                row.update({g: float(sub[mask, gi[g]].mean()) for g in _PANEL if g in gi})
            rows.append(row)
        print(f"  {path.name[:40]:<42} {cond:<8} {len(assign)} nuclei", flush=True)
    pb = pd.DataFrame(rows)

    def mwu(a: np.ndarray, c: np.ndarray) -> float:
        return float(mannwhitneyu(a, c, alternative="two-sided").pvalue) if len(a) > 2 and len(c) > 2 else np.nan

    n_als, n_ctl = pb[pb.cond == "ALS"].donor.nunique(), pb[pb.cond == "control"].donor.nunique()
    lines = [
        "Single-nucleus localisation of the panel in ALS cervical spinal cord (GSE287257)",
        "=" * 78,
        f"Donors: {n_als} ALS, {n_ctl} control (the GSE287256 bulk donors). Nuclei with >= {MIN_COUNTS}",
        "counts; CP10k + log1p; cell type by canonical marker score (argmax).",
        "",
        "(a) COMPOSITION: nucleus fraction, ALS vs control (MWU on donors)",
        f"{'cell type':<17}{'ALS':>8}{'control':>9}{'p':>9}",
    ]
    comp = {}
    for ct in _CT:
        a = pb[(pb.celltype == ct) & (pb.cond == "ALS")].frac.to_numpy()
        c = pb[(pb.celltype == ct) & (pb.cond == "control")].frac.to_numpy()
        comp[ct] = (a.mean(), c.mean())
        lines.append(f"{ct:<17}{a.mean():>8.3f}{c.mean():>9.3f}{mwu(a, c):>9.3g}")

    means = pb.groupby("celltype")[[g for g in _PANEL if g in pb]].mean().reindex(_CT)
    lines += ["", "(b) LOCALISATION: mean log-normalised expression per cell type (donor average)",
              f"{'gene':<9}" + "".join(f"{ct[:6]:>8}" for ct in _CT) + f"{'top':>17}{'share':>7}"]
    top = {}
    for g in means.columns:
        v = means[g].fillna(0)
        top[g] = v.idxmax()
        share = v.max() / v.sum() if v.sum() > 0 else np.nan
        lines.append(f"{g:<9}" + "".join(f"{x:>8.3f}" for x in v) + f"{top[g]:>17}{share:>7.2f}")

    lines += ["", "(c) PER-CELL CHANGE in the top cell type: ALS vs control donor pseudobulk (MWU, BH)",
              f"{'gene':<9}{'cell type':<17}{'ALS':>8}{'control':>9}{'log2FC':>8}{'p':>9}{'q':>8}"]
    test = []
    for g, ct in top.items():
        s = pb[(pb.celltype == ct) & pb[g].notna()]
        a, c = s[s.cond == "ALS"][g].to_numpy(), s[s.cond == "control"][g].to_numpy()
        test.append((g, ct, a.mean(), c.mean(), mwu(a, c)))
    ps = np.array([t[4] for t in test])
    ok = ~np.isnan(ps)
    qs = np.full(len(ps), np.nan)
    qs[ok] = multipletests(ps[ok], method="fdr_bh")[1]
    percell = []
    for (g, ct, a, c, p), q in zip(test, qs):
        lfc = np.log2((np.expm1(a) + 1e-3) / (np.expm1(c) + 1e-3))
        lines.append(f"{g:<9}{ct:<17}{a:>8.3f}{c:>9.3f}{lfc:>8.2f}{p:>9.3g}{q:>8.3g}")
        percell.append(dict(gene=g, celltype=ct, als=a, control=c, log2fc=lfc, p=p, q=q))
    # Read by drug_targets.py (step 07), which ranks drug candidates by this test.
    pd.DataFrame(percell).to_csv(SCRIPT_DIR / "snrna_spinal_cord_percell.csv", index=False)

    (SCRIPT_DIR / "snrna_spinal_cord_statistics.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    _plot(comp, means)


def _plot(comp: dict, means: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1, 1.5]})
    x = np.arange(len(_CT))
    ax.bar(x - 0.2, [comp[c][0] for c in _CT], 0.4, color="#d62728", alpha=0.8, label="ALS")
    ax.bar(x + 0.2, [comp[c][1] for c in _CT], 0.4, color="#1f77b4", alpha=0.8, label="control")
    ax.set_xticks(x, _CT, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("nucleus fraction")
    ax.set_title("(a) Cell-type composition, cervical spinal cord")
    ax.legend()
    rel = means.div(means.max(axis=0), axis=1).T  # genes x cell types, row-scaled
    im = bx.imshow(rel.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    bx.set_xticks(np.arange(len(_CT)), _CT, rotation=40, ha="right", fontsize=8)
    bx.set_yticks(np.arange(len(rel)), rel.index, fontsize=8)
    bx.set_title("(b) Panel-gene expression by cell type (scaled to each gene's maximum)")
    fig.colorbar(im, ax=bx, fraction=0.03)
    plt.tight_layout()
    fig.savefig(SCRIPT_DIR / "snrna_spinal_cord.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        main()
