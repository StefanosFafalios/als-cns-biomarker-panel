# ruff: noqa: E402
"""Sporadic/non-C9 snRNA per-cell validation (GSE212630) — generalises #1.

Experiment #1 used C9orf72 ALS (GSE219280). This repeats the per-cell validation
in an independent, NON-C9 TDP-43-proteinopathy cohort: GSE212630 (prefrontal
cortex snRNA, 10x), comparing non-neurological Control (n=7) vs TDPhigh (n=8;
strongest regional pTDP-43 pathology). Same method as #1 (marker-based
per-nucleus assignment + pseudobulk), reusing the .h5 reader and gene sets.

Asks whether the #1 findings generalise beyond C9: (a) endothelial cell-type
assignments, (b) SERTAD1 per-cell up-regulation, (c) FCN3 absence from CNS
nuclei (plasma-leak), and the composition shift.

Outputs
-------
  snrna_validation_gse212630_statistics.txt
  snrna_validation_gse212630.png
"""

from __future__ import annotations

import glob
import re
import warnings
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
_H5_DIR = SCRIPT_DIR.parent / "resources" / "GSE212630" / "suppl"
MIN_COUNTS = 500
_DIS = "TDPhigh"  # disease group (vs Control)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.stats import mannwhitneyu
    from snrna_percell_validation import _MARKERS, _PANEL, _TEST, _read_h5_targets

    files = sorted(glob.glob(str(_H5_DIR / "*_filtered_feature_bc_matrix.h5")))
    if not files:
        print(f"No .h5 in {_H5_DIR}; run the download first.")
        return
    print(f"Reading {len(files)} GSE212630 snRNA samples ...")

    pb_rows = []
    spec_accum: dict[str, list[np.ndarray]] = {ct: [] for ct in _MARKERS}
    genes_ref: list[str] = []
    for path in files:
        fn = Path(path).name
        m = re.search(r"_(Control|TDPhigh)-", fn)
        if not m:
            continue
        cond = m.group(1)
        samp = fn.split("_filtered")[0]
        sub, totals, genes = _read_h5_targets(path)
        if not genes_ref:
            genes_ref = genes
        gidx = {g: j for j, g in enumerate(genes)}
        sub = sub[totals >= MIN_COUNTS]
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
            row = {
                "sample": samp,
                "cond": cond,
                "celltype": ct,
                "frac": mask.sum() / n_cells,
                "n": int(mask.sum()),
            }
            if mask.sum() >= 10:
                means = sub[mask].mean(0)
                for g in _TEST + _PANEL:
                    if g in gidx:
                        row[g] = float(means[gidx[g]])
                spec_accum[ct].append(sub[mask].mean(0))
            pb_rows.append(row)
        print(f"  {samp[:42]:<44} {cond:<9} {n_cells} nuclei")

    pb = pd.DataFrame(pb_rows)
    n_dis = pb[pb.cond == _DIS]["sample"].nunique()
    n_ctrl = pb[pb.cond == "Control"]["sample"].nunique()

    lines = [
        "Sporadic/non-C9 snRNA per-cell validation (GSE212630) — generalises #1",
        "=" * 68,
        f"Prefrontal cortex snRNA; Control (n={n_ctrl}) vs {_DIS} (n={n_dis}; "
        "strongest pTDP-43 pathology). Marker-based per-nucleus assignment.",
        "",
        f"(a) COMPOSITION: cell-type fraction, {_DIS} vs Control (MWU)",
        "-" * 68,
        f"{'cell type':<16}{_DIS + ' frac':>13}{'Ctrl frac':>12}{'p (MWU)':>12}",
    ]
    for ct in _MARKERS:
        a = pb[(pb.celltype == ct) & (pb.cond == _DIS)]["frac"].values
        c = pb[(pb.celltype == ct) & (pb.cond == "Control")]["frac"].values
        try:
            _, p = mannwhitneyu(a, c, alternative="two-sided")
        except ValueError:
            p = np.nan
        lines.append(f"{ct:<16}{np.mean(a):>13.3f}{np.mean(c):>12.3f}{p:>12.3g}")

    lines += [
        "",
        "(b) MARKER-GENE CELL-TYPE SPECIFICITY (mean log-norm across all nuclei)",
        "-" * 68,
        f"{'gene':<10}" + "".join(f"{ct[:5]:>9}" for ct in _MARKERS),
    ]
    spec = {
        ct: (np.mean(spec_accum[ct], 0) if spec_accum[ct] else np.zeros(len(genes_ref)))
        for ct in _MARKERS
    }
    gref = {g: j for j, g in enumerate(genes_ref)}
    for g in [
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
        if g in gref:
            lines.append(
                f"{g:<10}" + "".join(f"{spec[ct][gref[g]]:>9.2f}" for ct in _MARKERS)
            )

    lines += [
        "",
        f"(c) FCN3 & SERTAD1 -- per-cell-type expression + {_DIS} vs Control (MWU)",
        "-" * 68,
    ]
    for g in _TEST:
        lines.append(f"{g}:")
        for ct in _MARKERS:
            if g not in pb.columns:
                continue
            sp = pb[(pb.celltype == ct) & pb[g].notna()]
            a = sp[sp.cond == _DIS][g].values
            c = sp[sp.cond == "Control"][g].values
            if len(a) < 3 or len(c) < 3:
                continue
            try:
                _, p = mannwhitneyu(a, c, alternative="two-sided")
            except ValueError:
                p = np.nan
            d = "UP in disease" if np.mean(a) > np.mean(c) else "down in disease"
            lines.append(
                f"  {ct:<15} {_DIS}={np.mean(a):.3f} Ctrl={np.mean(c):.3f} "
                f"p={p:.3g}  ({d})"
            )
    lines += [
        "",
        "Interpretation: generalisation of #1 (GSE219280 C9ALS) to a non-C9 "
        "TDP-43-proteinopathy cortex cohort. Compare endothelial specificity, "
        "SERTAD1 per-cell direction, and FCN3 absence to #1.",
    ]
    (SCRIPT_DIR / "snrna_validation_gse212630_statistics.txt").write_text(
        "\n".join(lines) + "\n"
    )
    print("\n" + "\n".join(lines))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cts = list(_MARKERS)
    x = np.arange(len(cts))
    af = [pb[(pb.celltype == ct) & (pb.cond == _DIS)]["frac"].mean() for ct in cts]
    cf = [pb[(pb.celltype == ct) & (pb.cond == "Control")]["frac"].mean() for ct in cts]
    axes[0].bar(x - 0.2, af, 0.4, label=_DIS, color="#d62728", alpha=0.8)
    axes[0].bar(x + 0.2, cf, 0.4, label="Control", color="#1f77b4", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([c[:5] for c in cts], fontsize=8)
    axes[0].set_ylabel("nucleus fraction")
    axes[0].set_title(f"(a) Composition (GSE212630): {_DIS} vs Control")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")
    for gi, g in enumerate(_TEST):
        if g not in pb.columns:
            continue
        av = [pb[(pb.celltype == ct) & (pb.cond == _DIS)][g].mean() for ct in cts]
        cv = [pb[(pb.celltype == ct) & (pb.cond == "Control")][g].mean() for ct in cts]
        off = -0.2 if gi == 0 else 0.2
        axes[1].bar(x + off, av, 0.18, label=f"{g} {_DIS}", alpha=0.85)
        axes[1].bar(x + off + 0.18, cv, 0.18, label=f"{g} Ctrl", alpha=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([c[:5] for c in cts], fontsize=8)
    axes[1].set_ylabel("mean log-norm expression")
    axes[1].set_title("(c) FCN3 / SERTAD1 per cell type")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(
        SCRIPT_DIR / "snrna_validation_gse212630.png", dpi=150, bbox_inches="tight"
    )
    print("\nSaved -> snrna_validation_gse212630.{txt,png}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        main()
