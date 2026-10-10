"""Replicate the critical panel's per-cell changes in two cortex snRNA cohorts (Step 105g).

The spinal-cord per-cell test (snrna_spinal_cord.py, GSE287257) uses the donors of the
GSE287256 bulk cohort, which helped choose the critical panel. These two cortex cohorts
share no donors with any panel-selection step, so they test whether the same genes
change within the same cell types in independent donors and another CNS region:

  GSE219280  C9orf72 ALS vs control, frontal + motor cortex (6 vs 6 donors, 2 regions each)
  GSE212630  TDP-43-high vs control, prefrontal cortex (8 vs 7 donors)

Analysis fixed before the run:
  * Cell typing exactly as snrna_spinal_cord.py (same marker sets; the endothelial set
    excludes the panel genes CLDN5 and VWF), per-nucleus argmax of marker z-scores.
  * Each critical-panel gene is tested in its spinal-cord top cell type
    (snrna_spinal_cord_percell.csv), so the cell type is not chosen on these data.
  * Donor pseudobulk: mean log-normalised expression per (sample, cell type), then the
    mean over a donor's regions; types with < 10 nuclei in a sample are skipped.
  * Two-sided Mann-Whitney U on donors, BH across the panel genes within each cohort.

Outputs: snrna_cortex_percell.csv, snrna_cortex_percell_statistics.txt
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

from scipy.stats import mannwhitneyu  # noqa: E402
from snrna_percell_validation import MIN_COUNTS, _read_h5_targets  # noqa: E402
from snrna_spinal_cord import _MARKERS, quasi_poisson  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402
from utils import critical_panel_symbols  # noqa: E402

_RES = SCRIPT_DIR.parent / "resources"
# dataset -> (h5 glob, filename regex giving (condition, donor), disease label)
_COHORTS = {
    "GSE219280": (
        _RES / "GSE219280" / "suppl",
        "*cellBender_corrected_filtered.h5",
        r"snRNA_(C9ALS|Control)_([A-Za-z0-9]+)_(?:Frontal|Motor)Cortex",
        "C9ALS",
    ),
    "GSE212630": (
        _RES / "GSE212630" / "suppl",
        "*_filtered_feature_bc_matrix.h5",
        r"_(Control|TDPhigh)-([^_]+)",
        "TDPhigh",
    ),
}
_CT = list(_MARKERS)


def _donor_pseudobulk(files: list[Path], pattern: str, genes: list[str]) -> pd.DataFrame:
    """One row per (donor, cell type): mean of the per-sample cell-type means."""
    rows = []
    for path in files:
        m = re.search(pattern, path.name)
        if not m:
            continue
        cond, donor = m.group(1), f"{m.group(1)}_{m.group(2)}"
        sub, totals, present = _read_h5_targets(str(path), sorted(set(sum(_MARKERS.values(), [])) | set(genes)))
        keep = totals >= MIN_COUNTS
        sub, totals = sub[keep], totals[keep]
        gi = {g: j for j, g in enumerate(present)}
        scores = np.full((sub.shape[0], len(_CT)), -np.inf, dtype=np.float32)
        for k, mk in enumerate(_MARKERS.values()):
            cols = [gi[g] for g in mk if g in gi]
            if cols:
                block = sub[:, cols]
                scores[:, k] = ((block - block.mean(0)) / (block.std(0) + 1e-8)).mean(1)
        assign = np.array(_CT)[scores.argmax(1)]
        for ct in _CT:
            mask = assign == ct
            if mask.sum() >= 10:
                raw = np.rint(np.expm1(sub[mask]) * (totals[mask, None] / 1e4))  # exact counts
                rows.append(dict(donor=donor, cond=cond, celltype=ct, lib=float(totals[mask].sum()),
                                 **{g: float(sub[mask, gi[g]].mean()) for g in genes if g in gi},
                                 **{f"{g}__n": float(raw[:, gi[g]].sum()) for g in genes if g in gi}))
        print(f"  {path.name[:50]:<52} {cond:<8} {sub.shape[0]} nuclei", flush=True)
    df = pd.DataFrame(rows)
    keys = ["donor", "cond", "celltype"]
    summed = [c for c in df.columns if c.endswith("__n") or c == "lib"]  # counts add over a donor's samples
    return df.groupby(keys, as_index=False).agg({c: ("sum" if c in summed else "mean") for c in df.columns if c not in keys})


def cortex_tests(genes: list[str], top: dict[str, str]) -> pd.DataFrame:
    """Each gene in cell type *top[gene]*: disease vs control donor pseudobulk per cortex cohort (MWU, BH)."""
    out = []
    for name, (folder, glob, pattern, dis) in _COHORTS.items():
        files = sorted(folder.glob(glob))
        if not files:
            raise SystemExit(f"No .h5 in {folder}; run reproduce/download_geo_data.sh first.")
        print(f"{name}: {len(files)} files")
        pb = _donor_pseudobulk(files, pattern, genes)
        tests = []
        for g in genes:
            s = pb[(pb.celltype == top[g])]
            a = s[s.cond == dis][g].dropna().to_numpy() if g in s else np.array([])
            c = s[s.cond == "Control"][g].dropna().to_numpy() if g in s else np.array([])
            p = float(mannwhitneyu(a, c, alternative="two-sided").pvalue) if len(a) > 2 and len(c) > 2 else np.nan
            lfc = float(np.log2((np.expm1(a.mean()) + 1e-3) / (np.expm1(c.mean()) + 1e-3))) if len(a) and len(c) else np.nan
            rec = dict(cohort=name, gene=g, celltype=top[g], n_dis=len(a), n_ctl=len(c),
                       dis=a.mean() if len(a) else np.nan, ctl=c.mean() if len(c) else np.nan,
                       log2fc=lfc, p=p)
            sc = s[s.cond.isin([dis, "Control"])]
            if f"{g}__n" in sc and len(sc) > 3 and (sc[f"{g}__n"] > 0).any():
                rec["glm_log2fc"], se, rec["glm_p"] = quasi_poisson(
                    sc[f"{g}__n"].to_numpy(float), sc.lib.to_numpy(float), (sc.cond == dis).to_numpy())
            tests.append(rec)
        ps = np.array([t["p"] for t in tests])
        q = np.full(len(ps), np.nan)
        ok = ~np.isnan(ps)
        q[ok] = multipletests(ps[ok], method="fdr_bh")[1]
        for t, qi in zip(tests, q):
            t["q"] = qi
        out += tests
    return pd.DataFrame(out)


def main() -> None:
    crit = critical_panel_symbols(protein_coding_only=True)
    top = pd.read_csv(SCRIPT_DIR / "snrna_spinal_cord_percell.csv").set_index("gene")["celltype"].to_dict()
    # The spinal table covers every measured 25-panel gene, so a critical gene missing from it is not in
    # the nuclei's gene reference (ENSG00000280893 has no symbol) and cannot be localised.
    unmeasured = [g for g in crit if g not in top]
    if unmeasured:
        print(f"not measured in the spinal nuclei, skipped: {', '.join(unmeasured)}")
    crit = [g for g in crit if g in top]
    df = cortex_tests(crit, top)
    df.to_csv(SCRIPT_DIR / "snrna_cortex_percell.csv", index=False)
    sc = pd.read_csv(SCRIPT_DIR / "snrna_spinal_cord_percell.csv").set_index("gene")
    lines = [
        "Per-cell replication of the critical panel in cortex snRNA (independent donors)",
        "=" * 80,
        "Cell type per gene = its spinal-cord top cell type (snrna_spinal_cord_percell.csv);",
        "donor pseudobulk; two-sided MWU; BH across the panel genes within each cohort.",
        "",
        f"{'gene':<9}{'cell type':<17}{'spinal log2FC':>14}"
        + "".join(f"{c + ' log2FC':>17}{'p':>9}{'q':>8}" for c in _COHORTS),
    ]
    for g in crit:
        row = f"{g:<9}{top[g]:<17}{sc.loc[g, 'log2fc']:>14.2f}"
        for c in _COHORTS:
            r = df[(df.cohort == c) & (df.gene == g)].iloc[0]
            row += f"{r.log2fc:>17.2f}{r.p:>9.3g}{r.q:>8.3g}"
        lines.append(row)
    lines += ["", "Donors per cohort (disease / control) in the tested cell types:"]
    for c in _COHORTS:
        s = df[df.cohort == c]
        lines.append(f"  {c}: {int(s.n_dis.max())} / {int(s.n_ctl.max())}")
    (SCRIPT_DIR / "snrna_cortex_percell_statistics.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
