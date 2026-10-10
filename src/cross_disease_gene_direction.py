"""Do the critical-panel genes move the same way in other neurodegenerative diseases as in ALS?

cross_disease_specificity.py shows the panel *detects* Huntington's disease; a
classifier can do that with genes moving in other directions, so detection alone
does not show a shared pathway. Here each gene's Hedges' g (disease against the
cohort's own controls) is set beside its ALS direction from the nine-unit
meta-analysis (multicohort_meta_analysis.csv).

Outputs: cross_disease_gene_direction.csv and _statistics.txt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

import cross_disease_specificity as cds  # noqa: E402
from utils import critical_panel_symbols  # noqa: E402

PREFIX = SCRIPT_DIR / "cross_disease_gene_direction"


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    """Bias-corrected standardised mean difference of ``x`` (cases) over ``y`` (controls)."""
    n1, n0 = len(x), len(y)
    sp = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n0 - 1) * y.var(ddof=1)) / (n1 + n0 - 2))
    if not sp:
        return float("nan")
    return float((1 - 3 / (4 * (n1 + n0) - 9)) * (x.mean() - y.mean()) / sp)


def main() -> None:
    """Write per-gene disease-vs-control effects for every cross-disease cohort."""
    panel = pd.read_csv(SCRIPT_DIR / "lgbm_core25_panel.csv")
    crit = set(critical_panel_symbols())
    panel = panel[panel.symbol.isin(crit)]
    meta = pd.read_csv(SCRIPT_DIR / "multicohort_meta_analysis.csv", index_col="gene")
    als_g = meta.g.reindex(panel.symbol)

    cohorts = {
        "GSE64810": ("Huntington's", cds._load_deseq_cohort(
            "GSE64810",
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE64nnn/GSE64810/suppl/"
            "GSE64810_mlhd_DESeq2_norm_counts_adjust.txt.gz",
            "GSE64810_mlhd_DESeq2_norm_counts_adjust.txt.gz", "H_", has_symbol_col=False)),
        "GSE68719": ("Parkinson's", cds._load_deseq_cohort(
            "GSE68719",
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE68nnn/GSE68719/suppl/"
            "GSE68719_mlpd_PCG_DESeq2_norm_counts.txt.gz",
            "GSE68719_mlpd_PCG_DESeq2_norm_counts.txt.gz", "P_", has_symbol_col=True)),
        "GSE125583": ("Alzheimer's", cds._load_gse125583()),
        "GSE13162": ("FTLD-TDP", cds._load_gse13162()),
    }

    rows = []
    for gse, (disease, (expr, y, _)) in cohorts.items():
        info = cds.COHORT_META[gse]
        tr, te, syms = cds._match_panel(
            panel.feature.tolist(), panel.symbol.tolist(), set(expr.columns), info["id_type"]
        )
        x = expr[te].to_numpy(dtype=float)
        if info["assay"] != "microarray (HG-U133A_2)":
            x = np.log1p(np.clip(x, 0, None))
        for j, s in enumerate(syms):
            a, c = x[y == 1, j], x[y == 0, j]
            rows.append(dict(cohort=gse, disease=disease, gene=s, n_cases=len(a), n_controls=len(c),
                             g=hedges_g(a, c), p=mannwhitneyu(a, c).pvalue, als_g=als_g.get(s)))
    df = pd.DataFrame(rows)
    df["same_as_als"] = np.sign(df.g) == np.sign(df.als_g)
    df.to_csv(f"{PREFIX}.csv", index=False)

    lines = ["Critical-panel genes: disease vs own controls (Hedges' g on log expression)",
             "als_g: nine-unit random-effects g (multicohort_meta_analysis.csv)", ""]
    for gse, d in df.groupby("cohort", sort=False):
        ok = d.dropna(subset=["als_g"])
        r = np.corrcoef(ok.g, ok.als_g)[0, 1] if len(ok) > 2 else float("nan")
        lines.append(f"{gse} {d.disease.iloc[0]} (n {d.n_cases.iloc[0]}/{d.n_controls.iloc[0]}): "
                     f"{int(ok.same_as_als.sum())}/{len(ok)} genes in the ALS direction, "
                     f"Pearson r with ALS g = {r:.3f}")
        for _, row in d.iterrows():
            lines.append(f"    {row.gene:16s} g {row.g:+.2f}  p {row.p:.2g}  ALS g {row.als_g:+.2f}")
    text = "\n".join(lines)
    Path(f"{PREFIX}_statistics.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
