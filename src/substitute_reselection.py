"""Do the nested discovery folds pick a critical gene's substitutes in its place?

Only 2 of the 21 critical genes are re-selected in at least half of the nested folds (step 30b).
If co-expressed genes stand in for one another, the folds should pick a gene's non-inferior
substitutes (step 27j) more often than other candidates tested for it. Per critical gene, the
summed nested selection frequency of its non-inferior substitutes is compared with that of
random sets of the same size drawn from the same gene's own swap candidates, which are already
co-expressed or highly ranked genes, so the null is conservative.

Outputs ``substitute_reselection.csv`` and ``substitute_reselection_statistics.txt``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from utils import critical_panel_size  # noqa: E402

B = 2000
OUT_CSV = SCRIPT_DIR / "substitute_reselection.csv"
OUT_TXT = SCRIPT_DIR / "substitute_reselection_statistics.txt"


def family_test(freq: np.ndarray, non_inferior: np.ndarray, rng: np.random.Generator, b: int = B) -> tuple[float, float, float]:
    """(observed summed frequency, null mean, one-sided p) for one gene's substitutes.

    Args:
        freq: Nested selection frequency of each swap candidate of the gene.
        non_inferior: Mask of the candidates that were non-inferior substitutes.
        rng: Random generator.
        b: Number of random sets.
    """
    k = int(non_inferior.sum())
    real = float(freq[non_inferior].sum())
    if k == 0:
        return real, 0.0, 1.0
    null = np.array([freq[rng.choice(len(freq), k, replace=False)].sum() for _ in range(b)])
    return real, float(null.mean()), float((1 + (null >= real).sum()) / (1 + b))


def main() -> None:
    """Run the comparison for every critical gene and write the outputs."""
    rep = pd.read_csv(SCRIPT_DIR / "critical_replacement_pooled.csv")
    sel = pd.read_csv(SCRIPT_DIR / "nested_panel_cv_selection.csv")
    sel = sel[sel.panel_size == critical_panel_size()]
    f = dict(zip(sel.feature.str.split(".").str[0], sel.frequency))
    rep["freq"] = rep.ensg.map(f).fillna(0.0)
    rng = np.random.default_rng(42)
    rows = []
    for gene, x in rep.groupby("panel_gene", sort=False):
        ni = x.non_inferior.to_numpy(dtype=bool)
        real, mean, p = family_test(x.freq.to_numpy(), ni, rng)
        rows.append(dict(gene=gene, candidates=len(x), non_inferior=int(ni.sum()), picks=real, null_mean=mean, p=p))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    ni_all = rep.non_inferior.astype(bool)
    above, ties = int((df.picks > df.null_mean).sum()), int((df.picks == df.null_mean).sum())
    sign = binomtest(above, len(df) - ties, 0.5, alternative="greater").pvalue
    lines = ["Nested re-selection of the critical genes' substitutes (30b x 27j)", "=" * 66,
             f"Mean nested pick rate: non-inferior substitutes {rep.freq[ni_all].mean():.4f}, "
             f"other candidates {rep.freq[~ni_all].mean():.4f}",
             f"Genes whose substitutes are picked more than random same-size candidate sets: {above} of "
             f"{len(df)} ({ties} tied); sign test p = {sign:.3f}", "",
             df.to_string(index=False, float_format=lambda v: f"{v:.3f}")]
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
