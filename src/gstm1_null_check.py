"""Is the GSTM1 signal disease biology or the common GSTM1 gene deletion? (Step 105g)

About half of people of European ancestry carry two copies of the GSTM1 deletion and
express no GSTM1. A difference in that genotype's frequency between ALS and control
donors would show up as "GSTM1 down in ALS" without any disease effect on the gene.

Analysis fixed before the run (GPL24676 and the donor-disjoint GPL16791 donors):
  * Donor GSTM1 level = median over the donor's samples of log1p(RSEM count).
  * Non-expressor (deletion homozygote proxy): donor level < 1.0, i.e. under ~2 reads;
    sensitivity thresholds 0.5 and 2.0.
  * (a) Non-expressor frequency, ALS vs control donors: Fisher's exact test.
  * (b) Among expressors only: per-sample log1p z-scored within tissue, averaged per
    donor, ALS vs control by two-sided Mann-Whitney U (rank-biserial effect).
  * (c) The same test over all donors, for reference.

Output: gstm1_null_check_statistics.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

SCRIPT_DIR = Path(__file__).parent
ALS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset  # noqa: E402

_GSTM1 = "ENSG00000134184"
_THRESHOLDS = (1.0, 0.5, 2.0)  # primary first


def _donor_table(platform: str, exclude_donors: set[str] | None = None) -> pd.DataFrame:
    """One row per donor: label, median log1p GSTM1, mean tissue-z GSTM1."""
    (ds,) = load_dataset("GSE153960", platform=platform, resources_dir=ALS_DIR / "resources")
    col = next(c for c in ds.X.columns if c.split(".")[0] == _GSTM1)
    meta = ds.meta.loc[ds.y.index]
    df = pd.DataFrame({
        "donor": meta["subject id"].astype(str).to_numpy(),
        "tissue": meta["tissue"].astype(str).to_numpy() if "tissue" in meta else "all",
        "y": ds.y.to_numpy().astype(int),
        "x": np.log1p(ds.X[col].to_numpy(dtype=float)),
    })
    if exclude_donors:
        df = df[~df.donor.isin(exclude_donors)]
    df["z"] = df.groupby("tissue").x.transform(lambda v: (v - v.mean()) / (v.std() or 1.0))
    return df.groupby("donor").agg(y=("y", "first"), level=("x", "median"), z=("z", "mean")).reset_index()


def _rb(a: np.ndarray, c: np.ndarray) -> tuple[float, float]:
    u, p = mannwhitneyu(a, c, alternative="two-sided")
    return 2 * u / (len(a) * len(c)) - 1, float(p)


def _report(name: str, d: pd.DataFrame) -> list[str]:
    als, ctl = d[d.y == 1], d[d.y == 0]
    rb_all, p_all = _rb(als.z.to_numpy(), ctl.z.to_numpy())
    lines = [f"{name}: {len(als)} ALS / {len(ctl)} control donors",
             f"  (c) all donors, ALS vs control: rank-biserial {rb_all:+.3f}  p={p_all:.3g}"]
    for t in _THRESHOLDS:
        null_a, null_c = int((als.level < t).sum()), int((ctl.level < t).sum())
        odds, p_f = fisher_exact([[null_a, len(als) - null_a], [null_c, len(ctl) - null_c]])
        ea, ec = als[als.level >= t], ctl[ctl.level >= t]
        rb, p = _rb(ea.z.to_numpy(), ec.z.to_numpy()) if len(ea) > 2 and len(ec) > 2 else (np.nan, np.nan)
        tag = "primary" if t == _THRESHOLDS[0] else "sensitivity"
        lines += [
            f"  threshold {t} ({tag}): non-expressors ALS {null_a}/{len(als)} ({null_a / len(als):.0%}), "
            f"control {null_c}/{len(ctl)} ({null_c / len(ctl):.0%}); Fisher OR {odds:.2f} p={p_f:.3g}",
            f"    (b) expressors only ({len(ea)} ALS / {len(ec)} control): rank-biserial {rb:+.3f}  p={p:.3g}",
        ]
    return lines


def main() -> None:
    d24 = _donor_table("GPL24676")
    d16 = _donor_table("GPL16791", exclude_donors=set(d24.donor))
    lines = [
        "GSTM1 deletion check: genotype-frequency imbalance vs disease effect",
        "=" * 70,
        "Donor level = median log1p(RSEM) over the donor's samples; non-expressor if below threshold.",
        "Effect = rank-biserial of tissue-z GSTM1 (donor mean), positive = higher in ALS.",
        "",
        *_report("GPL24676 (discovery)", d24),
        "",
        *_report("GPL16791 (donors absent from GPL24676)", d16),
    ]
    (SCRIPT_DIR / "gstm1_null_check_statistics.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
