"""Do the robust core genes keep their ALS effect after donor covariates?

Core = genes re-selected in >= 6 of 8 leave-one-dataset-out trims (lodo_trim_statistics.txt).
For each core gene (prefilter matrix, z-scored), OLS ``gene ~ ALS + tissue [+ covariates]`` on
GPL24676 with donor-clustered SEs. The base and adjusted fits use the same complete-case rows,
so the change in the ALS coefficient is the covariates' share. Covariates come from the NYGC
donor table (ST1, as nygc_clinical_association.py): sex, age at death, cause-of-death group,
collection site, plus any post-mortem-interval or RIN column the table has.

Only aggregate coefficients are printed. Outputs ``core_gene_covariates.csv`` and
``core_gene_covariates_statistics.txt``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

SCRIPT_DIR = Path(__file__).parent
ALS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ALS_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from nygc_clinical_association import _st1  # noqa: E402
from utils import load_dataset  # noqa: E402

LODO_TXT = SCRIPT_DIR / "lodo_trim_statistics.txt"
MIN_LODO = 6
COVARS = {"sex_call": "C(sex)", "age_death": "age", "cause_of_death_group": "C(cod)",
          "site_sample_collected": "C(site)"}
OUT_CSV = SCRIPT_DIR / "core_gene_covariates.csv"
OUT_TXT = SCRIPT_DIR / "core_gene_covariates_statistics.txt"


def core_genes(text: str, min_hits: int = MIN_LODO) -> list[str]:
    """Symbols re-selected in at least ``min_hits`` LODO units, from lodo_trim_statistics.txt."""
    return [g for g, n in re.findall(r"^\s+(\S+)\s+(\d+)/\d+", text, re.M) if int(n) >= min_hits]


def als_effect(df: pd.DataFrame, gene: str, terms: list[str]) -> dict:
    """ALS coefficient of ``gene ~ als + C(tissue) + terms`` with donor-clustered SEs."""
    fit = smf.ols(" + ".join([f"{gene} ~ als", "C(tissue)", *terms]), df).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(df.donor)[0]})
    lo, hi = fit.conf_int().loc["als"]
    return dict(beta=fit.params["als"], lo=lo, hi=hi, p=fit.pvalues["als"])


def main() -> None:
    """Fit base and adjusted models per core gene and write the outputs."""
    (ds,) = load_dataset("GSE153960", platform="GPL24676", resources_dir=ALS_DIR / "resources")
    meta = ds.meta.loc[ds.y.index]
    names = (SCRIPT_DIR / "lgbm_prefilter_names.txt").read_text().splitlines()
    X = np.load(SCRIPT_DIR / "lgbm_prefilter_X.npy", mmap_mode="r")
    sym = pd.read_csv(SCRIPT_DIR / "lgbm_core25_panel.csv").set_index("symbol").feature
    genes = [g for g in core_genes(LODO_TXT.read_text()) if g in sym.index]

    st1 = _st1()
    extra = [c for c in st1.columns if re.search(r"\bpmi\b|post.?mortem|\brin\b", c, re.I)]
    covars = {**COVARS, **{c: re.sub(r"\W", "_", c).lower() for c in extra}}
    df = pd.DataFrame({"als": ds.y.to_numpy().astype(int), "tissue": meta.tissue.to_numpy(),
                       "donor": meta["subject id"].to_numpy()})
    for col, term in covars.items():
        name = re.sub(r"C\((\w+)\)", r"\1", term)
        vals = st1[col].reindex(df.donor).to_numpy()
        df[name] = pd.to_numeric(vals, errors="coerce") if term == name else vals
    for g in genes:
        x = np.asarray(X[:, names.index(sym[g])], dtype=float)
        df[g.replace("-", "_")] = (x - x.mean()) / x.std()
    keep = df.dropna(subset=[re.sub(r"C\((\w+)\)", r"\1", t) for t in covars.values()])

    rows = []
    for g in genes:
        col = g.replace("-", "_")
        base, adj = als_effect(keep, col, []), als_effect(keep, col, list(covars.values()))
        # one covariate at a time, and all but cause of death (a possible mediator: ALS deaths are respiratory)
        single = {t: als_effect(keep, col, [t]) for t in covars.values()}
        single["all but C(cod)"] = als_effect(keep, col, [t for t in covars.values() if t != "C(cod)"])
        rows.append(dict(gene=g, n=len(keep), **{f"base_{k}": v for k, v in base.items()},
                         **{f"adj_{k}": v for k, v in adj.items()},
                         retained=adj["beta"] / base["beta"],
                         **{f"{t}_{k}": v for t, e in single.items() for k, v in e.items()}))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    lines = ["Core genes vs donor covariates (GPL24676, OLS, donor-clustered SE, gene z-scored)",
             "=" * 78,
             f"Core: LODO re-selected >= {MIN_LODO}/8: {', '.join(genes)}",
             f"Adjusted for tissue + {', '.join(covars.values())}"
             f"{'' if extra else '  (no PMI/RIN column in ST1)'}",
             f"Complete cases: {len(keep)} samples, {keep.donor.nunique()} donors "
             f"(of {len(df)} / {df.donor.nunique()})", "",
             f"{'gene':<9} {'ALS beta base':>22} {'p':>9} {'ALS beta adjusted':>24} {'p':>9} {'kept':>6}"]
    for r in out.itertuples():
        lines.append(f"{r.gene:<9} {r.base_beta:+.2f} [{r.base_lo:+.2f}, {r.base_hi:+.2f}] {r.base_p:9.2e} "
                     f"{r.adj_beta:+.2f} [{r.adj_lo:+.2f}, {r.adj_hi:+.2f}] {r.adj_p:9.2e} {r.retained:6.0%}")
    terms = [*covars.values(), "all but C(cod)"]
    lines += ["", "ALS beta (p) with tissue + one covariate, and with all but cause of death:",
              f"{'gene':<9}" + "".join(f"{t:>20}" for t in terms)]
    for _, r in out.iterrows():
        lines.append(f"{r.gene:<9}" + "".join(f"{r[f'{t}_beta']:+8.2f} ({r[f'{t}_p']:8.1e})" for t in terms))
    text = "\n".join(lines)
    OUT_TXT.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
