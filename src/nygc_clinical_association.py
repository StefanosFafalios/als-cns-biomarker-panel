"""Panel score and panel genes vs C9orf72 status, disease duration and onset site.

Post-hoc and hypothesis-generating (Step 83). Clinical labels come from the NYGC
ALS Consortium donor table (Zenodo 20025792, ST1), joined to GSE153960 by
``subject id`` == ``donor_id``.

Score: out-of-fold probability from the critical panel, with folds grouped by
donor (Step 78), so no sample is scored by a model that saw its donor. Only ALS
samples are analysed. GPL24676 profiles several regions per donor, and the
score depends on region, so every value is z-scored within tissue and then
averaged per donor. The donor is the unit of every test.

C9orf72 carrier: ExpansionHunter max repeat >= 30 *or* RP-PCR "Yes". A
non-carrier needs both negative. The two assays disagree for some donors
(counted in the report). ALS-FTD is C9-enriched and the panel separates ALS-FTD
from pure ALS (Step 15), so C9 is also tested within pure ALS.

Outputs ``nygc_clinical_association.csv`` and ``_statistics.txt``.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.model_selection import StratifiedGroupKFold

SCRIPT_DIR = Path(__file__).parent
ALS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ALS_DIR))

from utils import (  # noqa: E402
    PanelLGBMClassifier,
    adjust_colsample,
    critical_panel_features,
    critical_panel_label,
    load_dataset,
)

from utils import selected_lgbm_params  # noqa: E402
ST1 = ALS_DIR / "resources" / "NYGC_zenodo_20025792" / "NYGC_Flagship_ST_1_3_Metadata.xlsx"
PARAMS = selected_lgbm_params()
OUT_CSV = SCRIPT_DIR / "nygc_clinical_association.csv"
OUT_TXT = SCRIPT_DIR / "nygc_clinical_association_statistics.txt"
C9_REPEATS = 30


def _st1() -> pd.DataFrame:
    raw = pd.read_excel(ST1, sheet_name="ST1", header=None)
    h = raw.index[raw.iloc[:, 0].astype(str).str.strip().eq("dna_id")][0]
    df = raw.iloc[h + 1 :].copy()
    df.columns = raw.iloc[h].astype(str).str.strip()
    df = df.dropna(subset=["donor_id"]).set_index("donor_id")
    eh = pd.to_numeric(df["c9orf72_max_call"], errors="coerce")
    pcr = df["c9orf72_rp_pcr"].astype(str).str.strip().str.lower()
    df["c9_eh"] = eh >= C9_REPEATS
    df["c9_pcr_yes"] = pcr.eq("yes")
    df["c9"] = np.where(df.c9_eh | df.c9_pcr_yes, 1.0, np.where(eh.notna(), 0.0, np.nan))
    df["duration"] = pd.to_numeric(df["disease_duration_in_months"], errors="coerce")
    df["age_death"] = pd.to_numeric(df["age_at_death"], errors="coerce")
    onset = df["site_of_motor_onset"].astype(str).str.strip()
    df["bulbar"] = np.where(onset.eq("Bulbar"), 1.0, np.where(onset.eq("Limb"), 0.0, np.nan))
    return df


def _oof_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    params = adjust_colsample(PARAMS, X.shape[1])
    oof = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, groups):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = PanelLGBMClassifier(**params).fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def _donor_means(values: pd.DataFrame, tissue: pd.Series, donor: pd.Series) -> pd.DataFrame:
    """Z-score each column within tissue, then average per donor."""
    z = values.groupby(tissue.values).transform(lambda c: (c - c.mean()) / (c.std(ddof=0) or 1.0))
    return z.groupby(donor.values).mean()


def _partial_spearman(x: pd.Series, y: pd.Series, cov: pd.Series) -> tuple[float, float, int]:
    """Spearman of x and y after removing a covariate from both ranks."""
    d = pd.concat([x, y, cov], axis=1).dropna()
    if len(d) < 10:
        return np.nan, np.nan, len(d)
    r = d.rank()
    c = np.column_stack([np.ones(len(r)), r.iloc[:, 2]])
    res = [r.iloc[:, i] - c @ np.linalg.lstsq(c, r.iloc[:, i], rcond=None)[0] for i in (0, 1)]
    rho, p = spearmanr(res[0], res[1])
    return float(rho), float(p), len(d)


def _tests(values: pd.DataFrame, donors: pd.DataFrame) -> list[dict]:
    rows = []
    pure = donors.disease_group.eq("ALS")
    for name in values.columns:
        v = values[name]
        for label, mask in (("C9 (all ALS-spectrum)", donors.c9.notna()),
                            ("C9 (pure ALS)", donors.c9.notna() & pure)):
            a, b = v[mask & donors.c9.eq(1)].dropna(), v[mask & donors.c9.eq(0)].dropna()
            if len(a) >= 5 and len(b) >= 5:
                u, p = mannwhitneyu(a, b)
                rows.append(dict(feature=name, test=label, n1=len(a), n0=len(b),
                                 effect=2 * u / (len(a) * len(b)) - 1, p=p))
        a, b = v[donors.bulbar.eq(1)].dropna(), v[donors.bulbar.eq(0)].dropna()
        if len(a) >= 5 and len(b) >= 5:
            u, p = mannwhitneyu(a, b)
            rows.append(dict(feature=name, test="bulbar vs limb onset", n1=len(a), n0=len(b),
                             effect=2 * u / (len(a) * len(b)) - 1, p=p))
        d = pd.concat([v, donors.duration], axis=1).dropna()
        if len(d) >= 10:
            rho, p = spearmanr(d.iloc[:, 0], d.iloc[:, 1])
            rows.append(dict(feature=name, test="disease duration", n1=len(d), n0=0,
                             effect=rho, p=p))
            rho, p, n = _partial_spearman(v, donors.duration, donors.age_death)
            rows.append(dict(feature=name, test="disease duration | age at death",
                             n1=n, n0=0, effect=rho, p=p))
    return rows


def _bh(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    m = len(p)
    q = (p[order] * m / np.arange(1, m + 1))[::-1].cummin()[::-1].clip(upper=1)
    return q.reindex(p.index)


def main() -> None:
    feats = critical_panel_features()
    label = critical_panel_label()
    (ds,) = load_dataset("GSE153960", platform="GPL24676", resources_dir=ALS_DIR / "resources")
    assert list(ds.X.index) == list(ds.y.index)
    meta = ds.meta.loc[ds.y.index]
    y = ds.y.to_numpy().astype(int)
    donor, tissue = meta["subject id"], meta["tissue"]

    names = (SCRIPT_DIR / "lgbm_prefilter_names.txt").read_text().splitlines()
    col = {n: i for i, n in enumerate(names)}
    X_pre = np.load(SCRIPT_DIR / "lgbm_prefilter_X.npy", mmap_mode="r")
    assert X_pre.shape[0] == len(y)
    oof = _oof_scores(np.asarray(X_pre[:, [col[f] for f in feats]], dtype=np.float32), y,
                      donor.to_numpy())

    base = {c.split(".")[0]: c for c in ds.X.columns}
    expr = np.log1p(ds.X[[base[f.split(".")[0]] for f in feats]].astype(np.float32))
    symbols = pd.read_csv(SCRIPT_DIR / "panel_critical.csv").set_index("feature").symbol
    expr.columns = [symbols[f] for f in feats]

    als = y == 1
    values = pd.concat([pd.Series(oof, index=expr.index, name=f"{label} score"), expr], axis=1)
    donor_vals = _donor_means(values[als], tissue[als], donor[als])

    st1 = _st1()
    donors = st1.reindex(donor_vals.index)
    matched = donors.c9.notna().sum()
    rows = _tests(donor_vals, donors)
    df = pd.DataFrame(rows)
    # BH within each test family across the panel genes; the score row is the primary test.
    df["q_genes"] = np.nan
    for t, g in df[df.feature != f"{label} score"].groupby("test"):
        df.loc[g.index, "q_genes"] = _bh(g.p)
    df.to_csv(OUT_CSV, index=False)

    pcr_yes = donors.c9_pcr_yes.fillna(False).astype(bool)
    disc = pcr_yes & ~donors.c9_eh.fillna(False).astype(bool)
    lines = [
        "Panel vs NYGC clinical metadata (donor level, ALS samples, tissue-z then donor mean)",
        "=" * 78,
        f"ALS samples {int(als.sum())} from {donor_vals.shape[0]} donors; "
        f"matched to ST1 with C9 call: {int(matched)}",
        f"C9 carriers {int(donors.c9.eq(1).sum())} / non-carriers {int(donors.c9.eq(0).sum())} "
        f"(RP-PCR positive but EH < {C9_REPEATS}: {int(disc.sum())}); "
        f"disease duration known: {int(donors.duration.notna().sum())}; "
        f"bulbar {int(donors.bulbar.eq(1).sum())} / limb {int(donors.bulbar.eq(0).sum())}",
        "effect = rank-biserial (positive: higher in C9 carriers / bulbar) or Spearman rho",
        "",
        f"Primary: {label} score (donor-grouped out-of-fold probability)",
    ]
    for _, r in df[df.feature == f"{label} score"].iterrows():
        lines.append(f"  {r.test:<34} n={int(r.n1)}/{int(r.n0)}  effect={r.effect:+.3f}  p={r.p:.2e}")
    lines += ["", "Panel genes with BH q < 0.10 (within test family):"]
    hits = df[(df.feature != f"{label} score") & (df.q_genes < 0.10)].sort_values("q_genes")
    for _, r in hits.iterrows():
        lines.append(f"  {r.feature:<14} {r.test:<34} effect={r.effect:+.3f}  "
                     f"p={r.p:.2e}  q={r.q_genes:.3f}")
    if hits.empty:
        lines.append("  none")
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
