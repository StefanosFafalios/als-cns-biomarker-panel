"""Can a critical-panel gene be swapped for another gene without losing cross-cohort performance,
and is the substitute a drug target in the restoring direction? (Step 105j)

Step 27 screened substitutes inside the 25-gene panel on the discovery cohort, where
thousands of genes carry the ALS contrast. The critical panel was chosen for its
performance across the nine external datasets, so substitutes are judged there:

  * candidates per panel gene: step 27's (gene_replacement_results.csv: genes with
    |r| >= 0.5 to the panel gene in GPL24676, plus the SHAP top-500), minus step 27a's
    confirmed artifacts, other panel genes, and genes missing from any of the nine
    datasets (every critical gene is measured in all nine);
  * score: the 8-gene panel with the gene swapped, pooled 10-fold CV over the nine
    merged datasets on the split the trimming never saw (pooled_cv_random_null.pooled_score);
  * non-inferior: score >= panel score - 0.01, the trimming's own margin;
  * for every non-inferior substitute: its own ALS direction (random-effects Hedges' g
    over nine donor-level units, multicohort_meta_analysis.csv) and its DGIdb agents in the
    restoring direction (activator/agonist/inducer if lower in ALS, inhibitor if higher),
    as for the panel genes (drug_targets.py).

Point estimates on one split, in-sample for the panel; the per-cell direction of the
substitutes is not tested here.

``python critical_replacement_pooled.py smoke`` scores three candidates per gene.
Outputs: critical_replacement_pooled.csv, critical_replacement_pooled_statistics.txt
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

from drug_targets import _query_dgidb  # noqa: E402
from pooled_cv_random_null import load_full, pooled_score  # noqa: E402
from utils import critical_panel_features, critical_panel_symbols  # noqa: E402
from utils import selected_lgbm_params  # noqa: E402

MARGIN = 0.01
_RAISE = ("activator", "agonist", "inducer")


def matched_agents(interactions: list[dict], need: str) -> tuple[list[str], list[str]]:
    """(restoring agents, approved restoring agents) from DGIdb interactions for direction *need*."""
    def ok(types: str) -> bool:
        t = types.lower()
        return any(k in t for k in _RAISE) if need == "raise" else "inhibitor" in t

    hits = [i for i in interactions if ok(i["interaction_types"])]
    names = list(dict.fromkeys(i["drug"] for i in hits))
    return names, list(dict.fromkeys(i["drug"] for i in hits if i["approved"]))


def main(smoke: bool = False) -> None:
    warnings.filterwarnings("ignore")
    params = selected_lgbm_params()
    crit = [f.split(".")[0] for f in critical_panel_features()]
    crit_sym = critical_panel_symbols()
    full, sym = load_full()
    base, fold = pooled_score(full, crit, params)
    print(f"critical panel W.mean {base:.4f}", flush=True)

    audit = pd.read_csv(SCRIPT_DIR / "shap500_artifact_audit.csv")
    artifacts = set(audit.loc[audit.classification == "confirmed_artifact", "ensg_base"])
    cand = pd.read_csv(SCRIPT_DIR / "gene_replacement_results.csv")
    cand["base"] = cand.candidate_ensg.str.split(".").str[0]
    everywhere = set.intersection(*(set(v[1]) for v in full.values()))

    rows = []
    for g_sym, g in zip(crit_sym, crit, strict=True):
        c = cand[cand.panel_symbol == g_sym].drop_duplicates("base")
        pool = [b for b in c.base if b in everywhere and b not in artifacts and b not in crit]
        r = dict(zip(c.base, c.pearson_r, strict=True))
        for b in pool[:3] if smoke else pool:
            panel = [b if x == g else x for x in crit]
            v, _ = pooled_score(full, panel, params, fold)
            rows.append({"panel_gene": g_sym, "substitute": sym.get(b, b), "ensg": b, "pearson_r": r[b],
                         "wmean": v, "delta": v - base, "non_inferior": v >= base - MARGIN,
                         "n_candidates": len(pool)})
        print(f"  {g_sym:<8} {len(pool)} candidates scored", flush=True)
    df = pd.DataFrame(rows)

    meta = pd.read_csv(SCRIPT_DIR / "multicohort_meta_analysis.csv").set_index("gene")
    ni = df[df.non_inferior].copy()
    dg = {}
    syms = sorted(set(ni.substitute))
    for i in range(0, len(syms), 50):
        dg.update(_query_dgidb(syms[i:i + 50]))
    for col in ("meta_g", "meta_q", "need", "agents", "approved"):
        ni[col] = None
    for idx, row in ni.iterrows():
        if row.substitute in meta.index:
            g_, q_ = meta.loc[row.substitute, ["g", "q"]]
            need = "raise" if g_ < 0 else "lower"
            agents, approved = matched_agents(dg.get(row.substitute, []), need)
            ni.loc[idx, ["meta_g", "meta_q", "need", "agents", "approved"]] = [g_, q_, need, "|".join(agents), "|".join(approved)]
    df = df.merge(ni[["ensg", "panel_gene", "meta_g", "meta_q", "need", "agents", "approved"]],
                  on=["ensg", "panel_gene"], how="left")
    if smoke:
        print(df.to_string())
        return
    df.to_csv(SCRIPT_DIR / "critical_replacement_pooled.csv", index=False)
    _report(df, base)


def _report(df: pd.DataFrame, base: float) -> None:
    lines = ["Replaceability of the critical-panel genes across the nine external datasets", "=" * 76,
             f"Panel W.mean {base:.4f} (pooled 10-fold CV, split unseen by the trimming). Non-inferior: within "
             f"{MARGIN} AUC.", "Restoring agents: DGIdb activator/agonist/inducer for a gene lower in ALS (bulk "
             "meta g < 0), inhibitor for one higher.", "",
             f"{'gene':<9}{'tested':>7}{'non-inf':>8}{'share':>7}  best substitute (W.mean)"]
    for g, s in df.groupby("panel_gene", sort=False):
        best = s.sort_values("wmean", ascending=False).iloc[0]
        n_ni = int(s.non_inferior.sum())
        lines.append(f"{g:<9}{len(s):>7}{n_ni:>8}{n_ni / len(s):>7.0%}  {best.substitute} ({best.wmean:.4f})")
    drug = df[df.non_inferior & df.agents.fillna("").astype(bool)].copy()
    drug["robust"] = drug.meta_q < 0.05
    lines += ["", "Non-inferior substitutes with a restoring agent (bulk direction q < 0.05 marked *):"]
    for g, s in drug.sort_values(["robust", "wmean"], ascending=False).groupby("panel_gene", sort=False):
        lines.append(f"  {g}:")
        lines += [f"    {'*' if r.robust else ' '} {r.substitute:<10} g {r.meta_g:+.2f} (q {r.meta_q:.2g}) {r.need:<5} "
                  f"W.mean {r.wmean:.4f}  approved: {r.approved or '-'}  all: {'|'.join(r.agents.split('|')[:4])}"
                  for r in s.itertuples()]
    (SCRIPT_DIR / "critical_replacement_pooled_statistics.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(smoke=len(sys.argv) > 1 and sys.argv[1] == "smoke")
