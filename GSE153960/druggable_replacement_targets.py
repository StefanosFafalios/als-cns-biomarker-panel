# ruff: noqa: E402
"""Druggable-replacement target prioritisation.

For each high-signal protein-coding panel gene, scan its performance-preserving
(non-inferior) substitutes from the genome-wide replacement screen
(gene_replacement_results.csv) and rank them by a four-criterion scheme:

  1. Performance preserved   — is_replacement (delta_vs_baseline within 0.002);
                               substituted 5-fold CV AUC.
  2. Druggable               — DGIdb drug count + interaction types + approval.
  3. Favourable modulation   — drug acts opposite to the gene's ALS direction
                               (inhibitor/antagonist of an up-in-ALS gene;
                               agonist/activator of a down-in-ALS gene).
  4. Safety / maturity       — presence of an FDA-approved drug (regulatory
                               safety review passed) as a reproducible
                               first-order signal; the lead drugs' documented
                               black-box / withdrawal status is annotated in the
                               manuscript text.

Idea (user): for the high-replaceability mechanism-level genes (many co-expressed
substitutes), this picks the most druggable + expression-modulating + safe module
member to target in place of the panel gene; low-replaceability genes have no
substitute, so the gene itself must be drugged.

Reuses the DGIdb query from drug_targets.py.

Outputs
-------
  druggable_replacement_targets.csv
  druggable_replacement_targets.txt
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

ALS_DIR = Path(__file__).parents[1]
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(ALS_DIR))

from drug_targets import _query_dgidb

_REPL_CSV = SCRIPT_DIR / "gene_replacement_results.csv"
_PROFILING_CSV = SCRIPT_DIR / "gene_profiling_summary.csv"

# 12 protein-coding members of the 15-gene critical panel (drug-relevant set)
_HIGH_SIGNAL = [
    "FCN3",
    "PROS1",
    "ANGPT2",
    "TINAGL1",
    "CKMT2",
    "CLDN5",
    "NR4A1",
    "SOHLH2",
    "HEXB",
    "MCEE",
    "SLC37A2",
    "SERTAD1",
]
_TOP_K = 30  # top co-expressed substitutes per gene to screen for druggability

# Quantification-artifact gene families (regex), aligned with
# cross_cohort_substitution._ARTIFACT_FAMILY_PATTERNS and explicitly including the
# keratin-associated (KRTAP*) family flagged during this analysis (e.g. KRTAP6-2).
_ARTIFACT_RE = [
    re.compile(p)
    for p in (
        r"^OR\d",  # olfactory receptors (avoid ORAI/ORC/ORM)
        r"^TAS",  # taste receptors (TAS1R/TAS2R)
        r"^PATE",  # prostate/testis-expressed
        r"^DEFA",
        r"^DEFB",  # defensins
        r"^KRTAP",  # keratin-associated proteins (e.g. KRTAP6-2)
        r"^KRT\d",  # keratins KRT1..KRT86 (not KRTCAP/KRTDAP)
        r"^LCE",  # late-cornified envelope
        r"^SPRR",  # small proline-rich
        r"^VN1R",  # vomeronasal receptors
    )
]

# drugs withdrawn for safety / never marketed -- excluded from the safety-positive count
_WITHDRAWN = {
    "triparanol",
    "cerivastatin",
    "rofecoxib",
    "valdecoxib",
    "cisapride",
    "terfenadine",
    "thalidomide",
    "phenformin",
    "troglitazone",
}

_INHIBIT = {
    "inhibitor",
    "antagonist",
    "blocker",
    "negative modulator",
    "suppressor",
    "inverse agonist",
    "antibody",
    "cleavage",
}
_ACTIVATE = {
    "agonist",
    "activator",
    "positive modulator",
    "inducer",
    "stimulator",
    "partial agonist",
}


def _is_artifact(sym: str) -> bool:
    return any(p.match(sym) for p in _ARTIFACT_RE)


def main() -> None:
    import pandas as pd

    repl = pd.read_csv(_REPL_CSV)
    prof = pd.read_csv(_PROFILING_CSV)
    dir_of = {
        r["symbol"]: ("up" if str(r["direction"]).startswith("up") else "down")
        for _, r in prof.iterrows()
    }

    # candidate substitutes per high-signal gene: non-inferior AND genuinely
    # co-expressed (correlation source, |r| >= 0.5) -- i.e. real module members,
    # not arbitrary AUC-preserving genes from the SHAP pool.
    cand = repl[
        (repl["panel_symbol"].isin(_HIGH_SIGNAL))
        & (repl["is_replacement"])
        & (repl["source"] == "correlation")
        & (repl["pearson_r"] >= 0.5)
    ].copy()
    print(
        f"Non-inferior co-expressed substitute rows for high-signal genes: {len(cand)}"
    )

    # resolve candidate ENSG -> symbol + biotype
    import mygene

    bases = sorted({str(e).split(".")[0] for e in cand["candidate_ensg"]})
    mg = mygene.MyGeneInfo()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        hits = mg.querymany(
            bases,
            scopes="ensembl.gene",
            fields="symbol,type_of_gene",
            species="human",
            verbose=False,
        )
    sym_map, biotype = {}, {}
    for h in hits:
        q = h.get("query")
        sym_map[q] = h.get("symbol", q)
        biotype[q] = h.get("type_of_gene", "")
    cand["cand_sym"] = cand["candidate_ensg"].map(
        lambda e: sym_map.get(str(e).split(".")[0], "")
    )
    cand["cand_biotype"] = cand["candidate_ensg"].map(
        lambda e: biotype.get(str(e).split(".")[0], "")
    )

    # keep protein-coding, non-artifact, named
    cand = cand[(cand["cand_biotype"] == "protein-coding") & (cand["cand_sym"] != "")]
    cand = cand[~cand["cand_sym"].map(_is_artifact)]
    # top-K per gene by co-expression strength
    cand = (
        cand.sort_values("pearson_r", ascending=False)
        .groupby("panel_symbol")
        .head(_TOP_K)
        .reset_index(drop=True)
    )
    cand_syms = sorted(cand["cand_sym"].unique())
    print(f"Protein-coding non-artifact candidates to evaluate: {len(cand_syms)}")

    # DGIdb for all candidate symbols
    dg = _query_dgidb(cand_syms)

    def favourable(types: list[str], gene_dir: str) -> bool:
        t = {x.lower() for x in types}
        if gene_dir == "up":
            return bool(t & _INHIBIT)
        return bool(t & _ACTIVATE)

    # score candidates and pick best per gene
    rows = []
    for _, r in cand.iterrows():
        sym = r["cand_sym"]
        gdir = dir_of.get(r["panel_symbol"], "up")
        intx = dg.get(sym, [])
        n_drug = len(intx)
        approved = sorted({i["drug"] for i in intx if i.get("approved") and i["drug"]})
        withdrawn = sorted(d for d in approved if d.lower() in _WITHDRAWN)
        approved_safe = [d for d in approved if d.lower() not in _WITHDRAWN]
        types = [t for i in intx for t in i["interaction_types"].split("|")]
        fav = favourable(types, gdir)
        # favourable approved drugs (right direction + approved)
        fav_drugs = sorted(
            {
                i["drug"]
                for i in intx
                if i.get("approved")
                and i["drug"]
                and favourable(i["interaction_types"].split("|"), gdir)
            }
        )
        score = (
            2
            * int(bool(approved_safe))  # approved, non-withdrawn drug (safety-reviewed)
            + 2 * int(fav)  # favourable direction available
            + int(bool(fav_drugs))  # favourable AND approved
            + min(n_drug, 5) / 5.0  # breadth of chemistry
            + float(r["replacement_auc"])  # performance preserved
            - int(bool(withdrawn))  # penalise withdrawn-only chemistry
        )
        rows.append(
            {
                "panel_gene": r["panel_symbol"],
                "panel_dir": gdir,
                "substitute": sym,
                "pearson_r": round(float(r["pearson_r"]), 2),
                "subst_auc": round(float(r["replacement_auc"]), 4),
                "dgidb_n": n_drug,
                "approved_drugs": "|".join(approved_safe[:4]),
                "withdrawn": "|".join(withdrawn),
                "favourable_dir": fav,
                "favourable_approved": "|".join(fav_drugs[:3]),
                "score": round(score, 3),
            }
        )

    df = pd.DataFrame(rows).sort_values(
        ["panel_gene", "score"], ascending=[True, False]
    )
    df.to_csv(SCRIPT_DIR / "druggable_replacement_targets.csv", index=False)

    # best per gene -- only among druggable substitutes (DGIdb interactions > 0)
    drugg = df[df["dgidb_n"] > 0]
    best = drugg.sort_values("score", ascending=False).groupby("panel_gene").head(1)
    best = best.set_index("panel_gene").reindex(_HIGH_SIGNAL).reset_index()

    lines = [
        "Druggable-replacement target prioritisation",
        "=" * 70,
        "Best druggable + direction-appropriate + safety-reviewed substitute per",
        "high-signal gene (non-inferior substitutes only; DGIdb + approval).",
        "Score = approved-drug + favourable-direction + favourable&approved +",
        "        chemistry-breadth + substituted-AUC.",
        "",
        f"{'gene':9}{'dir':5}{'best substitute':16}{'r':>5}{'AUC':>8}{'DGIdb':>6}"
        f"{'fav':>5}  approved drugs",
        "-" * 90,
    ]
    for _, b in best.iterrows():
        if pd.isna(b.get("substitute")):
            lines.append(
                f"{b['panel_gene']:9}{'':5}(no druggable co-expressed substitute "
                f"-- target the gene itself)"
            )
            continue
        lines.append(
            f"{b['panel_gene']:9}{b['panel_dir']:5}{b['substitute']:16}{b['pearson_r']:>5}"
            f"{b['subst_auc']:>8}{b['dgidb_n']:>6}{('yes' if b['favourable_dir'] else 'no'):>5}  "
            f"{b['approved_drugs'] or '(none approved)'}"
        )
    lines += [
        "",
        "Low-replaceability gene-level targets (CKMT2, NR4A1, TINAGL1, SERTAD1)",
        "have few/no non-inferior protein-coding substitutes -> drug the gene itself.",
    ]
    (SCRIPT_DIR / "druggable_replacement_targets.txt").write_text(
        "\n".join(lines) + "\n"
    )
    print("\n".join(lines))
    print("\nSaved -> druggable_replacement_targets.{csv,txt}")


if __name__ == "__main__":
    main()
