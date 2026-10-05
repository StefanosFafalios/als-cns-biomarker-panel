# ruff: noqa: E402
"""Drug target annotation for the protein-coding members of the critical ALS panel.

Queries two databases for each protein-coding panel gene:

  DGIdb (Drug-Gene Interaction database)
    GraphQL endpoint: https://dgidb.org/api/graphql (DGIdb v5)
    Returns: drug names, interaction types (inhibitor/activator/etc.),
             FDA approval status, clinical phase.

  OpenTargets Platform (GraphQL)
    Endpoint: https://api.platform.opentargets.org/api/v4/graphql
    Returns: tractability tiers, known drugs in clinical trials,
             target-disease association scores.

Only protein-coding genes with a known HGNC symbol are queried.
Pseudogenes, ncRNAs, snoRNAs, and uncharacterised TEC entries
are excluded from drug target annotation.

Candidate ranking (user decision 2026-10-05): primary evidence is the per-cell
ALS vs control change in each gene's top cell type in spinal-cord single-nucleus
data (snrna_spinal_cord.py, step 41), which tests directly whether the cells
themselves change. The bulk deconvolution label (cell_type_deconvolution.py) is
secondary: it only infers that from bulk tissue, and on this panel the two disagree.

Outputs
-------
  drug_targets_summary.csv     — full drug-gene interaction table
  drug_targets_statistics.txt  — human-readable priority summary
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

ALS_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ALS_DIR))

from utils import critical_panel_directions, load_critical_panel

SCRIPT_DIR = Path(__file__).parent

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PROFILING_CSV = SCRIPT_DIR / "gene_profiling_summary.csv"

_DGIDB_URL = "https://dgidb.org/api/graphql"
_OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"

_REQUEST_DELAY = 1.0  # seconds between API calls

# ---------------------------------------------------------------------------
# Panel gene metadata: the protein-coding members of the critical panel
# ---------------------------------------------------------------------------


def _panel_genes() -> list[dict[str, str]]:
    """Return the protein-coding critical-panel members with their panel metrics.

    Generated rather than hand-kept. The hand-kept table listed the 25-gene panel
    and drifted from the critical panel (PROS1/SERTAD1 dropped, EMP1/MECOM in).
    ``loo_delta`` and ``shap_rank`` are the 25-gene panel's within-panel LOO and
    SHAP rank, as before. Ensembl IDs are base IDs (no version suffix).
    """
    core = pd.read_csv(_PANEL_CSV).set_index("feature")
    directions = critical_panel_directions(protein_coding_only=True)
    return [
        {
            "symbol": row.symbol,
            "ensg": row.feature.split(".")[0],
            "direction": f"{directions[row.symbol]}_in_ALS",
            "loo_delta": f"{core.loc[row.feature, 'loo_delta_auc']:+.4f}",
            "shap_rank": str(int(core.loc[row.feature, 'rank'])),
        }
        for row in load_critical_panel(protein_coding_only=True).itertuples()
    ]


_PROTEIN_CODING_GENES: list[dict[str, str]] = _panel_genes()


# ---------------------------------------------------------------------------
# Cell-type deconvolution annotation (from cell_type_deconvolution.py)
#
# dominant_ct  : cell type with highest Spearman ρ against gene expression
# rho          : that Spearman ρ (all 874 samples)
# ct_als_dir   : direction of dominant cell type in ALS vs Control
#                "up" / "down" (MWU significant, p < 0.05)
#                "ns" (not significant)
#
# sign_reversal: gene direction contradicts ct_als_dir
#   → e.g. gene is UP in ALS but its dominant cell type is DOWN in ALS
#   → cannot be explained by composition alone → per-cell dysregulation likely
# ---------------------------------------------------------------------------

# Per-gene composition label written by cell_type_deconvolution.py (step 11).
_DECONV: dict[str, dict] = (
    pd.read_csv(SCRIPT_DIR / "cell_type_deconvolution_genes.csv").set_index("symbol").to_dict("index")
)


# Per-cell change in each gene's top cell type, spinal-cord snRNA (step 41; 8 ALS /
# 4 control donors, the GSE287256 bulk donors). Read in main() so a missing file
# fails with the step order rather than at import.
_SNRNA_CSV = SCRIPT_DIR / "snrna_spinal_cord_percell.csv"
_SNRNA_ORDER = {"confirmed": 0, "suggestive": 1, "none": 2, "not_measured": 2, "opposite": 3}
_DECONV_ORDER = {"sign_reversal": 0, "ambiguous_ct": 1, "composition_driven": 2}


def _snrna_evidence(gene_dir: str, rec: dict | None) -> str:
    """Grade the single-nucleus per-cell test for one gene.

    Returns "confirmed" (q < 0.05, same direction as bulk), "suggestive" (p < 0.05,
    q >= 0.05, same direction), "opposite" (p < 0.05 against the bulk direction),
    "none" (p >= 0.05) or "not_measured".
    """
    if rec is None or pd.isna(rec["p"]):
        return "not_measured"
    if rec["p"] >= 0.05:
        return "none"
    if (rec["log2fc"] > 0) != (gene_dir == "up_in_ALS"):
        return "opposite"
    return "confirmed" if rec["q"] < 0.05 else "suggestive"


def _per_cell_evidence(gene_dir: str, deconv: dict) -> str:
    """Classify per-cell signal strength from deconvolution data.

    Returns one of:
      "sign_reversal"      — gene direction contradicts dominant cell-type ALS shift
      "ambiguous_ct"       — dominant cell type not significantly changed in ALS
      "composition_driven" — gene direction is fully consistent with cell-type shift
    """
    ct_dir = deconv["ct_als_dir"]
    if ct_dir == "ns":
        return "ambiguous_ct"
    gene_up = gene_dir == "up_in_ALS"
    ct_up = ct_dir == "up"
    if gene_up != ct_up:
        return "sign_reversal"
    return "composition_driven"


# ---------------------------------------------------------------------------
# DGIdb
# ---------------------------------------------------------------------------


_DGIDB_QUERY = """
query GeneInteractions($names: [String!]!) {
  genes(names: $names) {
    nodes {
      name
      interactions {
        drug { name approved }
        interactionTypes { type directionality }
        sources { sourceDbName }
      }
    }
  }
}
"""


def _query_dgidb(symbols: list[str]) -> dict[str, list[dict]]:
    """Query DGIdb v5 GraphQL for drug-gene interactions. Returns symbol → interactions."""
    payload = json.dumps(
        {"query": _DGIDB_QUERY, "variables": {"names": symbols}}
    ).encode()
    req = urllib.request.Request(
        _DGIDB_URL,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        print(f"  [WARN] DGIdb request failed: {exc}")
        return {s: [] for s in symbols}

    result: dict[str, list[dict]] = {s: [] for s in symbols}
    nodes = (data.get("data") or {}).get("genes", {}).get("nodes") or []
    for node in nodes:
        gene_sym = node.get("name", "")
        if gene_sym not in result:
            continue
        for intx in node.get("interactions") or []:
            drug_obj = intx.get("drug") or {}
            drug_name = drug_obj.get("name", "")
            approved = bool(drug_obj.get("approved", False))
            types = [t.get("type", "") for t in (intx.get("interactionTypes") or [])]
            sources = [s.get("sourceDbName", "") for s in (intx.get("sources") or [])]
            result[gene_sym].append(
                {
                    "drug": drug_name,
                    "interaction_types": "|".join(types) if types else "unspecified",
                    "approved": approved,
                    "sources": "|".join(sources),
                }
            )
    return result


# ---------------------------------------------------------------------------
# OpenTargets
# ---------------------------------------------------------------------------

_OT_QUERY = """
query TargetDrugs($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    tractability {
      label
      modality
      value
    }
    drugAndClinicalCandidates {
      count
      rows {
        maxClinicalStage
        drug { name }
      }
    }
  }
}
"""


def _query_opentargets(ensg: str) -> dict[str, Any]:
    """Query OpenTargets for one target. Returns parsed JSON or {}."""
    payload = json.dumps(
        {"query": _OT_QUERY, "variables": {"ensemblId": ensg}}
    ).encode()
    req = urllib.request.Request(
        _OT_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        print(f"  [WARN] OpenTargets request failed for {ensg}: {exc}")
        return {}


def _parse_tractability(tractability: list[dict]) -> str:
    """Summarise tractability tiers into short string."""
    active = []
    for item in tractability or []:
        if item.get("value"):
            mod = item.get("modality") or item.get("label") or "?"
            active.append(str(mod))
    return ",".join(sorted(set(active))) if active else "none"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Drug target annotation — critical ALS panel (protein-coding members)")
    print("=" * 60)

    symbols = [g["symbol"] for g in _PROTEIN_CODING_GENES]
    if not _SNRNA_CSV.exists():
        raise SystemExit(f"{_SNRNA_CSV.name} missing -- run snrna_spinal_cord.py (step 41) first")
    snrna = pd.read_csv(_SNRNA_CSV).set_index("gene").to_dict("index")
    # Independent-donor replication in cortex snRNA (snrna_cortex_percell.py, step 41a).
    cortex_csv = SCRIPT_DIR / "snrna_cortex_percell.csv"
    if not cortex_csv.exists():
        raise SystemExit(f"{cortex_csv.name} missing -- run snrna_cortex_percell.py (step 41a) first")
    cortex = pd.read_csv(cortex_csv)

    # ------------------------------------------------------------------
    # 1. DGIdb batch query
    # ------------------------------------------------------------------
    print(f"\n[1] Querying DGIdb for {len(symbols)} genes ...")
    dgidb_results = _query_dgidb(symbols)
    time.sleep(_REQUEST_DELAY)

    # ------------------------------------------------------------------
    # 2. OpenTargets per-gene query
    # ------------------------------------------------------------------
    print(f"\n[2] Querying OpenTargets for {len(symbols)} genes ...")
    ot_results: dict[str, dict] = {}
    for gene in _PROTEIN_CODING_GENES:
        sym, ensg = gene["symbol"], gene["ensg"]
        print(f"  {sym} ({ensg}) ...", end=" ", flush=True)
        resp = _query_opentargets(ensg)
        ot_results[sym] = resp.get("data", {}).get("target", {}) or {}
        n_drugs = ot_results[sym].get("drugAndClinicalCandidates", {}).get("count", 0)
        print(f"{n_drugs} drugs")
        time.sleep(_REQUEST_DELAY)

    # ------------------------------------------------------------------
    # 3. Compile table
    # ------------------------------------------------------------------
    print("\n[3] Compiling results ...")
    rows: list[dict] = []

    for gene in _PROTEIN_CODING_GENES:
        sym = gene["symbol"]
        ensg = gene["ensg"]
        direction = gene["direction"]
        loo = gene["loo_delta"]
        shap_rank = gene["shap_rank"]

        # DGIdb
        intx_list = dgidb_results.get(sym, [])
        approved_drugs = [i["drug"] for i in intx_list if i.get("approved")]
        clinical_drugs = [
            i["drug"]
            for i in intx_list
            if not i.get("approved")
            and any(
                src in i.get("sources", "")
                for src in ["CIViC", "ChEMBL", "ClinicalTrials", "FDA"]
            )
        ]
        inhibitors = [
            i["drug"]
            for i in intx_list
            if "inhibitor" in i.get("interaction_types", "").lower()
        ]
        activators = [
            i["drug"]
            for i in intx_list
            if any(
                t in i.get("interaction_types", "").lower()
                for t in ["activator", "agonist", "inducer"]
            )
        ]

        # OpenTargets
        ot = ot_results.get(sym, {})
        tractability = _parse_tractability(ot.get("tractability", []))
        known_drugs_ot = ot.get("drugAndClinicalCandidates", {})
        n_ot_drugs = known_drugs_ot.get("count", 0)
        _approval_stages = {"APPROVAL", "APPROVED", "4", "PHASE_4"}
        phase3_drugs = [
            r["drug"].get("name", "")
            for r in known_drugs_ot.get("rows", [])
            if r.get("maxClinicalStage", "") in _approval_stages
            or str(r.get("maxClinicalStage", "")).startswith("PHASE_3")
            or str(r.get("maxClinicalStage", "")).startswith("PHASE_4")
        ]

        # Deconvolution annotation
        dc = _DECONV.get(sym, {})
        dominant_ct = dc.get("dominant_ct", "unknown")
        deconv_rho = dc.get("rho", 0.0)
        ct_als_dir = dc.get("ct_als_dir", "ns")
        per_cell = _per_cell_evidence(direction, dc) if dc else "unknown"
        sn = snrna.get(sym)

        rows.append(
            {
                "shap_rank": shap_rank,
                "symbol": sym,
                "ensembl_id": ensg,
                "direction": direction,
                "loo_delta": loo,
                "tractability": tractability,
                "n_dgidb_interactions": len(intx_list),
                "approved_drugs_dgidb": "|".join(approved_drugs[:5])
                if approved_drugs
                else "",
                "inhibitors_dgidb": "|".join(inhibitors[:5]) if inhibitors else "",
                "activators_dgidb": "|".join(activators[:5]) if activators else "",
                "n_ot_drugs": n_ot_drugs,
                "phase3_drugs_ot": "|".join(phase3_drugs[:5]) if phase3_drugs else "",
                "has_approved_drug": bool(approved_drugs or phase3_drugs),
                "dominant_ct": dominant_ct,
                "deconv_rho": deconv_rho,
                "ct_als_dir": ct_als_dir,
                "per_cell_evidence": per_cell,
                "snrna_celltype": sn["celltype"] if sn else "",
                "snrna_log2fc": sn["log2fc"] if sn else float("nan"),
                "snrna_p": sn["p"] if sn else float("nan"),
                "snrna_q": sn["q"] if sn else float("nan"),
                "snrna_evidence": _snrna_evidence(direction, sn),
                **{
                    f"cortex_{r.cohort}": f"{r.log2fc:+.2f} (p {r.p:.2g})"
                    for r in cortex[cortex.gene == sym].itertuples()
                },
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("shap_rank")
    summary_df.to_csv(SCRIPT_DIR / "drug_targets_summary.csv", index=False)

    # ------------------------------------------------------------------
    # 4. Write statistics
    # ------------------------------------------------------------------
    # Candidate priority: single-nucleus per-cell evidence first, bulk deconvolution
    # second, SHAP rank last.
    ranked = summary_df.assign(
        _k1=summary_df.snrna_evidence.map(_SNRNA_ORDER).fillna(4),
        _k2=summary_df.per_cell_evidence.map(_DECONV_ORDER).fillna(3),
    ).sort_values(["_k1", "_k2", "shap_rank"])

    lines: list[str] = [
        "Drug Target Annotation — Critical ALS Panel (protein-coding members)",
        "=" * 70,
        f"Protein-coding genes queried: {len(symbols)}",
        "Sources: DGIdb v5 GraphQL + OpenTargets Platform GraphQL",
        "Per-cell evidence (primary): spinal-cord snRNA GSE287257, ALS vs control donor pseudobulk",
        "  in each gene's top cell type (snrna_spinal_cord.py). 8 ALS / 4 control donors (the GSE287256",
        "  bulk donors, a trimming cohort); the smallest attainable two-sided MWU p is 0.004.",
        "Bulk deconvolution (secondary): z-score module scoring (cell_type_deconvolution.py)",
        "Cortex replication (reported, not used for tiers): the same cell type in two cortex snRNA",
        "  cohorts with independent donors (snrna_cortex_percell.py): " + ", ".join(sorted(cortex.cohort.unique())),
        "",
        "CANDIDATE PRIORITY TABLE (snRNA per-cell evidence > bulk deconvolution > SHAP rank)",
        "-" * 100,
        f"{'Symbol':<9} {'Dir':<12} {'snRNA cell type':<16} {'log2FC':>7} {'q':>7}  {'per-cell':<13}"
        f"{'bulk deconv':<19} {'Tract.':<10} {'DGIdb n':>7}",
        "-" * 100,
    ]
    for _, r in ranked.iterrows():
        lines.append(
            f"{r['symbol']:<9} {r['direction']:<12} {r['snrna_celltype']:<16} {r['snrna_log2fc']:>7.2f} "
            f"{r['snrna_q']:>7.3g}  {r['snrna_evidence']:<13}{r['per_cell_evidence']:<19} "
            f"{r['tractability']:<10} {r['n_dgidb_interactions']:>7}"
        )

    lines += [
        "",
        "SHAP-RANK TABLE",
        "-" * 80,
        f"{'Rank':<5} {'Symbol':<10} {'Dir':<14} {'LOO Δ':<10} "
        f"{'Tract.':<12} {'DGIdb n':<9} {'Approved drugs':<35} {'OT phase3'}",
        "-" * 80,
    ]
    for _, r in summary_df.iterrows():
        approved = r["approved_drugs_dgidb"] or r["phase3_drugs_ot"] or "—"
        if len(approved) > 34:
            approved = approved[:31] + "..."
        lines.append(
            f"{r['shap_rank']:<5} {r['symbol']:<10} {r['direction']:<14} "
            f"{r['loo_delta']:<10} {r['tractability']:<12} "
            f"{r['n_dgidb_interactions']:<9} {approved:<35} "
            f"{r['phase3_drugs_ot'] or '—'}"
        )

    tiers = [
        ("TIER 1 — PER-CELL CHANGE CONFIRMED (snRNA q < 0.05, same direction as bulk)", ("confirmed",)),
        ("TIER 2 — PER-CELL CHANGE SUGGESTIVE (snRNA p < 0.05, q >= 0.05)", ("suggestive",)),
        ("TIER 3 — NO PER-CELL CHANGE DETECTED (or not measured, or opposite to bulk)",
         ("none", "not_measured", "opposite")),
    ]
    tier_genes: dict[str, list[str]] = {}
    for title, grades in tiers:
        lines += ["", title, "-" * 70]
        sub = ranked[ranked.snrna_evidence.isin(grades)]
        tier_genes[title] = sub.symbol.tolist()
        if sub.empty:
            lines.append("  None.")
        for _, r in sub.iterrows():
            approved = r["approved_drugs_dgidb"] or r["phase3_drugs_ot"]
            lines.append(
                f"  {r['symbol']} ({r['direction']}, SHAP rank {r['shap_rank']}): snRNA "
                f"{r['snrna_celltype'] or 'n/a'} log2FC {r['snrna_log2fc']:+.2f} (p {r['snrna_p']:.3g}, "
                f"q {r['snrna_q']:.3g}); bulk deconv {r['per_cell_evidence']} "
                f"({r['dominant_ct']}, ρ={r['deconv_rho']:.3f}, CT {r['ct_als_dir']} in ALS)"
            )
            lines.append(
                "    Cortex replication (log2FC, p): "
                + "; ".join(f"{c[7:]} {r[c]}" for c in summary_df.columns if c.startswith("cortex_"))
            )
            lines.append(f"    Tractability: {r['tractability']}  DGIdb interactions: {r['n_dgidb_interactions']}")
            if approved:
                lines.append(f"    Approved / phase>=3: {approved}")
            if r["inhibitors_dgidb"]:
                lines.append(f"    Inhibitors: {r['inhibitors_dgidb']}")
            if r["activators_dgidb"]:
                lines.append(f"    Activators: {r['activators_dgidb']}")

    stats_path = SCRIPT_DIR / "drug_targets_statistics.txt"
    stats_path.write_text("\n".join(lines) + "\n")

    actionable = summary_df[summary_df["has_approved_drug"]]
    tier1_genes = tier_genes[tiers[0][0]]
    print("\n  Summary CSV   → drug_targets_summary.csv")
    print("  Statistics    → drug_targets_statistics.txt")
    print(f"\n  Tier 1 (snRNA per-cell change confirmed): {tier1_genes}")
    print(f"  Actionable (approved/phase-3 drug): {actionable['symbol'].tolist()}")
    print("\nDone.")


if __name__ == "__main__":
    main()
