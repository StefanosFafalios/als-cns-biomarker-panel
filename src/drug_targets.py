# ruff: noqa: E402
"""Drug target annotation for the core 25-gene ALS panel.

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

SCRIPT_DIR = Path(__file__).parent

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PROFILING_CSV = SCRIPT_DIR / "gene_profiling_summary.csv"

_DGIDB_URL = "https://dgidb.org/api/graphql"
_OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"

_REQUEST_DELAY = 1.0  # seconds between API calls

# ---------------------------------------------------------------------------
# Panel gene metadata (protein-coding only, from findings.md annotation)
# Ensembl IDs are base IDs (no version suffix)
# ---------------------------------------------------------------------------

_PROTEIN_CODING_GENES: list[dict[str, str]] = [
    {"symbol": "MECOM",   "ensg": "ENSG00000085276", "direction": "down_in_ALS",
     "loo_delta": "+0.0011", "shap_rank": "1"},
    {"symbol": "SERTAD1", "ensg": "ENSG00000197019", "direction": "up_in_ALS",
     "loo_delta": "+0.0038", "shap_rank": "4"},
    {"symbol": "FCN3",    "ensg": "ENSG00000142748", "direction": "up_in_ALS",
     "loo_delta": "+0.0020", "shap_rank": "6"},
    {"symbol": "PROS1",   "ensg": "ENSG00000184500", "direction": "up_in_ALS",
     "loo_delta": "+0.0004", "shap_rank": "7"},
    {"symbol": "ANGPT2",  "ensg": "ENSG00000091879", "direction": "down_in_ALS",
     "loo_delta": "+0.0012", "shap_rank": "8"},
    {"symbol": "EMP1",    "ensg": "ENSG00000134531", "direction": "up_in_ALS",
     "loo_delta": "+0.0020", "shap_rank": "10"},
    {"symbol": "TINAGL1", "ensg": "ENSG00000142910", "direction": "down_in_ALS",
     "loo_delta": "+0.0013", "shap_rank": "12"},
    {"symbol": "CKMT2",   "ensg": "ENSG00000131730", "direction": "down_in_ALS",
     "loo_delta": "+0.0010", "shap_rank": "13"},
    {"symbol": "VWF",     "ensg": "ENSG00000110799", "direction": "down_in_ALS",
     "loo_delta": "+0.0009", "shap_rank": "15"},
    {"symbol": "CLDN5",   "ensg": "ENSG00000184113", "direction": "down_in_ALS",
     "loo_delta": "+0.0009", "shap_rank": "16"},
    {"symbol": "NR4A1",   "ensg": "ENSG00000123358", "direction": "down_in_ALS",
     "loo_delta": "+0.0005", "shap_rank": "17"},
    {"symbol": "SOHLH2",  "ensg": "ENSG00000120669", "direction": "up_in_ALS",
     "loo_delta": "-0.0018", "shap_rank": "19"},
    {"symbol": "HEXB",    "ensg": "ENSG00000049860", "direction": "up_in_ALS",
     "loo_delta": "+0.0002", "shap_rank": "21"},
    {"symbol": "MCEE",    "ensg": "ENSG00000124370", "direction": "up_in_ALS",
     "loo_delta": "+0.0008", "shap_rank": "24"},
    {"symbol": "SLC37A2", "ensg": "ENSG00000134955", "direction": "up_in_ALS",
     "loo_delta": "+0.0012", "shap_rank": "25"},
]


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

_DECONV: dict[str, dict] = {
    "MECOM":   {"dominant_ct": "Endothelial", "rho": 0.929, "ct_als_dir": "down"},
    "SERTAD1": {"dominant_ct": "Endothelial", "rho": 0.809, "ct_als_dir": "down"},
    "FCN3":    {"dominant_ct": "Endothelial", "rho": 0.650, "ct_als_dir": "down"},
    "PROS1":   {"dominant_ct": "Microglia",   "rho": 0.849, "ct_als_dir": "up"},
    "ANGPT2":  {"dominant_ct": "Endothelial", "rho": 0.670, "ct_als_dir": "down"},
    "EMP1":    {"dominant_ct": "Microglia",   "rho": 0.771, "ct_als_dir": "up"},
    "TINAGL1": {"dominant_ct": "Endothelial", "rho": 0.932, "ct_als_dir": "down"},
    "CKMT2":   {"dominant_ct": "OPC",         "rho": 0.785, "ct_als_dir": "ns"},
    "VWF":     {"dominant_ct": "Endothelial", "rho": 0.957, "ct_als_dir": "down"},
    "CLDN5":   {"dominant_ct": "Endothelial", "rho": 0.964, "ct_als_dir": "down"},
    "NR4A1":   {"dominant_ct": "Pericyte",    "rho": 0.786, "ct_als_dir": "down"},
    "SOHLH2":  {"dominant_ct": "Astrocyte",   "rho": 0.525, "ct_als_dir": "ns"},
    "HEXB":    {"dominant_ct": "Microglia",   "rho": 0.775, "ct_als_dir": "up"},
    "MCEE":    {"dominant_ct": "Astrocyte",   "rho": 0.623, "ct_als_dir": "ns"},
    "SLC37A2": {"dominant_ct": "Microglia",   "rho": 0.905, "ct_als_dir": "up"},
}


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
            result[gene_sym].append({
                "drug": drug_name,
                "interaction_types": "|".join(types) if types else "unspecified",
                "approved": approved,
                "sources": "|".join(sources),
            })
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
    print("Drug target annotation — core 25-gene ALS panel")
    print("=" * 60)

    symbols = [g["symbol"] for g in _PROTEIN_CODING_GENES]

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
            i["drug"] for i in intx_list
            if not i.get("approved") and any(
                src in i.get("sources", "") for src in
                ["CIViC", "ChEMBL", "ClinicalTrials", "FDA"]
            )
        ]
        inhibitors = [
            i["drug"] for i in intx_list
            if "inhibitor" in i.get("interaction_types", "").lower()
        ]
        activators = [
            i["drug"] for i in intx_list
            if any(t in i.get("interaction_types", "").lower()
                   for t in ["activator", "agonist", "inducer"])
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

        rows.append({
            "shap_rank": shap_rank,
            "symbol": sym,
            "ensembl_id": ensg,
            "direction": direction,
            "loo_delta": loo,
            "tractability": tractability,
            "n_dgidb_interactions": len(intx_list),
            "approved_drugs_dgidb": "|".join(approved_drugs[:5]) if approved_drugs else "",
            "inhibitors_dgidb": "|".join(inhibitors[:5]) if inhibitors else "",
            "activators_dgidb": "|".join(activators[:5]) if activators else "",
            "n_ot_drugs": n_ot_drugs,
            "phase3_drugs_ot": "|".join(phase3_drugs[:5]) if phase3_drugs else "",
            "has_approved_drug": bool(approved_drugs or phase3_drugs),
            "dominant_ct": dominant_ct,
            "deconv_rho": deconv_rho,
            "ct_als_dir": ct_als_dir,
            "per_cell_evidence": per_cell,
        })

    summary_df = pd.DataFrame(rows).sort_values("shap_rank")
    summary_df.to_csv(SCRIPT_DIR / "drug_targets_summary.csv", index=False)

    # ------------------------------------------------------------------
    # 4. Write statistics
    # ------------------------------------------------------------------
    # Priority order for deconv-adjusted ranking:
    # sign_reversal > ambiguous_ct > composition_driven
    _priority_order = {"sign_reversal": 0, "ambiguous_ct": 1, "composition_driven": 2}
    deconv_sorted = summary_df.sort_values(
        "per_cell_evidence",
        key=lambda s: s.map(_priority_order).fillna(3),
    )

    lines: list[str] = [
        "Drug Target Annotation — Core 25-gene ALS Panel",
        "=" * 70,
        f"Protein-coding genes queried: {len(symbols)}",
        "Sources: DGIdb v5 GraphQL + OpenTargets Platform GraphQL",
        "Deconvolution: z-score module scoring (cell_type_deconvolution.py)",
        "",
        "DECONVOLUTION-ADJUSTED PRIORITY TABLE",
        "(sign_reversal > ambiguous_ct > composition_driven)",
        "-" * 80,
        f"{'Symbol':<10} {'Dir':<14} {'LOO Δ':<10} {'Tract.':<12} "
        f"{'Dominant CT':<15} {'ρ':<7} {'CT dir':<8} {'Per-cell evidence'}",
        "-" * 80,
    ]

    for _, r in deconv_sorted.iterrows():
        lines.append(
            f"{r['symbol']:<10} {r['direction']:<14} {r['loo_delta']:<10} "
            f"{r['tractability']:<12} {r['dominant_ct']:<15} "
            f"{r['deconv_rho']:<7.3f} {r['ct_als_dir']:<8} {r['per_cell_evidence']}"
        )

    lines += [
        "",
        "SHAP-RANK TABLE (with deconv annotation)",
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

    lines += [
        "",
        "TIER 1 — SIGN REVERSAL (per-cell signal confirmed by deconvolution)",
        "-" * 70,
        "  Gene direction contradicts dominant cell-type ALS shift →",
        "  composition change cannot explain the expression difference.",
        "",
    ]
    tier1 = deconv_sorted[deconv_sorted["per_cell_evidence"] == "sign_reversal"]
    if tier1.empty:
        lines.append("  None.")
    else:
        for _, r in tier1.iterrows():
            lines.append(f"  {r['symbol']} ({r['direction']}, SHAP rank {r['shap_rank']}):")
            lines.append(
                f"    Dominant CT: {r['dominant_ct']} (ρ={r['deconv_rho']:.3f}, "
                f"CT is {r['ct_als_dir']} in ALS) — gene goes OPPOSITE → per-cell"
            )
            lines.append(f"    LOO Δ={r['loo_delta']}  Tractability: {r['tractability']}")
            if r["approved_drugs_dgidb"]:
                lines.append(f"    DGIdb approved: {r['approved_drugs_dgidb']}")
            if r["inhibitors_dgidb"]:
                lines.append(f"    Inhibitors: {r['inhibitors_dgidb']}")
            if r["phase3_drugs_ot"]:
                lines.append(f"    OT phase≥3: {r['phase3_drugs_ot']}")
            lines.append("")

    lines += [
        "TIER 2 — AMBIGUOUS CT (dominant cell type not significantly changed)",
        "-" * 70,
        "  Cannot resolve per-cell vs composition from bulk data alone.",
        "",
    ]
    tier2 = deconv_sorted[deconv_sorted["per_cell_evidence"] == "ambiguous_ct"]
    if tier2.empty:
        lines.append("  None.")
    else:
        for _, r in tier2.iterrows():
            lines.append(f"  {r['symbol']} ({r['direction']}, SHAP rank {r['shap_rank']}):")
            lines.append(
                f"    Dominant CT: {r['dominant_ct']} (ρ={r['deconv_rho']:.3f}, "
                f"CT not significant in ALS)"
            )
            lines.append(f"    LOO Δ={r['loo_delta']}  Tractability: {r['tractability']}")
            if r["approved_drugs_dgidb"]:
                lines.append(f"    DGIdb approved: {r['approved_drugs_dgidb']}")
            if r["phase3_drugs_ot"]:
                lines.append(f"    OT phase≥3: {r['phase3_drugs_ot']}")
            lines.append("")

    lines += [
        "TIER 3 — COMPOSITION DRIVEN (expression consistent with cell-type shift)",
        "-" * 70,
        "  Dysregulation may reflect cell-type abundance change, not per-cell effect.",
        "  Still valid biomarkers; therapeutic interpretation requires single-cell data.",
        "",
    ]
    tier3 = deconv_sorted[deconv_sorted["per_cell_evidence"] == "composition_driven"]
    if tier3.empty:
        lines.append("  None.")
    else:
        for _, r in tier3.iterrows():
            approved = r["approved_drugs_dgidb"] or r["phase3_drugs_ot"]
            flag = " [ACTIONABLE]" if approved else ""
            lines.append(
                f"  {r['symbol']:<10} CT={r['dominant_ct']:<15} "
                f"ρ={r['deconv_rho']:.3f}  "
                f"tract={r['tractability']}  "
                f"DGIdb={r['n_dgidb_interactions']}{flag}"
            )

    stats_path = SCRIPT_DIR / "drug_targets_statistics.txt"
    stats_path.write_text("\n".join(lines) + "\n")

    actionable = summary_df[summary_df["has_approved_drug"]]
    tier1_genes = tier1["symbol"].tolist() if not tier1.empty else []
    print(f"\n  Summary CSV   → drug_targets_summary.csv")
    print(f"  Statistics    → drug_targets_statistics.txt")
    print(f"\n  Tier 1 (sign reversal / per-cell): {tier1_genes}")
    print(f"  Actionable (approved/phase-3 drug): {actionable['symbol'].tolist()}")
    print("\nDone.")


if __name__ == "__main__":
    main()
