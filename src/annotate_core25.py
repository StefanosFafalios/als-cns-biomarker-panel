# ruff: noqa: E402
"""Annotate the 25-gene core panel using MyGeneInfo, UniProt, and OpenTargets.

For each gene this script fetches:
  - Full name, aliases, gene type (protein-coding / lncRNA / pseudogene)
  - UniProt function summary (first 600 chars)
  - OpenTargets tractability (small molecule / antibody)
  - OpenTargets known disease associations (ALS / MND / neurodegeneration)

Outputs
-------
  core25_annotations.tsv   — one row per gene, all fields
  core25_annotations.txt   — human-readable summary for findings.md
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

ALS_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ALS_DIR))

SCRIPT_DIR = Path(__file__).parent
_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"

_MYGENE_URL = "https://mygene.info/v3/query"
_UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"
_OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"

_REQUEST_DELAY = 0.4   # seconds between external calls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mygene_lookup(symbols: list[str]) -> dict[str, dict]:
    """Batch query MyGeneInfo for gene name, type, Ensembl ID, and UniProt accession."""
    import mygene

    mg = mygene.MyGeneInfo()
    hits = mg.querymany(
        symbols,
        scopes="symbol",
        fields="name,type_of_gene,ensembl.gene,uniprot.Swiss-Prot,alias,other_names",
        species="human",
        verbose=False,
    )
    result: dict[str, dict] = {}
    for h in hits:
        sym = h.get("query", "")
        result[sym] = {
            "name": h.get("name", ""),
            "type_of_gene": h.get("type_of_gene", ""),
            "ensembl_id": (h.get("ensembl") or {}).get("gene", ""),
            "uniprot_acc": _first(h.get("uniprot", {}).get("Swiss-Prot")),
            "aliases": "; ".join(_listify(h.get("alias", []))),
        }
    return result


def _first(val: str | list | None) -> str:
    if val is None:
        return ""
    if isinstance(val, list):
        return val[0] if val else ""
    return str(val)


def _listify(val: str | list) -> list:
    if isinstance(val, list):
        return val
    return [val] if val else []


def _uniprot_function(acc: str) -> str:
    """Return the first function annotation from UniProt for a given accession."""
    if not acc:
        return ""
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.json"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return ""
        data = resp.json()
        for comment in data.get("comments", []):
            if comment.get("commentType") == "FUNCTION":
                texts = comment.get("texts", [])
                if texts:
                    full = texts[0].get("value", "")
                    return full[:700]
        return ""
    except Exception:
        return ""


def _opentargets_target(ensembl_id: str) -> dict:
    """Query OpenTargets Platform for tractability and top disease associations."""
    if not ensembl_id:
        return {}
    query = """
    query Target($id: String!) {
      target(ensemblId: $id) {
        approvedName
        tractability {
          label
          modality
          value
        }
        associatedDiseases(page: {size: 5, index: 0}) {
          rows {
            disease { name }
            score
          }
        }
      }
    }
    """
    try:
        resp = requests.post(
            _OT_URL,
            json={"query": query, "variables": {"id": ensembl_id}},
            timeout=20,
        )
        if resp.status_code != 200:
            return {}
        data = resp.json().get("data", {}).get("target") or {}
        tractability_flags = [
            t["modality"]
            for t in data.get("tractability", [])
            if t.get("value") is True
        ]
        diseases = [
            f"{r['disease']['name']} ({r['score']:.2f})"
            for r in data.get("associatedDiseases", {}).get("rows", [])
        ]
        return {
            "ot_name": data.get("approvedName", ""),
            "tractability": "; ".join(dict.fromkeys(tractability_flags)),
            "top_diseases": " | ".join(diseases),
        }
    except Exception:
        return {}


def _fallback_ensembl_info(ensembl_id: str) -> dict:
    """Query Ensembl REST for gene name when MyGeneInfo returns nothing."""
    if not ensembl_id:
        return {}
    url = f"https://rest.ensembl.org/lookup/id/{ensembl_id}?content-type=application/json"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return {}
        data = resp.json()
        return {
            "name": data.get("description", ""),
            "type_of_gene": data.get("biotype", ""),
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import pandas as pd

    panel = pd.read_csv(_PANEL_CSV)
    symbols: list[str] = panel["symbol"].tolist()
    features: list[str] = panel["feature"].tolist()
    shap_vals: list[float] = panel["mean_abs_shap"].tolist()

    # Separate clean gene symbols from Ensembl-ID-only entries
    clean_symbols = [s for s in symbols if not s.startswith("ENSG")]
    ensg_only = [(i, s) for i, s in enumerate(symbols) if s.startswith("ENSG")]

    print("=" * 65)
    print("Core-25 Gene Annotation")
    print("=" * 65)

    print("\n[1/4] MyGeneInfo batch query ...")
    mg_data = _mygene_lookup(clean_symbols)

    print("[2/4] Ensembl REST fallback for ENSG-only symbols ...")
    ensg_mg: dict[str, dict] = {}
    for _, ensg_sym in ensg_only:
        base = ensg_sym.split(".")[0]
        fb = _fallback_ensembl_info(base)
        ensg_mg[ensg_sym] = {
            "name": fb.get("name", ""),
            "type_of_gene": fb.get("type_of_gene", ""),
            "ensembl_id": base,
            "uniprot_acc": "",
            "aliases": "",
        }
        time.sleep(_REQUEST_DELAY)

    print("[3/4] UniProt function summaries ...")
    uniprot_cache: dict[str, str] = {}
    all_mg = {**mg_data, **ensg_mg}
    for sym, info in all_mg.items():
        acc = info.get("uniprot_acc", "")
        if acc and acc not in uniprot_cache:
            uniprot_cache[acc] = _uniprot_function(acc)
            time.sleep(_REQUEST_DELAY)

    print("[4/4] OpenTargets tractability + disease associations ...")
    ot_cache: dict[str, dict] = {}
    for sym, info in all_mg.items():
        eid = info.get("ensembl_id", "")
        if eid and eid not in ot_cache:
            ot_cache[eid] = _opentargets_target(eid)
            time.sleep(_REQUEST_DELAY)

    # Build output rows
    rows = []
    for sym, feat, shap in zip(symbols, features, shap_vals):
        info = all_mg.get(sym, {})
        acc = info.get("uniprot_acc", "")
        eid = info.get("ensembl_id", "") or feat.split(".")[0]
        ot = ot_cache.get(eid, {})
        rows.append(
            {
                "symbol": sym,
                "feature": feat,
                "mean_abs_shap": round(shap, 6),
                "gene_name": info.get("name") or ot.get("ot_name", ""),
                "type_of_gene": info.get("type_of_gene", ""),
                "ensembl_id": eid,
                "uniprot_acc": acc,
                "aliases": info.get("aliases", ""),
                "uniprot_function": uniprot_cache.get(acc, ""),
                "tractability": ot.get("tractability", ""),
                "top_diseases_ot": ot.get("top_diseases", ""),
            }
        )

    df = pd.DataFrame(rows)
    tsv_out = SCRIPT_DIR / "core25_annotations.tsv"
    df.to_csv(tsv_out, sep="\t", index=False)
    print(f"\nSaved → {tsv_out.name}")

    # Human-readable text summary
    txt_out = SCRIPT_DIR / "core25_annotations.txt"
    lines = [
        "Core 25-gene Panel — Biological Annotations",
        "=" * 65,
        "",
    ]
    for _, row in df.iterrows():
        lines.append(f"{'─' * 60}")
        lines.append(f"Symbol       : {row['symbol']}")
        lines.append(f"Feature ID   : {row['feature']}")
        lines.append(f"SHAP rank    : {row.name + 1}/25  (mean |SHAP| = {row['mean_abs_shap']:.6f})")
        lines.append(f"Gene name    : {row['gene_name']}")
        lines.append(f"Gene type    : {row['type_of_gene']}")
        lines.append(f"Ensembl ID   : {row['ensembl_id']}")
        lines.append(f"UniProt      : {row['uniprot_acc']}")
        if row["aliases"]:
            lines.append(f"Aliases      : {row['aliases']}")
        if row["uniprot_function"]:
            lines.append(f"Function     : {row['uniprot_function'][:500]}")
        if row["tractability"]:
            lines.append(f"Tractability : {row['tractability']}")
        if row["top_diseases_ot"]:
            lines.append(f"Top diseases : {row['top_diseases_ot']}")
        lines.append("")

    txt_out.write_text("\n".join(lines))
    print(f"Saved → {txt_out.name}")

    print("\n" + "=" * 65)
    print("DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
