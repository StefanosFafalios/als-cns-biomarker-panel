"""Known ALS genes, removed before discovery so the panel is drawn from genes not already established in ALS.

Writes known_als_exclusion.csv (one row per Ensembl gene; any source suffices):

  open_targets  Open Targets Platform overall association score >= 0.1 with ALS (MONDO_0004976,
                whose indirect score includes the familial and juvenile subtypes) or with motor
                neuron disorder (MONDO_0020128). The score (0-1) combines genetic association, known
                drugs, animal models, expression changes, pathways and text-mined literature, each
                summed with diminishing returns per piece of evidence and weighted by type, so a
                gene reaches 0.1 through genetic or experimental evidence or a large literature.
  biomarker     established ALS fluid or tissue biomarkers, whatever their score.
  literature    genes named in the title or abstract of at least 10 ALS papers, each paper checked
                by hand (added when the gene reached the discovery top 60).

lgbm_iterative_pipeline.py adds its 18 earlier exclusions (KNOWN_ALS_FEATURES) to these.

Usage: python build_known_als_exclusion.py   (network: Open Targets and MyGeneInfo; the
Open Targets answer is cached in opentargets_als_datatypes.json, delete it to re-query)
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
OUT = SCRIPT_DIR / "known_als_exclusion.csv"
OT_CACHE = SCRIPT_DIR / "opentargets_als_datatypes.json"
OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"
OT_IDS = ("MONDO_0004976", "MONDO_0020128")
OT_THRESHOLD = 0.1
BIOMARKERS = ("CHIT1", "CHI3L1", "CHI3L2", "GPNMB", "NEFL", "NEFM", "NEFH", "UCHL1", "LGALS3", "CXCL10")
# symbol -> evidence; only genes whose ALS title/abstract papers were read and confirmed
LITERATURE: dict[str, str] = {
    # round 1 (step-02 top 500, Europe PMC 2026-10-10); checked by hand, gene itself studied in ALS
    "HGF": "~25 ALS papers: intrathecal recombinant HGF (KP-100) and HGF gene-therapy (VM202) trials, HGF/c-Met in ALS motor neurons and CSF",
    "TSPO": "~10 ALS papers: TSPO-PET of microglial activation in patients and SOD1 mice (PK11195, PBR28, DPA-714) and olesoxime, a TSPO ligand trialled in ALS",
    # round 2
    "APOE": "~28 ALS-titled papers: APOE e4 and bulbar onset, age at onset, survival, cognition (1994-2026)",
    "PPARG": "11 ALS-specific papers: pioglitazone phase II and pilot trials, PPARgamma agonists in SOD1 mice and a TDP-43 fly model",
}


def _post(query: str, variables: dict | None = None) -> dict:
    req = urllib.request.Request(OT_URL, data=json.dumps({"query": query, "variables": variables or {}}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())["data"]


def open_targets_scores() -> dict[str, dict]:
    """{Ensembl ID: {symbol, overall, literature}}, the best score over OT_IDS (cached)."""
    if OT_CACHE.exists():
        return json.loads(OT_CACHE.read_text())
    q = ("query($id:String!,$i:Int!){disease(efoId:$id){associatedTargets(page:{index:$i,size:500})"
         "{rows{score target{id approvedSymbol} datatypeScores{id score}}}}}")
    rows: dict[str, dict] = {}
    for did in OT_IDS:
        for i in range(500):
            dis = _post(q, {"id": did, "i": i})["disease"]
            if dis is None:
                raise ValueError(f"Open Targets has no disease {did}")
            page = dis["associatedTargets"]["rows"]
            if not page:
                break
            for r in page:
                t, lit = r["target"], {d["id"]: d["score"] for d in r["datatypeScores"]}.get("literature", 0.0)
                cur = rows.setdefault(t["id"], {"symbol": t["approvedSymbol"], "overall": 0.0, "literature": 0.0})
                cur["overall"], cur["literature"] = max(cur["overall"], r["score"]), max(cur["literature"], lit)
    OT_CACHE.write_text(json.dumps(rows))
    return rows


def _ensembl(symbols: list[str]) -> dict[str, str]:
    import mygene

    out: dict[str, str] = {}
    for r in mygene.MyGeneInfo().querymany(symbols, scopes="symbol", fields="ensembl.gene", species="human",
                                           verbose=False):
        e = r.get("ensembl")
        e = e if isinstance(e, list) else [e] if e else []
        if e and r["query"] not in out:
            out[r["query"]] = e[0]["gene"]
    missing = set(symbols) - set(out)
    assert not missing, f"no Ensembl ID for {sorted(missing)}"
    return out


def build() -> pd.DataFrame:
    """Every excluded gene with its source(s)."""
    ot = open_targets_scores()
    rows = [dict(ensembl_id=k, symbol=v["symbol"], source="open_targets", detail=f"overall {v['overall']:.3f}")
            for k, v in ot.items() if v["overall"] >= OT_THRESHOLD]
    named = {**{s: "established biomarker" for s in BIOMARKERS}, **LITERATURE}
    ens = _ensembl(sorted(named))
    for s, note in named.items():
        rows.append(dict(ensembl_id=ens[s], symbol=s, source="biomarker" if s in BIOMARKERS else "literature",
                         detail=f"{note}; Open Targets {ot.get(ens[s], {}).get('overall', 0.0):.3f}"))
    df = pd.DataFrame(rows).groupby(["ensembl_id", "symbol"], as_index=False).agg(
        {"source": lambda s: "+".join(sorted(set(s))), "detail": "; ".join})
    return df.sort_values("symbol").reset_index(drop=True)


def main() -> None:
    """Write known_als_exclusion.csv and print the counts per source."""
    v = _post("{meta{dataVersion{year month}}}")["meta"]["dataVersion"]
    df = build()
    df["open_targets_release"] = f"{v['year']}.{v['month']}"
    df.to_csv(OUT, index=False)
    print(f"{len(df)} genes -> {OUT.name} (Open Targets {v['year']}.{v['month']})")
    print(df.source.str.split("+").explode().value_counts().to_string())


if __name__ == "__main__":
    main()
