"""Annotate panel genes and write the 25-gene panel restricted to annotated genes.

Pseudogenes and unannotated loci are excluded by a fixed rule: no symbol, a ``LOC``/Ensembl symbol, or an NCBI/Ensembl gene type
marking a pseudogene (MyGeneInfo, cached in ``ensg_biotype_cache.json``). Writes
``lgbm_core25_panel_annotated.csv`` (17 genes), the input of the annotated-panel random
gene-set baseline in ``random_panel_null_cohorts.py``.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

SCRIPT_DIR = Path(__file__).parent
_BIOTYPE_CACHE = SCRIPT_DIR / "ensg_biotype_cache.json"
OUT_CSV = SCRIPT_DIR / "lgbm_core25_panel_annotated.csv"


def _annotate(bases: list[str]) -> dict[str, dict]:
    """Symbol and gene types per Ensembl base ID, cached across runs."""
    cache = json.loads(_BIOTYPE_CACHE.read_text()) if _BIOTYPE_CACHE.exists() else {}
    missing = [b for b in dict.fromkeys(bases) if b not in cache]
    for i in range(0, len(missing), 500):
        batch = missing[i : i + 500]
        r = requests.post(
            "https://mygene.info/v3/gene",
            json={"ids": batch, "fields": "symbol,type_of_gene,ensembl.type_of_gene",
                  "species": "human"},
            timeout=60,
        )
        r.raise_for_status()
        for item in r.json():
            q = item.get("query", "")
            if q in cache or item.get("notfound"):
                continue
            ens = item.get("ensembl", {})
            ens = ens if isinstance(ens, list) else [ens]
            cache[q] = dict(
                symbol=item.get("symbol", ""), ncbi=item.get("type_of_gene", ""),
                ensembl=";".join(sorted({e.get("type_of_gene", "") for e in ens} - {""})),
            )
        for b in batch:
            cache.setdefault(b, {})  # not found: unannotated
        time.sleep(0.25)
    if missing:
        _BIOTYPE_CACHE.write_text(json.dumps(cache))
    return {b: cache[b] for b in bases}


def _exclusion(a: dict) -> str:
    sym = a.get("symbol", "")
    if not sym or re.fullmatch(r"LOC\d+|ENSG\d+", sym):
        return "unannotated"
    if "pseudo" in a.get("ncbi", "") or "pseudogene" in a.get("ensembl", ""):
        return "pseudogene"
    return ""


def main() -> None:
    """Write the annotated-gene restriction of the 25-gene panel."""
    panel = pd.read_csv(SCRIPT_DIR / "lgbm_core25_panel.csv")
    ann = _annotate([f.split(".")[0] for f in panel.feature])
    reason = {f: _exclusion(ann[f.split(".")[0]]) for f in panel.feature}
    for f, sym in zip(panel.feature, panel.symbol, strict=True):
        if reason[f]:
            print(f"  excluded {sym:<16} {reason[f]}")
    keep = panel[[not reason[f] for f in panel.feature]][["feature", "symbol"]]
    keep.to_csv(OUT_CSV, index=False)
    print(f"{len(keep)} of {len(panel)} genes kept -> {OUT_CSV.name}")


if __name__ == "__main__":
    main()
