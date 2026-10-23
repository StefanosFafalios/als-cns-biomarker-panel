"""Screen the discovery candidates for an established ALS literature (the exclusion's literature source).

For each gene of the step-02 SHAP top 500 (the pool the panel is drawn from), Europe PMC counts the
papers whose title or abstract names the gene (symbol, official name, or an alias with a digit and
at least four characters) together with amyotrophic lateral sclerosis or motor neuron disease.
Genes with at least LITERATURE_MIN hits are listed with their most cited titles for a hand check:
aliases and generic names over-count (TINAGL1's alias ARG1 is arginase 1), so only genes whose
papers are confirmed to be about that gene in ALS enter build_known_als_exclusion.LITERATURE.

Output: known_als_literature_screen.csv (gene, counts) and _statistics.txt (flagged genes + titles).
Usage: python known_als_literature_screen.py [N]   (N candidates, default 500; network)
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
OUT = SCRIPT_DIR / "known_als_literature_screen"
LITERATURE_MIN = 10
_EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
_ALS = '(TITLE_ABS:"amyotrophic lateral sclerosis" OR TITLE_ABS:"motor neuron disease")'


def gene_terms(symbol: str, name: str, aliases: list[str]) -> list[str]:
    """Search terms for a gene: symbol, official name, and aliases that are unlikely to be words."""
    keep = [a for a in aliases if len(a) >= 4 and re.search(r"[A-Za-z]", a) and re.search(r"\d", a)]
    return [t for t in dict.fromkeys([symbol, name, *keep]) if t]


def _search(terms: list[str], size: int = 1) -> dict:
    q = "(" + " OR ".join(f'TITLE_ABS:"{t}"' for t in terms) + ") AND " + _ALS
    url = _EPMC + urllib.parse.urlencode({"query": q, "format": "json", "pageSize": size, "sort": "CITED desc"})
    for attempt in range(4):
        try:
            return json.loads(urllib.request.urlopen(url, timeout=60).read())
        except OSError:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Europe PMC unreachable for {terms[0]}")


def main() -> None:
    """Count ALS papers per candidate and list the genes that need a hand check."""
    import mygene

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    feats = pd.read_csv(SCRIPT_DIR / "lgbm_iter_final_shap_ranking.csv").feature.head(n)
    base = [f.split(".")[0] for f in feats]
    info: dict[str, dict] = {}
    for r in mygene.MyGeneInfo().querymany(base, scopes="ensembl.gene", fields="symbol,name,alias",
                                           species="human", verbose=False):
        a = r.get("alias", [])
        info.setdefault(r["query"], {"symbol": r.get("symbol", ""), "name": r.get("name", ""),
                                     "alias": [a] if isinstance(a, str) else a})
    rows, flagged = [], []
    for rank, b in enumerate(base, 1):
        g = info.get(b, {"symbol": "", "name": "", "alias": []})
        terms = gene_terms(g["symbol"], g["name"], g["alias"]) if g["symbol"] else []
        hits = _search(terms)["hitCount"] if terms else 0
        rows.append(dict(rank=rank, ensembl_id=b, symbol=g["symbol"] or b, als_papers=hits, terms="; ".join(terms)))
        if hits >= LITERATURE_MIN:
            top = _search(terms, size=8)["resultList"]["result"]
            flagged.append(f"{rank:3d} {g['symbol']} ({hits}): " + " | ".join(
                f"{x.get('pubYear', '')} {x.get('title', '')[:110]}" for x in top))
        time.sleep(0.1)
    pd.DataFrame(rows).to_csv(f"{OUT}.csv", index=False)
    text = (f"Step-02 SHAP top {n}: genes named with ALS/MND in >= {LITERATURE_MIN} titles/abstracts "
            f"(Europe PMC, {time.strftime('%Y-%m-%d')}); confirm each by hand before excluding.\n\n"
            + ("\n".join(flagged) if flagged else "none"))
    Path(f"{OUT}_statistics.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
