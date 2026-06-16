# ruff: noqa: E402
"""DGIdb + OpenTargets annotation for the non-coding critical members'
protein-coding surrogates (XAF1, TM4SF18, SLC6A18).

The non-coding critical-panel members (HERC2P8, SMG1P5 pseudogenes; SNORD97
snoRNA) are not protein-level targets, but the cross-cohort surrogate
substitution analysis maps each to a protein-coding substitute that preserves
(HERC2P8->XAF1: improves) cross-cohort discrimination. This script annotates
those surrogates as Tier-S drug-target candidates, reusing the exact DGIdb /
OpenTargets query functions from drug_targets.py.

Output
------
  drug_targets_surrogates.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

ALS_DIR = Path(__file__).parents[1]
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(ALS_DIR))

from drug_targets import _parse_tractability, _query_dgidb, _query_opentargets

# surrogate -> (replaced non-coding member, Ensembl ID, biology)
_SURROGATES = [
    (
        "XAF1",
        "HERC2P8",
        "ENSG00000132530",
        "XIAP-associated factor 1; type I interferon / apoptosis",
    ),
    ("TM4SF18", "SMG1P5", "ENSG00000157570", "tetraspanin (TM4SF family)"),
    ("SLC6A18", "SNORD97", "ENSG00000164363", "amino-acid (SLC6) transporter"),
]


def main() -> None:
    symbols = [s[0] for s in _SURROGATES]
    print(f"Querying DGIdb for surrogates: {symbols}")
    dgidb = _query_dgidb(symbols)

    lines = [
        "Non-coding critical-member surrogates — DGIdb + OpenTargets annotation",
        "=" * 70,
        f"{'Surrogate':10}{'replaces':10}{'DGIdb_n':>8}  {'approved/clinical drugs':32}{'tractability':18}",
        "-" * 70,
    ]
    for sym, repl, ensg, bio in _SURROGATES:
        intx = dgidb.get(sym, [])
        n = len(intx)
        approved = sorted({i["drug"] for i in intx if i.get("approved") and i["drug"]})
        drug_str = ", ".join(approved[:4]) if approved else "(none approved)"
        try:
            ot = (_query_opentargets(ensg).get("data") or {}).get("target") or {}
            tract = _parse_tractability(ot.get("tractability", []))
        except Exception as exc:
            tract = f"OT-fail:{type(exc).__name__}"
        print(f"  {sym:10}{repl:10}{n:>8}  {drug_str:32}{tract:18} | {bio}")
        lines.append(f"{sym:10}{repl:10}{n:>8}  {drug_str:32}{tract:18}")
        lines.append(f"           biology: {bio}")
        if intx:
            itypes = sorted(
                {
                    t
                    for i in intx
                    for t in i["interaction_types"].split("|")
                    if t and t != "unspecified"
                }
            )
            lines.append(
                f"           interaction types: {', '.join(itypes) if itypes else 'unspecified'}"
            )
    (SCRIPT_DIR / "drug_targets_surrogates.txt").write_text("\n".join(lines) + "\n")
    print("\nSaved -> drug_targets_surrogates.txt")


if __name__ == "__main__":
    main()
