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

import re
import sys
from pathlib import Path

ALS_DIR = Path(__file__).parents[1]
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(ALS_DIR))

from drug_targets import _parse_tractability, _query_dgidb, _query_opentargets

_SUBSTITUTION_STATS = SCRIPT_DIR / "cross_cohort_substitution_statistics.txt"

# Short biology notes, keyed by surrogate symbol. Only annotation lives here;
# WHICH gene is the surrogate for WHICH member is read from the substitution
# screen, never written down. The previous hardcoded triple (XAF1/HERC2P8,
# TM4SF18/SMG1P5, SLC6A18/SNORD97) was derived from a run in which GSE76220 was
# collapsed by the scaler bug, and two of its three assignments did not survive
# the corrected screen.
# Hand-curated one-line gene function, keyed by surrogate symbol. This is domain
# annotation, not a derived quantity -- but its *keys* go stale whenever the
# substitution screen returns a different surrogate set, which is what happened at
# the k=18 redefinition (the previous five entries were XAF1, TM4SF18, PACS2,
# RPL41 and SLC6A18, of which only SLC6A18 is still selected). The manuscript's
# druggability table quotes these strings, so this dict is the single source for
# them; _check_biology_coverage below makes any gap visible rather than silent.
_BIOLOGY: dict[str, str] = {
    "SLC25A24": "mitochondrial ATP-Mg/Pi carrier (SLC25 family)",
    "EEF1AKMT3": "eEF1A lysine methyltransferase",
    "SEC31B": "COPII coat component (ER-to-Golgi transport)",
    "MOV10L1": "RNA helicase, piRNA pathway",
    "SLC6A18": "amino-acid (SLC6) transporter",
}


def _check_biology_coverage(rows: list[tuple[str, str, str, str]]) -> None:
    """Warn loudly when a selected surrogate has no curated function note.

    Args:
        rows: (symbol, replaces, ensg, biology) tuples as returned by
            ``_load_surrogates``.
    """
    gaps = [r[0] for r in rows if r[3].startswith("(no annotation")]
    if gaps:
        print(
            f"  [warn] no _BIOLOGY entry for {', '.join(gaps)} -- the manuscript "
            "druggability table quotes this field; add them before citing it."
        )


def _load_surrogates() -> list[tuple[str, str, str, str]]:
    """Read the winning surrogate per non-coding member from the substitution screen.

    Returns:
        ``(surrogate_symbol, replaced_member, ensembl_id, biology)`` per gap,
        de-duplicated on the surrogate symbol and ordered by best delta AUC.

    Raises:
        SystemExit: if the substitution statistics file is absent, which means
            cross_cohort_substitution.py has not been run against the current panel.
    """
    if not _SUBSTITUTION_STATS.exists():
        raise SystemExit(
            f"{_SUBSTITUTION_STATS.name} missing -- run cross_cohort_substitution.py "
            "first; the surrogate assignments are derived from it, not hardcoded."
        )
    rows: list[tuple[float, str, str]] = []
    for line in _SUBSTITUTION_STATS.read_text().splitlines():
        m = re.match(
            r"^\S+\s+(\S+)\s+[\d.]+ \[[^\]]+\]\s+(\S+)\s+[\d.]+ \[[^\]]+\]\s+([+-][\d.]+)",
            line,
        )
        if m:
            rows.append((float(m.group(3)), m.group(2), m.group(1)))
    if not rows:
        raise SystemExit(f"no substitution rows parsed from {_SUBSTITUTION_STATS.name}")

    best: dict[str, tuple[float, str]] = {}
    for delta, surrogate, replaced in sorted(rows, reverse=True):
        if surrogate not in best:
            best[surrogate] = (delta, replaced)

    ensg = _ensg_for(sorted(best))
    return [
        (sym, repl, ensg.get(sym, ""), _BIOLOGY.get(sym, "(no annotation on file)"))
        for sym, (_, repl) in sorted(best.items(), key=lambda kv: -kv[1][0])
    ]


def _ensg_for(symbols: list[str]) -> dict[str, str]:
    """Resolve HGNC symbols to Ensembl gene IDs via MyGeneInfo."""
    import mygene

    res = mygene.MyGeneInfo().querymany(
        symbols, scopes="symbol", fields="ensembl.gene", species="human", verbose=False
    )
    out: dict[str, str] = {}
    for r in res:
        ens = r.get("ensembl")
        if isinstance(ens, list):
            ens = ens[0] if ens else None
        if isinstance(ens, dict):
            out[r["query"]] = ens.get("gene", "")
    return out


def main() -> None:
    surrogates = _load_surrogates()
    _check_biology_coverage(surrogates)
    symbols = [s[0] for s in surrogates]
    print(f"Querying DGIdb for surrogates: {symbols}")
    dgidb = _query_dgidb(symbols)

    lines = [
        "Non-coding critical-member surrogates — DGIdb + OpenTargets annotation",
        "=" * 70,
        f"{'Surrogate':10}{'replaces':10}{'DGIdb_n':>8}  {'approved/clinical drugs':32}{'tractability':18}",
        "-" * 70,
    ]
    for sym, repl, ensg, bio in surrogates:
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
