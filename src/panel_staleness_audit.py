"""Report which analysis outputs predate the critical-panel definition they depend on.

Two earlier audits in this study missed stale scripts because they grepped for a
*symptom spelling* -- a gene symbol, a scaler variable name -- and so found only the
call sites that happened to be named after the thing being searched for. This audit
works on the dependency relation instead:

1. A script depends on the panel *directly* if it names one of the ``utils`` panel
   accessors or ``panel_critical.csv``.
2. A script depends on it *transitively* if it imports (at any nesting depth, via
   AST rather than a line-anchored regex) a local module that does.
3. A script depends on it at *second order* if it reads an output artefact written
   by a dependent script -- ``cross_cohort_loo_sensitivity.py`` parses
   ``iterative_panel_elimination_statistics.txt`` and never imports anything, which
   is how a two-month-stale file survived two audits.

A script is flagged STALE when its newest output artefact is older than
``panel_critical.csv``. Because ``greedy_significance_stop.py`` rewrites that file on
every run even when membership is unchanged, an mtime-based flag can over-report:
treat STALE as "prove this one was rerun", not as "this one is definitely wrong".

Transitive dependency is not the same as dependency on panel *content* -- a script may
import only a loader from a panel-aware module. The report prints the import chain so
that judgement stays with the reader.

Usage:
    python panel_staleness_audit.py
"""

from __future__ import annotations

import ast
import re
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
ALS_DIR = SCRIPT_DIR.parent
_PANEL_CSV = SCRIPT_DIR / "panel_critical.csv"

_READERS = re.compile(
    r"critical_panel_symbols|critical_panel_features|"
    r"load_critical_panel|critical_panel_directions|panel_critical\.csv"
)

# Scripts keyed to the 25-gene panel (lgbm_core25_panel.csv) rather than the critical
# panel are unaffected by a critical-panel redefinition even though they import utils.
_TWENTYFIVE_GENE_ONLY = frozenset(
    {"top1_axis_check", "reciprocal_discovery", "incremental_over_composition"}
)


def _pipeline_scripts() -> frozenset[str]:
    """Script stems that run_all.sh actually invokes.

    Derived by parsing run_all.sh rather than kept as a hand-maintained list, so
    adding or retiring a step needs no edit here. Scripts outside the pipeline are
    exploratory: they are never rerun, so they are permanently older than
    panel_critical.csv and would otherwise sit in the report as STALE forever,
    training the reader to ignore the column that matters.
    """
    sh = SCRIPT_DIR / "reproduce" / "run_all.sh"
    if not sh.exists():
        return frozenset()
    return frozenset(re.findall(r"([A-Za-z0-9_]+)\.py", sh.read_text()))


def _local_imports(tree: ast.AST, known: set[str]) -> set[str]:
    """Module names imported anywhere in the tree, restricted to local scripts."""
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    return mods & known


def _artefact_reads(src: str, stems: set[str]) -> set[str]:
    """Script stems whose output artefacts this source reads by filename.

    Looks for ``<stem>...<ext>`` string literals, which is how the output-parsing
    scripts in this study locate their inputs. Import-based dependency is handled
    separately; this catches the ones with no import at all.

    Args:
        src: Source text of the script being examined.
        stems: Known script stems in this directory.

    Returns:
        The stems whose artefacts appear as filenames in ``src``.
    """
    found = set()
    for m in re.finditer(
        r'["\'](\w+?)((?:_\w+)*?)\.(?:txt|csv|tsv|json|npz)["\']', src
    ):
        name = m.group(1) + m.group(2)
        for stem in stems:
            if name == stem or name.startswith(stem + "_"):
                found.add(stem)
    return found


def _declared_outputs(src: str, stem: str, other_stems: set[str]) -> set[Path]:
    """Artefact paths the script names literally, excluding other scripts' outputs.

    Name-prefix matching alone silently loses any script whose artefacts are not
    named after it -- explainability_figures writes explainability_covariates.png,
    srp064478_figure writes manuscript/srp064478_validation.png. Those four steps
    were reported "no artefact found" rather than STALE, so the audit's headline
    could never have caught them going stale. A guard with an invisible coverage
    hole is the same defect this audit exists to catch.

    Args:
        src: Source text of the script.
        stem: The script's own stem.
        other_stems: Stems of every other known script, whose artefacts must not
            be credited to this one.

    Returns:
        Existing artefact paths named in the source.
    """
    out: set[Path] = set()
    for m in re.finditer(r'["\']([\w./-]+\.(?:png|txt|csv|tsv|json|npz|pdf))["\']', src):
        name = Path(m.group(1)).name
        # The panel CSV is every dependent script's *input*; crediting it as an
        # output would make each of them trivially "0.0h newer" than itself.
        if name == _PANEL_CSV.name or any(
            name == s or name.startswith(s + "_") for s in other_stems
        ):
            continue
        for base in (SCRIPT_DIR, SCRIPT_DIR / "manuscript"):
            if (base / name).is_file():
                out.add(base / name)
    return out


def _newest_output(stem: str, declared: set[Path] = frozenset()) -> Path | None:
    """Newest non-source artefact for ``stem``: name-prefixed or declared in source."""
    outs = [
        f
        for f in SCRIPT_DIR.glob(f"{stem}*")
        if f.suffix not in (".py", ".pyc") and f.is_file()
    ]
    outs += [f for f in declared if f.is_file()]
    return max(outs, key=lambda f: f.stat().st_mtime) if outs else None


def main() -> None:
    pipeline = _pipeline_scripts()
    """Print the staleness report."""
    if not _PANEL_CSV.exists():
        sys.exit(
            f"{_PANEL_CSV.name} missing -- run greedy_significance_stop.py first; "
            "it is the single source of truth this audit compares against."
        )
    panel_mt = _PANEL_CSV.stat().st_mtime

    scripts = sorted(p for p in SCRIPT_DIR.glob("*.py"))
    known = {p.stem for p in scripts} | {"utils"}
    trees = {}
    direct: set[str] = set()
    for p in scripts:
        src = p.read_text()
        try:
            trees[p.stem] = ast.parse(src)
        except SyntaxError:
            continue
        if _READERS.search(src):
            direct.add(p.stem)
    if (ALS_DIR / "utils.py").exists():
        trees["utils"] = ast.parse((ALS_DIR / "utils.py").read_text())

    imports = {s: _local_imports(t, known) for s, t in trees.items()}
    # Discard self-reads: a script naming its own output file is not a dependency.
    reads = {p.stem: _artefact_reads(p.read_text(), known) - {p.stem} for p in scripts}
    # An edge is either "imports" or "reads the outputs of". Both propagate
    # staleness; only the first is visible to an import-based audit.
    edges = {s: imports.get(s, set()) | reads.get(s, set()) for s in trees}

    # Transitive closure over both edge kinds.
    dependent = set(direct)
    while True:
        grown = {s for s, mods in edges.items() if mods & dependent} - dependent
        if not grown:
            break
        dependent |= grown

    print(
        f"panel_critical.csv  {time.strftime('%Y-%m-%d %H:%M', time.localtime(panel_mt))}"
    )
    print(f"{len(dependent)} scripts depend on it ({len(direct)} directly)\n")

    rows = []
    for stem in sorted(dependent):
        src = (SCRIPT_DIR / f"{stem}.py").read_text()
        out = _newest_output(stem, _declared_outputs(src, stem, known - {stem}))
        age_h = (out.stat().st_mtime - panel_mt) / 3600 if out else None
        rows.append((stem, age_h, out.name if out else "-"))

    for stem, age_h, out in sorted(rows, key=lambda r: (r[1] is None, r[1] or 0)):
        if stem in direct:
            how = "direct"
        else:
            via_import = sorted(imports.get(stem, set()) & dependent)
            via_read = sorted(reads.get(stem, set()) & dependent - set(via_import))
            parts = []
            if via_import:
                parts.append("imports " + ",".join(via_import))
            if via_read:
                parts.append("reads " + ",".join(via_read))
            how = "; ".join(parts) or "?"
        if stem in _TWENTYFIVE_GENE_ONLY:
            flag, detail = "n/a  ", "keyed to lgbm_core25_panel.csv"
        elif stem not in pipeline:
            flag, detail = "expl ", "exploratory; not a run_all.sh step"
        elif age_h is None:
            flag, detail = "?    ", "no artefact found"
        elif age_h < 0:
            flag, detail = "STALE", f"{-age_h:.1f}h older  ({out})"
        else:
            flag, detail = "ok   ", f"{age_h:.1f}h newer  ({out})"
        print(f"  {flag}  {stem:34s} {detail:44s} [{how}]")


if __name__ == "__main__":
    main()
