"""Re-annotate the array probes to current genes by exact identity, not by the platform's old symbol.

The platform tables name each probe with the gene symbol of their release, so a gene renamed
since then, or one that had no symbol yet, looks absent from the array. Two exact routes:

  GPL6480 (Agilent, E-MTAB-8635/2325): every 60-mer is matched base for base against the
      GENCODE transcripts (both strands are scanned; the strand with more hits to the
      platform's own symbols is the probe strand). A probe whose exact matches all fall in
      transcripts of one gene is assigned to that gene.
  GPL6255 (Illumina HumanRef-8 v2, GSE26927): the GEO table carries no sequences, only the
      RefSeq mRNA each probe was designed against. The accession is resolved to its current
      gene (MyGeneInfo); a probe whose accession maps to one gene is assigned to it.

Probes with no exact match, matches in several genes, or (GPL6480) a platform symbol that is still
the current symbol of another gene keep the platform symbol. Labels follow
the panel's convention (MyGeneInfo symbol, else the Ensembl base ID).

Outputs: resources/platforms/{GPL}_probe_genes.csv (probe, ensembl_id, label).
Usage: python probe_reannotation.py   (downloads GENCODE transcripts once, ~180 MB)
"""

from __future__ import annotations

import gzip
import shutil
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
PLATFORMS = SCRIPT_DIR.parent / "resources" / "platforms"
GENCODE_URL = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50/gencode.v50.transcripts.fa.gz"
GENCODE_FA = SCRIPT_DIR.parent / "resources" / "gencode" / "gencode.v50.transcripts.fa.gz"
_RC = str.maketrans("ACGTN", "TGCAN")


def _table(gpl: str) -> pd.DataFrame:
    """The GEO platform table (resources/platforms/{gpl}_annot.txt) as strings."""
    lines = (PLATFORMS / f"{gpl}_annot.txt").read_text(errors="ignore").splitlines()
    head = next(i for i, ln in enumerate(lines) if ln.startswith("ID\t"))
    cols = lines[head].split("\t")
    rows = [ln.split("\t") for ln in lines[head + 1:] if ln and not ln.startswith("!")]
    return pd.DataFrame([r + [""] * (len(cols) - len(r)) for r in rows], columns=cols)


def _transcripts(path: Path):
    """Yield (gene ID without version, sequence) from a GENCODE transcripts FASTA."""
    gene, seq = None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if gene:
                    yield gene, "".join(seq)
                gene, seq = line[1:].split("|")[1].split(".")[0], []
            else:
                seq.append(line.strip().upper())
    if gene:
        yield gene, "".join(seq)


def exact_matches(probes: dict[str, str], transcripts) -> dict[str, dict[str, set[str]]]:
    """Genes whose transcripts contain each probe exactly, per strand.

    Args:
        probes: probe ID -> sequence (one length).
        transcripts: iterable of (gene, transcript sequence).

    Returns:
        ``{"sense": {probe: genes}, "antisense": {probe: genes}}``.
    """
    k = len(next(iter(probes.values())))
    by_seq: dict[str, list[tuple[str, str]]] = {}
    for p, s in probes.items():
        by_seq.setdefault(s, []).append(("sense", p))
        by_seq.setdefault(s.translate(_RC)[::-1], []).append(("antisense", p))
    hits: dict[str, dict[str, set[str]]] = {"sense": {}, "antisense": {}}
    get = by_seq.get
    for gene, t in transcripts:
        for i in range(len(t) - k + 1):
            found = get(t[i:i + k])
            if found:
                for strand, p in found:
                    hits[strand].setdefault(p, set()).add(gene)
    return hits


def unique_gene(hits: dict[str, set[str]]) -> dict[str, str]:
    """Probe -> gene for probes whose matches all fall in one gene."""
    return {p: next(iter(g)) for p, g in hits.items() if len(g) == 1}


def symbol_conflicts(pairs: dict[str, tuple[str, str]], lookup=None) -> set[str]:
    """Probes whose platform symbol is a current symbol of a gene other than their exact match.

    A 60-mer can match a processed pseudogene exactly yet differ by one base from its abundant
    parent mRNA, which then supplies the signal (FTL -> FTLP3, RPS3A -> RPS3AP6); such probes keep
    the platform symbol. Obsolete names (LOC..., FLJ..., renamed symbols) do not conflict.

    Args:
        pairs: probe -> (platform symbol, Ensembl ID of the exact match).
        lookup: symbol -> Ensembl IDs it currently names; MyGeneInfo when None.
    """
    if lookup is None:
        import mygene

        lookup = {}
        for r in mygene.MyGeneInfo().querymany(sorted({s for s, _ in pairs.values()}), scopes="symbol",
                                               fields="ensembl.gene", species="human", verbose=False):
            e = r.get("ensembl")
            e = e if isinstance(e, list) else [e] if e else []
            lookup.setdefault(r["query"], set()).update(x["gene"] for x in e)
    return {p for p, (s, g) in pairs.items() if lookup.get(s) and g not in lookup[s]}


def _labels(ensembl_ids: list[str]) -> dict[str, str]:
    """Ensembl ID -> MyGeneInfo symbol, else the ID (the panel's convention)."""
    import mygene

    out: dict[str, str] = {}
    for r in mygene.MyGeneInfo().querymany(sorted(set(ensembl_ids)), scopes="ensembl.gene", fields="symbol",
                                           species="human", verbose=False):
        out.setdefault(r["query"], r.get("symbol") or r["query"])
    return {e: out.get(e, e) for e in ensembl_ids}


def build_gpl6480() -> pd.DataFrame:
    """Agilent 60-mers matched exactly against GENCODE transcripts."""
    if not GENCODE_FA.exists():
        GENCODE_FA.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {GENCODE_URL} ...")
        with urllib.request.urlopen(GENCODE_URL, timeout=600) as r, open(GENCODE_FA, "wb") as fh:
            shutil.copyfileobj(r, fh)
    t = _table("GPL6480")
    t = t[(t.CONTROL_TYPE == "FALSE") & (t.SEQUENCE.str.len() == 60)]
    probes = dict(zip(t.ID, t.SEQUENCE.str.upper()))
    print(f"GPL6480: scanning {len(probes)} probes against {GENCODE_FA.name} ...")
    hits = exact_matches(probes, _transcripts(GENCODE_FA))
    sym = dict(zip(t.ID, t.GENE_SYMBOL))
    agree = {}
    for strand, h in hits.items():
        lab = _labels([g for gs in h.values() for g in gs])
        agree[strand] = sum(sym[p] in {lab[g] for g in gs} for p, gs in h.items())
    strand = max(agree, key=agree.get)
    print(f"  probe strand: {strand} (agreement with platform symbols {agree})")
    gene = unique_gene(hits[strand])
    conflict = symbol_conflicts({p: (sym[p], g) for p, g in gene.items() if sym[p]})
    print(f"  {len(conflict)} probes keep their platform symbol (a current gene other than the exact match)")
    gene = {p: g for p, g in gene.items() if p not in conflict}
    lab = _labels(list(gene.values()))
    return pd.DataFrame({"probe": list(gene), "ensembl_id": list(gene.values()),
                         "label": [lab[g] for g in gene.values()]})


def build_gpl6255() -> pd.DataFrame:
    """Illumina probes resolved through the RefSeq mRNA they were designed against."""
    import mygene

    t = _table("GPL6255")
    acc = dict(zip(t.ID, t.GB_ACC.str.split(".").str[0]))
    genes: dict[str, set[str]] = {}
    for r in mygene.MyGeneInfo().querymany(sorted(set(acc.values())), scopes="refseq.rna,accession.rna",
                                           fields="ensembl.gene", species="human", verbose=False):
        e = r.get("ensembl")
        e = e if isinstance(e, list) else [e] if e else []
        genes.setdefault(r["query"], set()).update(x["gene"] for x in e)
    gene = {p: next(iter(genes[a])) for p, a in acc.items() if len(genes.get(a, ())) == 1}
    lab = _labels(list(gene.values()))
    return pd.DataFrame({"probe": list(gene), "ensembl_id": list(gene.values()),
                         "label": [lab[g] for g in gene.values()]})


def probe_labels(gpl: str) -> dict[str, str]:
    """Probe -> exact current gene label, for probes with one exact gene (empty if not built)."""
    path = PLATFORMS / f"{gpl}_probe_genes.csv"
    return dict(pd.read_csv(path, dtype=str)[["probe", "label"]].itertuples(index=False)) if path.exists() else {}


def main() -> None:
    """Build both re-annotation tables and report how many probes changed gene."""
    for gpl, build in (("GPL6480", build_gpl6480), ("GPL6255", build_gpl6255)):
        if len(sys.argv) > 1 and gpl not in sys.argv[1:]:
            continue
        df = build()
        df.to_csv(PLATFORMS / f"{gpl}_probe_genes.csv", index=False)
        t = _table(gpl)
        old = dict(zip(t.ID, t["GENE_SYMBOL" if "GENE_SYMBOL" in t else "SYMBOL"]))
        changed = Counter("unlabelled before" if not old.get(p) else "renamed or reassigned"
                          for p, lab in zip(df.probe, df.label) if old.get(p) != lab)
        print(f"{gpl}: {len(df)} probes with one exact gene; {dict(changed)}")


if __name__ == "__main__":
    main()
