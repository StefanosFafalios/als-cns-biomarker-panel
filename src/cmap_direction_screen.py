"""Direction-matched connectivity-map screen for the per-cell changed genes (Step 105i).

Asks which perturbations move the critical-panel genes with a confirmed per-cell change
(tier 1 in drug_direction_table.csv: per-cell q < 0.05 in ALS spinal cord) back towards
the control state: raise the genes that fall in ALS, lower the ones that rise.

Sources (Enrichr gene-set libraries built from LINCS L1000, cached in resources/LINCS):
  * LINCS_L1000_Chem_Pert_Consensus_Sigs: one up and one down set per compound;
  * LINCS_L1000_CRISPR_KO_Consensus_Sigs: the same per gene knockout (a knockout that
    reverses the signature names a target whose inhibition might);
  * LINCS_L1000_Chem_Pert_up/down: single signatures with their cell line, used for
    EMP1 in astrocytes (ASC). L1000 holds no endothelial cell line.
Every target gene is inferred, not measured, by L1000 (none is among its 978 landmark
genes; GSE92742 gene info), so hits are hypotheses for direct measurement.

Score per perturbagen: genes moved in the needed direction (right) and against it
(wrong). p = probability of at least that many right hits if each gene were included
independently at its background rate (the fraction of perturbagens whose set in that
direction contains it), a Poisson-binomial tail; BH across perturbagens. Run with all
tier-1 genes and without ANGPT2, whose fall in ALS may be compensatory (Discussion).

Outputs: cmap_direction_screen.csv, cmap_direction_screen_statistics.txt
"""

from __future__ import annotations

import gzip
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

SCRIPT_DIR = Path(__file__).parent
LINCS = SCRIPT_DIR.parent / "resources" / "LINCS"
_ENRICHR = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={}"
_GENE_INFO = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/"
              "GSE92742_Broad_LINCS_gene_info.txt.gz")


def _fetch(name: str, url: str) -> Path:
    path = LINCS / name
    if not path.exists():
        LINCS.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(url, timeout=600).read())
    return path


def _library(lib: str) -> dict[str, set[str]]:
    path = _fetch(f"{lib}.txt", _ENRICHR.format(lib))
    out = {}
    for line in path.read_text().splitlines():
        term, _, *genes = line.split("\t")
        out[term] = {g for g in genes if g}
    return out


def _consensus(lib: str) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Split '<name> Up' / '<name> Down' terms into up and down sets keyed by name."""
    up, down = {}, {}
    for term, genes in _library(lib).items():
        name, _, d = term.rpartition(" ")
        (up if d.lower() == "up" else down)[name] = genes
    return up, down


def poisson_binom_sf(k: int, probs: list[float]) -> float:
    """P(at least k successes) for independent Bernoulli trials with *probs*."""
    dist = np.zeros(len(probs) + 1)
    dist[0] = 1.0
    for p in probs:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= 1 - p
    return float(dist[k:].sum())


def reversal_scores(up: dict[str, set[str]], down: dict[str, set[str]], need: dict[str, str]) -> pd.DataFrame:
    """Right/wrong hits, Poisson-binomial p and BH q per perturbagen.

    Args:
        up: Perturbagen -> genes it raises.
        down: Perturbagen -> genes it lowers.
        need: Gene -> "raise" or "lower", the therapeutic direction.

    Returns:
        One row per perturbagen present in both *up* and *down*, sorted by p.
    """
    names = sorted(set(up) & set(down))
    right_sets = {g: (up if d == "raise" else down) for g, d in need.items()}
    wrong_sets = {g: (down if d == "raise" else up) for g, d in need.items()}
    rate = {g: np.mean([g in right_sets[g][n] for n in names]) for g in need}
    rows = []
    for n in names:
        right = [g for g in need if g in right_sets[g][n]]
        wrong = [g for g in need if g in wrong_sets[g][n]]
        rows.append({"perturbagen": n, "n_right": len(right), "n_wrong": len(wrong),
                     "right": ";".join(right), "wrong": ";".join(wrong),
                     "p": poisson_binom_sf(len(right), [rate[g] for g in need])})
    df = pd.DataFrame(rows)
    df["q"] = multipletests(df.p, method="fdr_bh")[1]
    return df.sort_values(["p", "n_wrong"]).reset_index(drop=True)


def _astrocyte_emp1(need: str) -> pd.DataFrame:
    """Per compound: ASC signatures moving EMP1 the needed way vs the other way."""
    up, down = _library("LINCS_L1000_Chem_Pert_up"), _library("LINCS_L1000_Chem_Pert_down")
    asc = [t for t in up if t.split(" ")[1:2] == ["ASC"] and t in down]
    right, wrong = (down, up) if need == "lower" else (up, down)
    base = np.mean([("EMP1" in right[t]) for t in asc])
    rows = {}
    for t in asc:
        cpd = t.split("-", 1)[1].rsplit("-", 1)[0]  # 'CPC001 ASC 24H-<compound>-<dose>'
        r = rows.setdefault(cpd, [0, 0, 0])
        r[0] += 1
        r[1] += "EMP1" in right[t]
        r[2] += "EMP1" in wrong[t]
    df = pd.DataFrame([(c, *v) for c, v in rows.items()], columns=["compound", "n_sig", "n_right", "n_wrong"])
    df["p"] = [poisson_binom_sf(r, [base] * n) for r, n in zip(df.n_right, df.n_sig)]
    df["q"] = multipletests(df.p, method="fdr_bh")[1]
    df.attrs.update(n_sig=len(asc), base=base)
    return df.sort_values(["p", "n_wrong"]).reset_index(drop=True)


def main() -> None:
    tab = pd.read_csv(SCRIPT_DIR / "drug_direction_table.csv")
    t1 = tab[tab.tier == 1]
    need_all = dict(zip(t1.gene, t1.therapeutic_direction, strict=True))
    info = pd.read_csv(gzip.open(_fetch("GSE92742_Broad_LINCS_gene_info.txt.gz", _GENE_INFO)), sep="\t")
    info = info.set_index("pr_gene_symbol")
    status = {g: ("landmark" if info.loc[g, "pr_is_lm"] == 1 else "inferred (BING)" if info.loc[g, "pr_is_bing"] == 1
                  else "inferred") if g in info.index else "absent" for g in need_all}

    chem, ko = _consensus("LINCS_L1000_Chem_Pert_Consensus_Sigs"), _consensus("LINCS_L1000_CRISPR_KO_Consensus_Sigs")
    sets = {"all tier-1": need_all, "without ANGPT2": {g: d for g, d in need_all.items() if g != "ANGPT2"}}
    frames, lines = [], [
        "Direction-matched connectivity-map screen (LINCS L1000 via Enrichr)", "=" * 70,
        "Target genes (therapeutic direction; L1000 status):",
        *[f"  {g:<8} {d:<6} {status[g]}" for g, d in need_all.items()], "",
    ]
    for label, need in sets.items():
        for src, (up, down) in (("compound", chem), ("CRISPR knockout", ko)):
            df = reversal_scores(up, down, need).assign(signature=label, source=src)
            frames.append(df)
            top = df[(df.n_right >= 2) & (df.n_wrong == 0)].head(15)
            lines += [f"[{src}, {label}: {', '.join(need)}] {len(df)} perturbagens; "
                      f"q < 0.05: {int((df.q < 0.05).sum())}; all {len(need)} right, none wrong: "
                      f"{int(((df.n_right == len(need)) & (df.n_wrong == 0)).sum())}"]
            lines += [f"    {r.perturbagen:<40} right {r.right:<26} p {r.p:.2e}  q {r.q:.3f}" for r in top.itertuples()]
            lines.append("")
    if "EMP1" in need_all:
        asc = _astrocyte_emp1(need_all["EMP1"])
        frames.append(asc.rename(columns={"compound": "perturbagen"}).assign(signature="EMP1", source="ASC single signatures"))
        lines += [f"[EMP1 in astrocytes (ASC): {asc.attrs['n_sig']} signatures, background "
                  f"{asc.attrs['base']:.3f} move EMP1 the needed way; q < 0.05: {int((asc.q < 0.05).sum())}]"]
        lines += [f"    {r.compound:<40} {r.n_right}/{r.n_sig} right, {r.n_wrong} wrong  p {r.p:.2e}  q {r.q:.3f}"
                  for r in asc[asc.n_right > 0].head(10).itertuples()]
    pd.concat(frames, ignore_index=True).to_csv(SCRIPT_DIR / "cmap_direction_screen.csv", index=False)
    (SCRIPT_DIR / "cmap_direction_screen_statistics.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
