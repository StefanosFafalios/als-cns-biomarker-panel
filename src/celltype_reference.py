"""Reference-based cell typing for spinal-cord nuclei (Step 105k).

Replaces the four-marker argmax with the healthy human lumbar spinal-cord atlas of
Yadav et al. 2023 (GSE190442, 55,289 nuclei, 64 subtypes), which separates
capillary/venous from arterial endothelium, pericytes/smooth muscle and meningeal cells.

  * build: the dense reference counts are streamed once into per-subtype pseudobulk
    profiles (log CPM); each subtype contributes its N_MARKERS genes with the largest
    log2 ratio over the next-highest subtype (CPM >= 10). Cached as an .npz.
  * assign: each nucleus goes to the subtype whose profile it correlates with best
    (Spearman over the marker genes it has), then to a group in GROUP.

Correlation to reference centroids is the SingleR idea without its fine-tuning step.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
_REF = SCRIPT_DIR.parent / "resources" / "GSE190442" / "suppl"
_CACHE = _REF / "reference_centroids.npz"
N_MARKERS = 30


def group(subtype: str) -> str:
    """Analysis group for a reference subtype (endothelium split by vessel segment)."""
    s = subtype.lower()
    for key, g in (("arterial endo", "endothelial_arterial"), ("ven/cap endo", "endothelial_capven"),
                   ("pericyte", "pericyte"), ("mening", "fibroblast"), ("astro", "astrocyte"),
                   ("microglia", "microglia"), ("macrophage", "microglia"), ("oligo", "oligodendrocyte"), ("opc", "OPC"),
                   ("schwann", "schwann"), ("lymph", "lymphocyte"), ("ependym", "ependymal")):
        if key in s:
            return g
    return "neuron"


def markers(cpm: pd.DataFrame, n: int = N_MARKERS, min_cpm: float = 10.0) -> list[str]:
    """Union of each subtype's top-*n* genes by log2 ratio over the next-highest subtype.

    Args:
        cpm: genes x subtypes pseudobulk CPM.
        n: Markers per subtype.
        min_cpm: Minimum CPM in the subtype.
    """
    lg = np.log2(cpm + 1)
    out: list[str] = []
    for t in cpm.columns:
        other = lg.drop(columns=t).max(axis=1)
        ratio = (lg[t] - other)[cpm[t] >= min_cpm]
        out += [g for g in ratio.nlargest(n).index if g not in out]
    return out


def build(chunk: int = 300) -> tuple[np.ndarray, list[str], list[str]]:
    """(log CPM centroids [subtypes x markers], marker genes, subtype names), cached."""
    if _CACHE.exists():
        z = np.load(_CACHE, allow_pickle=True)
        return z["cent"], list(z["genes"]), list(z["types"])
    meta = pd.read_csv(_REF / "GSE190442_aggregated_metadata_postqc.csv.gz", index_col=0)
    meta.index = meta.index.str.replace("-", ".", regex=False)  # the counts header went through R's make.names
    with gzip.open(_REF / "GSE190442_aggregated_counts_postqc.csv.gz", "rt") as fh:
        cells = [c.strip('"') for c in fh.readline().rstrip("\n").split(",")[1:]]
    sub = meta.subtype_annotation.reindex(cells).fillna("unassigned").to_numpy()
    if (sub == "unassigned").mean() > 0.5:
        raise ValueError(f"only {(sub != 'unassigned').sum()} of {len(cells)} reference cells match the metadata")
    types = sorted(set(sub) - {"unassigned"})
    onehot = np.stack([sub == t for t in types], axis=1).astype(np.float32)  # cells x types
    sums, genes = [], []
    for df in pd.read_csv(_REF / "GSE190442_aggregated_counts_postqc.csv.gz", index_col=0, chunksize=chunk):
        sums.append(df.to_numpy(np.float32) @ onehot)
        genes += list(df.index)
    tot = np.vstack(sums)  # genes x types
    cpm = pd.DataFrame(tot / tot.sum(axis=0) * 1e6, index=genes, columns=types)
    mk = markers(cpm)
    cent = np.log1p(cpm.loc[mk].to_numpy().T)  # types x markers
    np.savez(_CACHE, cent=cent, genes=np.array(mk, dtype=object), types=np.array(types, dtype=object))
    return cent, mk, types


def assign(logn: np.ndarray, genes: list[str], cent: np.ndarray, ref_genes: list[str],
           types: list[str]) -> np.ndarray:
    """Group label per nucleus: best Spearman correlation to a reference subtype.

    Args:
        logn: nuclei x genes log-normalised expression.
        genes: Column names of *logn*.
        cent: types x ref_genes reference profiles.
        ref_genes: Marker genes of *cent*.
        types: Reference subtype names.
    """
    from scipy.stats import rankdata

    common = [g for g in ref_genes if g in set(genes)]
    gi = {g: i for i, g in enumerate(genes)}
    ri = {g: i for i, g in enumerate(ref_genes)}
    q = rankdata(logn[:, [gi[g] for g in common]], axis=1)
    r = rankdata(cent[:, [ri[g] for g in common]], axis=1)
    q = (q - q.mean(1, keepdims=True)) / (q.std(1, keepdims=True) + 1e-9)
    r = (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + 1e-9)
    best = (q @ r.T).argmax(axis=1)
    return np.array([group(types[i]) for i in best])
