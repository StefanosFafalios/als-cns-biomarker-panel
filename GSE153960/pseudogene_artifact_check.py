# ruff: noqa: E402
"""Artifact assessment for pseudogenes and uncharacterized IDs in the core-25 panel.

For each of the 7 genes (5 pseudogenes + 2 uncharacterized ENSG-only):
  1. Query Ensembl REST for biotype and parent gene (if pseudogene).
  2. Check whether the parent gene is present in the prefilter feature set.
  3. Compute Pearson r and Spearman ρ between the pseudogene column and the
     parent gene column in the prefiltered expression matrix.
  4. Report the parent gene's univariate AUC and SHAP rank if it appears in
     the top-500.
  5. For uncharacterized IDs: fetch genomic region and nearby gene annotations.

Interpretation:
  - High |r| / |ρ| with parent  AND  parent highly expressed/ALS-relevant
    → likely cross-mapping artifact; SHAP importance reflects parent signal.
  - Low |r| / |ρ|  OR  parent absent / unexpressed
    → evidence of genuine independent transcription at the locus.

Outputs
-------
  pseudogene_artifact_report.txt  — human-readable verdict per gene
  pseudogene_artifact_report.tsv  — machine-readable summary
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import requests

ALS_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ALS_DIR))

SCRIPT_DIR = Path(__file__).parent

_PREFILTER_X_NPY = SCRIPT_DIR / "lgbm_prefilter_X.npy"
_PREFILTER_NAMES_TXT = SCRIPT_DIR / "lgbm_prefilter_names.txt"
_SHAP_RANKING_CSV = SCRIPT_DIR / "lgbm_top500_final_shap_ranking.csv"
_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PROFILING_CSV = SCRIPT_DIR / "gene_profiling_summary.csv"

_ENSEMBL_URL = "https://rest.ensembl.org"
_REQUEST_DELAY = 0.4

# The 7 genes to audit (symbol → Ensembl base ID)
_AUDIT_GENES: dict[str, str] = {
    "HERC2P8":       "ENSG00000261599",
    "SMG1P5":        "ENSG00000183604",
    "RPL21P75":      "ENSG00000213860",
    "RPL15P11":      "ENSG00000223718",
    "RHOT1P2":       "ENSG00000203616",
    "ENSG00000280893": "ENSG00000280893",
    "ENSG00000279656": "ENSG00000279656",
}


# ---------------------------------------------------------------------------
# Ensembl helpers
# ---------------------------------------------------------------------------


def _ensembl_lookup(ensembl_id: str) -> dict:
    """Return gene record from Ensembl /lookup/id (includes biotype, description)."""
    url = f"{_ENSEMBL_URL}/lookup/id/{ensembl_id}?content-type=application/json&expand=1"
    try:
        r = requests.get(url, timeout=15)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def _ensembl_overlap(ensembl_id: str, region: str) -> list[dict]:
    """Return overlapping genes in ±500 kb window around the locus."""
    # region format: "chr:start-end:strand"
    url = (
        f"{_ENSEMBL_URL}/overlap/region/human/{region}"
        f"?feature=gene&content-type=application/json"
    )
    try:
        r = requests.get(url, timeout=15)
        return r.json() if r.status_code == 200 and isinstance(r.json(), list) else []
    except Exception:
        return []


def _ensembl_xref(ensembl_id: str, db: str = "EntrezGene") -> list[dict]:
    """Cross-references for a gene (e.g. parent gene via HGNC xref)."""
    url = (
        f"{_ENSEMBL_URL}/xrefs/id/{ensembl_id}"
        f"?content-type=application/json&external_db={db}"
    )
    try:
        r = requests.get(url, timeout=15)
        return r.json() if r.status_code == 200 and isinstance(r.json(), list) else []
    except Exception:
        return []


def _lookup_parent(info: dict) -> str | None:
    """Try to extract a parent gene name from the Ensembl description field.

    Ensembl pseudogene descriptions often contain the parent gene name in
    parentheses: 'HERC2 pseudogene 8 [Source:HGNC...]' → 'HERC2'.
    """
    desc = info.get("description", "") or ""
    name = info.get("display_name", "") or ""
    for candidate in [name, desc]:
        parts = candidate.split()
        if "pseudogene" in parts:
            idx = parts.index("pseudogene")
            if idx > 0:
                return parts[idx - 1]
    return None


# ---------------------------------------------------------------------------
# Expression correlation
# ---------------------------------------------------------------------------


def _load_column_index() -> tuple[np.ndarray, dict[str, int]]:
    """Return (X_full, name→col_index) for the prefilter matrix (lazy load)."""
    names = _PREFILTER_NAMES_TXT.read_text().splitlines()
    name_to_col = {n: i for i, n in enumerate(names)}
    return name_to_col


def _pearson_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from scipy.stats import pearsonr, spearmanr

    r, _ = pearsonr(x, y)
    rho, _ = spearmanr(x, y)
    return float(r), float(rho)


def _versioned_key(base_id: str, name_to_col: dict[str, int]) -> str | None:
    """Find the versioned feature key for a base Ensembl ID (e.g. ENSG00000261599.6)."""
    for k in name_to_col:
        if k.startswith(base_id + ".") or k == base_id:
            return k
    return None


# ---------------------------------------------------------------------------
# SHAP / profiling lookups
# ---------------------------------------------------------------------------


def _load_references() -> tuple[dict[str, float], dict[str, int], dict[str, float]]:
    """Return (feature→univariate_auc, feature→shap_rank, feature→mean_abs_shap)."""
    import pandas as pd

    prof = pd.read_csv(_PROFILING_CSV)
    # profiling CSV has 'feature' column (versioned Ensembl ID)
    auc_map: dict[str, float] = {}
    for _, row in prof.iterrows():
        auc_map[row["feature"]] = float(row["univariate_auc"])

    shap_df = pd.read_csv(_SHAP_RANKING_CSV)
    shap_rank: dict[str, int] = {}
    shap_val: dict[str, float] = {}
    for rank, row in shap_df.iterrows():
        shap_rank[row["feature"]] = int(rank) + 1
        shap_val[row["feature"]] = float(row["mean_abs_shap"])

    return auc_map, shap_rank, shap_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import pandas as pd
    from scipy.stats import pearsonr  # noqa: F401 — trigger import check early

    print("=" * 68)
    print("Pseudogene / Uncharacterized ID — Artifact Assessment")
    print("=" * 68)

    print("\nLoading prefilter feature index ...")
    name_to_col = _load_column_index()

    print("Loading SHAP and profiling references ...")
    auc_map, shap_rank, shap_val = _load_references()

    # Lazy-load X only if we find a parent with a column
    X_cache: dict[str, np.ndarray] = {}  # feature_key → expression vector

    def _get_expr(feature_key: str | None) -> np.ndarray | None:
        if feature_key is None:
            return None
        if feature_key not in X_cache:
            if not X_cache:
                # First access — load full matrix (874×47,822 float32 ~200 MB)
                print("  [loading prefilter matrix — this may take a moment]")
                X_full = np.load(_PREFILTER_X_NPY, mmap_mode="r")
                X_cache["__X__"] = X_full
            col = name_to_col.get(feature_key)
            if col is None:
                return None
            X_cache[feature_key] = X_cache["__X__"][:, col].astype(np.float64)
        return X_cache[feature_key]

    rows: list[dict] = []

    for symbol, base_id in _AUDIT_GENES.items():
        print(f"\n{'─' * 60}")
        print(f"Auditing: {symbol}  ({base_id})")

        # 1. Ensembl lookup
        info = _ensembl_lookup(base_id)
        time.sleep(_REQUEST_DELAY)
        biotype = info.get("biotype", "unknown")
        description = info.get("description", "")
        chrom = info.get("seq_region_name", "")
        start = info.get("start", 0)
        end = info.get("end", 0)
        strand = info.get("strand", 0)
        print(f"  biotype   : {biotype}")
        print(f"  description: {description[:120]}")
        print(f"  location  : chr{chrom}:{start}-{end} (strand {strand})")

        # 2. Identify parent gene (from description or display name)
        parent_sym = _lookup_parent(info)
        print(f"  parent (parsed): {parent_sym}")

        # 3. Cross-reference: check if parent is in prefilter set
        pseudo_feat_key = _versioned_key(base_id, name_to_col)
        parent_feat_key = None
        parent_base_id = None

        if parent_sym:
            # Try to find parent Ensembl ID by querying Ensembl
            search_url = (
                f"{_ENSEMBL_URL}/lookup/symbol/homo_sapiens/{parent_sym}"
                f"?content-type=application/json"
            )
            try:
                pr = requests.get(search_url, timeout=15)
                if pr.status_code == 200:
                    pdata = pr.json()
                    parent_base_id = pdata.get("id", "")
                    if parent_base_id:
                        parent_feat_key = _versioned_key(parent_base_id, name_to_col)
                        print(f"  parent Ensembl: {parent_base_id}")
                        print(f"  parent in prefilter: {parent_feat_key is not None}")
            except Exception:
                pass
            time.sleep(_REQUEST_DELAY)

        # 4. Parent SHAP / AUC
        parent_shap_rank = shap_rank.get(parent_feat_key, None) if parent_feat_key else None
        parent_shap_val  = shap_val.get(parent_feat_key, None)  if parent_feat_key else None
        parent_auc       = auc_map.get(parent_feat_key, None)   if parent_feat_key else None
        if parent_shap_rank is not None:
            print(f"  parent SHAP rank (top-500): {parent_shap_rank}  (mean |SHAP|={parent_shap_val:.4f})")
        if parent_auc is not None:
            print(f"  parent univariate AUC: {parent_auc:.4f}")

        # 5. Expression correlation (pseudo vs parent)
        r_val = rho_val = None
        if pseudo_feat_key and parent_feat_key:
            x_pseudo = _get_expr(pseudo_feat_key)
            x_parent = _get_expr(parent_feat_key)
            if x_pseudo is not None and x_parent is not None:
                r_val, rho_val = _pearson_spearman(x_pseudo, x_parent)
                print(f"  Pearson r  (pseudo vs parent): {r_val:+.4f}")
                print(f"  Spearman ρ (pseudo vs parent): {rho_val:+.4f}")

        # 6. For uncharacterized: overlap analysis (nearby protein-coding genes)
        nearby_coding: list[str] = []
        if biotype in ("", "unknown", "novel_transcript") or symbol.startswith("ENSG"):
            window = 500_000
            region_str = f"{chrom}:{max(1, start - window)}-{end + window}"
            overlaps = _ensembl_overlap(base_id, region_str)
            time.sleep(_REQUEST_DELAY)
            nearby_coding = [
                g.get("external_name", g.get("id", ""))
                for g in overlaps
                if g.get("biotype") == "protein_coding"
                   and g.get("external_name", g.get("id", "")) != symbol
            ][:10]
            if nearby_coding:
                print(f"  Nearby protein-coding (±500 kb): {', '.join(nearby_coding)}")

        # 7. Verdict
        artifact = False
        verdict = ""
        if r_val is not None and abs(r_val) >= 0.85:
            artifact = True
            verdict = (
                f"LIKELY ARTIFACT — Pearson r={r_val:+.3f} with parent {parent_sym}. "
                f"Reads from {parent_sym} almost certainly cross-map to this pseudogene locus."
            )
        elif r_val is not None and abs(r_val) >= 0.60:
            artifact = True
            verdict = (
                f"PROBABLE ARTIFACT — moderate correlation r={r_val:+.3f} with parent {parent_sym}. "
                f"Partial cross-mapping cannot be excluded; treat as proxy for {parent_sym}."
            )
        elif r_val is not None and abs(r_val) < 0.60:
            artifact = False
            verdict = (
                f"LIKELY GENUINE — low correlation r={r_val:+.3f} with parent {parent_sym}. "
                f"Expression pattern is largely independent of the parent gene."
            )
        elif biotype in ("processed_pseudogene", "pseudogene", "unprocessed_pseudogene"):
            verdict = (
                "UNRESOLVED — pseudogene with no detectable parent in prefilter set. "
                "Cannot compute correlation; cross-mapping possible but unverifiable."
            )
        else:
            verdict = (
                "UNRESOLVED — uncharacterized locus; no parent gene identified. "
                "May be a novel lncRNA or unannotated transcript."
            )
        print(f"\n  VERDICT: {verdict}")

        rows.append(
            {
                "symbol": symbol,
                "ensembl_id": base_id,
                "biotype": biotype,
                "parent_symbol": parent_sym or "",
                "parent_ensembl": parent_base_id or "",
                "parent_in_prefilter": parent_feat_key is not None,
                "parent_shap_rank": parent_shap_rank,
                "parent_univariate_auc": parent_auc,
                "pearson_r_vs_parent": round(r_val, 4) if r_val is not None else None,
                "spearman_rho_vs_parent": round(rho_val, 4) if rho_val is not None else None,
                "artifact": artifact,
                "verdict": verdict,
                "nearby_protein_coding": "; ".join(nearby_coding),
            }
        )

    # Output
    df = pd.DataFrame(rows)
    tsv_out = SCRIPT_DIR / "pseudogene_artifact_report.tsv"
    df.to_csv(tsv_out, sep="\t", index=False)

    txt_out = SCRIPT_DIR / "pseudogene_artifact_report.txt"
    lines = [
        "Pseudogene / Uncharacterized ID — Artifact Assessment",
        "=" * 68,
        "",
        "Cross-mapping criterion: Pearson |r| ≥ 0.85 → LIKELY ARTIFACT",
        "                         Pearson |r| ≥ 0.60 → PROBABLE ARTIFACT",
        "                         Pearson |r| < 0.60 → LIKELY GENUINE",
        "",
    ]
    for _, row in df.iterrows():
        lines.append(f"{'─' * 60}")
        lines.append(f"Gene          : {row['symbol']} ({row['ensembl_id']})")
        lines.append(f"Biotype       : {row['biotype']}")
        lines.append(f"Parent        : {row['parent_symbol']} ({row['parent_ensembl']})")
        lines.append(f"Parent in prefilter: {row['parent_in_prefilter']}")
        if row["parent_shap_rank"]:
            lines.append(f"Parent SHAP rank: {row['parent_shap_rank']}/500  (top-500 model)")
        if row["parent_univariate_auc"]:
            lines.append(f"Parent univariate AUC: {row['parent_univariate_auc']:.4f}")
        if row["pearson_r_vs_parent"] is not None:
            lines.append(f"Pearson r (pseudo vs parent): {row['pearson_r_vs_parent']:+.4f}")
            lines.append(f"Spearman ρ (pseudo vs parent): {row['spearman_rho_vs_parent']:+.4f}")
        if row["nearby_protein_coding"]:
            lines.append(f"Nearby coding (±500 kb): {row['nearby_protein_coding']}")
        lines.append(f"VERDICT: {row['verdict']}")
        lines.append("")

    txt_out.write_text("\n".join(lines))
    print(f"\nSaved → {tsv_out.name}")
    print(f"Saved → {txt_out.name}")
    print("\n" + "=" * 68)
    print("DONE")
    print("=" * 68)


if __name__ == "__main__":
    main()
