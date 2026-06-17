"""Artifact contamination audit of the SHAP top-500 non-panel gene pool.

OR4D6 (olfactory receptor, chr11 tandem cluster) and PATE1 (prostate/testis
expressed) were identified as quantification artifacts in the universal
replacement gene profiling. This script screens all 475 non-panel SHAP
top-500 genes for similar artifacts using three filters:

  Filter A — Symbol pattern: OR* prefix (olfactory receptors), or known
              testis/sperm-specific families (PATE*, TEKT*, SPATA*, SPAM*,
              TSSK*, PRSS*, ADAM2, ZPBP, etc.)
  Filter B — MyGeneInfo biotype: pseudogene or uncharacterised that is
              NOT already in the core panel (panel pseudogenes were
              explicitly assessed in pseudogene_artifact_check.py)
  Filter C — GTEx max TPM < 1.0 across all 54 tissues for any gene
              flagged by A or B (confirms near-zero expression)

Outputs
-------
  shap500_artifact_audit.txt   — full report with per-gene classification
  shap500_artifact_audit.csv   — machine-readable table (all 475 genes)
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

SCRIPT_DIR = Path(__file__).parent

_SHAP_CSV   = SCRIPT_DIR / "lgbm_top500_final_shap_ranking.csv"
_REPL_CSV   = SCRIPT_DIR / "gene_replacement_results.csv"

# Core panel ENSG bases (version-stripped) — excluded from audit
PANEL_BASES = {
    "ENSG00000085276", "ENSG00000261599", "ENSG00000183604",
    "ENSG00000197019", "ENSG00000213860", "ENSG00000142748",
    "ENSG00000184500", "ENSG00000091879", "ENSG00000236682",
    "ENSG00000134531", "ENSG00000238622", "ENSG00000142910",
    "ENSG00000131730", "ENSG00000280893", "ENSG00000110799",
    "ENSG00000184113", "ENSG00000123358", "ENSG00000223718",
    "ENSG00000120669", "ENSG00000203616", "ENSG00000049860",
    "ENSG00000279656", "ENSG00000226308", "ENSG00000124370",
    "ENSG00000134955",
}

# Olfactory receptor + testis/sperm-specific symbol prefix patterns
_ARTIFACT_PREFIXES = (
    "OR",                        # olfactory receptors
    "PATE",                      # prostate/testis expressed
    "TEKT",                      # tektin (sperm flagellum)
    "SPATA",                     # spermatogenesis associated
    "SPAM",                      # sperm adhesion molecule
    "TSSK",                      # testis-specific serine kinase
    "PRSS",                      # serine proteases (many testis-specific)
    "ZPBP",                      # zona pellucida binding protein
    "ADAM2",                     # fertilin (sperm-egg fusion)
    "CRISP",                     # cysteine-rich secretory protein (testis)
    "OLFM",                      # olfactomedin (distinct from olfactory receptors but flag)
    "OLF",                       # short olfactory prefix
)

GTEX_API = "https://gtexportal.org/api/v2"
MAX_TPM_ARTIFACT_THRESH = 1.0   # genes with max GTEx TPM < 1 in ALL tissues flagged

MYGENE_BATCH = 500  # MyGeneInfo max genes per POST request


# ---------------------------------------------------------------------------
# MyGeneInfo batch symbol + biotype lookup
# ---------------------------------------------------------------------------

def _mygeneinfo_batch(ensg_ids: list[str]) -> dict[str, dict]:
    """Return {ensg_base: {symbol, gene_type}} for a list of ENSG IDs."""
    url = "https://mygene.info/v3/gene"
    results: dict[str, dict] = {}
    for i in range(0, len(ensg_ids), MYGENE_BATCH):
        chunk = ensg_ids[i: i + MYGENE_BATCH]
        try:
            r = requests.post(
                url,
                json={"ids": chunk, "fields": "symbol,type_of_gene,biotype,name"},
                timeout=30,
            )
            r.raise_for_status()
            for item in r.json():
                ensg = item.get("query", "").split(".")[0]
                results[ensg] = {
                    "symbol":       item.get("symbol", ""),
                    "gene_type":    item.get("type_of_gene", ""),
                    "biotype":      item.get("biotype", ""),
                    "name":         item.get("name", ""),
                    "found":        not item.get("notfound", False),
                }
        except Exception as e:
            print(f"  [WARN] MyGeneInfo batch error: {e}")
        time.sleep(0.2)
    return results


# ---------------------------------------------------------------------------
# GTEx max-TPM query for a single gene symbol
# ---------------------------------------------------------------------------

def _gtex_max_tpm(symbol: str) -> float:
    """Return max median TPM across all GTEx v8 tissues for a gene symbol."""
    try:
        r = requests.get(f"{GTEX_API}/reference/gene",
                         params={"geneId": symbol}, timeout=15)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            return -1.0
        gencode_id = data[0]["gencodeId"]
    except Exception:
        return -1.0
    time.sleep(0.25)
    try:
        r2 = requests.get(
            f"{GTEX_API}/expression/medianGeneExpression",
            params={"gencodeId": gencode_id, "datasetId": "gtex_v8"},
            timeout=30,
        )
        r2.raise_for_status()
        rows = r2.json().get("data", [])
        if not rows:
            return 0.0
        return float(max(row["median"] for row in rows))
    except Exception:
        return -1.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("SHAP Top-500 Artifact Contamination Audit")
    print("=" * 72)

    # Load SHAP ranking
    shap_df = pd.read_csv(_SHAP_CSV)
    shap_df["ensg_base"] = shap_df["feature"].str.split(".").str[0]
    non_panel = shap_df[~shap_df["ensg_base"].isin(PANEL_BASES)].copy()
    print(f"\nNon-panel SHAP top-500 genes: {len(non_panel)}")

    # -----------------------------------------------------------------------
    # Step 1: MyGeneInfo batch lookup — symbols + biotype
    # -----------------------------------------------------------------------
    print("\n[1] Querying MyGeneInfo for symbol + biotype ...")
    ensg_ids = non_panel["ensg_base"].tolist()
    gene_info = _mygeneinfo_batch(ensg_ids)

    non_panel["symbol"]    = non_panel["ensg_base"].map(lambda e: gene_info.get(e, {}).get("symbol", ""))
    non_panel["gene_type"] = non_panel["ensg_base"].map(lambda e: gene_info.get(e, {}).get("gene_type", ""))
    non_panel["biotype"]   = non_panel["ensg_base"].map(lambda e: gene_info.get(e, {}).get("biotype", ""))
    non_panel["name"]      = non_panel["ensg_base"].map(lambda e: gene_info.get(e, {}).get("name", ""))
    non_panel["mg_found"]  = non_panel["ensg_base"].map(lambda e: gene_info.get(e, {}).get("found", False))

    # -----------------------------------------------------------------------
    # Step 2: Apply symbol-pattern filter (A) and biotype filter (B)
    # -----------------------------------------------------------------------
    def _symbol_flag(sym: str) -> bool:
        s = str(sym).upper()
        return any(s.startswith(p.upper()) for p in _ARTIFACT_PREFIXES)

    _SUSPICIOUS_BIOTYPES = {
        "pseudogene", "transcribed_unprocessed_pseudogene",
        "unprocessed_pseudogene", "processed_pseudogene",
        "polymorphic_pseudogene", "transcribed_processed_pseudogene",
        "unitary_pseudogene",
    }

    non_panel["flag_symbol"]  = non_panel["symbol"].apply(_symbol_flag)
    non_panel["flag_biotype"] = non_panel["biotype"].str.lower().isin(_SUSPICIOUS_BIOTYPES)
    non_panel["flag_nosym"]   = non_panel["symbol"].str.match(r"^(ENSG|$)", na=True)

    non_panel["any_flag"] = (
        non_panel["flag_symbol"] | non_panel["flag_biotype"] | non_panel["flag_nosym"]
    )

    n_flagged = non_panel["any_flag"].sum()
    print(f"\n  Symbol-pattern flagged (Filter A): {non_panel['flag_symbol'].sum()}")
    print(f"  Pseudogene biotype flagged (Filter B): {non_panel['flag_biotype'].sum()}")
    print(f"  No HGNC symbol flagged: {non_panel['flag_nosym'].sum()}")
    print(f"  Total flagged (any filter): {n_flagged}")

    # -----------------------------------------------------------------------
    # Step 3: GTEx max-TPM for flagged genes (Filter C)
    # -----------------------------------------------------------------------
    flagged = non_panel[non_panel["any_flag"]].copy()
    print(f"\n[2] Querying GTEx v8 max TPM for {len(flagged)} flagged genes ...")

    gtex_max: dict[str, float] = {}
    for _, row in flagged.iterrows():
        sym = row["symbol"]
        # Use symbol if available, otherwise try ENSG ID directly
        query = sym if sym and not sym.startswith("ENSG") else row["ensg_base"]
        tpm = _gtex_max_tpm(query)
        gtex_max[row["ensg_base"]] = tpm
        flag_str = "ARTIFACT" if (0 <= tpm < MAX_TPM_ARTIFACT_THRESH) else (
            "borderline" if (0 <= tpm < 5) else ("expressed" if tpm >= 5 else "not_found")
        )
        print(f"  {query:30s}  max TPM={tpm:7.2f}  → {flag_str}")

    non_panel["gtex_max_tpm"] = non_panel["ensg_base"].map(gtex_max)

    # -----------------------------------------------------------------------
    # Step 4: Classify each flagged gene
    # -----------------------------------------------------------------------
    def _classify(row: pd.Series) -> str:
        if not row["any_flag"]:
            return "clean"
        tpm = row["gtex_max_tpm"]
        if pd.isna(tpm) or tpm < 0:
            return "flagged_no_gtex"
        if tpm < MAX_TPM_ARTIFACT_THRESH:
            return "confirmed_artifact"
        if tpm < 5.0:
            return "borderline"
        return "flagged_but_expressed"

    non_panel["classification"] = non_panel.apply(_classify, axis=1)

    # -----------------------------------------------------------------------
    # Step 5: Replacement impact — how many panel gene replacement
    #         recommendations pointed to confirmed artifacts?
    # -----------------------------------------------------------------------
    repl_df = pd.read_csv(_REPL_CSV)
    artifact_bases = set(
        non_panel.loc[non_panel["classification"] == "confirmed_artifact", "ensg_base"]
    )

    repl_df["candidate_base"] = repl_df["candidate_ensg"].str.split(".").str[0]
    affected = repl_df[
        repl_df["candidate_base"].isin(artifact_bases) & repl_df["is_replacement"]
    ]
    affected_panel_genes = sorted(affected["panel_symbol"].unique())
    n_best_affected = 0
    # Check how many panel genes had an artifact as their BEST replacement
    best_repl = (
        repl_df[repl_df["is_replacement"]]
        .sort_values("replacement_auc", ascending=False)
        .groupby("panel_symbol")
        .first()
        .reset_index()
    )
    best_repl["candidate_base"] = best_repl["candidate_ensg"].str.split(".").str[0]
    best_artifact = best_repl[best_repl["candidate_base"].isin(artifact_bases)]
    n_best_affected = len(best_artifact)

    # -----------------------------------------------------------------------
    # Step 6: Write outputs
    # -----------------------------------------------------------------------
    out_csv = SCRIPT_DIR / "shap500_artifact_audit.csv"
    non_panel.to_csv(out_csv, index=False)
    print(f"\nCSV saved: {out_csv}")

    # Summary report
    class_counts = non_panel["classification"].value_counts()

    lines = [
        "SHAP Top-500 Non-Panel Artifact Contamination Audit",
        "=" * 72,
        "",
        f"Total non-panel SHAP top-500 genes audited: {len(non_panel)}",
        "",
        "Classification summary:",
    ]
    for cls, cnt in class_counts.items():
        lines.append(f"  {cls:35s}: {cnt}")
    lines += [
        "",
        f"Confirmed artifacts (max GTEx TPM < {MAX_TPM_ARTIFACT_THRESH}): "
        f"{class_counts.get('confirmed_artifact', 0)}",
        f"Borderline (max GTEx TPM 1–5): {class_counts.get('borderline', 0)}",
        f"Flagged but expressed (max GTEx TPM ≥ 5): "
        f"{class_counts.get('flagged_but_expressed', 0)}",
        "",
        "Confirmed artifact genes:",
    ]
    confirmed = non_panel[non_panel["classification"] == "confirmed_artifact"]
    for _, row in confirmed.sort_values("rank").iterrows():
        lines.append(
            f"  Rank {row['rank']:3d}  {row['ensg_base']:20s}  "
            f"symbol={row['symbol']:20s}  "
            f"type={row['gene_type']:15s}  "
            f"max_TPM={row['gtex_max_tpm']:.2f}  "
            f"flag={'sym' if row['flag_symbol'] else ''}"
            f"{'bio' if row['flag_biotype'] else ''}"
            f"{'nosym' if row['flag_nosym'] else ''}"
        )
    lines += [
        "",
        "Replacement analysis impact:",
        f"  Panel genes with ≥1 artifact in their replacement list: "
        f"{len(affected_panel_genes)} / 25",
        f"  Affected panel genes: {', '.join(affected_panel_genes) if affected_panel_genes else 'none'}",
        f"  Panel genes whose BEST replacement is an artifact: {n_best_affected}",
    ]
    if n_best_affected > 0:
        for _, row in best_artifact.iterrows():
            lines.append(
                f"    {row['panel_symbol']:15s} → best repl = "
                f"{row['candidate_ensg']}  AUC={row['replacement_auc']:.4f}"
            )
    lines += [
        "",
        "Borderline genes (expressed 1–5 TPM; needs case-by-case review):",
    ]
    borderline = non_panel[non_panel["classification"] == "borderline"]
    for _, row in borderline.sort_values("rank").iterrows():
        lines.append(
            f"  Rank {row['rank']:3d}  {row['ensg_base']:20s}  "
            f"symbol={row['symbol']:20s}  max_TPM={row['gtex_max_tpm']:.2f}"
        )

    out_txt = SCRIPT_DIR / "shap500_artifact_audit.txt"
    out_txt.write_text("\n".join(lines))
    print(f"Report saved: {out_txt}")

    # Print summary
    print("\n" + "\n".join(lines[: lines.index("Confirmed artifact genes:") + 1]))
    print(f"  (see {out_txt} for full list)")


if __name__ == "__main__":
    main()
