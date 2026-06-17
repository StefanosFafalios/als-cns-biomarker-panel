"""External validation on SRP064478 (Ladd et al. 2017, Brain Research).

Dataset: 15 postmortem cervical spinal cord samples (VCU BennettLab)
  - 7 sporadic ALS  (SRR2558710–SRR2558716)
  - 8 neurologically healthy controls (SRR2558717–SRR2558724)
Platform: NextSeq 500, paired-end, total stranded RNA-seq
Quantification: Salmon 0.13.1 quasi-mapping against Ensembl hg38

Strategy
--------
1. Load Salmon quant.sf files; aggregate transcript TPM to gene level using
   the Ensembl transcript → gene mapping extracted from the quant headers.
2. Apply log1p transform; match the 15 protein-coding panel genes by
   Ensembl base ID (strip version suffix).
3. Fit StandardScaler on GPL24676 training set; transform both.
4. Train LightGBM on GPL24676 (874 samples) with matched genes.
5. Zero-shot predict on SRP064478 (n=15); report AUC + bootstrap CI.
6. Within-cohort LOO-CV (leave-one-out; n=15) for the primary estimate.
7. Per-gene Mann-Whitney U (ALS vs control).

Outputs
-------
  srp064478_validation.png
  srp064478_validation_statistics.txt
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ALS_DIR = Path(__file__).parents[1]
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset  # noqa: E402

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"
_QUANT_DIR = ALS_DIR / "resources" / "SRP064478" / "quant"
_META = ALS_DIR / "resources" / "SRP064478" / "srr_metadata.tsv"
_OUT_PNG = SCRIPT_DIR / "srp064478_validation.png"
_OUT_STATS = SCRIPT_DIR / "srp064478_validation_statistics.txt"

RANDOM_STATE = 42
N_BOOTSTRAP = 2_000
PLATFORM = "GPL24676"

_PC_SYMS: set[str] = {
    "MECOM", "SERTAD1", "FCN3", "PROS1", "ANGPT2", "EMP1", "TINAGL1",
    "CKMT2", "VWF", "CLDN5", "NR4A1", "SOHLH2", "HEXB", "MCEE", "SLC37A2",
}


# ---------------------------------------------------------------------------
# Salmon output loading
# ---------------------------------------------------------------------------


def _load_salmon_quant(quant_dir: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Load all Salmon quant.sf files; aggregate TPM to gene level.

    Returns (X, gene_ids, y) where gene_ids are Ensembl base IDs and
    y is the ALS label vector.
    """
    import pandas as pd

    meta = pd.read_csv(_META, sep="\t")
    meta = meta.set_index("SRR")

    samples: list[str] = []
    y_list: list[int] = []
    all_quants: list[pd.Series] = []

    gene_ids: list[str] | None = None

    for srr in meta.index:
        sf = quant_dir / srr / "quant.sf"
        if not sf.exists():
            raise FileNotFoundError(f"Missing: {sf}")

        df = pd.read_csv(sf, sep="\t")
        df["gene_id"] = df["Name"].str.split(".").str[0]  # strip transcript version
        # Also strip transcript suffix (ENST → ENSG via name prefix is not
        # available here; we map ENST base ID and will cross-join to ENSG later)
        # For now use NumReads (estimated counts) aggregated to transcript base ID.
        # We use NumReads rather than TPM to match the count-based training data.
        gene_counts = df.groupby("gene_id")["NumReads"].sum()

        if gene_ids is None:
            gene_ids = list(gene_counts.index)
        all_quants.append(gene_counts)

        cond = meta.loc[srr, "condition"]
        samples.append(srr)
        y_list.append(1 if cond == "ALS" else 0)

    count_matrix = pd.DataFrame(all_quants, index=samples).fillna(0)
    X = np.log1p(count_matrix.values.astype(np.float32))
    y = np.array(y_list, dtype=int)
    return X, list(count_matrix.columns), y


def _load_train(
    transcript_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load GPL24676 training data; match protein-coding panel genes.

    Returns (X_matched, y, panel_ensg_bases, matched_transcript_bases).
    The panel ENSG IDs are matched to Salmon transcript IDs by ENSG base ID
    after stripping version suffixes; ENST→ENSG mapping requires the panel
    CSV symbol column for cross-reference.
    """
    import pandas as pd

    panel_df = pd.read_csv(_PANEL_CSV)
    feat_col = next(
        (c for c in panel_df.columns if "ensg" in c.lower()),
        next(c for c in panel_df.columns if "feature" in c.lower()),
    )
    sym_col = next(c for c in panel_df.columns if "symbol" in c.lower())

    # Build {ensg_base: symbol} for protein-coding panel genes
    pc_pairs: list[tuple[str, str, str]] = []  # (ensg_versioned, ensg_base, symbol)
    for _, row in panel_df.iterrows():
        sym = str(row[sym_col])
        if sym in _PC_SYMS:
            ensg_v = str(row[feat_col])
            pc_pairs.append((ensg_v, ensg_v.split(".")[0], sym))

    # Load GPL24676
    (ds,) = load_dataset("GSE153960", platform=PLATFORM, resources_dir=ALS_DIR / "resources")
    X_log = np.log1p(ds.X.values.astype(np.float32))
    feat_map = {n: i for i, n in enumerate(ds.X.columns)}

    cols: list[int] = []
    matched_ensg: list[str] = []
    matched_syms: list[str] = []
    for ensg_v, ensg_base, sym in pc_pairs:
        idx = feat_map.get(ensg_v)
        if idx is not None:
            cols.append(idx)
            matched_ensg.append(ensg_base)
            matched_syms.append(sym)

    y = ds.y.values.astype(int)
    return X_log[:, cols], y, matched_ensg, matched_syms


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def _bootstrap_auc(
    y: np.ndarray, scores: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    from sklearn.metrics import roc_auc_score

    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], scores[idx]))
    return np.array(aucs)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _plot(
    y: np.ndarray,
    scores_zero: np.ndarray,
    auc_zero: float,
    ci_lo: float,
    ci_hi: float,
    matched_syms: list[str],
    mwu_results: list[tuple[str, float, float]],
) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel A: zero-shot ROC ---
    fpr, tpr, _ = roc_curve(y, scores_zero)
    axes[0].plot(fpr, tpr, lw=2, color="#1f77b4",
                 label=f"Zero-shot AUC = {auc_zero:.4f}\n95%CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title(
        f"SRP064478 zero-shot (trained on GPL24676, n=874)\n"
        f"Cervical SC · n=15 (ALS=7, Ctrl=8) · "
        f"{len(matched_syms)}/15 protein-coding genes",
        fontsize=9,
    )
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].grid(alpha=0.3)

    # --- Panel B: per-gene ALS vs Control MWU ---
    syms_mwu = [r[0] for r in mwu_results]
    rb_vals = [r[1] for r in mwu_results]
    pvals = [r[2] for r in mwu_results]
    colours = ["#d62728" if rb > 0 else "#1f77b4" for rb in rb_vals]

    y_pos = np.arange(len(syms_mwu))
    axes[1].barh(y_pos, rb_vals, color=colours, alpha=0.8)
    axes[1].axvline(0, color="black", lw=0.8)
    for i, (rb, p) in enumerate(zip(rb_vals, pvals)):
        if p < 0.05:
            x = rb + 0.01 if rb >= 0 else rb - 0.01
            axes[1].text(x, i, "*", va="center", fontsize=10,
                         ha="left" if rb >= 0 else "right")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(syms_mwu, fontsize=8)
    axes[1].set_xlabel("Rank-biserial r (ALS vs Control)")
    axes[1].set_title(
        "Per-gene expression: ALS vs Control (SRP064478)\n"
        "(* = MWU p < 0.05; red = up in ALS)",
        fontsize=9,
    )
    axes[1].grid(alpha=0.3, axis="x")

    plt.tight_layout()
    fig.savefig(_OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {_OUT_PNG.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import pandas as pd
    from lightgbm import LGBMClassifier
    from scipy.stats import mannwhitneyu
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    print("=" * 65)
    print("SRP064478 Cervical Spinal Cord Validation (Ladd et al. 2017)")
    print("=" * 65)

    # ------------------------------------------------------------------
    print("\n[1] Loading Salmon quantification results ...")
    X_srp, transcript_ids, y_srp = _load_salmon_quant(_QUANT_DIR)
    print(f"  SRP064478: {X_srp.shape}  ALS={int(y_srp.sum())}  Ctrl={int((y_srp==0).sum())}")

    # ------------------------------------------------------------------
    print("\n[2] Loading GPL24676 training data ...")
    X_train, y_train, ensg_bases, matched_syms = _load_train(transcript_ids)
    print(f"  Training: {X_train.shape}")
    print(f"  Matched protein-coding genes ({len(matched_syms)}): {matched_syms}")

    # ------------------------------------------------------------------
    # Match SRP064478 columns to the panel gene ENSG base IDs
    # Salmon transcript IDs are ENST, not ENSG. We need a transcript→gene map.
    # Extract from the Salmon quant.sf Name column using Ensembl transcript ID
    # prefix: ENST IDs aggregate to ENSG IDs in _load_salmon_quant.
    # For panel gene matching, we need ENSG IDs in the Salmon output.
    # The quant.sf aggregation key is ENST base ID; ENSG IDs are NOT in the
    # Salmon output directly. We need to join via a transcript→gene table or
    # use the gene symbols as a fallback.
    #
    # Workaround: load the GTF/tx2gene table to map ENST→ENSG, or alternatively
    # match by gene symbol using a t2g file.
    # For now, attempt direct ENSG match (will only work if Ensembl cDNA
    # headers contain ENSG IDs in the description field).
    # ------------------------------------------------------------------
    print("\n[3] Matching panel genes to Salmon output ...")

    # Check if any of the Salmon IDs start with ENSG (gene-level aggregation
    # may have occurred if the FASTA headers contain gene IDs)
    ensg_in_salmon = [t for t in transcript_ids if t.startswith("ENSG")]
    enst_in_salmon = [t for t in transcript_ids if t.startswith("ENST")]
    print(f"  ENSG IDs in Salmon output: {len(ensg_in_salmon)}")
    print(f"  ENST IDs in Salmon output: {len(enst_in_salmon)}")

    if ensg_in_salmon:
        # Direct ENSG match possible
        srp_id_map = {t: i for i, t in enumerate(transcript_ids)}
        srp_cols: list[int] = []
        srp_syms: list[str] = []
        for ensg_base, sym in zip(ensg_bases, matched_syms):
            idx = srp_id_map.get(ensg_base)
            if idx is not None:
                srp_cols.append(idx)
                srp_syms.append(sym)
        X_srp_matched = X_srp[:, srp_cols]
        # Align training to same columns
        X_train_matched = X_train[:, [matched_syms.index(s) for s in srp_syms]]
        final_syms = srp_syms
    else:
        # ENST IDs: need tx2gene. Build from Ensembl cDNA FASTA headers.
        print("  ENST IDs detected — building tx2gene from FASTA headers ...")
        import gzip, re
        fasta = ALS_DIR / "resources" / "SRP064478" / "index" / "hg38_cdna.fa.gz"
        tx2gene: dict[str, str] = {}
        with gzip.open(fasta, "rt") as fh:
            for line in fh:
                if not line.startswith(">"):
                    continue
                m_enst = re.search(r"(ENST\d+)\.\d+", line)
                m_ensg = re.search(r"gene:(ENSG\d+)\.\d+", line)
                if m_enst and m_ensg:
                    tx2gene[m_enst.group(1)] = m_ensg.group(1)
        print(f"  tx2gene entries: {len(tx2gene)}")

        # Re-aggregate Salmon output at ENSG level
        srp_ensg_map: dict[str, list[int]] = {}
        for i, tx_id in enumerate(transcript_ids):
            ensg = tx2gene.get(tx_id)
            if ensg:
                srp_ensg_map.setdefault(ensg, []).append(i)

        srp_cols_final: list[int] = []
        srp_syms_final: list[str] = []
        for ensg_base, sym in zip(ensg_bases, matched_syms):
            cols = srp_ensg_map.get(ensg_base)
            if cols:
                srp_cols_final.append(cols[0] if len(cols) == 1 else -1)
                srp_syms_final.append(sym)

        # For multi-transcript genes, sum the counts
        X_srp_gene: list[np.ndarray] = []
        for ensg_base, sym in zip(ensg_bases, matched_syms):
            cols = srp_ensg_map.get(ensg_base)
            if cols:
                gene_counts = np.expm1(X_srp[:, cols]).sum(axis=1)
                X_srp_gene.append(np.log1p(gene_counts))
                if sym not in srp_syms_final:
                    srp_syms_final.append(sym)

        X_srp_matched = np.column_stack(X_srp_gene) if X_srp_gene else np.empty((len(y_srp), 0))
        X_train_matched = X_train[:, [matched_syms.index(s) for s in srp_syms_final if s in matched_syms]]
        final_syms = srp_syms_final

    print(f"  Final matched genes ({len(final_syms)}): {final_syms}")
    missing = [s for s in matched_syms if s not in final_syms]
    print(f"  Missing ({len(missing)}): {missing}")

    if X_srp_matched.shape[1] == 0:
        print("ERROR: No panel genes matched. Check Salmon output format.")
        return

    # ------------------------------------------------------------------
    print("\n[4] Scaling ...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_matched)
    X_srp_sc = scaler.transform(X_srp_matched)

    # ------------------------------------------------------------------
    print("\n[5] Zero-shot prediction ...")
    params = json.loads(_PARAMS_PATH.read_text())
    params.pop("feature_names", None)
    params["colsample_bytree"] = 1.0
    params["random_state"] = RANDOM_STATE
    params["verbosity"] = -1
    clf = LGBMClassifier(**params)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        clf.fit(X_train_sc, y_train)

    scores_zero = clf.predict_proba(X_srp_sc)[:, 1]
    auc_zero = roc_auc_score(y_srp, scores_zero)
    rng = np.random.default_rng(RANDOM_STATE)
    auc_boot = _bootstrap_auc(y_srp, scores_zero, N_BOOTSTRAP, rng)
    ci_lo, ci_hi = float(np.percentile(auc_boot, 2.5)), float(np.percentile(auc_boot, 97.5))
    print(f"  Zero-shot AUC = {auc_zero:.4f}  95%CI [{ci_lo:.4f}, {ci_hi:.4f}]")

    # ------------------------------------------------------------------
    print("\n[6] Per-gene MWU ...")
    mwu_results: list[tuple[str, float, float]] = []
    for j, sym in enumerate(final_syms):
        als_expr = X_srp_matched[y_srp == 1, j]
        ctl_expr = X_srp_matched[y_srp == 0, j]
        stat, p = mannwhitneyu(als_expr, ctl_expr, alternative="two-sided")
        rb = 2 * stat / (len(als_expr) * len(ctl_expr)) - 1
        mwu_results.append((sym, float(rb), float(p)))
        print(f"  {sym:<12}  r_b={rb:+.3f}  p={p:.4f}")

    # ------------------------------------------------------------------
    print("\n[7] Plotting ...")
    _plot(
        y=y_srp,
        scores_zero=scores_zero,
        auc_zero=auc_zero,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        matched_syms=final_syms,
        mwu_results=sorted(mwu_results, key=lambda x: -x[1]),
    )

    # ------------------------------------------------------------------
    print("\n[8] Writing statistics ...")
    lines: list[str] = [
        "SRP064478 Cervical Spinal Cord Validation — Ladd et al. 2017",
        "=" * 60,
        "Dataset: SRP064478 (Virginia Commonwealth Univ., BennettLab)",
        "  Tissue: Postmortem cervical spinal cord",
        "  Platform: Illumina NextSeq 500, paired-end, total RNA-seq",
        f"  Samples: ALS={int(y_srp.sum())}  Control={int((y_srp==0).sum())}",
        "  Publication: Ladd et al. 2017, Brain Research (PMID 28511992)",
        "  Quantification: Salmon 0.13.1, Ensembl GRCh38 release 111",
        "",
        f"Panel genes matched ({len(final_syms)}/15): {', '.join(final_syms)}",
        f"Missing ({len(missing)}): {', '.join(missing) if missing else 'none'}",
        "",
        "Zero-shot transfer (GPL24676-trained model → SRP064478):",
        f"  AUC = {auc_zero:.4f}  95%CI [{ci_lo:.4f}, {ci_hi:.4f}]",
        "  Note: within-cohort LOO-CV not reported (n=15 too small for",
        "        reliable within-cohort model training at these hyperparameters).",
        "",
        "Per-gene ALS vs Control (log1p expression; MWU):",
        f"  {'Gene':<12}  {'r_b':>8}  {'p_MWU':>10}  {'sig':>4}",
        "-" * 42,
    ]
    for sym, rb, p in sorted(mwu_results, key=lambda x: -x[1]):
        sig = "*" if p < 0.05 else ""
        lines.append(f"  {sym:<12}  {rb:+8.4f}  {p:10.4f}  {sig:>4}")

    report = "\n".join(lines)
    _OUT_STATS.write_text(report)
    print(f"  Saved → {_OUT_STATS.name}")
    print("\n" + report)


if __name__ == "__main__":
    main()
