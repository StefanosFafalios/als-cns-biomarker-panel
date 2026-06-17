"""Regenerate srp064478_validation.png with canonical 25-gene and 15-crit results.

Replicates the SRP064478 loading and gene-matching logic from lr_vs_lgbm_zeroshot.py
and produces a two-panel figure:
  Left  : ROC curves for 25-gene (21/25) and 15-crit (14/15) LGBM panels.
  Right : Per-gene Mann-Whitney rank-biserial r_b (ALS vs Control) for the
          15-crit genes available in SRP064478 (14 genes, SNORD97 absent
          from the Salmon Ensembl-cDNA index).
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).parent
ALS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset  # noqa: E402

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"
_SRP_QUANT_DIR = ALS_DIR / "resources" / "SRP064478" / "quant"
_SRP_META = ALS_DIR / "resources" / "SRP064478" / "srr_metadata.tsv"
_SRP_FASTA = ALS_DIR / "resources" / "SRP064478" / "index" / "hg38_cdna.fa.gz"
_OUT_PNG = SCRIPT_DIR / "manuscript" / "srp064478_validation.png"

RANDOM_STATE = 42
N_BOOTSTRAP = 2_000
CRITICAL_IDX: list[int] = [1, 2, 3, 5, 6, 7, 10, 11, 12, 15, 16, 18, 20, 23, 24]


def _bootstrap_auc(
    y: np.ndarray, scores: np.ndarray, n: int = N_BOOTSTRAP, seed: int = RANDOM_STATE,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(float(roc_auc_score(y[idx], scores[idx])))
    a = np.array(aucs)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def _safe_scale(sc: StandardScaler, X: np.ndarray) -> np.ndarray:
    return np.nan_to_num(sc.transform(X), nan=0.0)


def _load_srp064478() -> tuple[np.ndarray, list[str], np.ndarray]:
    meta_df = pd.read_csv(_SRP_META, sep="\t").set_index("SRR")
    tx2gene: dict[str, str] = {}
    with gzip.open(str(_SRP_FASTA), "rt") as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            m_enst = re.search(r"(ENST\d+)", line)
            m_ensg = re.search(r"gene:(ENSG\d+)", line)
            if m_enst and m_ensg:
                tx2gene[m_enst.group(1)] = m_ensg.group(1)
    gene_counts: dict[str, dict[str, float]] = {}
    y_list: list[int] = []
    for srr in meta_df.index:
        sf = _SRP_QUANT_DIR / srr / "quant.sf"
        df = pd.read_csv(sf, sep="\t")
        df["gene_base"] = df["Name"].str.split(".").str[0].map(tx2gene)
        df = df.dropna(subset=["gene_base"])
        sums = df.groupby("gene_base")["NumReads"].sum()
        gene_counts[srr] = sums.to_dict()
        y_list.append(1 if meta_df.loc[srr, "condition"] == "ALS" else 0)
    all_genes = sorted({g for d in gene_counts.values() for g in d})
    X_raw = np.array(
        [[gene_counts[srr].get(g, 0.0) for g in all_genes] for srr in meta_df.index],
        dtype=np.float32,
    )
    return np.log1p(X_raw), all_genes, np.array(y_list, dtype=int)


def _train_predict(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, params: dict,
) -> np.ndarray:
    from lightgbm import LGBMClassifier
    sc = StandardScaler().fit(X_tr)
    clf = LGBMClassifier(**params)
    clf.fit(_safe_scale(sc, X_tr), y_tr)
    return clf.predict_proba(_safe_scale(sc, X_te))[:, 1]


def main() -> None:
    # Load panel definition
    panel_df = pd.read_csv(_PANEL_CSV)
    feat25_bases = [e.split(".")[0] for e in panel_df["feature"]]
    feat25_syms = list(panel_df["symbol"])
    base_crit = [feat25_bases[i] for i in CRITICAL_IDX]
    sym_crit = [feat25_syms[i] for i in CRITICAL_IDX]

    params = json.loads(_PARAMS_PATH.read_text())
    params.update({"colsample_bytree": 1.0, "n_jobs": -1, "verbose": -1,
                   "random_state": RANDOM_STATE})

    # Load GPL24676 training (log1p only, no CTD — same as lr_vs_lgbm_zeroshot for SRP)
    print("Loading GPL24676 ...")
    (ds,) = load_dataset("GSE153960", platform="GPL24676", resources_dir=ALS_DIR / "resources")
    X_train_log = np.log1p(ds.X.values.astype(np.float32))
    y_train = ds.y.values.astype(int)
    feat_train = list(ds.X.columns)
    feat_bases = [f.split(".")[0] for f in feat_train]
    b2i_tr = {b: i for i, b in enumerate(feat_bases)}

    # Load SRP064478
    print("Loading SRP064478 ...")
    Xsrp_log, vocab_srp, y_srp = _load_srp064478()
    v_srp = {v: i for i, v in enumerate(vocab_srp)}

    # Match 25-gene panel
    m25_tr = [b2i_tr[b] for b in feat25_bases if b in b2i_tr]
    m25_te_idx = [i for i, b in enumerate(feat25_bases) if b in v_srp]
    m25_te_cols = [v_srp[feat25_bases[i]] for i in m25_te_idx]
    m25_tr_cols = [b2i_tr[feat25_bases[i]] for i in m25_te_idx]
    n_25 = len(m25_te_idx)

    # Match 15-crit panel
    m16_te_idx = [i for i, b in enumerate(base_crit) if b in v_srp]
    m16_te_cols = [v_srp[base_crit[i]] for i in m16_te_idx]
    m16_tr_cols = [b2i_tr[base_crit[i]] for i in m16_te_idx if base_crit[i] in b2i_tr]
    matched_crit_syms = [sym_crit[i] for i in m16_te_idx]
    n_16 = len(m16_te_idx)

    print(f"  25-gene: {n_25}/25 matched; 15-crit: {n_16}/15 matched")
    print(f"  15-crit matched genes: {matched_crit_syms}")

    Xtr_25 = X_train_log[:, m25_tr_cols]
    Xte_25 = Xsrp_log[:, m25_te_cols]

    Xtr_16 = X_train_log[:, m16_tr_cols]
    Xte_16 = Xsrp_log[:, m16_te_cols]

    # Train and predict
    print("Training 25-gene LGBM ...")
    sc25 = StandardScaler()
    scores_25 = _train_predict(Xtr_25, y_train, Xte_25, params)
    auc_25 = float(roc_auc_score(y_srp, scores_25))
    ci_lo_25, ci_hi_25 = _bootstrap_auc(y_srp, scores_25)
    fpr_25, tpr_25, _ = roc_curve(y_srp, scores_25)
    print(f"  AUC={auc_25:.4f}  95%CI [{ci_lo_25:.4f}, {ci_hi_25:.4f}]")

    print("Training 15-crit LGBM ...")
    scores_16 = _train_predict(Xtr_16, y_train, Xte_16, params)
    auc_16 = float(roc_auc_score(y_srp, scores_16))
    ci_lo_16, ci_hi_16 = _bootstrap_auc(y_srp, scores_16)
    fpr_16, tpr_16, _ = roc_curve(y_srp, scores_16)
    print(f"  AUC={auc_16:.4f}  95%CI [{ci_lo_16:.4f}, {ci_hi_16:.4f}]")

    # Per-gene Mann-Whitney for 15-crit matched genes
    als_mask = y_srp == 1
    ctrl_mask = y_srp == 0
    rb_vals, pvals, syms_plot = [], [], []
    for j, sym in enumerate(matched_crit_syms):
        g_als = Xsrp_log[als_mask, m16_te_cols[j]]
        g_ctrl = Xsrp_log[ctrl_mask, m16_te_cols[j]]
        stat, pval = mannwhitneyu(g_als, g_ctrl, alternative="two-sided")
        n1, n2 = len(g_als), len(g_ctrl)
        rb = 1 - 2 * stat / (n1 * n2)
        rb_vals.append(rb)
        pvals.append(pval)
        syms_plot.append(sym if sym and not sym.startswith("ENSG") else sym[:15])

    order = np.argsort(rb_vals)
    rb_sorted = [rb_vals[i] for i in order]
    sym_sorted = [syms_plot[i] for i in order]
    pval_sorted = [pvals[i] for i in order]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(fpr_25, tpr_25, lw=2, color="#1f77b4",
            label=f"25-gene ({n_25}/25)  AUC = {auc_25:.4f}\n"
                  f"95% CI [{ci_lo_25:.3f}, {ci_hi_25:.3f}]")
    ax.plot(fpr_16, tpr_16, lw=2, color="#d62728",
            label=f"15-crit ({n_16}/15)  AUC = {auc_16:.4f}\n"
                  f"95% CI [{ci_lo_16:.3f}, {ci_hi_16:.3f}]")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("SRP064478 zero-shot ROC\n(7 ALS vs 8 controls; Cervical SC; Salmon)")
    ax.legend(fontsize=8, loc="lower right")

    ax2 = axes[1]
    colours = ["#d62728" if r > 0 else "#1f77b4" for r in rb_sorted]
    bars = ax2.barh(range(len(sym_sorted)), rb_sorted, color=colours)
    for k, (r, p) in enumerate(zip(rb_sorted, pval_sorted)):
        if p < 0.05:
            ax2.text(r + 0.01 * np.sign(r), k, "*", va="center", ha="left", fontsize=9)
    ax2.set_yticks(range(len(sym_sorted)))
    ax2.set_yticklabels(sym_sorted, fontsize=8)
    ax2.axvline(0, color="k", lw=0.8)
    ax2.set_xlabel("Rank-biserial $r_b$ (ALS vs Control)")
    ax2.set_title(f"Per-gene effect size — 15-crit panel\n({n_16}/15 genes matched)")
    ax2.set_xlim(-1.05, 1.05)

    fig.suptitle("SRP064478 Cervical SC validation ($n = 15$)", fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(_OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {_OUT_PNG}")


if __name__ == "__main__":
    main()
