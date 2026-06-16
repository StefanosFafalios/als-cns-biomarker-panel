"""
Additional Cohort Validation — GSE122649 (motor cortex, postmortem, n=38)

GSE122649 (Tam et al. 2019, Cell Reports):
  "Postmortem Cortex Samples Identify Distinct Molecular Subtypes of ALS:
   Retrotransporon Activation, Oxidative Stress, and Activated Glia"
  Platform: GPL18573 (Illumina NextSeq 500)
  Tissue: Motor cortex (postmortem)
  Samples: 22 sALS + 4 sALS/FTD + 12 Non-neurological controls
  Features: ENSG ID count files in GSE122649_RAW.tar

Two zero-shot conditions reported:
  A) StandardScaler, 17 natively matched genes
  B) StandardScaler, 25 genes (17 native + 8 de-duplicated replacements)

Replacement genes (from gene_replacement_results.csv, clean pool,
de-duplicated assignment sorted by |Pearson r| descending then AUC):
  HERC2P8         → HERC2P4   (ENSG00000230267)  |r|=0.872
  RPL21P75        → SLC6A18   (ENSG00000164363)
  MAP3K2-DT       → KRTAP6-2  (ENSG00000186930)
  ENSG00000280893 → LINC01318 (ENSG00000237790)
  RPL15P11        → C1orf185  (ENSG00000204006)
  RHOT1P2         → INSL6     (ENSG00000120210)
  ENSG00000279656 → LINC02556 (ENSG00000236611)
  LOC112268270    → PPP1R1B   (ENSG00000131771)  |r|=0.573

Within-cohort LOO-CV uses the native matched genes only (StandardScaler).
"""

from __future__ import annotations

import gzip
import sys
import tarfile
import time
import warnings
from pathlib import Path

import numpy as np
import requests

ALS_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset  # noqa: E402

SCRIPT_DIR = Path(__file__).parent

_RAW_TAR_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE122nnn/GSE122649/suppl/"
    "GSE122649_RAW.tar"
)
_SOFT_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE122nnn/GSE122649/soft/"
    "GSE122649_family.soft.gz"
)

_CACHE_DIR = ALS_DIR / "resources" / "GSE122649"
_RAW_TAR = _CACHE_DIR / "GSE122649_RAW.tar"
_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"
_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"

RANDOM_STATE = 42
N_BOOTSTRAP = 2_000

# De-duplicated clean surrogates (confirmed artifacts AND artifact gene
# families excluded), matching the canonical assignment in
# adaptive_panel_validation.py for this cohort. The keratin-associated artifact
# KRTAP6-2 is filtered, so MAP3K2-DT takes its next clean candidate (C1orf185)
# and the de-dup cascade reassigns RPL15P11/RHOT1P2/ENSG279656 accordingly.
REPLACEMENTS: dict[str, tuple[str, str]] = {
    "HERC2P8": ("HERC2P4", "ENSG00000230267"),  # |r|=0.872
    "RPL21P75": ("SLC6A18", "ENSG00000164363"),
    "MAP3K2-DT": ("C1orf185", "ENSG00000204006"),
    "ENSG00000280893": ("LINC01318", "ENSG00000237790"),
    "RPL15P11": ("INSL6", "ENSG00000120210"),
    "RHOT1P2": ("LINC02556", "ENSG00000236611"),
    "ENSG00000279656": ("OAS2", "ENSG00000111335"),
    "LOC112268270": ("PPP1R1B", "ENSG00000131771"),  # |r|=0.573
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download_with_retry(url: str, dest: Path, desc: str = "") -> None:
    """Download url to dest with retry and progress."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            print(f"  Downloading {desc or url.split('/')[-1]} ...")
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = 100 * downloaded / total
                        print(f"    {pct:.0f}%", end="\r", flush=True)
            print(f"    {downloaded / 1e6:.1f} MB saved to {dest}")
            return
        except Exception as e:
            if attempt == 2:
                raise
            print(f"  Retry {attempt + 1}/3 after error: {e}")
            time.sleep(5)


# ---------------------------------------------------------------------------
# SOFT metadata parser
# ---------------------------------------------------------------------------


def _parse_soft() -> dict[str, str]:
    """Return {GSM_accession: diagnosis} from soft file."""
    import re

    print("  Parsing sample metadata ...")
    r = requests.get(_SOFT_URL, timeout=60, stream=True)
    content = gzip.decompress(r.content).decode("utf-8", errors="ignore")
    samples = re.split(r"\^SAMPLE", content)[1:]
    meta: dict[str, str] = {}
    for s in samples:
        acc_m = re.search(r"!Sample_geo_accession = (GSM\d+)", s)
        diag_m = re.search(r"!Sample_characteristics_ch1 = diagnosis: (.+)", s)
        if acc_m and diag_m:
            meta[acc_m.group(1)] = diag_m.group(1).strip()
    print(
        f"  Metadata: {len(meta)} samples — {dict((v, list(meta.values()).count(v)) for v in set(meta.values()))}"
    )
    return meta


# ---------------------------------------------------------------------------
# Count matrix extraction from RAW.tar
# ---------------------------------------------------------------------------


def _extract_counts(
    tar_path: Path, gsm_to_diag: dict[str, str]
) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    """
    Extract per-sample HTSeq/TEtools count files from RAW.tar.
    Returns (X_counts, gene_symbols, gsm_ids, y_labels).
    """
    import re

    print("  Extracting count files from RAW.tar ...")
    gene_symbols: list[str] = []
    sample_counts: dict[str, np.ndarray] = {}

    with tarfile.open(tar_path, "r") as tf:
        members = tf.getmembers()
        print(f"  Files in archive: {len(members)}")
        for member in members:
            fname = member.name
            gsm_m = re.search(r"(GSM\d+)", fname)
            if not gsm_m:
                continue
            gsm = gsm_m.group(1)
            if gsm not in gsm_to_diag:
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            raw = f.read()
            if fname.endswith(".gz"):
                raw = gzip.decompress(raw)
            text = raw.decode("utf-8", errors="ignore")
            lines = text.splitlines()
            rows = []
            for line in lines[1:]:
                line = line.strip()
                if not line or line.startswith("__"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                try:
                    count = float(parts[1])
                except ValueError:
                    continue
                symbol = parts[0].strip('"').strip("'")
                rows.append((symbol, count))
            if not rows:
                continue
            syms = [r[0] for r in rows]
            counts = np.array([r[1] for r in rows], dtype=np.float32)
            if not gene_symbols:
                gene_symbols = syms
            elif len(syms) != len(gene_symbols):
                print(f"  WARNING: {gsm} gene count mismatch, skipping")
                continue
            sample_counts[gsm] = counts

    if not sample_counts:
        raise RuntimeError("No count files parsed from RAW.tar.")

    gsm_ids = list(sample_counts.keys())
    X = np.stack([sample_counts[g] for g in gsm_ids], axis=0)
    diag_map = {"sALS": 1, "sALS/FTD": 1, "Non-neurological control": 0}
    y = np.array([diag_map.get(gsm_to_diag[g], -1) for g in gsm_ids])
    valid = y >= 0
    gsm_ids = [g for g, v in zip(gsm_ids, valid) if v]
    X = X[valid]
    y = y[valid]
    print(f"  Matrix: {X.shape} | ALS: {int(y.sum())} | Control: {int((y == 0).sum())}")
    return X, gene_symbols, gsm_ids, y


# ---------------------------------------------------------------------------
# Training data loader — returns raw (unlogged) values
# ---------------------------------------------------------------------------


def _load_train() -> tuple[np.ndarray, list[str], np.ndarray]:
    """Load GPL24676 training data as raw (unlogged) RSEM counts."""
    (ds,) = load_dataset(
        "GSE153960", platform="GPL24676", resources_dir=ALS_DIR / "resources"
    )
    return ds.X.values.astype(np.float32), list(ds.X.columns), ds.y.values.astype(int)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bootstrap_auc(
    y_true: np.ndarray, y_score: np.ndarray, n: int = 2000, seed: int = 42
) -> tuple[float, float]:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    aucs_arr = np.array(aucs)
    return float(np.percentile(aucs_arr, 2.5)), float(np.percentile(aucs_arr, 97.5))


def _zeroshot(X_tr_sc, y_tr, X_te_sc, y_te, params):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    m = lgb.LGBMClassifier(**params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X_tr_sc, y_tr)
    prob = m.predict_proba(X_te_sc)[:, 1]
    auc = roc_auc_score(y_te, prob)
    ci_lo, ci_hi = _bootstrap_auc(y_te, prob, n=N_BOOTSTRAP)
    return auc, ci_lo, ci_hi, prob


def main() -> None:
    import json

    import lightgbm as lgb
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.preprocessing import StandardScaler

    print("=" * 65)
    print("Additional Cohort Validation — GSE122649")
    print("Tam et al. 2019 | Motor Cortex | Postmortem | n=38")
    print("=" * 65)

    if not _RAW_TAR.exists():
        print("\n[1] Downloading GSE122649_RAW.tar ...")
        _download_with_retry(_RAW_TAR_URL, _RAW_TAR, "GSE122649_RAW.tar")
    else:
        print(f"\n[1] Using cached RAW.tar ({_RAW_TAR.stat().st_size / 1e6:.0f} MB)")

    print("\n[2] Parsing sample metadata ...")
    gsm_to_diag = _parse_soft()

    print("\n[3] Extracting count matrix ...")
    X_test_raw, gene_ids, gsm_ids, y_test = _extract_counts(_RAW_TAR, gsm_to_diag)

    print("\n[4] Loading GPL24676 training data ...")
    X_train_raw, train_ensg, y_train = _load_train()

    panel_df = pd.read_csv(_PANEL_CSV)
    sym_col = next((c for c in panel_df.columns if "symbol" in c.lower()), None)
    ensg_col = next(
        (c for c in panel_df.columns if "ensg" in c.lower() or "ENSG" in c), None
    )
    if ensg_col is None:
        ensg_col = panel_df.columns[1]
    panel_ensg: list[str] = panel_df[ensg_col].tolist()
    panel_symbols: list[str] = (
        panel_df[sym_col].tolist() if sym_col else [""] * len(panel_ensg)
    )

    test_sym_map = {s: i for i, s in enumerate(gene_ids)}
    train_ensg_base_map = {g.split(".")[0]: i for i, g in enumerate(train_ensg)}

    # ---------------------------------------------------------------------------
    # Gene matching
    # ---------------------------------------------------------------------------
    print("\n[5] Matching panel genes ...")

    native_train_cols: list[int] = []
    native_test_cols: list[int] = []
    native_genes: list[str] = []
    missing_labels: list[str] = []

    for ensg, sym in zip(panel_ensg, panel_symbols):
        tr_col = train_ensg_base_map.get(ensg.split(".")[0], -1)
        if tr_col < 0:
            continue
        te_col = test_sym_map.get(sym, -1) if sym else -1
        label = sym if sym else ensg.split(".")[0]
        if te_col >= 0:
            native_train_cols.append(tr_col)
            native_test_cols.append(te_col)
            native_genes.append(label)
        else:
            missing_labels.append(label)

    # Extended: add replacements for missing genes
    repl_train_cols: list[int] = list(native_train_cols)
    repl_test_cols: list[int] = list(native_test_cols)
    repl_genes: list[str] = list(native_genes)
    repl_pairs: list[str] = []

    for label in missing_labels:
        if label not in REPLACEMENTS:
            continue
        repl_sym, repl_ensg = REPLACEMENTS[label]
        tr_col = train_ensg_base_map.get(repl_ensg.split(".")[0], -1)
        te_col = test_sym_map.get(repl_sym, -1)
        if tr_col >= 0 and te_col >= 0:
            repl_train_cols.append(tr_col)
            repl_test_cols.append(te_col)
            repl_genes.append(f"{label}→{repl_sym}")
            repl_pairs.append(f"{label} → {repl_sym} ({repl_ensg})")

    n_repl = len(repl_train_cols) - len(native_train_cols)
    print(f"  Native matches  : {len(native_train_cols)}/25")
    print(
        f"  With replacements: {len(repl_train_cols)}/25 ({len(native_train_cols)} native + {n_repl} replacements)"
    )

    if len(native_train_cols) < 5:
        print("ERROR: Fewer than 5 panel genes matched.")
        return

    params = json.loads(_PARAMS_PATH.read_text())
    params.update(
        {
            "colsample_bytree": 1.0,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1,
        }
    )

    # ---------------------------------------------------------------------------
    # Condition A: native genes, StandardScaler
    # ---------------------------------------------------------------------------
    X_tr_A = np.log1p(X_train_raw[:, native_train_cols])
    X_te_A = np.log1p(X_test_raw[:, native_test_cols])
    sc_A = StandardScaler()
    X_tr_A_sc = sc_A.fit_transform(X_tr_A)
    X_te_A_sc = sc_A.transform(X_te_A)
    auc_A, lo_A, hi_A, prob_A = _zeroshot(X_tr_A_sc, y_train, X_te_A_sc, y_test, params)

    # ---------------------------------------------------------------------------
    # Condition B: native + replacements, StandardScaler
    # ---------------------------------------------------------------------------
    X_tr_B = np.log1p(X_train_raw[:, repl_train_cols])
    X_te_B = np.log1p(X_test_raw[:, repl_test_cols])
    sc_B = StandardScaler()
    X_tr_B_sc = sc_B.fit_transform(X_tr_B)
    X_te_B_sc = sc_B.transform(X_te_B)
    auc_B, lo_B, hi_B, prob_B = _zeroshot(X_tr_B_sc, y_train, X_te_B_sc, y_test, params)

    print("\n[6] Zero-shot transfer ...")
    print(
        f"  A) {len(native_train_cols):2d} native genes : AUC = {auc_A:.4f}  [{lo_A:.3f}, {hi_A:.3f}]"
    )
    print(
        f"  B) {len(repl_train_cols):2d} genes (+repl) : AUC = {auc_B:.4f}  [{lo_B:.3f}, {hi_B:.3f}]"
    )

    # ---------------------------------------------------------------------------
    # Within-cohort LOO-CV: native genes, StandardScaler
    # ---------------------------------------------------------------------------
    from sklearn.model_selection import LeaveOneOut

    loo_params = {**params, "min_child_samples": 1}
    X_te_native_log = np.log1p(X_test_raw[:, native_test_cols].astype(np.float32))
    loo_probs = np.zeros(len(y_test))
    for lo_idx, va_idx in LeaveOneOut().split(X_te_native_log):
        X_lo, X_va = X_te_native_log[lo_idx], X_te_native_log[va_idx]
        sc = StandardScaler()
        X_lo_sc = sc.fit_transform(X_lo)
        X_va_sc = sc.transform(X_va)
        m = lgb.LGBMClassifier(**loo_params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X_lo_sc, y_lo := y_test[lo_idx])
        loo_probs[va_idx] = m.predict_proba(X_va_sc)[:, 1]

    loo_auc = roc_auc_score(y_test, loo_probs)
    loo_lo, loo_hi = _bootstrap_auc(y_test, loo_probs, n=N_BOOTSTRAP)
    print(f"\n[7] Within-cohort LOO-CV ({len(native_train_cols)} native genes) ...")
    print(f"  AUC = {loo_auc:.4f}  [{loo_lo:.3f}, {loo_hi:.3f}]")

    # ---------------------------------------------------------------------------
    # Plot: 3 panels
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def _roc_ax(ax, y, prob, auc, lo, hi, color, title):
        fpr, tpr, _ = roc_curve(y, prob)
        ax.plot(
            fpr,
            tpr,
            color=color,
            linewidth=2,
            label=f"AUC = {auc:.4f}\n95% CI [{lo:.3f}, {hi:.3f}]",
        )
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

    _roc_ax(
        axes[0],
        y_test,
        prob_A,
        auc_A,
        lo_A,
        hi_A,
        "#7B1FA2",
        f"Zero-Shot A: {len(native_train_cols)} native genes\nStandardScaler",
    )
    _roc_ax(
        axes[1],
        y_test,
        prob_B,
        auc_B,
        lo_B,
        hi_B,
        "#388E3C",
        f"Zero-Shot B: {len(repl_train_cols)} genes (+{n_repl} replacements)\nStandardScaler",
    )
    _roc_ax(
        axes[2],
        y_test,
        loo_probs,
        loo_auc,
        loo_lo,
        loo_hi,
        "#1565C0",
        f"LOO-CV: {len(native_train_cols)} native genes\n(within-cohort only)",
    )

    plt.tight_layout()
    plt.savefig(
        SCRIPT_DIR / "additional_cohort_gse122649.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    # ---------------------------------------------------------------------------
    # Statistics file
    # ---------------------------------------------------------------------------
    repl_lines = [f"  {p}" for p in repl_pairs]
    lines = [
        "Additional Cohort Validation — GSE122649 (Tam et al. 2019)",
        "=" * 65,
        "Dataset: GSE122649 | Motor cortex, postmortem",
        "Platform: GPL18573 (Illumina NextSeq 500)",
        f"Samples: {len(y_test)} (ALS: {int(y_test.sum())}, Control: {int((y_test == 0).sum())})",
        f"Panel genes matched natively: {len(native_train_cols)}/25",
        f"Panel genes with replacements: {len(repl_train_cols)}/25",
        "",
        "Replacement gene mapping:",
        *repl_lines,
        "",
        "ZERO-SHOT VALIDATION (panel trained on GPL24676, n=874)",
        "-" * 65,
        f"Condition A — StandardScaler, {len(native_train_cols)} native genes:",
        f"  AUC-ROC : {auc_A:.4f}  95% CI [{lo_A:.4f}, {hi_A:.4f}]",
        f"Condition B — StandardScaler, {len(repl_train_cols)} genes (+ {n_repl} replacements):",
        f"  AUC-ROC : {auc_B:.4f}  95% CI [{lo_B:.4f}, {hi_B:.4f}]",
        "",
        "WITHIN-COHORT LOO-CV (trained and evaluated on GSE122649 only)",
        "-" * 65,
        f"Native genes ({len(native_train_cols)}), StandardScaler:",
        f"  AUC-ROC : {loo_auc:.4f}  95% CI [{loo_lo:.4f}, {loo_hi:.4f}]",
        "",
        f"Native matched genes: {', '.join(native_genes)}",
        f"All matched genes (native + replacements): {', '.join(repl_genes)}",
    ]
    (SCRIPT_DIR / "additional_cohort_gse122649_statistics.txt").write_text(
        "\n".join(lines)
    )
    print(
        "\nDone. Outputs: additional_cohort_gse122649.png  additional_cohort_gse122649_statistics.txt"
    )


if __name__ == "__main__":
    main()
