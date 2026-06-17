"""Panel variant CV and zero-shot comparison.

For each of the 25 panel genes, creates a variant where that gene is swapped
for its best non-artifact replacement, then evaluates:

  1. 5-fold stratified CV on GPL24676 (fast model, n_estimators=100)
  2. Zero-shot AUC on GSE76220  (if replacement has symbol present in cohort)
  3. Zero-shot AUC on GSE122649 (if replacement has symbol present in cohort)

Also identifies the best *protein-coding* replacement per panel gene —
the highest-AUC non-artifact candidate with biotype="protein-coding".
Protein-coding replacements are the primary druggability candidates since
many panel genes are lncRNAs, pseudogenes, or poorly characterised.

Baseline: original 25-gene panel evaluated with the same fast model.

Outputs
-------
  panel_variant_cv.png
  panel_variant_cv_statistics.txt
"""

from __future__ import annotations

import gzip
import json
import re
import sys
import tarfile
import time
import warnings
from pathlib import Path

import numpy as np

ALS_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

SCRIPT_DIR   = Path(__file__).parent
ALS_DIR_RES  = ALS_DIR / "resources"

_PREFILTER_X     = SCRIPT_DIR / "lgbm_prefilter_X.npy"
_PREFILTER_NAMES = SCRIPT_DIR / "lgbm_prefilter_names.txt"
_PANEL_CSV       = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PARAMS_PATH     = SCRIPT_DIR / "lgbm_top500_best_params.json"
_REPL_CSV        = SCRIPT_DIR / "gene_replacement_results.csv"
_AUDIT_CSV       = SCRIPT_DIR / "shap500_artifact_audit.csv"

RANDOM_STATE = 42
N_FOLDS      = 5
FAST_N_EST   = 100
PANEL_CV_AUC = 0.9621  # reference AUC for full 25-gene panel (full model, 5-fold CV)

_GSE122649_RAW_TAR = ALS_DIR_RES / "GSE122649" / "GSE122649_RAW.tar"
_SOFT_URL_122649 = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE122nnn/GSE122649/soft/"
    "GSE122649_family.soft.gz"
)
_RAW_URL_122649 = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE122nnn/GSE122649/suppl/"
    "GSE122649_RAW.tar"
)


# ---------------------------------------------------------------------------
# Replacement lookup and gene info
# ---------------------------------------------------------------------------


def _build_replacement_lookup(
    repl_csv: Path,
    audit_csv: Path,
) -> dict[str, list[tuple[float, str]]]:
    """panel_symbol → [(auc, ensg_base), ...] sorted desc, non-artifact only."""
    repl = pd.read_csv(repl_csv)
    audit = pd.read_csv(audit_csv)
    artifact_bases = set(
        audit.loc[audit["classification"] == "confirmed_artifact", "ensg_base"]
    )
    repl["candidate_base"] = repl["candidate_ensg"].str.split(".").str[0]
    clean = repl[
        repl["is_replacement"] & ~repl["candidate_base"].isin(artifact_bases)
    ]
    lookup: dict[str, list[tuple[float, str]]] = {}
    for sym, grp in clean.groupby("panel_symbol"):
        grp_s = grp.sort_values("replacement_auc", ascending=False)
        lookup[str(sym)] = list(zip(
            grp_s["replacement_auc"].tolist(),
            grp_s["candidate_base"].tolist(),
        ))
    return lookup


def _build_gene_info(
    audit_csv: Path,
    ensg_bases: set[str],
) -> dict[str, dict]:
    """Return {ensg_base → {symbol, biotype}} for all requested bases.

    Tries audit CSV first, then MyGeneInfo for the remainder.
    biotype values: 'protein-coding', 'lncRNA', 'pseudogene', 'other', ''
    """
    audit = pd.read_csv(audit_csv)
    info: dict[str, dict] = {}

    for _, row in audit.iterrows():
        base = str(row["ensg_base"])
        if base in ensg_bases:
            sym = str(row.get("symbol", "") or "").strip()
            btype = str(row.get("type_of_gene", "") or "").strip()
            info[base] = {
                "symbol":  sym if sym != "nan" else "",
                "biotype": btype if btype != "nan" else "",
            }

    unresolved = [b for b in ensg_bases if b not in info]
    if unresolved:
        print(f"  MyGeneInfo lookup for {len(unresolved)} unresolved ENSGs ...")
        for i in range(0, len(unresolved), 500):
            batch = unresolved[i : i + 500]
            try:
                r = requests.post(
                    "https://mygene.info/v3/gene",
                    json={
                        "ids": batch,
                        "fields": "symbol,type_of_gene",
                        "species": "human",
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=60,
                )
                r.raise_for_status()
                for item in r.json():
                    qid = item.get("query", "")
                    if qid:
                        info[qid] = {
                            "symbol":  item.get("symbol", ""),
                            "biotype": item.get("type_of_gene", ""),
                        }
            except Exception as exc:
                print(f"  WARNING: MyGeneInfo failed: {exc}")
            time.sleep(0.25)
        # Fill anything still missing
        for b in unresolved:
            if b not in info:
                info[b] = {"symbol": "", "biotype": ""}

    return info


def _find_best_and_best_coding(
    ranked: list[tuple[float, str]],
    gene_info: dict[str, dict],
) -> tuple[tuple[float, str] | None, tuple[float, str] | None]:
    """Return (best_overall, best_protein_coding) from a ranked list."""
    best = ranked[0] if ranked else None
    best_coding = None
    for auc, base in ranked:
        btype = gene_info.get(base, {}).get("biotype", "")
        if btype == "protein-coding":
            best_coding = (auc, base)
            break
    return best, best_coding


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _load_prefilter() -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    """lgbm_prefilter_X.npy → (X, ensg_base_to_col, y)."""
    names = _PREFILTER_NAMES.read_text().splitlines()
    base_to_col = {n.split(".")[0]: i for i, n in enumerate(names)}
    X = np.load(_PREFILTER_X, mmap_mode="r")
    (ds,) = load_dataset("GSE153960", platform="GPL24676", resources_dir=ALS_DIR_RES)
    y = ds.y.values.astype(int)
    return X, base_to_col, y


def _load_train_raw() -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    """GPL24676 log1p RSEM for zero-shot training → (X, ensg_base_to_col, y)."""
    (ds,) = load_dataset("GSE153960", platform="GPL24676", resources_dir=ALS_DIR_RES)
    X = np.log1p(ds.X.values.astype(np.float32))
    base_to_col = {n.split(".")[0]: i for i, n in enumerate(ds.X.columns)}
    return X, base_to_col, ds.y.values.astype(int)


def _load_gse76220() -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    """GSE76220 log1p RPKM → (X, sym_to_col, y)."""
    (ds,) = load_dataset("GSE76220", resources_dir=ALS_DIR_RES)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        X = np.log1p(ds.X.values.astype(np.float32))
    sym_to_col = {s: i for i, s in enumerate(ds.X.columns)}
    return X, sym_to_col, ds.y.values.astype(int)


def _download_if_needed(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {dest.name} ...")
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    print(f"  Saved {dest.stat().st_size / 1e6:.0f} MB")


def _load_gse122649() -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    """GSE122649 log1p raw counts → (X, sym_to_col, y)."""
    _download_if_needed(_RAW_URL_122649, _GSE122649_RAW_TAR)
    r = requests.get(_SOFT_URL_122649, timeout=60, stream=True)
    content = gzip.decompress(r.content).decode("utf-8", errors="ignore")
    samples = re.split(r"\^SAMPLE", content)[1:]
    gsm_to_diag: dict[str, str] = {}
    for s in samples:
        acc_m = re.search(r"!Sample_geo_accession = (GSM\d+)", s)
        diag_m = re.search(r"!Sample_characteristics_ch1 = diagnosis: (.+)", s)
        if acc_m and diag_m:
            gsm_to_diag[acc_m.group(1)] = diag_m.group(1).strip()

    gene_symbols: list[str] = []
    sample_counts: dict[str, np.ndarray] = {}
    with tarfile.open(_GSE122649_RAW_TAR, "r") as tf:
        for member in tf.getmembers():
            gsm_m = re.search(r"(GSM\d+)", member.name)
            if not gsm_m or gsm_m.group(1) not in gsm_to_diag:
                continue
            gsm = gsm_m.group(1)
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            raw = fobj.read()
            if member.name.endswith(".gz"):
                raw = gzip.decompress(raw)
            lines = raw.decode("utf-8", errors="ignore").splitlines()
            rows = []
            for line in lines[1:]:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                try:
                    rows.append((parts[0].strip('"\''), float(parts[1])))
                except ValueError:
                    continue
            if not rows:
                continue
            syms = [r[0] for r in rows]
            counts = np.array([r[1] for r in rows], dtype=np.float32)
            if not gene_symbols:
                gene_symbols = syms
            elif len(syms) != len(gene_symbols):
                continue
            sample_counts[gsm] = counts

    gsm_ids = list(sample_counts.keys())
    X_raw = np.stack([sample_counts[g] for g in gsm_ids], axis=0)
    diag_map = {"sALS": 1, "sALS/FTD": 1, "Non-neurological control": 0}
    y = np.array([diag_map.get(gsm_to_diag[g], -1) for g in gsm_ids])
    valid = y >= 0
    X_log = np.log1p(X_raw[valid].astype(np.float32))
    sym_to_col = {s: i for i, s in enumerate(gene_symbols)}
    return X_log, sym_to_col, y[valid]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _fast_cv_auc(
    X_pre: np.ndarray,
    y: np.ndarray,
    col_indices: list[int],
    params: dict,
) -> tuple[float, float]:
    """5-fold stratified CV on prefilter matrix columns col_indices.

    Returns (mean_auc, std_auc).
    """
    X = np.asarray(X_pre[:, col_indices], dtype=np.float32)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs: list[float] = []
    for tr, te in skf.split(X, y):
        clf = LGBMClassifier(**params)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            clf.fit(X[tr], y[tr])
        aucs.append(float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])))
    return float(np.mean(aucs)), float(np.std(aucs))


def _zeroshot_auc(
    X_train_log: np.ndarray,
    y_train: np.ndarray,
    X_test_log: np.ndarray,
    y_test: np.ndarray,
    tr_cols: list[int],
    te_cols: list[int],
    params: dict,
) -> float:
    """Zero-shot: train on GPL24676, predict on external cohort."""
    X_tr = X_train_log[:, tr_cols].astype(np.float32)
    X_te = X_test_log[:, te_cols].astype(np.float32)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    clf = LGBMClassifier(**{**params, "verbose": -1})
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        clf.fit(X_tr_sc, y_train)
    return float(roc_auc_score(y_test, clf.predict_proba(X_te_sc)[:, 1]))


# ---------------------------------------------------------------------------
# Build variant list
# ---------------------------------------------------------------------------


def _build_variants(
    panel_df: pd.DataFrame,
    replacement_lookup: dict[str, list[tuple[float, str]]],
    gene_info: dict[str, dict],
    prefilter_base_to_col: dict[str, int],
    train_base_to_col: dict[str, int],
    cohort_sym_to_cols: dict[str, dict[str, int]],
) -> list[dict]:
    """Build a list of variant descriptors.

    Each variant has:
      panel_symbol   : gene being replaced
      repl_ensg      : replacement ENSG base
      repl_symbol    : replacement HGNC symbol (may be empty)
      repl_biotype   : protein-coding / lncRNA / etc.
      repl_auc_train : AUC of replacement in GPL24676 context (from gene_replacement_results)
      kind           : 'best_clean' or 'best_coding'
      pre_cols       : column indices in prefilter matrix for 25-gene variant
      train_cols     : column indices in GPL24676 log1p matrix for 25-gene variant
      cohort_cols    : {cohort_name → column indices} for external cohorts
      cohort_has_repl: {cohort_name → bool} whether replacement is in cohort
    """
    # Base (original panel) column indices
    panel_pre_cols: list[int] = []
    panel_train_cols: list[int] = []
    panel_cohort_te: dict[str, list[int]] = {name: [] for name in cohort_sym_to_cols}
    panel_cohort_has: dict[str, bool] = {name: True for name in cohort_sym_to_cols}

    for _, row in panel_df.iterrows():
        feat = str(row["feature"])
        sym = str(row["symbol"])
        base = feat.split(".")[0]
        panel_pre_cols.append(prefilter_base_to_col[base])
        panel_train_cols.append(train_base_to_col[base])
        for cname, sym_to_col in cohort_sym_to_cols.items():
            panel_cohort_te[cname].append(sym_to_col.get(sym, -1))

    for cname in cohort_sym_to_cols:
        panel_cohort_has[cname] = all(c >= 0 for c in panel_cohort_te[cname])

    variants: list[dict] = []

    for idx, row in panel_df.iterrows():
        sym = str(row["symbol"])
        feat = str(row["feature"])
        orig_base = feat.split(".")[0]

        if sym not in replacement_lookup:
            continue

        ranked = replacement_lookup[sym]
        best, best_coding = _find_best_and_best_coding(ranked, gene_info)

        for kind, candidate in [("best_clean", best), ("best_coding", best_coding)]:
            if candidate is None:
                continue
            repl_auc, repl_base = candidate
            gi = gene_info.get(repl_base, {})
            repl_sym = gi.get("symbol", "")
            repl_bio = gi.get("biotype", "")

            # Prefilter columns: swap gene at idx
            pre_cols = list(panel_pre_cols)
            if repl_base not in prefilter_base_to_col:
                continue  # replacement not in prefilter matrix
            pre_cols[idx] = prefilter_base_to_col[repl_base]

            # Train (log1p) columns: swap
            train_cols = list(panel_train_cols)
            if repl_base not in train_base_to_col:
                continue
            train_cols[idx] = train_base_to_col[repl_base]

            # Cohort test columns: use replacement if present, else keep original slot
            cohort_te: dict[str, list[int]] = {}
            cohort_has: dict[str, bool] = {}  # True = replacement actually used in test
            for cname, sym_to_col in cohort_sym_to_cols.items():
                te_cols = list(panel_cohort_te[cname])
                if repl_sym and repl_sym in sym_to_col:
                    te_cols[idx] = sym_to_col[repl_sym]
                    cohort_has[cname] = True
                else:
                    cohort_has[cname] = False  # original gene kept in this slot
                cohort_te[cname] = te_cols

            variants.append({
                "panel_symbol":    sym,
                "repl_ensg":       repl_base,
                "repl_symbol":     repl_sym,
                "repl_biotype":    repl_bio,
                "repl_auc_train":  repl_auc,
                "kind":            kind,
                "pre_cols":        pre_cols,
                "train_cols":      train_cols,
                "cohort_te":       cohort_te,
                "cohort_has":      cohort_has,
            })

    return variants, {
        "pre_cols":   panel_pre_cols,
        "train_cols": panel_train_cols,
        "cohort_te":  panel_cohort_te,
        "cohort_has": panel_cohort_has,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _make_figure(results_df: pd.DataFrame) -> None:
    """Horizontal grouped bar chart: CV AUC for best_clean and best_coding variants."""
    baseline = PANEL_CV_AUC

    best_clean = results_df[results_df["kind"] == "best_clean"].set_index("panel_symbol")
    best_coding = results_df[results_df["kind"] == "best_coding"].set_index("panel_symbol")

    genes = results_df["panel_symbol"].unique().tolist()

    fig, axes = plt.subplots(1, 2, figsize=(18, 10), sharey=False)

    for ax, df_sub, title, colour in [
        (axes[0], best_clean,  "Best clean replacement (any biotype)", "#4A90D9"),
        (axes[1], best_coding, "Best protein-coding replacement",       "#E05C5C"),
    ]:
        rows = [(g, df_sub.loc[g, "cv_auc"] if g in df_sub.index else float("nan"))
                for g in genes]
        rows_valid = [(g, a) for g, a in rows if not np.isnan(a)]
        rows_sorted = sorted(rows_valid, key=lambda x: x[1])

        syms = [r[0] for r in rows_sorted]
        aucs = [r[1] for r in rows_sorted]
        y_pos = np.arange(len(syms))
        deltas = [a - baseline for a in aucs]
        bar_colours = ["#d62728" if d < -0.005 else "#2ca02c" if d > 0.005 else colour
                       for d in deltas]

        ax.barh(y_pos, aucs, color=bar_colours, alpha=0.8)
        ax.axvline(baseline, color="black", lw=1.5, ls="--", label=f"Baseline {baseline:.4f}")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(syms, fontsize=7.5)
        ax.set_xlabel("5-fold CV AUC (fast model)", fontsize=9)
        ax.set_xlim(0.85, 1.00)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="x", color="#EEEEEE", lw=0.6)

        # Annotate with Δ AUC
        for i, (a, d) in enumerate(zip(aucs, deltas)):
            ax.text(a + 0.001, i, f"{d:+.4f}", va="center", fontsize=6.5,
                    color="#333333")

    fig.suptitle(
        "Panel variant CV AUC: single-gene swap with best non-artifact replacement\n"
        "Green = variant ≥ baseline + 0.005 · Red = variant ≤ baseline − 0.005",
        fontsize=9,
    )
    plt.tight_layout()
    out = SCRIPT_DIR / "panel_variant_cv.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {out.name}")


# ---------------------------------------------------------------------------
# Statistics file
# ---------------------------------------------------------------------------


def _write_stats(
    baseline_cv: tuple[float, float],
    baseline_zs: dict[str, float],
    results_df: pd.DataFrame,
) -> None:
    lines = [
        "Panel Variant CV and Zero-Shot Comparison",
        "=" * 80,
        f"Fast model: n_estimators={FAST_N_EST}, 5-fold stratified CV, random_state={RANDOM_STATE}",
        f"Artifact-free pool: 41 confirmed artifacts excluded.",
        "",
        "BASELINE (original 25-gene panel)",
        "-" * 80,
        f"  Fast-model 5-fold CV AUC : {baseline_cv[0]:.4f} ± {baseline_cv[1]:.4f}",
    ]
    for cname, auc in baseline_zs.items():
        lines.append(f"  Zero-shot {cname:<12}: {auc:.4f}")
    lines += [
        f"  (Reference full-model CV AUC: {PANEL_CV_AUC:.4f})",
        "",
    ]

    for kind, kind_label in [
        ("best_clean",  "BEST CLEAN REPLACEMENT (any biotype, highest AUC)"),
        ("best_coding", "BEST PROTEIN-CODING REPLACEMENT"),
    ]:
        sub = results_df[results_df["kind"] == kind].sort_values("cv_delta")
        lines += [
            kind_label,
            "-" * 80,
            f"{'Gene':<22} {'Replacement':<18} {'Biotype':<16} {'ReplAUC':>8}"
            f"  {'CV AUC':>8}  {'Δ CV':>7}",
            "-" * 80,
        ]
        for _, r in sub.iterrows():
            repl_label = (r["repl_symbol"] or r["repl_ensg"])[:17]
            bio = (r["repl_biotype"] or "?")[:15]
            cv = f"{r['cv_auc']:.4f}" if not np.isnan(r["cv_auc"]) else "  N/A "
            delta = f"{r['cv_delta']:+.4f}" if not np.isnan(r["cv_delta"]) else "  N/A "
            lines.append(
                f"{r['panel_symbol']:<22} {repl_label:<18} {bio:<16}"
                f"  {r['repl_auc_train']:>8.4f}  {cv:>8}  {delta:>7}"
            )
            for cname in [c for c in r.index if c.startswith("zs_") and
                          not c.endswith("_repl_in_cohort")]:
                cohort = cname[3:]
                auc = r[cname]
                in_cohort = r.get(f"{cname}_repl_in_cohort", False)
                if not np.isnan(auc):
                    b_auc = baseline_zs.get(cohort, float("nan"))
                    delta_zs = auc - b_auc if not np.isnan(b_auc) else float("nan")
                    marker = "* repl tested" if in_cohort else "~ orig tested"
                    lines.append(
                        f"    {cohort:<20} zero-shot={auc:.4f}  Δ={delta_zs:+.4f}  ({marker})"
                    )
        lines.append("")

    # Summary: best variants overall
    best_sub = results_df[results_df["kind"] == "best_clean"].nlargest(5, "cv_auc")
    lines += [
        "TOP 5 VARIANTS BY CV AUC (best clean replacement)",
        "-" * 80,
    ]
    for _, r in best_sub.iterrows():
        repl_label = r["repl_symbol"] or r["repl_ensg"]
        lines.append(
            f"  Swap {r['panel_symbol']:<18} → {repl_label:<20} "
            f"CV={r['cv_auc']:.4f} (Δ{r['cv_delta']:+.4f})"
        )

    out = SCRIPT_DIR / "panel_variant_cv_statistics.txt"
    out.write_text("\n".join(lines))
    print(f"Statistics saved: {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 80)
    print("Panel Variant CV and Zero-Shot Comparison")
    print("=" * 80)

    panel_df = pd.read_csv(_PANEL_CSV).reset_index(drop=True)
    base_params = json.loads(_PARAMS_PATH.read_text())
    fast_params = {
        **base_params,
        "n_estimators": FAST_N_EST,
        "colsample_bytree": 1.0,
        "verbose": -1,
        "n_jobs": 1,
    }
    zs_params = {**fast_params, "n_estimators": 300, "n_jobs": -1}

    # Build replacement lookup
    print("\n[1] Building clean replacement lookup ...")
    replacement_lookup = _build_replacement_lookup(_REPL_CSV, _AUDIT_CSV)
    n_entries = sum(len(v) for v in replacement_lookup.values())
    print(f"  {n_entries} clean replacement candidate entries across {len(replacement_lookup)} panel genes")

    # Collect all unique candidate bases for gene info lookup
    all_bases: set[str] = {
        base for cands in replacement_lookup.values() for _, base in cands
    }
    print(f"\n[2] Resolving gene info for {len(all_bases)} unique replacement candidates ...")
    gene_info = _build_gene_info(_AUDIT_CSV, all_bases)
    n_coding = sum(1 for v in gene_info.values() if v.get("biotype") == "protein-coding")
    n_sym = sum(1 for v in gene_info.values() if v.get("symbol"))
    print(f"  {n_sym}/{len(gene_info)} have HGNC symbol, {n_coding} protein-coding")

    # Load prefilter matrix for fast CV
    print("\n[3] Loading prefilter matrix ...")
    X_pre, pre_base_to_col, y_pre = _load_prefilter()
    print(f"  X_pre: {X_pre.shape}")

    # Load GPL24676 log1p for zero-shot training
    print("\n[4] Loading GPL24676 log1p for zero-shot training ...")
    X_tr_log, tr_base_to_col, y_tr = _load_train_raw()
    print(f"  n={len(y_tr)}  ALS={int(y_tr.sum())}  Ctrl={int((y_tr==0).sum())}")

    # Load external cohorts
    print("\n[5] Loading GSE76220 ...")
    X_76220, sym_76220, y_76220 = _load_gse76220()
    print(f"  n={len(y_76220)}")

    print("\n[6] Loading GSE122649 ...")
    X_122649, sym_122649, y_122649 = _load_gse122649()
    print(f"  n={len(y_122649)}")

    cohort_data = {
        "GSE76220":  (X_76220,  sym_76220,  y_76220),
        "GSE122649": (X_122649, sym_122649, y_122649),
    }
    cohort_sym_to_cols = {
        "GSE76220":  sym_76220,
        "GSE122649": sym_122649,
    }

    # Build variant list
    print("\n[7] Building variant panel list ...")
    variants, baseline_info = _build_variants(
        panel_df, replacement_lookup, gene_info,
        pre_base_to_col, tr_base_to_col, cohort_sym_to_cols,
    )
    print(f"  {len(variants)} variants to evaluate")

    # Evaluate baseline
    print("\n[8] Evaluating baseline (original 25-gene panel) ...")
    baseline_pre_cols = baseline_info["pre_cols"]
    baseline_cv_auc, baseline_cv_std = _fast_cv_auc(X_pre, y_pre, baseline_pre_cols, fast_params)
    print(f"  Baseline fast-model CV AUC: {baseline_cv_auc:.4f} ± {baseline_cv_std:.4f}")

    baseline_zs: dict[str, float] = {}
    for cname, (X_te, sym_map, y_te) in cohort_data.items():
        # Use whatever cohort columns are available (skip slots with -1)
        tr_cols = [c for c in baseline_info["train_cols"] if c >= 0]
        te_cols_raw = baseline_info["cohort_te"][cname]
        # Match: only use slots where BOTH train and test have valid columns
        paired = [(tc, ec) for tc, ec in zip(baseline_info["train_cols"], te_cols_raw)
                  if tc >= 0 and ec >= 0]
        if len(paired) < 5:
            baseline_zs[cname] = float("nan")
            continue
        tr_c, te_c = zip(*paired)
        auc = _zeroshot_auc(X_tr_log, y_tr, X_te, y_te, list(tr_c), list(te_c), zs_params)
        baseline_zs[cname] = auc
        print(f"  Baseline zero-shot {cname}: {auc:.4f}")

    # Evaluate all variants
    print(f"\n[9] Evaluating {len(variants)} variants ...")
    records: list[dict] = []
    for i, v in enumerate(variants):
        sym = v["panel_symbol"]
        kind = v["kind"]
        print(f"  [{i+1:3d}/{len(variants)}] {sym} ({kind}) → {v['repl_symbol'] or v['repl_ensg']}", end="  ", flush=True)

        cv_auc, cv_std = _fast_cv_auc(X_pre, y_pre, v["pre_cols"], fast_params)
        print(f"CV={cv_auc:.4f}", end="", flush=True)

        rec: dict = {
            "panel_symbol":   sym,
            "repl_ensg":      v["repl_ensg"],
            "repl_symbol":    v["repl_symbol"],
            "repl_biotype":   v["repl_biotype"],
            "repl_auc_train": v["repl_auc_train"],
            "kind":           kind,
            "cv_auc":         cv_auc,
            "cv_std":         cv_std,
            "cv_delta":       cv_auc - baseline_cv_auc,
        }

        for cname, (X_te, sym_map, y_te) in cohort_data.items():
            paired = [(tc, ec) for tc, ec in zip(v["train_cols"], v["cohort_te"][cname])
                      if tc >= 0 and ec >= 0]
            if len(paired) < 5:
                rec[f"zs_{cname}"] = float("nan")
                rec[f"zs_{cname}_repl_in_cohort"] = False
                continue
            tr_c, te_c = zip(*paired)
            zs_auc = _zeroshot_auc(X_tr_log, y_tr, X_te, y_te, list(tr_c), list(te_c), zs_params)
            rec[f"zs_{cname}"] = zs_auc
            rec[f"zs_{cname}_repl_in_cohort"] = v["cohort_has"].get(cname, False)
            marker = "*" if v["cohort_has"].get(cname, False) else "~"
            print(f"  ZS_{cname}={marker}{zs_auc:.4f}", end="", flush=True)

        print()
        records.append(rec)

    results_df = pd.DataFrame(records)

    print("\n[10] Generating figure and statistics ...")
    _make_figure(results_df)
    _write_stats(
        (baseline_cv_auc, baseline_cv_std),
        baseline_zs,
        results_df,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
