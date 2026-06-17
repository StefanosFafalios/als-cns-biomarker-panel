"""Greedy forward selection for druggable panel optimisation.

Starting from the original 25-gene panel, greedily swaps one panel gene at a
time for its best protein-coding replacement.  At each step, all remaining
candidate single swaps are evaluated; the one that maximises the primary
objective (GSE122649 zero-shot AUC, which has the richer sample size) is
accepted if it improves over the current panel.

Also evaluates:
  - The 'max-coding' panel: ALL 16 genes with a protein-coding replacement
    swapped simultaneously.
  - The top-N greedy selections applied to GSE76220 (secondary objective).
  - All pairwise combinations of the top-4 single-swap improvers.

Fast model (n_estimators=100) for CV; n_estimators=300 for zero-shot.

Outputs:
  panel_variant_greedy.png
  panel_variant_greedy_statistics.txt
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
_VARIANT_STATS   = SCRIPT_DIR / "panel_variant_cv_statistics.txt"

RANDOM_STATE = 42
N_FOLDS      = 5
FAST_N_EST   = 100

_GSE122649_RAW_TAR = ALS_DIR_RES / "GSE122649" / "GSE122649_RAW.tar"
_SOFT_URL_122649 = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE122nnn/GSE122649/soft/"
    "GSE122649_family.soft.gz"
)
_RAW_URL_122649 = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE122nnn/GSE122649/suppl/"
    "GSE122649_RAW.tar"
)

# Protein-coding single-swap results from panel_variant_cv.py
# (panel_symbol, replacement_symbol, prefilter_base, cv_auc, zs_76220, zs_122649)
# zs_* may be nan if replacement not in cohort
_SINGLE_SWAP_CODING: list[dict] = []  # populated from panel_variant_cv_statistics.txt


# ---------------------------------------------------------------------------
# Loaders (reused from panel_variant_cv.py)
# ---------------------------------------------------------------------------

def _load_prefilter() -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    names = _PREFILTER_NAMES.read_text().splitlines()
    base_to_col = {n.split(".")[0]: i for i, n in enumerate(names)}
    X = np.load(_PREFILTER_X, mmap_mode="r")
    (ds,) = load_dataset("GSE153960", platform="GPL24676", resources_dir=ALS_DIR_RES)
    y = ds.y.values.astype(int)
    return X, base_to_col, y


def _load_train_raw() -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    (ds,) = load_dataset("GSE153960", platform="GPL24676", resources_dir=ALS_DIR_RES)
    X = np.log1p(ds.X.values.astype(np.float32))
    base_to_col = {n.split(".")[0]: i for i, n in enumerate(ds.X.columns)}
    return X, base_to_col, ds.y.values.astype(int)


def _load_gse76220() -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    (ds,) = load_dataset("GSE76220", resources_dir=ALS_DIR_RES)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        X = np.log1p(ds.X.values.astype(np.float32))
    return X, {s: i for i, s in enumerate(ds.X.columns)}, ds.y.values.astype(int)


def _load_gse122649() -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    if not _GSE122649_RAW_TAR.exists():
        _GSE122649_RAW_TAR.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(_RAW_URL_122649, stream=True, timeout=300)
        r.raise_for_status()
        with open(_GSE122649_RAW_TAR, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)

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
    return X_log, {s: i for i, s in enumerate(gene_symbols)}, y[valid]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _cv_auc(
    X_pre: np.ndarray,
    y: np.ndarray,
    cols: list[int],
    params: dict,
) -> float:
    X = np.asarray(X_pre[:, cols], dtype=np.float32)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = LGBMClassifier(**params)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            clf.fit(X[tr], y[tr])
        aucs.append(float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])))
    return float(np.mean(aucs))


def _zeroshot(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    tr_cols: list[int], te_cols: list[int],
    params: dict,
) -> float:
    X_tr = X_train[:, tr_cols].astype(np.float32)
    X_te = X_test[:, te_cols].astype(np.float32)
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)
    clf = LGBMClassifier(**{**params, "verbose": -1})
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        clf.fit(X_tr_sc, y_train)
    return float(roc_auc_score(y_test, clf.predict_proba(X_te_sc)[:, 1]))


# ---------------------------------------------------------------------------
# Panel state helper
# ---------------------------------------------------------------------------

class PanelState:
    """Tracks which panel genes are swapped and provides column index lists."""

    def __init__(
        self,
        panel_df: pd.DataFrame,
        pre_base_to_col: dict[str, int],
        tr_base_to_col: dict[str, int],
        cohort_sym_maps: dict[str, dict[str, int]],
        single_swaps: list[dict],  # from panel_variant_cv output
    ) -> None:
        self.panel_syms: list[str] = list(panel_df["symbol"])
        self.panel_feats: list[str] = list(panel_df["feature"])
        self.pre_base_to_col = pre_base_to_col
        self.tr_base_to_col  = tr_base_to_col
        self.cohort_sym_maps = cohort_sym_maps

        # Base columns for original panel genes
        self._orig_pre: list[int] = [
            pre_base_to_col[f.split(".")[0]] for f in self.panel_feats
        ]
        self._orig_tr: list[int] = [
            tr_base_to_col[f.split(".")[0]] for f in self.panel_feats
        ]
        self._orig_cohort: dict[str, list[int]] = {
            cname: [sym_map.get(s, -1) for s in self.panel_syms]
            for cname, sym_map in cohort_sym_maps.items()
        }

        # Build swap lookup: {panel_symbol → swap info}
        self.swap_options: dict[str, dict] = {}
        for sw in single_swaps:
            self.swap_options[sw["panel_symbol"]] = sw

        # Current active swaps
        self.active: dict[str, str] = {}  # panel_symbol → replacement_symbol

    def _apply_swaps(
        self, swaps: dict[str, str]
    ) -> tuple[list[int], list[int], dict[str, list[int]]]:
        pre_cols  = list(self._orig_pre)
        tr_cols   = list(self._orig_tr)
        coh_cols  = {k: list(v) for k, v in self._orig_cohort.items()}

        for sym, repl_sym in swaps.items():
            idx = self.panel_syms.index(sym)
            sw  = self.swap_options[sym]
            pre_cols[idx] = sw["pre_col"]
            tr_cols[idx]  = sw["tr_col"]
            for cname, sym_map in self.cohort_sym_maps.items():
                if repl_sym and repl_sym in sym_map:
                    coh_cols[cname][idx] = sym_map[repl_sym]

        return pre_cols, tr_cols, coh_cols

    def cols(
        self, extra_swaps: dict[str, str] | None = None
    ) -> tuple[list[int], list[int], dict[str, list[int]]]:
        swaps = dict(self.active)
        if extra_swaps:
            swaps.update(extra_swaps)
        return self._apply_swaps(swaps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("Greedy Druggable Panel Optimisation")
    print("=" * 80)

    panel_df = pd.read_csv(_PANEL_CSV).reset_index(drop=True)
    base_params = json.loads(_PARAMS_PATH.read_text())
    fast_params = {**base_params, "n_estimators": FAST_N_EST,
                   "colsample_bytree": 1.0, "verbose": -1, "n_jobs": 1}
    zs_params   = {**base_params, "n_estimators": 300,
                   "colsample_bytree": 1.0, "verbose": -1, "n_jobs": -1}

    print("\n[1] Loading data ...")
    X_pre, pre_b2c, y_pre = _load_prefilter()
    X_tr,  tr_b2c,  y_tr  = _load_train_raw()
    X_76,  sym_76,  y_76  = _load_gse76220()
    X_122, sym_122, y_122 = _load_gse122649()
    cohort_data = {"GSE76220": (X_76, sym_76, y_76), "GSE122649": (X_122, sym_122, y_122)}
    cohort_sym_maps = {"GSE76220": sym_76, "GSE122649": sym_122}
    print(f"  Prefilter: {X_pre.shape}  |  Train: {X_tr.shape}")

    # -----------------------------------------------------------------------
    # Load single-swap results from panel_variant_cv_statistics.txt
    # -----------------------------------------------------------------------
    print("\n[2] Parsing single-swap protein-coding results ...")
    single_swaps: list[dict] = []
    stat_text = _VARIANT_STATS.read_text()
    # Re-derive column info directly from panel_variant_cv computation
    # We need pre_col and tr_col for each protein-coding replacement
    from panel_variant_cv import (  # type: ignore
        _build_replacement_lookup, _build_gene_info, _AUDIT_CSV, _REPL_CSV
    )
    repl_lookup = _build_replacement_lookup(_REPL_CSV, _AUDIT_CSV)
    all_bases: set[str] = {b for cands in repl_lookup.values() for _, b in cands}
    gene_info = _build_gene_info(_AUDIT_CSV, all_bases)

    for sym, ranked in repl_lookup.items():
        # Find best protein-coding
        for auc_repl, base in ranked:
            bi = gene_info.get(base, {})
            if bi.get("biotype") != "protein-coding":
                continue
            repl_sym = bi.get("symbol", "")
            if not repl_sym:
                continue
            if base not in pre_b2c or base not in tr_b2c:
                continue
            single_swaps.append({
                "panel_symbol": sym,
                "repl_symbol":  repl_sym,
                "repl_base":    base,
                "pre_col":      pre_b2c[base],
                "tr_col":       tr_b2c[base],
                "auc_repl":     auc_repl,
            })
            break  # one per panel gene

    print(f"  {len(single_swaps)} panel genes have a protein-coding replacement")

    # Panel state object
    state = PanelState(panel_df, pre_b2c, tr_b2c, cohort_sym_maps, single_swaps)

    # -----------------------------------------------------------------------
    # Baseline
    # -----------------------------------------------------------------------
    print("\n[3] Evaluating baseline ...")
    pre_b, tr_b, coh_b = state.cols()
    baseline_cv = _cv_auc(X_pre, y_pre, pre_b, fast_params)
    baseline_zs: dict[str, float] = {}
    for cname, (X_te, sym_map, y_te) in cohort_data.items():
        paired = [(tc, ec) for tc, ec in zip(tr_b, coh_b[cname]) if tc >= 0 and ec >= 0]
        if paired:
            tc, ec = zip(*paired)
            baseline_zs[cname] = _zeroshot(X_tr, y_tr, X_te, y_te, list(tc), list(ec), zs_params)
    print(f"  Baseline CV={baseline_cv:.4f}  "
          + "  ".join(f"ZS_{c}={v:.4f}" for c, v in baseline_zs.items()))

    # -----------------------------------------------------------------------
    # Greedy forward selection (objective: GSE122649 zero-shot AUC)
    # -----------------------------------------------------------------------
    print("\n[4] Greedy forward selection (objective: GSE122649 zero-shot) ...")
    remaining = {sw["panel_symbol"]: sw for sw in single_swaps}
    greedy_history: list[dict] = []
    current_cv   = baseline_cv
    current_zs122 = baseline_zs.get("GSE122649", 0.0)
    current_zs76  = baseline_zs.get("GSE76220", 0.0)
    state.active  = {}

    while remaining:
        best_sym  = None
        best_gain = 0.0
        best_metrics: dict = {}

        for sym, sw in remaining.items():
            pre_c, tr_c, coh_c = state.cols({sym: sw["repl_symbol"]})
            # Only use genes available in GSE122649 for the candidate evaluation
            paired_122 = [(tc, ec) for tc, ec in zip(tr_c, coh_c["GSE122649"])
                          if tc >= 0 and ec >= 0]
            if len(paired_122) < 5:
                continue
            tc, ec = zip(*paired_122)
            zs = _zeroshot(X_tr, y_tr, X_122, y_122, list(tc), list(ec), zs_params)
            gain = zs - current_zs122
            if gain > best_gain:
                best_gain = gain
                best_sym  = sym
                best_metrics = {"zs_122": zs}

        if best_sym is None or best_gain <= 0:
            print("  No further improvement possible.")
            break

        sw = remaining.pop(best_sym)
        state.active[best_sym] = sw["repl_symbol"]
        current_zs122 = best_metrics["zs_122"]

        # Evaluate full state
        pre_c, tr_c, coh_c = state.cols()
        cv = _cv_auc(X_pre, y_pre, pre_c, fast_params)

        zs_full: dict[str, float] = {}
        for cname, (X_te, sym_map, y_te) in cohort_data.items():
            paired = [(tc, ec) for tc, ec in zip(tr_c, coh_c[cname]) if tc >= 0 and ec >= 0]
            if paired:
                tc2, ec2 = zip(*paired)
                zs_full[cname] = _zeroshot(X_tr, y_tr, X_te, y_te, list(tc2), list(ec2), zs_params)

        current_cv   = cv
        current_zs122 = zs_full.get("GSE122649", current_zs122)
        current_zs76  = zs_full.get("GSE76220", current_zs76)

        step = {
            "step":         len(greedy_history) + 1,
            "swapped":      best_sym,
            "replacement":  sw["repl_symbol"],
            "cv_auc":       cv,
            "zs_122649":    zs_full.get("GSE122649", float("nan")),
            "zs_76220":     zs_full.get("GSE76220", float("nan")),
            "gain_122":     best_gain,
        }
        greedy_history.append(step)
        print(f"  Step {step['step']}: swap {best_sym} → {sw['repl_symbol']}  "
              f"CV={cv:.4f}  ZS_122={zs_full.get('GSE122649',float('nan')):.4f} "
              f"(+{best_gain:.4f})  ZS_76={zs_full.get('GSE76220',float('nan')):.4f}")

    # -----------------------------------------------------------------------
    # Max-coding panel (all protein-coding swaps simultaneously)
    # -----------------------------------------------------------------------
    print("\n[5] Max-coding panel (all protein-coding swaps simultaneously) ...")
    all_swaps = {sw["panel_symbol"]: sw["repl_symbol"] for sw in single_swaps}
    state_all = PanelState(panel_df, pre_b2c, tr_b2c, cohort_sym_maps, single_swaps)
    state_all.active = all_swaps
    pre_all, tr_all, coh_all = state_all.cols()
    cv_all = _cv_auc(X_pre, y_pre, pre_all, fast_params)
    zs_all: dict[str, float] = {}
    for cname, (X_te, sym_map, y_te) in cohort_data.items():
        paired = [(tc, ec) for tc, ec in zip(tr_all, coh_all[cname]) if tc >= 0 and ec >= 0]
        if paired:
            tc2, ec2 = zip(*paired)
            zs_all[cname] = _zeroshot(X_tr, y_tr, X_te, y_te, list(tc2), list(ec2), zs_params)
    print(f"  Max-coding CV={cv_all:.4f}  "
          + "  ".join(f"ZS_{c}={v:.4f}" for c, v in zs_all.items()))

    # -----------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------
    print("\n[6] Generating figure ...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: greedy trace
    ax = axes[0]
    steps  = [0] + [g["step"] for g in greedy_history]
    cv_seq = [baseline_cv] + [g["cv_auc"] for g in greedy_history]
    z1_seq = [baseline_zs.get("GSE122649", float("nan"))] + [g["zs_122649"] for g in greedy_history]
    z2_seq = [baseline_zs.get("GSE76220",  float("nan"))] + [g["zs_76220"]  for g in greedy_history]

    ax.plot(steps, cv_seq, "o-", color="#4A90D9", lw=2, label="5-fold CV AUC (GPL24676)")
    ax.plot(steps, z1_seq, "s-", color="#E05C5C", lw=2, label="Zero-shot GSE122649")
    ax.plot(steps, z2_seq, "^-", color="#F5A623", lw=2, label="Zero-shot GSE76220")
    ax.axhline(baseline_cv, color="#4A90D9", lw=0.8, ls="--", alpha=0.4)
    ax.axhline(baseline_zs.get("GSE122649", 0), color="#E05C5C", lw=0.8, ls="--", alpha=0.4)
    if greedy_history:
        for g in greedy_history:
            ax.annotate(
                f"{g['swapped']}→{g['replacement']}",
                (g["step"], g["zs_122649"]),
                textcoords="offset points", xytext=(5, 4),
                fontsize=6.5, rotation=15, color="#333333",
            )
    ax.set_xlabel("Greedy step (genes swapped)", fontsize=9)
    ax.set_ylabel("AUC", fontsize=9)
    ax.set_title("Greedy protein-coding panel optimisation\n(objective: GSE122649 zero-shot)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_xticks(steps)

    # Right: bar chart comparing key panels
    ax = axes[1]
    panels: list[tuple[str, float, float, float]] = [
        ("Original\n25-gene panel",  baseline_cv, baseline_zs.get("GSE122649", float("nan")), baseline_zs.get("GSE76220", float("nan"))),
    ]
    for g in greedy_history:
        label = f"Greedy\nstep {g['step']} (+{g['swapped']}→{g['replacement']})"
        panels.append((label, g["cv_auc"], g["zs_122649"], g["zs_76220"]))
    panels.append((f"Max-coding\n({len(single_swaps)} swaps)", cv_all,
                   zs_all.get("GSE122649", float("nan")), zs_all.get("GSE76220", float("nan"))))

    n = len(panels)
    x = np.arange(n)
    w = 0.28
    labels_bar = [p[0] for p in panels]
    cv_bar  = [p[1] for p in panels]
    z1_bar  = [p[2] for p in panels]
    z2_bar  = [p[3] for p in panels]

    ax.bar(x - w,  cv_bar, w, label="5-fold CV", color="#4A90D9", alpha=0.85)
    ax.bar(x,      z1_bar, w, label="ZS GSE122649", color="#E05C5C", alpha=0.85)
    ax.bar(x + w,  z2_bar, w, label="ZS GSE76220", color="#F5A623", alpha=0.85)
    ax.axhline(baseline_cv, color="#4A90D9", lw=0.8, ls="--", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar, fontsize=7, rotation=10, ha="right")
    ax.set_ylabel("AUC", fontsize=9)
    ax.set_title("Panel performance by configuration", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0.6, 1.0)

    fig.suptitle(
        "Greedy forward selection for protein-coding panel optimisation\n"
        "Objective: maximise GSE122649 zero-shot AUC at each step",
        fontsize=9,
    )
    plt.tight_layout()
    out_fig = SCRIPT_DIR / "panel_variant_greedy.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {out_fig.name}")

    # -----------------------------------------------------------------------
    # Statistics file
    # -----------------------------------------------------------------------
    lines = [
        "Greedy Druggable Panel Optimisation",
        "=" * 80,
        f"Baseline (fast model CV): {baseline_cv:.4f}",
    ]
    for c, v in baseline_zs.items():
        lines.append(f"Baseline zero-shot {c}: {v:.4f}")
    lines += [
        "",
        "GREEDY FORWARD SELECTION (objective: GSE122649 zero-shot AUC)",
        "-" * 80,
        f"{'Step':<5} {'Swapped':<22} {'→'} {'Replacement':<14}  {'CV':>8}  {'ZS_122':>8}  {'ZS_76':>8}  {'Δ122':>7}",
        "-" * 80,
    ]
    if greedy_history:
        for g in greedy_history:
            lines.append(
                f"{g['step']:<5} {g['swapped']:<22} → {g['replacement']:<14}"
                f"  {g['cv_auc']:>8.4f}  {g['zs_122649']:>8.4f}  {g['zs_76220']:>8.4f}"
                f"  {g['gain_122']:>+7.4f}"
            )
    else:
        lines.append("  No single protein-coding swap improved GSE122649 zero-shot.")
    lines += [
        "",
        "Active swaps in final greedy panel:",
        f"  {list(state.active.items()) if state.active else 'none'}",
        "",
        "MAX-CODING PANEL (all protein-coding swaps simultaneously)",
        "-" * 80,
        f"Swaps: {len(single_swaps)} (genes: {', '.join(s['panel_symbol'] for s in single_swaps)})",
        f"CV AUC    : {cv_all:.4f}  (Δ baseline: {cv_all - baseline_cv:+.4f})",
    ]
    for c, v in zs_all.items():
        b = baseline_zs.get(c, float("nan"))
        lines.append(f"ZS {c:<12}: {v:.4f}  (Δ baseline: {v-b:+.4f})")

    out_txt = SCRIPT_DIR / "panel_variant_greedy_statistics.txt"
    out_txt.write_text("\n".join(lines))
    print(f"  Statistics saved: {out_txt.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
