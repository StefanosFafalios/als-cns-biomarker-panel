"""Cross-cohort LOO on the 15-gene protein-coding replacement panel.

The replacement panel is the minimum transferable protein-coding panel with
the two weakest members swapped out for their greedy-identified replacements:

  SOHLH2 → KIF2A   (SOHLH2: most redundant; LOO Δ = -0.0018 within main cohort)
  VWF    → TM4SF1  (VWF: atypical cross-platform transfer; best swap for GSE76220)

This gives a clean 15-gene panel where every member is protein-coding, and
both new genes were already validated as part of the 4-swap greedy panel.

For each of the 15 panel genes, drops it and evaluates the 14-gene reduced
panel on two independent external cohorts:

  - GSE76220  (lumbar SC LCM,   n=20)
  - GSE122649 (motor cortex,    n=38)

Two evaluation conditions per cohort per removed gene:
  A) Zero-shot: retrain on GPL24676 using cohort-available genes; evaluate ZS.
  B) Within-cohort LOO-CV (min_child_samples=1).

Summary metric:
  Weighted mean AUC = (20 * AUC_76 + 38 * AUC_122) / 58
  Δ = weighted mean (14-gene) − weighted mean (full 15-gene baseline)
  Positive Δ → gene is dispensable; Negative Δ → gene is irreplaceable.

Outputs:
  replacement_panel_loo.txt  — results table
  replacement_panel_loo.png  — ZS and LOO Δ bar charts
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
ALS_DIR = SCRIPT_DIR.parent

sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset  # noqa: E402

from additional_cohort_gse122649 import (  # noqa: E402
    _RAW_TAR,
    _RAW_TAR_URL,
    _download_with_retry,
    _extract_counts,
    _parse_soft,
)

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"

RANDOM_STATE = 42
N_76220 = 20
N_122649 = 38
TOTAL_N = N_76220 + N_122649

# Original 15 protein-coding panel members (symbols)
_PROTEIN_CODING_PANEL = {
    "MECOM", "SERTAD1", "FCN3", "PROS1", "ANGPT2", "EMP1", "TINAGL1",
    "CKMT2", "VWF", "CLDN5", "NR4A1", "SOHLH2", "HEXB", "MCEE", "SLC37A2",
}

# Greedy replacement mappings (symbol_out → symbol_in)
_REPLACEMENTS = {
    "SOHLH2": ("KIF2A",  "ENSG00000068796"),
    "VWF":    ("TM4SF1", "ENSG00000169908"),
}


# ---------------------------------------------------------------------------
# Data loaders (identical to cross_cohort_panel_loo.py)
# ---------------------------------------------------------------------------


def _load_gpl24676_raw() -> tuple[np.ndarray, list[str], np.ndarray]:
    (ds,) = load_dataset(
        "GSE153960", platform="GPL24676", resources_dir=ALS_DIR / "resources"
    )
    return ds.X.values.astype(np.float32), list(ds.X.columns), ds.y.values.astype(int)


def _load_gse76220_log1p() -> tuple[np.ndarray, list[str], np.ndarray]:
    (ds,) = load_dataset("GSE76220", resources_dir=ALS_DIR / "resources")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        X = np.log1p(ds.X.values.astype(np.float32))
    return X, list(ds.X.columns), ds.y.values.astype(int)


# ---------------------------------------------------------------------------
# Evaluation helpers (identical to cross_cohort_panel_loo.py)
# ---------------------------------------------------------------------------


def _zeroshot_auc(
    X_tr_raw: np.ndarray,
    tr_cols: list[int],
    y_tr: np.ndarray,
    X_te_log: np.ndarray,
    te_cols: list[int],
    y_te: np.ndarray,
    params: dict,
) -> float:
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    X_tr = np.log1p(X_tr_raw[:, tr_cols])
    X_te = X_te_log[:, te_cols]
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr).astype(np.float32)
    X_te_sc = sc.transform(X_te).astype(np.float32)
    clf = LGBMClassifier(**params)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        clf.fit(X_tr_sc, y_tr)
    return float(roc_auc_score(y_te, clf.predict_proba(X_te_sc)[:, 1]))


def _loo_cv_auc(
    X_log: np.ndarray,
    cols: list[int],
    y: np.ndarray,
    params: dict,
) -> float:
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    loo_params = {**params, "min_child_samples": 1}
    X = X_log[:, cols]
    n = len(y)
    scores = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr]).astype(np.float32)
        X_va = sc.transform(X[[i]]).astype(np.float32)
        clf = LGBMClassifier(**loo_params)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            clf.fit(X_tr, y[tr])
        scores[i] = clf.predict_proba(X_va)[0, 1]
    return float(roc_auc_score(y, scores))


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------


def _build_replacement_panel(
    feat_train: list[str],
) -> list[tuple[str, str]]:
    """Return list of (ensg_versioned, symbol) for the 15-gene replacement panel.

    Reads the core 25-gene panel CSV, keeps protein-coding members, and
    applies the SOHLH2→KIF2A and VWF→TM4SF1 swaps.
    """
    import pandas as pd

    panel_df = pd.read_csv(_PANEL_CSV)
    base_to_versioned = {n.split(".")[0]: n for n in feat_train}

    entries: list[tuple[str, str]] = []
    for _, row in panel_df.iterrows():
        sym: str = str(row["symbol"])
        feat: str = str(row["feature"])

        if sym not in _PROTEIN_CODING_PANEL:
            continue  # skip non-coding

        if sym in _REPLACEMENTS:
            # Swap this gene out for its replacement
            new_sym, new_base = _REPLACEMENTS[sym]
            new_feat = base_to_versioned.get(new_base)
            if new_feat is None:
                raise RuntimeError(
                    f"Replacement gene {new_sym} ({new_base}) not found in "
                    "GPL24676 training feature space."
                )
            entries.append((new_feat, new_sym))
        else:
            entries.append((feat, sym))

    assert len(entries) == 15, f"Expected 15 panel genes, got {len(entries)}"
    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd

    matplotlib.use("Agg")

    print("=" * 70)
    print("Replacement Panel Cross-Cohort LOO")
    print("15-gene protein-coding panel: SOHLH2→KIF2A, VWF→TM4SF1")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load all datasets
    # ------------------------------------------------------------------
    print("\n[1] Loading GPL24676 training data ...")
    X_tr_raw, feat_train, y_tr = _load_gpl24676_raw()
    train_name_to_col = {n: i for i, n in enumerate(feat_train)}
    print(f"  n={len(y_tr)}  ALS={y_tr.sum()}  Ctrl={(y_tr == 0).sum()}")

    print("\n[2] Loading GSE76220 ...")
    X_76_log, feat_76, y_76 = _load_gse76220_log1p()
    sym_to_col_76 = {s: i for i, s in enumerate(feat_76)}
    print(f"  n={len(y_76)}  ALS={y_76.sum()}  Ctrl={(y_76 == 0).sum()}")

    print("\n[3] Loading GSE122649 ...")
    if not _RAW_TAR.exists():
        _download_with_retry(_RAW_TAR_URL, _RAW_TAR, "GSE122649_RAW.tar")
    else:
        print(f"  Using cached RAW.tar ({_RAW_TAR.stat().st_size / 1e6:.0f} MB)")
    gsm_to_diag = _parse_soft()
    X_122_raw, syms_122, _, y_122 = _extract_counts(_RAW_TAR, gsm_to_diag)
    X_122_log = np.log1p(X_122_raw.astype(np.float32))
    sym_to_col_122 = {s: i for i, s in enumerate(syms_122)}
    print(f"  n={len(y_122)}  ALS={y_122.sum()}  Ctrl={(y_122 == 0).sum()}")

    # ------------------------------------------------------------------
    # Build replacement panel and resolve column indices
    # ------------------------------------------------------------------
    print("\n[4] Building 15-gene replacement panel ...")
    panel_entries = _build_replacement_panel(feat_train)

    params = json.loads(_PARAMS_PATH.read_text())
    params.update({"colsample_bytree": 1.0, "random_state": RANDOM_STATE,
                   "n_jobs": -1, "verbose": -1})

    # (train_col, col_76, col_122) — None if not available in that dataset
    gene_cols: list[tuple[int | None, int | None, int | None]] = []
    for feat, sym in panel_entries:
        tr_col = train_name_to_col.get(feat)
        gene_cols.append((tr_col, sym_to_col_76.get(sym), sym_to_col_122.get(sym)))

    for (feat, sym), (tc, c76, c122) in zip(panel_entries, gene_cols):
        avail = f"train={'Y' if tc is not None else 'N'}  " \
                f"76220={'Y' if c76 is not None else 'N'}  " \
                f"122649={'Y' if c122 is not None else 'N'}"
        print(f"  {sym:<12}  {avail}")

    # Full-panel column lists (genes available in BOTH train and the cohort)
    full_tr_76 = [gc[0] for gc in gene_cols if gc[0] is not None and gc[1] is not None]
    full_te_76 = [gc[1] for gc in gene_cols if gc[0] is not None and gc[1] is not None]
    full_tr_122 = [gc[0] for gc in gene_cols if gc[0] is not None and gc[2] is not None]
    full_te_122 = [gc[2] for gc in gene_cols if gc[0] is not None and gc[2] is not None]

    native_76 = [sym for sym, gc in zip([e[1] for e in panel_entries], gene_cols)
                 if gc[0] is not None and gc[1] is not None]
    native_122 = [sym for sym, gc in zip([e[1] for e in panel_entries], gene_cols)
                  if gc[0] is not None and gc[2] is not None]

    print(f"\n  Native in GSE76220  ({len(full_tr_76)}/15): {', '.join(native_76)}")
    print(f"  Native in GSE122649 ({len(full_tr_122)}/15): {', '.join(native_122)}")

    # ------------------------------------------------------------------
    # Baseline — full replacement panel
    # ------------------------------------------------------------------
    print("\n[5] Baseline (full 15-gene replacement panel, native genes per cohort) ...")
    base_zs_76 = _zeroshot_auc(
        X_tr_raw, full_tr_76, y_tr, X_76_log, full_te_76, y_76, params
    )
    base_zs_122 = _zeroshot_auc(
        X_tr_raw, full_tr_122, y_tr, X_122_log, full_te_122, y_122, params
    )
    base_loo_76 = _loo_cv_auc(X_76_log, full_te_76, y_76, params)
    base_loo_122 = _loo_cv_auc(X_122_log, full_te_122, y_122, params)
    base_wm_zs = (N_76220 * base_zs_76 + N_122649 * base_zs_122) / TOTAL_N
    base_wm_loo = (N_76220 * base_loo_76 + N_122649 * base_loo_122) / TOTAL_N

    print(
        f"  Zero-shot:  76220={base_zs_76:.4f}  122649={base_zs_122:.4f}"
        f"  W.mean={base_wm_zs:.4f}"
    )
    print(
        f"  LOO-CV:     76220={base_loo_76:.4f}  122649={base_loo_122:.4f}"
        f"  W.mean={base_wm_loo:.4f}"
    )

    # ------------------------------------------------------------------
    # Panel LOO (15 iterations)
    # ------------------------------------------------------------------
    print(f"\n[6] Panel LOO ({len(panel_entries)} iterations) ...")
    results: list[dict] = []

    for i, (feat, sym) in enumerate(panel_entries):
        _, c76_i, c122_i = gene_cols[i]
        print(f"  [{i+1:2d}/15] Remove {sym:<12}", end=" ", flush=True)

        tr_76_r = [gc[0] for j, gc in enumerate(gene_cols)
                   if j != i and gc[0] is not None and gc[1] is not None]
        te_76_r = [gc[1] for j, gc in enumerate(gene_cols)
                   if j != i and gc[0] is not None and gc[1] is not None]
        tr_122_r = [gc[0] for j, gc in enumerate(gene_cols)
                    if j != i and gc[0] is not None and gc[2] is not None]
        te_122_r = [gc[2] for j, gc in enumerate(gene_cols)
                    if j != i and gc[0] is not None and gc[2] is not None]

        zs_76 = _zeroshot_auc(X_tr_raw, tr_76_r, y_tr, X_76_log, te_76_r, y_76, params)
        zs_122 = _zeroshot_auc(
            X_tr_raw, tr_122_r, y_tr, X_122_log, te_122_r, y_122, params
        )
        loo_76 = _loo_cv_auc(X_76_log, te_76_r, y_76, params)
        loo_122 = _loo_cv_auc(X_122_log, te_122_r, y_122, params)

        wm_zs = (N_76220 * zs_76 + N_122649 * zs_122) / TOTAL_N
        wm_loo = (N_76220 * loo_76 + N_122649 * loo_122) / TOTAL_N
        d_zs = wm_zs - base_wm_zs
        d_loo = wm_loo - base_wm_loo

        results.append({
            "symbol": sym,
            "replaced_from": "SOHLH2" if sym == "KIF2A"
                             else "VWF" if sym == "TM4SF1" else None,
            "in_76220": c76_i is not None,
            "in_122649": c122_i is not None,
            "zs_76220": zs_76,
            "zs_122649": zs_122,
            "wm_zs": wm_zs,
            "d_zs": d_zs,
            "loo_76220": loo_76,
            "loo_122649": loo_122,
            "wm_loo": wm_loo,
            "d_loo": d_loo,
        })
        print(f"ZS Δ={d_zs:+.4f}  LOO Δ={d_loo:+.4f}")

    # ------------------------------------------------------------------
    # Output table
    # ------------------------------------------------------------------
    df = pd.DataFrame(results)
    df_zs = df.sort_values("d_zs", ascending=False)

    lines = [
        "Replacement Panel Cross-Cohort LOO",
        "15-gene protein-coding panel: SOHLH2→KIF2A, VWF→TM4SF1",
        "=" * 70,
        "Baseline (full 15-gene replacement panel, native genes per cohort):",
        f"  Zero-shot : GSE76220={base_zs_76:.4f}  GSE122649={base_zs_122:.4f}"
        f"  Weighted mean={base_wm_zs:.4f}",
        f"  LOO-CV    : GSE76220={base_loo_76:.4f}  GSE122649={base_loo_122:.4f}"
        f"  Weighted mean={base_wm_loo:.4f}",
        f"  Weights: GSE76220 n={N_76220}, GSE122649 n={N_122649}, total={TOTAL_N}",
        "",
        "SORTED BY Δ zero-shot weighted mean (positive = dispensable):",
        (
            f"{'Gene':<12}  {'Replaces':<10}  {'76?':>3}  {'122?':>4}  "
            f"{'ZS_76':>7}  {'ZS_122':>7}  {'WM_ZS':>7}  {'Δ_ZS':>7}  "
            f"{'LOO_76':>7}  {'LOO_122':>8}  {'WM_LOO':>7}  {'Δ_LOO':>7}"
        ),
        "-" * 100,
    ]

    for _, r in df_zs.iterrows():
        repl = r["replaced_from"] if r["replaced_from"] else "—"
        lines.append(
            f"{r['symbol']:<12}  {repl:<10}  "
            f"{'Y' if r['in_76220'] else 'N':>3}  "
            f"{'Y' if r['in_122649'] else 'N':>4}  "
            f"{r['zs_76220']:.4f}  {r['zs_122649']:.4f}  "
            f"{r['wm_zs']:.4f}  {r['d_zs']:+.4f}  "
            f"{r['loo_76220']:.4f}  {r['loo_122649']:.4f}  "
            f"  {r['wm_loo']:.4f}  {r['d_loo']:+.4f}"
        )

    out_txt = SCRIPT_DIR / "replacement_panel_loo.txt"
    out_txt.write_text("\n".join(lines))
    print(f"\nSaved → {out_txt.name}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    df_loo = df.sort_values("d_loo", ascending=False)

    _CLR_NEW = "#ff7f0e"   # orange — replacement genes
    _CLR_POS = "#2ca02c"   # green — dispensable
    _CLR_NEG = "#d62728"   # red — irreplaceable

    def _bar_color(row: pd.Series, val_col: str) -> str:
        if row["symbol"] in ("KIF2A", "TM4SF1"):
            return _CLR_NEW
        return _CLR_POS if row[val_col] >= 0 else _CLR_NEG

    colors_zs = [_bar_color(r, "d_zs") for _, r in df_zs.iterrows()]
    colors_loo = [_bar_color(r, "d_loo") for _, r in df_loo.iterrows()]

    syms_zs = df_zs["symbol"].tolist()
    d_zs_vals = df_zs["d_zs"].tolist()
    syms_loo = df_loo["symbol"].tolist()
    d_loo_vals = df_loo["d_loo"].tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.barh(range(len(syms_zs)), d_zs_vals, color=colors_zs, edgecolor="none")
    ax1.axvline(0, color="black", lw=0.9, ls="--")
    ax1.set_yticks(range(len(syms_zs)))
    ax1.set_yticklabels(syms_zs, fontsize=9)
    ax1.set_xlabel("Δ Weighted Mean AUC")
    ax1.set_title(
        f"Zero-Shot Panel LOO\n"
        f"Baseline W.mean = {base_wm_zs:.4f} "
        f"(76220={base_zs_76:.4f}, 122649={base_zs_122:.4f})",
        fontsize=9,
    )
    ax1.grid(axis="x", alpha=0.3)

    ax2.barh(range(len(syms_loo)), d_loo_vals, color=colors_loo, edgecolor="none")
    ax2.axvline(0, color="black", lw=0.9, ls="--")
    ax2.set_yticks(range(len(syms_loo)))
    ax2.set_yticklabels(syms_loo, fontsize=9)
    ax2.set_xlabel("Δ Weighted Mean AUC")
    ax2.set_title(
        f"Within-Cohort LOO-CV Panel LOO\n"
        f"Baseline W.mean = {base_wm_loo:.4f} "
        f"(76220={base_loo_76:.4f}, 122649={base_loo_122:.4f})",
        fontsize=9,
    )
    ax2.grid(axis="x", alpha=0.3)

    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color=_CLR_NEW, label="Replacement gene (KIF2A / TM4SF1)"),
        mpatches.Patch(color=_CLR_POS, label="Dispensable (Δ ≥ 0)"),
        mpatches.Patch(color=_CLR_NEG, label="Irreplaceable (Δ < 0)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        "Replacement Panel Cross-Cohort LOO — 15-gene protein-coding panel\n"
        "SOHLH2→KIF2A, VWF→TM4SF1  ·  "
        "ALS Spectrum MND vs Non-Neurological Control\n"
        "GSE76220 (n=20) + GSE122649 (n=38)  ·  weighted mean AUC",
        fontsize=9.5,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out_png = SCRIPT_DIR / "replacement_panel_loo.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Saved → {out_png.name}")


if __name__ == "__main__":
    main()
