# ruff: noqa: E402
"""Explicit cross-platform batch correction for the GPL24676 -> GPL16791
zero-shot transfer (addresses manuscript Limitation 4).

Baseline pipeline (calibration_analysis.py / external_validation_gpl16791.py):
    raw RSEM -> log1p -> CTD compartment regression (fit on train)
             -> extract 25-gene panel -> LightGBM (top-500 params).

This script inserts an explicit batch-correction step between CTD
residualisation and panel extraction, and compares three configurations on the
identical train/test split and model:

  (A) none      — baseline, no batch correction (reproduces the manuscript
                  zero-shot number in-run so the comparison is self-contained);
  (B) zstd      — per-platform per-gene z-standardisation (simplest correction
                  of additive + multiplicative platform shift);
  (C) combat    — parametric empirical-Bayes ComBat (Johnson, Li & Rabinovic,
                  Biostatistics 2007), implemented from scratch (no external
                  dependency).

ComBat and z-standardisation are *unsupervised* in the class label: the batch
variable is the sequencing platform only. The ALS/control label is never used
to estimate correction parameters (it is unavailable for the test cohort in a
deployment setting), so there is no label leakage. ComBat parameters are
estimated across the top-500 SHAP feature space (well-powered empirical-Bayes
priors) and the corrected 25 panel genes are extracted for the model.

Caveat reported in the output: because both platforms are ALS-heavy
(GPL24676 78% ALS, GPL16791 86% ALS) and no biological covariate is preserved,
this is the conservative (lower-bound) correction; preserving the class
covariate is impossible without test labels.

Outputs
-------
  combat_batch_correction.png
  combat_batch_correction_statistics.txt
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

from calibration_analysis import _expected_calibration_error
from external_validation_gpl16791 import (
    _ctd_residualise,
    _extract_panel,
    _load_platform,
)

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"
_SHAP_RANKING = SCRIPT_DIR / "lgbm_top500_final_shap_ranking.csv"

N_BOOTSTRAP = 2_000
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Parametric ComBat (Johnson, Li & Rabinovic 2007) — from scratch
# ---------------------------------------------------------------------------


def _aprior(delta_hat: np.ndarray) -> float:
    m = delta_hat.mean()
    s2 = delta_hat.var()
    return (2.0 * s2 + m**2) / s2


def _bprior(delta_hat: np.ndarray) -> float:
    m = delta_hat.mean()
    s2 = delta_hat.var()
    return (m * s2 + m**3) / s2


def _postmean(
    g_hat: np.ndarray, g_bar: float, n: int, d_star: np.ndarray, t2: float
) -> np.ndarray:
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def _postvar(sum2: np.ndarray, n: int, a: float, b: float) -> np.ndarray:
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)


def _it_eb(
    sdat: np.ndarray,
    g_hat: np.ndarray,
    d_hat: np.ndarray,
    g_bar: float,
    t2: float,
    a: float,
    b: float,
    conv: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterative empirical-Bayes fixed point for one batch (parametric)."""
    n = sdat.shape[1]
    g_old = g_hat.copy()
    d_old = d_hat.copy()
    change = 1.0
    count = 0
    while change > conv and count < 500:
        g_new = _postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = ((sdat - g_new[:, None]) ** 2).sum(axis=1)
        d_new = _postvar(sum2, n, a, b)
        change = max(
            np.max(np.abs(g_new - g_old) / (np.abs(g_old) + 1e-12)),
            np.max(np.abs(d_new - d_old) / (np.abs(d_old) + 1e-12)),
        )
        g_old, d_old = g_new, d_new
        count += 1
    return g_old, d_old


def combat(data: np.ndarray, batch: np.ndarray) -> np.ndarray:
    """Parametric ComBat batch correction (no biological covariate).

    Args:
        data: Expression matrix, genes x samples (G x N).
        batch: Length-N integer batch labels.

    Returns:
        Batch-corrected matrix, genes x samples (G x N).
    """
    batches = np.unique(batch)
    batch_idx = [np.where(batch == b)[0] for b in batches]
    n_batches = np.array([len(i) for i in batch_idx], dtype=float)
    n_total = data.shape[1]

    # Design = batch one-hot (no separate intercept); B_hat rows are batch means
    design = np.zeros((n_total, len(batches)))
    for j, idx in enumerate(batch_idx):
        design[idx, j] = 1.0
    b_hat = np.linalg.lstsq(design, data.T, rcond=None)[0]  # (n_batch x G)

    grand_mean = (n_batches / n_total) @ b_hat  # (G,)
    fitted = (design @ b_hat).T  # (G x N)
    var_pooled = ((data - fitted) ** 2) @ np.ones(n_total) / n_total  # (G,)
    var_pooled = np.maximum(var_pooled, 1e-8)

    stand_mean = grand_mean[:, None]
    s_data = (data - stand_mean) / np.sqrt(var_pooled)[:, None]

    bayesdata = s_data.copy()
    for j, idx in enumerate(batch_idx):
        sdat = s_data[:, idx]
        g_hat = sdat.mean(axis=1)
        d_hat = np.maximum(sdat.var(axis=1), 1e-8)
        g_bar = float(g_hat.mean())
        t2 = float(g_hat.var())
        a_prior = _aprior(d_hat)
        b_prior = _bprior(d_hat)
        g_star, d_star = _it_eb(sdat, g_hat, d_hat, g_bar, t2, a_prior, b_prior)
        bayesdata[:, idx] = (sdat - g_star[:, None]) / np.sqrt(d_star)[:, None]

    return bayesdata * np.sqrt(var_pooled)[:, None] + stand_mean


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fit_lgbm(X_tr: np.ndarray, y_train: np.ndarray):
    from lightgbm import LGBMClassifier

    params = json.loads(_PARAMS_PATH.read_text())
    params["colsample_bytree"] = 1.0
    clf = LGBMClassifier(**params)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        clf.fit(X_tr, y_train)
    return clf


def _zstandardise_per_platform(
    X_tr_panel: np.ndarray, X_te_panel: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-platform per-gene z-score (each platform standardised to its own stats)."""
    mu_tr, sd_tr = X_tr_panel.mean(0), X_tr_panel.std(0) + 1e-9
    mu_te, sd_te = X_te_panel.mean(0), X_te_panel.std(0) + 1e-9
    return (X_tr_panel - mu_tr) / sd_tr, (X_te_panel - mu_te) / sd_te


def _bootstrap_auc(
    y: np.ndarray, scores: np.ndarray, n: int, rng: np.random.Generator
) -> tuple[float, float]:
    from sklearn.metrics import roc_auc_score

    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], scores[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _metrics(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    return {
        "AUC": float(roc_auc_score(y, s)),
        "brier": float(brier_score_loss(y, s)),
        "ECE": _expected_calibration_error(y, s, n_bins=10),
        "mean_neg": float(s[y == 0].mean()),
        "mean_pos": float(s[y == 1].mean()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import roc_curve

    print("\n" + "=" * 70)
    print("Cross-platform batch correction: GPL24676 -> GPL16791 (Limitation 4)")
    print("=" * 70)

    panel = pd.read_csv(_PANEL_CSV)
    panel_features: list[str] = panel["feature"].tolist()

    top500: list[str] = pd.read_csv(_SHAP_RANKING).head(500)["feature"].tolist()

    print("\nLoading platforms ...")
    X_train_raw, feat_train, y_train, _ = _load_platform("GPL24676")
    X_test_raw, feat_test, y_test, _ = _load_platform("GPL16791")
    assert feat_train == feat_test

    print("log1p + CTD compartment regression (fit on GPL24676) ...")
    X_tr_res, X_te_res = _ctd_residualise(
        np.log1p(X_train_raw), np.log1p(X_test_raw), feat_train
    )

    # ---- (A) baseline: no correction --------------------------------------
    X_tr_p = _extract_panel(X_tr_res, feat_train, panel_features)
    X_te_p = _extract_panel(X_te_res, feat_test, panel_features)
    clf = _fit_lgbm(X_tr_p, y_train)
    s_none = clf.predict_proba(X_te_p)[:, 1]

    # ---- (B) per-platform z-standardisation -------------------------------
    X_tr_z, X_te_z = _zstandardise_per_platform(X_tr_p, X_te_p)
    clf_z = _fit_lgbm(X_tr_z, y_train)
    s_zstd = clf_z.predict_proba(X_te_z)[:, 1]

    # ---- (C) ComBat on top-500 space, extract panel -----------------------
    name_to_col = {n: i for i, n in enumerate(feat_train)}
    cols500 = [name_to_col[f] for f in top500 if f in name_to_col]
    feat500 = [f for f in top500 if f in name_to_col]
    print(f"ComBat estimated across {len(feat500)} top-SHAP features ...")
    combined = np.vstack([X_tr_res[:, cols500], X_te_res[:, cols500]])  # (N x 500)
    batch = np.array([0] * len(y_train) + [1] * len(y_test))
    corrected = combat(combined.T, batch).T  # back to (N x 500)
    # platform-mean gap before/after on panel genes (sanity check)
    panel_in_500 = [i for i, f in enumerate(feat500) if f in set(panel_features)]
    gap_before = float(
        np.abs(
            combined[: len(y_train)][:, panel_in_500].mean(0)
            - combined[len(y_train) :][:, panel_in_500].mean(0)
        ).mean()
    )
    gap_after = float(
        np.abs(
            corrected[: len(y_train)][:, panel_in_500].mean(0)
            - corrected[len(y_train) :][:, panel_in_500].mean(0)
        ).mean()
    )
    col500_to_idx = {f: i for i, f in enumerate(feat500)}
    p_cols = [col500_to_idx[f] for f in panel_features if f in col500_to_idx]
    X_tr_c = corrected[: len(y_train)][:, p_cols]
    X_te_c = corrected[len(y_train) :][:, p_cols]
    clf_c = _fit_lgbm(X_tr_c, y_train)
    s_combat = clf_c.predict_proba(X_te_c)[:, 1]

    # ---- metrics ----------------------------------------------------------
    rng = np.random.default_rng(RANDOM_STATE)
    configs = [("none (baseline)", s_none), ("zstd", s_zstd), ("combat", s_combat)]
    rows = []
    for name, s in configs:
        m = _metrics(y_test, s)
        lo, hi = _bootstrap_auc(y_test, s, N_BOOTSTRAP, rng)
        m["ci_lo"], m["ci_hi"] = lo, hi
        rows.append((name, m))
        print(
            f"  {name:16s} AUC={m['AUC']:.4f} [{lo:.3f},{hi:.3f}] "
            f"Brier={m['brier']:.4f} ECE={m['ECE']:.4f} "
            f"mean_ctrl={m['mean_neg']:.3f}"
        )
    print(
        f"\n  Panel-gene mean platform gap: before={gap_before:.4f} "
        f"after={gap_after:.4f}"
    )

    # ---- plot -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    colours = {"none (baseline)": "#7f7f7f", "zstd": "#ff7f0e", "combat": "#1f77b4"}
    for name, s in configs:
        fpr, tpr, _ = roc_curve(y_test, s)
        m = dict(rows)[name]
        ax.plot(
            fpr,
            tpr,
            lw=2,
            color=colours[name],
            label=f"{name}: AUC={m['AUC']:.3f} [{m['ci_lo']:.3f},{m['ci_hi']:.3f}]",
        )
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        "Zero-shot GPL24676 -> GPL16791 (n=636)\nbatch-correction comparison",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    names = [n for n, _ in rows]
    eces = [m["ECE"] for _, m in rows]
    briers = [m["brier"] for _, m in rows]
    x = np.arange(len(names))
    ax2.bar(x - 0.2, eces, 0.4, label="ECE (10-bin)", color="#d62728")
    ax2.bar(x + 0.2, briers, 0.4, label="Brier", color="#2ca02c")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9)
    ax2.set_ylabel("Calibration error (lower = better)")
    ax2.set_title("Cross-platform calibration", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_png = SCRIPT_DIR / "combat_batch_correction.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"\n  Saved -> {out_png.name}")

    # ---- stats file -------------------------------------------------------
    lines = [
        "Cross-platform batch correction: GPL24676 -> GPL16791",
        "=" * 70,
        f"Train: GPL24676 n={len(y_train)} ALS={int(y_train.sum())} "
        f"Ctrl={int((y_train == 0).sum())}",
        f"Test : GPL16791 n={len(y_test)} ALS={int(y_test.sum())} "
        f"Ctrl={int((y_test == 0).sum())}",
        "Panel: 25 genes; ComBat estimated across top-500 SHAP features.",
        "Correction is unsupervised in the class label (batch = platform only).",
        "",
        f"{'config':18s}{'AUC':>8s}{'95% CI':>18s}{'Brier':>9s}"
        f"{'ECE':>8s}{'mean_ctrl':>11s}{'mean_ALS':>10s}",
    ]
    for name, m in rows:
        ci = f"[{m['ci_lo']:.3f},{m['ci_hi']:.3f}]"
        lines.append(
            f"{name:18s}{m['AUC']:>8.4f}{ci:>18s}"
            f"{m['brier']:>9.4f}{m['ECE']:>8.4f}"
            f"{m['mean_neg']:>11.3f}{m['mean_pos']:>10.3f}"
        )
    lines += [
        "",
        "Panel-gene mean platform gap (|mean_train - mean_test|, avg over panel):",
        f"  before correction: {gap_before:.4f}",
        f"  after  ComBat    : {gap_after:.4f}",
        "",
        "Note: both platforms are ALS-heavy (78% vs 86%); no biological covariate",
        "is preserved (test labels unavailable), so this is a conservative",
        "lower-bound correction. zstd = per-platform per-gene z-standardisation.",
    ]
    out_txt = SCRIPT_DIR / "combat_batch_correction_statistics.txt"
    out_txt.write_text("\n".join(lines))
    print(f"  Saved -> {out_txt.name}")
    print("\n" + "=" * 70 + "\nDONE\n" + "=" * 70)


if __name__ == "__main__":
    main()
