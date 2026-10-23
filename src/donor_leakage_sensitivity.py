"""Donor-leakage sensitivity for the 25-gene and critical panels (Step 78).

GPL24676 holds 874 samples from 246 donors, and every CV in the discovery
pipeline split by *sample*, so most test folds shared donors with training.
GPL16791, the cross-platform cohort, also shares donors with GPL24676. This
script measures both effects without touching any panel definition:

1. Internal CV -- the same panel, params and ``colsample`` rescaling as the
   incremental curve (``lgbm_top500_final_analysis.py``), scored with
   sample-level ``StratifiedKFold`` and with donor-grouped
   ``StratifiedGroupKFold`` over seeds 42..51.
2. GPL16791 zero-shot -- Figure 3's estimator (combined_validation_roc.py), scored on the
   full cohort and on the samples whose donor is absent from GPL24676. The
   scaler is fit on training data, so dropping test samples needs no refit.
   The three- and four-cohort equal-weight means are recomputed with the
   donor-disjoint GPL16791 value; the other cohorts are read from
   ``combined_validation_roc_statistics.txt``.

Outputs ``donor_leakage_sensitivity.csv`` and
``donor_leakage_sensitivity_statistics.txt``.
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

SCRIPT_DIR = Path(__file__).parent
ALS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ALS_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from combined_validation_roc import N_SEEDS as FIG3_SEEDS  # noqa: E402
from cross_cohort_substitution import _ensemble_scores  # noqa: E402
from panel_critical_eval import _bootstrap_auc  # noqa: E402
from utils import (  # noqa: E402
    PanelLGBMClassifier,
    adjust_colsample,
    critical_panel_features,
    critical_panel_label,
    load_dataset,
    zero_shot_scale,
)
from utils import CORE_PANEL_SIZE  # noqa: E402

from utils import selected_lgbm_params  # noqa: E402
PARAMS = selected_lgbm_params()
SEEDS = range(42, 52)
FOLDS = 5
ROC_STATS = SCRIPT_DIR / "combined_validation_roc_statistics.txt"
OUT_CSV = SCRIPT_DIR / "donor_leakage_sensitivity.csv"
OUT_TXT = SCRIPT_DIR / "donor_leakage_sensitivity_statistics.txt"
# Cohorts averaged into the headline transfer figure (SRP064478 excluded as
# saturated at n=15), and the four-cohort selection scheme.
THREE = ("GPL16791", "GSE76220", "GSE122649")
FOUR = (*THREE, "SRP064478")


def _subjects(ds) -> np.ndarray:
    return ds.meta.loc[ds.y.index, "subject id"].to_numpy()


def _cv_auc(X: np.ndarray, y: np.ndarray, splitter, groups=None) -> float:
    params = adjust_colsample(PARAMS, X.shape[1])
    aucs = []
    for tr, te in splitter.split(X, y, groups):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = PanelLGBMClassifier(**params).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))


def _published_zero_shot() -> dict[str, dict[str, float]]:
    """Read the per-cohort zero-shot AUCs (25-gene, critical) from Figure 3's artefact."""
    out: dict[str, dict[str, float]] = {f"{CORE_PANEL_SIZE}-gene": {}, "critical": {}}
    for m in re.finditer(
        r"^(\S+)\s+\d+\s+([\d.]+)\s+\[[^\]]+\]\s+([\d.]+)",
        ROC_STATS.read_text(),
        re.MULTILINE,
    ):
        out[f"{CORE_PANEL_SIZE}-gene"][m.group(1)] = float(m.group(2))
        out["critical"][m.group(1)] = float(m.group(3))
    missing = set(FOUR) - set(out["critical"])
    if missing:
        raise SystemExit(f"{ROC_STATS.name} lacks cohorts {missing}")
    return out


def main() -> None:
    panel25 = pd.read_csv(SCRIPT_DIR / "lgbm_core25_panel.csv").feature.tolist()
    panels = {f"{CORE_PANEL_SIZE}-gene": panel25, "critical": critical_panel_features()}
    crit_label = critical_panel_label()
    rows: list[dict] = []

    # ---- 1. internal CV on the discovery matrix --------------------------
    (ds24,) = load_dataset(
        "GSE153960", platform="GPL24676", resources_dir=ALS_DIR / "resources"
    )
    y24 = ds24.y.to_numpy().astype(int)
    subj24 = _subjects(ds24)
    names = (SCRIPT_DIR / "lgbm_prefilter_names.txt").read_text().splitlines()
    col = {n: i for i, n in enumerate(names)}
    X_pre = np.load(SCRIPT_DIR / "lgbm_prefilter_X.npy", mmap_mode="r")
    assert X_pre.shape[0] == len(y24) == len(subj24)

    leak = {}
    for label, feats in panels.items():
        X = np.asarray(X_pre[:, [col[f] for f in feats]], dtype=np.float32)
        for seed in SEEDS:
            for split, splitter, groups in (
                ("sample", StratifiedKFold(FOLDS, shuffle=True, random_state=seed), None),
                ("donor", StratifiedGroupKFold(FOLDS, shuffle=True, random_state=seed), subj24),
            ):
                auc = _cv_auc(X, y24, splitter, groups)
                rows.append(
                    dict(analysis="internal_cv", panel=label, split=split,
                         seed=seed, auc=auc)
                )
        leak[label] = float(np.mean([
            np.isin(subj24[te], subj24[tr]).mean()
            for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=42).split(X, y24)
        ]))

    # ---- 2. GPL16791 zero-shot, full vs donor-disjoint -------------------
    (ds16,) = load_dataset(
        "GSE153960", platform="GPL16791", resources_dir=ALS_DIR / "resources"
    )
    assert list(ds16.X.columns) == list(ds24.X.columns)
    y16 = ds16.y.to_numpy().astype(int)
    subj16 = _subjects(ds16)
    disjoint = ~np.isin(subj16, subj24)
    base_col = {c.split(".")[0]: j for j, c in enumerate(ds24.X.columns)}
    X24_log = np.log1p(ds24.X.to_numpy(dtype=np.float32))
    X16_log = np.log1p(ds16.X.to_numpy(dtype=np.float32))
    zs_params = dict(PARAMS, n_jobs=-1, verbose=-1)
    published = _published_zero_shot()

    zs: dict[str, dict[str, float]] = {}
    for label, feats in panels.items():
        cols = [base_col[f.split(".")[0]] for f in feats]
        Ztr, Zte = zero_shot_scale(X24_log[:, cols], X16_log[:, cols], "GPL16791")
        # Figure 3's seed ensemble: above 25 genes colsample < 1 makes the trees seed-dependent
        scores = _ensemble_scores(Ztr, y24, Zte, zs_params, seeds=tuple(range(FIG3_SEEDS)))
        full = float(roc_auc_score(y16, scores))
        # The full-cohort value must reproduce Figure 3 before the subset means anything.
        if abs(full - published[label]["GPL16791"]) > 5e-5:
            raise SystemExit(
                f"{label}: GPL16791 AUC {full:.4f} does not reproduce "
                f"{published[label]['GPL16791']:.4f} in {ROC_STATS.name}"
            )
        sub = float(roc_auc_score(y16[disjoint], scores[disjoint]))
        lo_f, hi_f = _bootstrap_auc(y16, scores)
        lo_s, hi_s = _bootstrap_auc(y16[disjoint], scores[disjoint])
        zs[label] = {"full": full, "disjoint": sub}
        rows += [
            dict(analysis="gpl16791_zero_shot", panel=label, split="all_donors",
                 n=len(y16), auc=full, ci_lo=lo_f, ci_hi=hi_f),
            dict(analysis="gpl16791_zero_shot", panel=label, split="donor_disjoint",
                 n=int(disjoint.sum()), auc=sub, ci_lo=lo_s, ci_hi=hi_s),
        ]

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    # ---- report ------------------------------------------------------------
    cv = df[df.analysis == "internal_cv"]
    shared16 = np.unique(subj16[~disjoint])
    lines = [
        f"Donor-leakage sensitivity — {CORE_PANEL_SIZE}-gene and critical panels",
        "=" * 64,
        f"GPL24676 (discovery): n={len(y24)} ({y24.sum()} ALS / {(y24 == 0).sum()} control) "
        f"from {len(np.unique(subj24))} donors",
        f"Sample-level folds (seed 42, mean over folds): {leak[f'{CORE_PANEL_SIZE}-gene']:.1%} of test "
        "samples have a same-donor sample in training",
        "",
        "1. Internal 5-fold CV AUC (discovery matrix, curve params)",
        f"{'panel':<10}{'split':<8}{'seed 42':>9}{'mean 42-51':>12}{'sd':>8}",
    ]
    for label in panels:
        for split in ("sample", "donor"):
            s = cv[(cv.panel == label) & (cv.split == split)]
            lines.append(
                f"{label:<10}{split:<8}{s[s.seed == 42].auc.iloc[0]:>9.4f}"
                f"{s.auc.mean():>12.4f}{s.auc.std(ddof=0):>8.4f}"
            )
    lines += [
        "",
        "Headline (donor-grouped, seed 42):",
    ]
    for label in panels:
        s = cv[(cv.panel == label) & (cv.split == "donor") & (cv.seed == 42)]
        name = crit_label if label == "critical" else label
        lines.append(f"  {name} donor-grouped CV AUC = {s.auc.iloc[0]:.4f}")

    zs_df = df[df.analysis == "gpl16791_zero_shot"]
    lines += [
        "",
        f"2. GPL16791 zero-shot ({FIG3_SEEDS}-seed LightGBM ensemble, as Figure 3)",
        f"   {len(shared16)} of {len(np.unique(subj16))} GPL16791 donors also appear in GPL24676; "
        f"their {int((~disjoint).sum())} samples are removed in the donor-disjoint row",
        f"   donor-disjoint cohort: n={int(disjoint.sum())} "
        f"({int(y16[disjoint].sum())} ALS / {int((y16[disjoint] == 0).sum())} control)",
        f"{'panel':<10}{'donors':<16}{'n':>5}{'AUC':>9}  95% CI",
    ]
    for _, r in zs_df.iterrows():
        lines.append(
            f"{r.panel:<10}{r.split:<16}{int(r.n):>5}{r.auc:>9.4f}  [{r.ci_lo:.3f}, {r.ci_hi:.3f}]"
        )
    lines += ["", "Equal-weight cohort means with GPL16791 donor-disjoint (other cohorts unchanged)"]
    for label in panels:
        for scheme, cohorts in (("3-cohort", THREE), ("4-cohort", FOUR)):
            vals = {c: published[label][c] for c in cohorts}
            before = float(np.mean(list(vals.values())))
            vals["GPL16791"] = zs[label]["disjoint"]
            after = float(np.mean(list(vals.values())))
            lines.append(
                f"  {label:<10}{scheme}: {before:.4f} -> {after:.4f} ({after - before:+.4f})"
            )
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
