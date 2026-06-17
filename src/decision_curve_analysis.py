"""Decision-curve analysis on the 25-gene panel.

Computes net-benefit curves for treat-all, treat-none, and model strategies
across threshold probabilities 0.05–0.95 for:
  - GPL24676 5-fold CV (training cohort)
  - GPL16791 zero-shot (primary external)

Uses score arrays saved in calibration_scores.npz by calibration_analysis.py.

Net-benefit = (TP / N) - (FP / N) * (p_t / (1 - p_t))
where p_t is the threshold probability.

Outputs:
  decision_curve.png
  decision_curve_statistics.txt
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).parent
_SCORES = SCRIPT_DIR / "calibration_scores.npz"


def _net_benefit(y_true: np.ndarray, scores: np.ndarray, p_t: float) -> float:
    """Net benefit at threshold p_t under model strategy."""
    n = len(y_true)
    pred = (scores >= p_t).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    if n == 0:
        return float("nan")
    return tp / n - fp / n * p_t / max(1 - p_t, 1e-9)


def _net_benefit_all(y_true: np.ndarray, p_t: float) -> float:
    """Net benefit of treat-all."""
    prev = y_true.mean()
    return prev - (1 - prev) * p_t / max(1 - p_t, 1e-9)


def main() -> None:
    data = np.load(_SCORES, allow_pickle=True)
    y_train = data["y_train"]
    s_train = data["train_cv_scores"]
    y_test = data["gpl16791_y"]
    s_test = data["gpl16791_scores"]

    thresholds = np.arange(0.05, 0.96, 0.025)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    rows = []
    for ax, (lbl, y, s) in zip(axes, [
        ("Train cohort (GPL24676; 5-fold CV)", y_train, s_train),
        ("GPL16791 zero-shot", y_test, s_test),
    ]):
        nb_model = [_net_benefit(y, s, p) for p in thresholds]
        nb_all = [_net_benefit_all(y, p) for p in thresholds]
        nb_none = [0.0] * len(thresholds)
        ax.plot(thresholds, nb_model, "-", color="#1f77b4", lw=2, label="25-gene panel (LGBM)")
        ax.plot(thresholds, nb_all, "--", color="#ff7f0e", lw=1.5, label="Treat-all")
        ax.plot(thresholds, nb_none, ":", color="gray", lw=1.5, label="Treat-none")
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
        ax.set_title(lbl, fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)

        # Find threshold range where model dominates both alternatives
        dominates = np.array([
            nb_model[i] > nb_all[i] and nb_model[i] > nb_none[i]
            for i in range(len(thresholds))
        ])
        if dominates.any():
            dom_t = thresholds[dominates]
            rows.append(f"{lbl}: model dominates over threshold range "
                        f"[{dom_t.min():.2f}, {dom_t.max():.2f}]")
        else:
            rows.append(f"{lbl}: model does not strictly dominate at any threshold")
    plt.tight_layout()
    fig.savefig(SCRIPT_DIR / "decision_curve.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    print("Decision-curve analysis:")
    for r in rows:
        print(" ", r)

    lines = [
        "Decision-curve analysis (DCA) — 25-gene panel (LGBM)",
        "=" * 60,
        "Net benefit computed across threshold probabilities 0.05–0.95.",
        "",
        "Strategies compared: model (25-gene LGBM), treat-all, treat-none.",
        "",
    ] + rows + [
        "",
        "Interpretation:",
        "  The model strategy is preferred over treat-all/treat-none across",
        "  the range of threshold probabilities where its net-benefit curve",
        "  lies above both alternatives. Within that range, the panel provides",
        "  clinical utility beyond a naive policy.",
    ]
    (SCRIPT_DIR / "decision_curve_statistics.txt").write_text("\n".join(lines))
    print("\nSaved -> decision_curve.{png,txt}")


if __name__ == "__main__":
    main()
