"""Forest plot of cohort_validation_table.csv (step 24b): per-cohort AUC, both panels, both modes.

Output: cohort_validation.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(SCRIPT_DIR / "cohort_validation_table.csv")
    cohorts = [c for c in dict.fromkeys(df.cohort) if c not in ("W.mean", "global")]
    panels = list(dict.fromkeys(df.panel))
    modes = ["zero-shot", "pooled LOO"]
    style = {(panels[0], modes[0]): ("#9e9e9e", "o"), (panels[0], modes[1]): ("#424242", "s"),
             (panels[1], modes[0]): ("#80cbc4", "o"), (panels[1], modes[1]): ("#00695c", "s")}
    fig, ax = plt.subplots(figsize=(9, 0.75 * len(cohorts) + 2.2))
    y0 = np.arange(len(cohorts))[::-1]
    offs = np.linspace(-0.27, 0.27, 4)
    k = 0
    for p in panels:
        for m in modes:
            sub = df[(df.panel == p) & (df["mode"] == m)].set_index("cohort")
            col, mk = style[(p, m)]
            for yi, c in zip(y0, cohorts):
                r = sub.loc[c]
                ax.plot([r.ci_lo, r.ci_hi], [yi + offs[k]] * 2, color=col, lw=1.2)
                ax.plot(r.auc, yi + offs[k], mk, color=col, ms=6)
            wm = sub.loc["W.mean", "auc"]
            ax.plot([], [], mk, color=col, label=f"{p}, {m} (weighted mean {wm:.3f})")
            k += 1
    n = df.drop_duplicates("cohort").set_index("cohort")
    ax.set_yticks(y0)
    ax.set_yticklabels([f"{c}  (n={n.loc[c, 'n']}, {n.loc[c, 'n_als']} ALS)" for c in cohorts])
    ax.axvline(0.5, color="grey", ls="--", lw=0.8)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("AUC-ROC (95% bootstrap CI)")
    ax.set_title("External CNS datasets: zero-shot transfer and pooled leave-one-donor-out\n"
                 "(critical-panel values are in-sample: these datasets chose its genes)", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, frameon=False)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(SCRIPT_DIR / "cohort_validation.png", dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    main()
