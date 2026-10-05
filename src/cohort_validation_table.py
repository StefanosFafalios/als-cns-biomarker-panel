"""Per-cohort validation of the 25-gene and critical panels: zero-shot and pooled LOO.

Study design (findings.md Step 105): GPL24676 supplies the 25-gene panel. Every other
CNS cohort then (1) validates it, zero-shot and by leave-one-out CV, and (2) is pooled
into the greedy leave-one-gene-out trimming that yields the critical panel (steps
24/24a). The critical-panel rows are therefore in-sample: those cohorts chose its genes.

  Zero-shot  : model trained on all GPL24676 samples, applied unchanged
               (utils.zero_shot_scale), exactly as in step 24a.
  Pooled LOO : the nine validation datasets merged into one (GPL24676 left out), leave
               one donor out. Most cohorts are too small for a within-cohort LOO, whose
               pooled AUC is biased low at n < 50 (Step 105b). Before merging, each cohort
               is z-scored against its own controls (pooled within-class SD), with the
               statistics taken from the training rows of every fold, so neither the
               held-out sample nor its label shapes its own scaling. Genes a platform lacks
               stay missing (LightGBM routes NaN). Sample weights give each trimming unit
               equal total weight (iterative_panel_elimination._W). AUC is computed within
               each cohort, since cohorts differ in ALS fraction and platform; a global AUC
               over all merged samples is a secondary line (cross-cohort pairs dominate it). Donors:
               GPL16791 subject id; E-MTAB-8635 and E-MTAB-2325 share 35 patients (SDRF
               patient codes), held out together.

Model: the presented one, PanelLGBMClassifier on lgbm_top500_best_params.json (leaf
size at the tuned fraction of each training set). Cohorts, loaders, merging and the
pooled CV are step 24's own (_trim_cohorts, merge_cohorts, pooled_oof). 95% CIs: bootstrap of the scores over samples (B=2000); for
GPL16791 this ignores donor clustering, as in step 24.

Outputs: cohort_validation_table.csv, cohort_validation_table.txt
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from utils import (  # noqa: E402
    PanelLGBMClassifier,
    critical_panel_symbols,
    zero_shot_scale,
)


def main() -> None:
    warnings.filterwarnings("ignore")
    params = json.loads((SCRIPT_DIR / "lgbm_top500_best_params.json").read_text())

    ipe = SCRIPT_DIR / "iterative_panel_elimination.py"
    ns: dict = {"__name__": "notmain", "__file__": str(ipe)}
    exec(compile(ipe.read_text().split("def main(")[0], "ipe", "exec"), ns)  # noqa: S102

    panel_df = pd.read_csv(SCRIPT_DIR / "lgbm_core25_panel.csv")
    sym25 = list(panel_df.symbol)
    feat25_bases = [f.split(".")[0] for f in panel_df.feature]
    X_log, y_train, _, feat_train = ns["_load_gpl24676_ctd"]()
    base_map = {f.split(".")[0]: j for j, f in enumerate(feat_train)}
    panel_train_cols = [base_map[b] for b in feat25_bases]
    Xtr25 = X_log[:, panel_train_cols].astype(np.float32)
    del X_log
    cohorts = ns["_trim_cohorts"](sym25, feat25_bases, panel_train_cols)

    m = ns["merge_cohorts"](cohorts, len(sym25))
    y, cid = m.y, m.cid

    crit = critical_panel_symbols()
    panels = {
        "25-gene": list(range(len(sym25))),
        f"critical ({len(crit)})": sorted(sym25.index(g) for g in crit),
    }

    rows = []
    for pname, pidx in panels.items():
        loo = ns["pooled_oof"](m, pidx, m.donor[None], params)
        for cname, (M, av, yc, _) in cohorts.items():
            common = [pi for pi in av if pi in set(pidx)]
            Ztr, Zte = zero_shot_scale(
                Xtr25[:, common], M[:, [av.index(pi) for pi in common]], cname
            )
            zs = (
                PanelLGBMClassifier(**params).fit(Ztr, y_train).predict_proba(Zte)[:, 1]
            )
            for mode, sc in (("zero-shot", zs), ("pooled LOO", loo[cid == cname])):
                lo, hi = ns["_bootstrap_auc"](yc, sc)
                rows.append(
                    {
                        "cohort": cname,
                        "panel": pname,
                        "mode": mode,
                        "n": len(yc),
                        "n_als": int(yc.sum()),
                        "genes": f"{len(common)}/{len(pidx)}",
                        "auc": roc_auc_score(yc, sc),
                        "ci_lo": lo,
                        "ci_hi": hi,
                    }
                )
            print(
                f"  {cname:<11} {pname:<13} ZS {rows[-2]['auc']:.4f}  LOO {rows[-1]['auc']:.4f}",
                flush=True,
            )
        for mode in ("zero-shot", "pooled LOO"):
            wm = sum(
                ns["_W"][r["cohort"]] * r["auc"]
                for r in rows
                if r["panel"] == pname and r["mode"] == mode
            )
            rows.append(
                {
                    "cohort": "W.mean",
                    "panel": pname,
                    "mode": mode,
                    "n": len(y),
                    "n_als": int(y.sum()),
                    "genes": "",
                    "auc": wm,
                    "ci_lo": np.nan,
                    "ci_hi": np.nan,
                }
            )
        # Secondary: one AUC over all merged samples. 60% of its ALS/control pairs cross
        # cohorts, so it mainly checks that one threshold holds across platforms; far
        # above W.mean would mean cohort identity leaks into the scores.
        lo, hi = ns["_bootstrap_auc"](y, loo)
        rows.append(
            {
                "cohort": "global",
                "panel": pname,
                "mode": "pooled LOO",
                "n": len(y),
                "n_als": int(y.sum()),
                "genes": "",
                "auc": roc_auc_score(y, loo),
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(
        SCRIPT_DIR / "cohort_validation_table.csv", index=False, float_format="%.4f"
    )
    lines = [
        "Per-cohort validation: zero-shot (GPL24676-trained) and pooled leave-one-donor-out",
        "Pooled LOO: nine cohorts merged, control-referenced z-scores per cohort. Critical-panel rows are in-sample.",
        "",
        f"{'Cohort':<12}{'Panel':<14}{'Mode':<12}{'n (ALS)':>9}{'genes':>8}{'AUC':>8}  95% CI",
    ]
    for r in rows:
        lines.append(
            f"{r['cohort']:<12}{r['panel']:<14}{r['mode']:<12}{r['n']:>4} ({r['n_als']:>2}){r['genes']:>8}"
            f"{r['auc']:>8.4f}  [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
        )
    (SCRIPT_DIR / "cohort_validation_table.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
