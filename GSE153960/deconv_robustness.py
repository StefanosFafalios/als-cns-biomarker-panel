# ruff: noqa: E402
"""Bootstrap robustness of the cell-type 'composition driver' assignment.

Addresses the question: is the single dominant cell type assigned to each
critical-panel gene (Table~bio_evidence) statistically robust, or is it a
near-tie produced by correlated glial module scores?

For each of the 13 labelled panel genes (12 critical + EMP1) we:
  1. Replicate the exact deconvolution pipeline (same loading, log1p, z-scored
     module scores) so point estimates match cell_type_deconvolution.py.
  2. Compute the per-cell-type Spearman rho (point estimate) and identify the
     top-1 / top-2 modules by |rho|.
  3. Bootstrap the profiled samples (n=999; B=2000) and record, per replicate:
       - the arg-max module (selection frequency = how often each module wins);
       - the gap |rho_top1| - |rho_top2| (95% percentile CI).
     Bootstrap uses the rank-pair resampling of Spearman (Pearson on resampled
     precomputed ranks) -- a standard, fast bootstrap for rank correlations.
  4. Classify the assignment as ROBUST only if the top-1 module wins >= 95% of
     bootstraps AND the gap CI excludes 0; otherwise AMBIGUOUS.
  5. Cross-reference whether the top-1 module shows a significant ALS
     composition shift (MWU) in the direction consistent with the gene, since a
     gene cannot be 'composition-driven' by a compartment that does not shift.

Output
------
  deconv_robustness.csv   — per-gene table
  deconv_robustness.txt   — human-readable summary
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ALS_DIR = Path(__file__).parents[1]
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(ALS_DIR))

from cell_type_deconvolution import (
    _DECONV_GENES,
    _MARKERS,
    _MATRIX_PATH,
    _PANEL_CSV,
    RESOURCES_DIR,
    _build_ensg_to_symbol,
    _compute_module_scores,
    _load_suppl_expression,
    _map_symbols_to_ensembl,
    _parse_series_matrix,
)

B = 2000
SEED = 42


def main() -> None:
    import pandas as pd
    from scipy.stats import mannwhitneyu, rankdata, spearmanr

    print("Replicating deconvolution loading ...")
    meta_df, _ = _parse_series_matrix(_MATRIX_PATH)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        expr_df = _load_suppl_expression("GSE153960", meta_df, RESOURCES_DIR)
    common = meta_df.index.intersection(expr_df.columns)
    meta_df = meta_df.loc[common]
    expr_df = expr_df[common]
    feature_names = list(expr_df.index)
    ensg_index = _build_ensg_to_symbol(feature_names)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        X_log = np.log1p(expr_df.values.T.astype(np.float32))
    mu = X_log.mean(axis=0, keepdims=True)
    sd = X_log.std(axis=0, keepdims=True) + 1e-8
    X_z = (X_log - mu) / sd

    all_symbols = sorted({s for syms in _MARKERS.values() for s in syms})
    sym_to_ensg = _map_symbols_to_ensembl(all_symbols)
    score_df = _compute_module_scores(X_z, ensg_index, sym_to_ensg, _MARKERS)
    cell_types = [ct for ct in _MARKERS if not score_df[ct].isna().all()]

    # ALS / control masks (same logic as cell_type_deconvolution.main)
    grp = meta_df["group"].fillna("").astype(str)
    als_mask = grp.str.match(r"^ALS Spectrum MND$", na=False) | grp.str.match(
        r"^ALS Spectrum MND, Other Neurological Dis", na=False
    )
    ctrl_mask = grp.str.match(r"^Non-Neurological Control", na=False)
    als_i = np.where(als_mask.values)[0]
    ctrl_i = np.where(ctrl_mask.values)[0]

    # significant composition shifts (for the consistency check)
    shift: dict[str, tuple[float, str]] = {}
    for ct in cell_types:
        a = score_df[ct].values[als_i]
        c = score_df[ct].values[ctrl_i]
        _, p = mannwhitneyu(a, c, alternative="two-sided")
        direction = "UP" if np.median(a) > np.median(c) else "DOWN"
        shift[ct] = (float(p), direction)

    # co-expression over all profiled samples (n=999), matching
    # cell_type_deconvolution.py
    n = X_log.shape[0]
    print(f"  n={n} profiled samples, {len(cell_types)} cell types")
    score_train = {ct: score_df[ct].values.astype(float) for ct in cell_types}

    # precompute rank matrix of module scores (rank-pair bootstrap)
    ct_rank = np.column_stack([rankdata(score_train[ct]) for ct in cell_types])

    panel = pd.read_csv(_PANEL_CSV)
    panel = panel[panel["symbol"].isin(_DECONV_GENES)]
    profiling = pd.read_csv(SCRIPT_DIR / "gene_profiling_summary.csv")
    dir_map = dict(zip(profiling["symbol"], profiling["direction"]))

    rng = np.random.default_rng(SEED)
    boot = np.stack([rng.integers(0, n, n) for _ in range(B)])  # (B, n)

    rows: list[dict] = []
    for _, r in panel.iterrows():
        base = str(r["feature"]).split(".")[0]
        sym = str(r["symbol"])
        if base not in ensg_index:
            continue
        gx = X_log[:, ensg_index[base]].astype(float)

        # point estimates (exact Spearman, matches main script)
        rho = np.array([spearmanr(gx, score_train[ct])[0] for ct in cell_types])
        order = np.argsort(-np.abs(rho))
        i1, i2 = order[0], order[1]
        ct1, ct2 = cell_types[i1], cell_types[i2]

        # bootstrap (rank-pair): recompute |rho| per ct on resampled ranks
        rgx = rankdata(gx)
        wins = np.zeros(len(cell_types), dtype=int)
        gaps = np.empty(B)
        for b in range(B):
            idx = boot[b]
            x = rgx[idx]
            xc = x - x.mean()
            xden = np.sqrt((xc * xc).sum()) + 1e-12
            C = ct_rank[idx]
            Cc = C - C.mean(axis=0)
            num = xc @ Cc
            den = xden * (np.sqrt((Cc * Cc).sum(axis=0)) + 1e-12)
            absr = np.abs(num / den)
            wins[absr.argmax()] += 1
            gaps[b] = absr[i1] - absr[i2]

        gap_lo, gap_hi = np.percentile(gaps, [2.5, 97.5])
        top1_freq = wins[i1] / B
        robust = bool(gap_lo > 0 and top1_freq >= 0.95)

        # consistency with a significant composition shift
        gdir = "UP" if str(dir_map.get(sym, "")).startswith("up") else "DOWN"
        p_shift, sdir = shift[ct1]
        consistent = (p_shift < 0.05) and (sdir == gdir)

        rows.append(
            {
                "gene": sym,
                "gene_dir": gdir,
                "top1_ct": ct1,
                "top1_rho": round(float(rho[i1]), 3),
                "top2_ct": ct2,
                "top2_rho": round(float(rho[i2]), 3),
                "gap": round(float(abs(rho[i1]) - abs(rho[i2])), 3),
                "gap_CI_lo": round(float(gap_lo), 3),
                "gap_CI_hi": round(float(gap_hi), 3),
                "top1_argmax_freq": round(float(top1_freq), 3),
                "assignment": "ROBUST" if robust else "AMBIGUOUS",
                "top1_shift_p": f"{p_shift:.2e}",
                "top1_shift_dir": sdir,
                "shift_consistent": consistent,
            }
        )
        print(
            f"  {sym:9} {ct1:12}({rho[i1]:+.3f}) vs {ct2:12}({rho[i2]:+.3f})  "
            f"gap CI[{gap_lo:+.3f},{gap_hi:+.3f}]  win={top1_freq:.2f}  "
            f"{'ROBUST' if robust else 'AMBIGUOUS':9}  "
            f"shift={sdir}({p_shift:.1e}) consistent={consistent}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(SCRIPT_DIR / "deconv_robustness.csv", index=False)

    lines = [
        "Cell-type 'composition driver' robustness — bootstrap (B=2000)",
        "=" * 66,
        "Rank-pair Spearman bootstrap; ROBUST = top-1 module wins >=95% of",
        "bootstraps AND gap(|rho1|-|rho2|) 95% CI excludes 0.",
        "shift_consistent = top-1 module shows a significant ALS composition",
        "shift (MWU p<0.05) in the gene's direction.",
        "",
        df.to_string(index=False),
        "",
        "AMBIGUOUS genes have no statistically robust single dominant cell type",
        "and should be labelled pan-glial rather than assigned one compartment.",
    ]
    (SCRIPT_DIR / "deconv_robustness.txt").write_text("\n".join(lines) + "\n")
    print("\nSaved -> deconv_robustness.{csv,txt}")


if __name__ == "__main__":
    main()
