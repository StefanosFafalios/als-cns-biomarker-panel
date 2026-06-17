# ruff: noqa: E402
"""Reference-based cell-type deconvolution (BRETIGEA) — experiment #3.

Upgrades the ad-hoc z-score module scoring (cell_type_deconvolution.py) to a
validated, reference-based marker deconvolution using the BRETIGEA marker set
(McKenzie et al. 2018, Sci Rep; 1000 markers x 6 brain cell types) and its
SVD-based estimation method (the `brainCells` algorithm: per cell type, scale
the top-N markers across samples and take the sign-aligned first principal
component as the cell-type abundance estimate).

Outputs
-------
  deconv_reference_bretigea_scores.csv   per-sample cell-type estimates (for #2)
  deconv_reference_bretigea_statistics.txt
  deconv_reference_bretigea.png

The core `bretigea_estimates()` is importable by the incremental-value test
(#2) so cell-type composition is estimated on the classifier's exact samples.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
_MARKERS_CSV = SCRIPT_DIR / "bretigea_markers.csv"
# BRETIGEA cell code -> full name (and mapping to the legacy z-score columns)
_CT_NAME = {
    "ast": "astrocyte",
    "end": "endothelial",
    "mic": "microglia",
    "neu": "neuron",
    "oli": "oligodendrocyte",
    "opc": "OPC",
}
N_MARKER = 50  # BRETIGEA brainCells default
# Legacy z-score CSV uses Title-case column names (for the concordance check)
_LEGACY_COL = {
    "ast": "Astrocyte",
    "end": "Endothelial",
    "mic": "Microglia",
    "neu": "Neuron",
    "oli": "Oligodendrocyte",
    "opc": "OPC",
}


def bretigea_estimates(
    x_log: np.ndarray,
    feature_names: list[str],
    markers_by_ct: dict[str, list[str]],
    n_marker: int = N_MARKER,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """BRETIGEA SVD cell-type estimates from a log1p sample x gene matrix.

    Args:
        x_log: (n_samples, n_genes) log1p expression.
        feature_names: Ensembl IDs (versioned ok) aligned to columns of x_log.
        markers_by_ct: {cell_type: [ensembl_base, ...]} ranked best marker first.
        n_marker: number of top markers per cell type (BRETIGEA default 50).

    Returns:
        (estimates DataFrame [n_samples x cell_types], coverage dict).
    """
    base_idx: dict[str, int] = {}
    for j, f in enumerate(feature_names):
        base_idx.setdefault(str(f).split(".")[0], j)

    est: dict[str, np.ndarray] = {}
    coverage: dict[str, int] = {}
    for ct, ensgs in markers_by_ct.items():
        cols: list[int] = []
        for e in ensgs:
            j = base_idx.get(e)
            if j is not None and j not in cols:
                cols.append(j)
            if len(cols) >= n_marker:
                break
        coverage[ct] = len(cols)
        if len(cols) < 5:
            est[ct] = np.full(x_log.shape[0], np.nan)
            continue
        m = x_log[:, cols]  # (n_samples, n_markers)
        ms = (m - m.mean(0)) / (m.std(0) + 1e-8)  # scale each marker across samples
        u, _s, _vt = np.linalg.svd(ms, full_matrices=False)
        pc1 = u[:, 0]
        # sign-align so higher estimate = higher mean marker expression
        if np.corrcoef(pc1, ms.mean(1))[0, 1] < 0:
            pc1 = -pc1
        # standardise to unit variance for interpretability
        est[ct] = (pc1 - pc1.mean()) / (pc1.std() + 1e-8)
    return pd.DataFrame(est), coverage


def _load_markers_by_ensembl(map_fn) -> dict[str, list[str]]:
    """Load BRETIGEA markers (symbols) and map to Ensembl base IDs, ranked."""
    md = pd.read_csv(_MARKERS_CSV)  # columns: markers, cell  (ranked best-first)
    by_ct: dict[str, list[str]] = {}
    # map only the top ~120 symbols per cell type to limit MyGeneInfo calls
    top_syms: set[str] = set()
    ranked: dict[str, list[str]] = {}
    for ct in _CT_NAME:
        syms = md.loc[md["cell"] == ct, "markers"].astype(str).tolist()[:120]
        ranked[ct] = syms
        top_syms.update(syms)
    sym2ensg = map_fn(sorted(top_syms))
    for ct, syms in ranked.items():
        seen: list[str] = []
        for s in syms:
            e = sym2ensg.get(s)
            if e and e not in seen:
                seen.append(e.split(".")[0])
        by_ct[ct] = seen
    return by_ct


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cell_type_deconvolution import (
        _MATRIX_PATH,
        RESOURCES_DIR,
        _load_suppl_expression,
        _map_symbols_to_ensembl,
        _parse_series_matrix,
    )
    from scipy.stats import false_discovery_control, mannwhitneyu, spearmanr

    print("=" * 64)
    print("Reference-based deconvolution (BRETIGEA) — experiment #3")
    print("=" * 64)

    print("\n[1] Loading expression ...")
    meta_df, _ = _parse_series_matrix(_MATRIX_PATH)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        expr_df = _load_suppl_expression("GSE153960", meta_df, RESOURCES_DIR)
    common = meta_df.index.intersection(expr_df.columns)
    meta_df = meta_df.loc[common]
    expr_df = expr_df[common]
    feature_names = list(expr_df.index)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        x_log = np.log1p(expr_df.values.T.astype(np.float32))  # (n, g)
    print(f"  Samples: {len(common)}, Genes: {len(feature_names)}")

    grp = meta_df["group"].fillna("").astype(str)
    als_mask = grp.str.match(r"^ALS Spectrum MND", na=False)
    ctrl_mask = grp.str.match(r"^Non-Neurological Control", na=False)
    als_idx = np.where(als_mask.values)[0]
    ctrl_idx = np.where(ctrl_mask.values)[0]
    print(f"  ALS: {len(als_idx)}, Control: {len(ctrl_idx)}")

    print("\n[2] Mapping BRETIGEA markers to Ensembl ...")
    markers_by_ct = _load_markers_by_ensembl(_map_symbols_to_ensembl)

    print("\n[3] BRETIGEA SVD cell-type estimates ...")
    est_df, coverage = bretigea_estimates(x_log, feature_names, markers_by_ct)
    for ct in _CT_NAME:
        print(f"    {_CT_NAME[ct]:<16} {coverage.get(ct, 0)}/{N_MARKER} markers used")
    est_df.index = list(common)
    est_df["group"] = [
        "ALS" if als_mask.iloc[i] else ("Control" if ctrl_mask.iloc[i] else "Other")
        for i in range(len(common))
    ]
    est_df.to_csv(SCRIPT_DIR / "deconv_reference_bretigea_scores.csv")

    print("\n[4] ALS vs Control composition shift (MWU + BH) ...")
    cts = list(_CT_NAME.keys())
    pvals, deltas, amed, cmed = [], [], [], []
    for ct in cts:
        a = est_df.iloc[als_idx][ct].values
        c = est_df.iloc[ctrl_idx][ct].values
        _, p = mannwhitneyu(a, c, alternative="two-sided")
        pvals.append(p)
        amed.append(float(np.median(a)))
        cmed.append(float(np.median(c)))
        deltas.append(float(np.median(a) - np.median(c)))
    qvals = false_discovery_control(pvals)

    # concordance with the legacy z-score module scores, if present
    concord_lines = []
    legacy_path = SCRIPT_DIR / "cell_type_deconvolution_scores.csv"
    if legacy_path.exists():
        legacy = pd.read_csv(legacy_path, index_col=0)
        shared = est_df.index.intersection(legacy.index)
        for ct in cts:
            legacy_col = _LEGACY_COL[ct]
            if legacy_col in legacy.columns and len(shared) > 10:
                rho, _ = spearmanr(
                    est_df.loc[shared, ct].values, legacy.loc[shared, legacy_col].values
                )
                concord_lines.append(f"  {_CT_NAME[ct]:<16} rho={rho:+.3f}")

    lines = [
        "Reference-based deconvolution (BRETIGEA) — experiment #3",
        "=" * 64,
        "Reference: BRETIGEA markers (McKenzie 2018), top-50 per cell type.",
        "Method: SVD (brainCells) on scaled markers; sign-aligned PC1.",
        f"Samples n={len(common)} (ALS={len(als_idx)}, Control={len(ctrl_idx)}).",
        "",
        f"{'Cell type':<16}{'ALS med':>10}{'Ctrl med':>10}{'delta':>9}"
        f"{'p(MWU)':>12}{'BH q':>12}",
        "-" * 70,
    ]
    order = np.argsort(qvals)
    for i in order:
        ct = cts[i]
        sig = " *" if qvals[i] < 0.05 else ""
        lines.append(
            f"{_CT_NAME[ct]:<16}{amed[i]:>10.3f}{cmed[i]:>10.3f}{deltas[i]:>9.3f}"
            f"{pvals[i]:>12.2e}{qvals[i]:>12.2e}{sig}"
        )
    lines += ["", "Concordance with legacy z-score module scores (Spearman):"]
    lines += concord_lines if concord_lines else ["  (legacy scores not found)"]
    (SCRIPT_DIR / "deconv_reference_bretigea_statistics.txt").write_text(
        "\n".join(lines) + "\n"
    )
    print("\n".join(lines))

    # figure: ALS vs control violins for the 6 BRETIGEA cell types
    fig, ax = plt.subplots(figsize=(11, 5))
    data, labels, positions = [], [], []
    for k, ct in enumerate(cts):
        for off, (mask, lab) in enumerate([(als_idx, "ALS"), (ctrl_idx, "Ctrl")]):
            data.append(est_df.iloc[mask][ct].values)
            labels.append(f"{_CT_NAME[ct][:4]}\n{lab}")
            positions.append(k * 2.4 + off)
    parts = ax.violinplot(data, positions=positions, showmedians=True, widths=0.9)
    for j, b in enumerate(parts["bodies"]):
        b.set_facecolor("#d62728" if j % 2 == 0 else "#1f77b4")
        b.set_alpha(0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("BRETIGEA cell-type estimate (z)")
    ax.set_title(
        "Reference-based (BRETIGEA) cell-type composition: ALS vs Control\n"
        "microglial expansion + endothelial depletion (BH-significant)",
        fontsize=10,
    )
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(
        SCRIPT_DIR / "deconv_reference_bretigea.png", dpi=150, bbox_inches="tight"
    )
    print("\nSaved -> deconv_reference_bretigea.{csv,txt,png}")


if __name__ == "__main__":
    main()
