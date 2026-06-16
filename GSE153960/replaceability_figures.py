"""Generate four replaceability-related figures for the manuscript.

(1) repl_vs_greedy.png — replaceability% vs greedy backward elimination step
    for the 15 protein-coding panel members. Shows the convergence between
    two independent criteria.

(2) repl_candidate_boxplot.png — distribution of replacement candidate AUCs
    per PC panel gene. Highlights the spread that the single replaceability
    % number collapses.

(3) substitution_roc.png — for GSE76220 and GSE122649, overlay the baseline
    14-of-15 ROC against the protein-coding-substituted ROC, showing visible
    gain.

(4) repl_module_heatmap.png — for each PC panel gene, the count of valid
    replacement candidates broken down by biotype (protein-coding /
    pseudogene / ncRNA / artifact) — shows that the "module density" varies
    not just in raw count but in candidate quality.
"""
from __future__ import annotations

import gzip
import json
import re
import sys
import tarfile
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ALS_DIR = Path(__file__).parents[1]
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset  # noqa: E402

# Reuse loaders from the substitution script
from cross_cohort_substitution import (  # noqa: E402
    _load_gpl24676_ctd,
    _load_gse76220,
    _load_gse122649,
    _ensemble_scores,
)

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"
_REPL_CSV = SCRIPT_DIR / "gene_replacement_results.csv"
_REPL_CLEAN = SCRIPT_DIR / "gene_replacement_clean_statistics.txt"
_ITER_STAT = SCRIPT_DIR / "iterative_panel_elimination_statistics.txt"

# 15-gene critical panel indices in the 25-gene ordered panel
_CRITICAL_IDX = [1, 2, 3, 5, 6, 7, 10, 11, 12, 15, 16, 18, 20, 23, 24]


# ---------------------------------------------------------------------------
# Data ingest
# ---------------------------------------------------------------------------

def _parse_drop_steps(text: str) -> dict[str, int]:
    """Parse iterative_panel_elimination_statistics.txt for per-gene drop step.

    Returns {gene_symbol: step_index}. The final singleton gene is assigned step 25.
    """
    drops: dict[str, int] = {}
    in_log = False
    final_gene = None
    for line in text.splitlines():
        if line.strip().startswith("Step") and "Dropped" in line:
            in_log = True
            continue
        if not in_log:
            continue
        if line.strip().startswith("Final minimal panel"):
            in_log = False
            continue
        if not line.strip() or line.startswith("-"):
            continue
        # Format:  N  GENE  D_ZS   ...
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            step = int(parts[0])
        except ValueError:
            continue
        gene = parts[1]
        drops[gene] = step
    # Identify the surviving final gene
    for line in text.splitlines():
        m = re.match(r"\s*Symbols:\s*(\S+)", line)
        if m:
            final_gene = m.group(1).rstrip(",")
            break
    if final_gene:
        drops[final_gene] = max(drops.values(), default=0) + 1
    return drops


def _parse_clean_repl(text: str) -> dict[str, float]:
    """Parse gene_replacement_clean_statistics.txt for {gene: replaceability%}."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        # Format:  GENE   orig%   clean%   ...
        m = re.match(r"^([A-Za-z0-9\.\-]+)\s+\d+\.\d+%\s+(\d+\.\d+)%", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


# ---------------------------------------------------------------------------
# Figure 1: replaceability vs greedy drop step
# ---------------------------------------------------------------------------

def fig_repl_vs_greedy() -> tuple[Path, dict]:
    drop_steps = _parse_drop_steps(_ITER_STAT.read_text())
    repl = _parse_clean_repl(_REPL_CLEAN.read_text())

    # PC subset of the 25 panel
    pc_genes = [
        "MECOM", "SERTAD1", "FCN3", "PROS1", "ANGPT2", "EMP1", "TINAGL1",
        "CKMT2", "VWF", "CLDN5", "NR4A1", "SOHLH2", "HEXB", "MCEE", "SLC37A2",
    ]
    critical_pc = {"FCN3", "PROS1", "ANGPT2", "TINAGL1", "CKMT2", "CLDN5",
                    "NR4A1", "SOHLH2", "HEXB", "MCEE", "SLC37A2", "SERTAD1"}

    points = []
    for g in pc_genes:
        if g in drop_steps and g in repl:
            points.append({
                "gene": g, "step": drop_steps[g], "repl": repl[g],
                "is_critical": g in critical_pc,
            })

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for p in points:
        colour = "#C62828" if p["is_critical"] else "#888"
        marker = "o" if p["is_critical"] else "s"
        ax.scatter(p["repl"], p["step"], s=110, c=colour, marker=marker,
                    edgecolor="black", lw=0.6, zorder=3)
        # label offset
        ax.annotate(p["gene"], (p["repl"], p["step"]),
                     xytext=(5, 5), textcoords="offset points", fontsize=8.5)

    # Quadrant guides
    ax.axhline(10.5, color="grey", ls="--", lw=0.7,
                label="Greedy peak k=15 (drop step 10/11 boundary)")
    ax.axvline(25, color="grey", ls=":", lw=0.7, label="Replaceability = 25%")

    ax.set_xlabel("Genome-wide replaceability (% of non-artifact candidates valid)")
    ax.set_ylabel("Greedy backward elimination drop step\n"
                    "(later = harder to drop = more anchored)")
    ax.set_title(
        "Convergence of two independent criteria — replaceability and "
        "cross-cohort greedy elimination\n"
        "The 4 hardest-to-replace genes (CKMT2, NR4A1, TINAGL1, SERTAD1; red) "
        "are all in the 15-crit panel; the 3 most-replaceable PC genes "
        "(MECOM, VWF, EMP1) are all dropped early.",
        fontsize=9,
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(-3, 105)
    ax.set_ylim(0, 26)
    plt.tight_layout()
    out = SCRIPT_DIR / "repl_vs_greedy.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    return out, {"n_points": len(points)}


# ---------------------------------------------------------------------------
# Figure 2: candidate-AUC distribution boxplot per PC panel gene
# ---------------------------------------------------------------------------

def fig_candidate_boxplot() -> tuple[Path, dict]:
    df = pd.read_csv(_REPL_CSV)
    pc_genes = [
        "MECOM", "SERTAD1", "FCN3", "PROS1", "ANGPT2", "EMP1", "TINAGL1",
        "CKMT2", "VWF", "CLDN5", "NR4A1", "SOHLH2", "HEXB", "MCEE", "SLC37A2",
    ]
    critical_pc = {"FCN3", "PROS1", "ANGPT2", "TINAGL1", "CKMT2", "CLDN5",
                    "NR4A1", "SOHLH2", "HEXB", "MCEE", "SLC37A2", "SERTAD1"}

    # Use only candidate AUCs (column "replacement_auc"); no baseline shift here.
    data, labels, colours = [], [], []
    repl_pct: dict[str, float] = _parse_clean_repl(_REPL_CLEAN.read_text())
    pc_genes_sorted = sorted(pc_genes, key=lambda g: repl_pct.get(g, 0))
    for g in pc_genes_sorted:
        sub = df[df["panel_symbol"] == g]
        if sub.empty:
            continue
        data.append(sub["replacement_auc"].values)
        labels.append(f"{g} ({repl_pct.get(g, 0):.0f}%)")
        colours.append("#C62828" if g in critical_pc else "#888")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False,
                     widths=0.7)
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    # Baseline reference
    baseline = 0.9469  # fast-LGBM baseline used by gene_replacement.py
    threshold = baseline - 0.002
    ax.axhline(baseline, color="black", ls="--", lw=0.8, label=f"Baseline AUC = {baseline:.4f}")
    ax.axhline(threshold, color="grey", ls=":", lw=0.7,
                label=f"Non-inferiority cutoff = {threshold:.4f}")
    ax.set_ylabel("Candidate replacement AUC (5-fold CV)")
    ax.set_title(
        "Distribution of replacement candidate AUCs per PC panel gene "
        "(sorted by genome-wide replaceability)\n"
        "Red = 15-crit panel member; grey = dropped by greedy elimination.",
        fontsize=10,
    )
    ax.tick_params(axis="x", rotation=35)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    out = SCRIPT_DIR / "repl_candidate_boxplot.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    return out, {"n_genes": len(data)}


# ---------------------------------------------------------------------------
# Figure 3: substitution-augmented ROC overlay
# ---------------------------------------------------------------------------

# These are the cross_cohort_substitution.py findings:
_GAP_WINS = [
    {"cohort": "GSE76220", "missing": "HERC2P8", "sub_symbol": "ZNF586",
      "sub_ensg": None,  # filled in at runtime
      "use_symbol_match": True, "baseline_auc": 0.9688, "sub_auc": 0.9792},
    {"cohort": "GSE122649", "missing": "HERC2P8", "sub_symbol": "XAF1",
      "sub_ensg": None,
      "use_symbol_match": True, "baseline_auc": 0.8269, "sub_auc": 0.8590},
]

# Need to find the ENSG IDs of ZNF586, XAF1 via gene_replacement_results.csv
def _resolve_sub_ensg(repl_df: pd.DataFrame) -> dict:
    """Pull the candidate ENSG for each substitution from the replacement table."""
    import mygene
    mg = mygene.MyGeneInfo()
    syms = list({w["sub_symbol"] for w in _GAP_WINS})
    res = mg.querymany(syms, scopes="symbol", fields="ensembl.gene",
                        species="human", verbose=False)
    out = {}
    for r in res:
        if r.get("notfound"):
            continue
        sym = r["query"]
        ens = r.get("ensembl", {})
        if isinstance(ens, list):
            ens = ens[0]
        eg = ens.get("gene") if ens else None
        if eg:
            out[sym] = eg
    return out


def fig_substitution_roc() -> tuple[Path, dict]:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_curve, roc_auc_score
    warnings.filterwarnings("ignore")

    params_raw = json.loads(_PARAMS_PATH.read_text())
    params = dict(params_raw, colsample_bytree=1.0, n_jobs=-1, verbose=-1)

    df_panel = pd.read_csv(_PANEL_CSV)
    feat_col = next(c for c in df_panel.columns
                     if "feature" in c.lower() or "ensg" in c.lower())
    sym_col = next(c for c in df_panel.columns if "symbol" in c.lower())
    feat25 = df_panel[feat_col].tolist()
    sym25 = df_panel[sym_col].tolist()
    feat25_bases = [f.split(".")[0] for f in feat25]
    panel_syms = [sym25[i] for i in _CRITICAL_IDX]
    panel_bases = [feat25_bases[i] for i in _CRITICAL_IDX]

    print("Loading GPL24676 ...")
    _, y_train, X_train_log, feat_train = _load_gpl24676_ctd()
    feat_base_to_col = {f.split(".")[0]: j for j, f in enumerate(feat_train)}
    print("Loading GSE76220 ...")
    X76, vocab76, y76 = _load_gse76220()
    v76 = {s: i for i, s in enumerate(vocab76)}
    print("Loading GSE122649 ...")
    X122, vocab122, y122 = _load_gse122649()
    v122 = {s: i for i, s in enumerate(vocab122)}

    sub_ensg = _resolve_sub_ensg(pd.read_csv(_REPL_CSV))

    cohorts = {
        "GSE76220":  (y76,  X76,  v76,  True),
        "GSE122649": (y122, X122, v122, True),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, win in zip(axes, [_GAP_WINS[0], _GAP_WINS[1]]):
        cohort = win["cohort"]
        y_te, X_te, v_map, by_sym = cohorts[cohort]
        # Baseline panel: drop everything cohort-missing.
        missing_in_cohort = set()
        for i, (sym, base) in enumerate(zip(panel_syms, panel_bases)):
            key = sym if by_sym else base
            if key not in v_map:
                missing_in_cohort.add(i)
        included = [i for i in range(15) if i not in missing_in_cohort]

        # Build baseline matrices
        train_cols = [feat_base_to_col[panel_bases[i]] for i in included]
        Xtr = X_train_log[:, train_cols]
        te_keys = [panel_syms[i] if by_sym else panel_bases[i] for i in included]
        Xte = X_te[:, [v_map[k] for k in te_keys]]
        sc = StandardScaler().fit(Xtr)
        scores_base = _ensemble_scores(sc.transform(Xtr), y_train,
                                          sc.transform(Xte), params,
                                          seeds=tuple(range(5)))
        auc_base = roc_auc_score(y_te, scores_base)
        fpr_b, tpr_b, _ = roc_curve(y_te, scores_base)

        # Substitution: add the substitute column to baseline
        eg = sub_ensg.get(win["sub_symbol"])
        if eg is None or eg not in feat_base_to_col:
            print(f"  No ENSG for {win['sub_symbol']} — skip plot for {cohort}")
            continue
        if win["sub_symbol"] not in v_map:
            print(f"  {win['sub_symbol']} not in {cohort} vocab — skip")
            continue
        tr_col = X_train_log[:, feat_base_to_col[eg]].reshape(-1, 1)
        te_col = X_te[:, v_map[win["sub_symbol"]]].reshape(-1, 1)
        Xtr2 = np.hstack([Xtr, tr_col])
        Xte2 = np.hstack([Xte, te_col])
        sc2 = StandardScaler().fit(Xtr2)
        scores_sub = _ensemble_scores(sc2.transform(Xtr2), y_train,
                                         sc2.transform(Xte2), params,
                                         seeds=tuple(range(5)))
        auc_sub = roc_auc_score(y_te, scores_sub)
        fpr_s, tpr_s, _ = roc_curve(y_te, scores_sub)

        # Plot
        ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=0.7)
        ax.plot(fpr_b, tpr_b, color="#888", lw=2.0,
                  label=f"Baseline 14/15 (drop {win['missing']})  AUC = {auc_base:.3f}")
        ax.plot(fpr_s, tpr_s, color="#1565C0", lw=2.0,
                  label=f"With {win['missing']}$\\to${win['sub_symbol']}  AUC = {auc_sub:.3f}")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"{cohort} (n={len(y_te)})", fontsize=11)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Substitution-augmented ROC: filling pseudogene gaps with protein-coding "
        "surrogates from the genome-wide replaceability screen",
        fontsize=11,
    )
    plt.tight_layout()
    out = SCRIPT_DIR / "substitution_roc.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    return out, {"plot": str(out)}


# ---------------------------------------------------------------------------
# Figure 4: candidate biotype heatmap
# ---------------------------------------------------------------------------

def fig_module_heatmap() -> tuple[Path, dict]:
    df = pd.read_csv(_REPL_CSV)
    df = df[df["is_replacement"]]
    pc_genes = [
        "MECOM", "SERTAD1", "FCN3", "PROS1", "ANGPT2", "EMP1", "TINAGL1",
        "CKMT2", "VWF", "CLDN5", "NR4A1", "SOHLH2", "HEXB", "MCEE", "SLC37A2",
    ]
    critical_pc = {"FCN3", "PROS1", "ANGPT2", "TINAGL1", "CKMT2", "CLDN5",
                    "NR4A1", "SOHLH2", "HEXB", "MCEE", "SLC37A2", "SERTAD1"}

    # Pre-resolve biotypes
    import mygene
    mg = mygene.MyGeneInfo()
    uniq = sorted(df["candidate_ensg"].unique().tolist())
    print(f"Resolving biotypes for {len(uniq)} candidate ENSG ...")
    res = mg.querymany(uniq, scopes="ensembl.gene", fields="type_of_gene,symbol",
                        species="human", verbose=False)
    biotype_map = {}
    artifact_prefixes = ("OR", "TAS", "PATE", "DEFB", "KRTAP", "VN1R",
                          "LCE", "SPRR", "MS4A")
    for r in res:
        if r.get("notfound"):
            biotype_map[r["query"]] = "unresolved"
            continue
        bt = r.get("type_of_gene", "")
        sym = r.get("symbol", "")
        if bt == "protein-coding":
            if any(sym.startswith(p) for p in artifact_prefixes):
                biotype_map[r["query"]] = "artifact-PC"
            else:
                biotype_map[r["query"]] = "protein-coding"
        elif bt in ("pseudo", ""):
            biotype_map[r["query"]] = "pseudogene"
        elif bt == "ncRNA":
            biotype_map[r["query"]] = "ncRNA"
        else:
            biotype_map[r["query"]] = "other"

    biotypes = ["protein-coding", "pseudogene", "ncRNA",
                 "artifact-PC", "other", "unresolved"]
    repl_pct = _parse_clean_repl(_REPL_CLEAN.read_text())
    pc_sorted = sorted(pc_genes, key=lambda g: repl_pct.get(g, 0))
    matrix = np.zeros((len(pc_sorted), len(biotypes)), dtype=int)
    for i, g in enumerate(pc_sorted):
        sub = df[df["panel_symbol"] == g]
        if sub.empty:
            continue
        for ensg in sub["candidate_ensg"]:
            bt = biotype_map.get(ensg, "unresolved")
            if bt in biotypes:
                matrix[i, biotypes.index(bt)] += 1

    fig, ax = plt.subplots(figsize=(9, 6.5))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(biotypes)))
    ax.set_xticklabels(biotypes, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pc_sorted)))
    ax.set_yticklabels(
        [f"{g}{'★' if g in critical_pc else ''} ({repl_pct.get(g, 0):.0f}%)"
         for g in pc_sorted], fontsize=9,
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=7.5,
                          color="white" if v > matrix.max() / 2 else "black")
    plt.colorbar(im, ax=ax, label="N valid replacement candidates")
    ax.set_title(
        "Valid replacement candidates per PC panel gene, broken down by biotype\n"
        "(★ = member of 15-gene greedy-tail critical panel; sorted by overall replaceability)",
        fontsize=10,
    )
    plt.tight_layout()
    out = SCRIPT_DIR / "repl_module_heatmap.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    return out, {"shape": matrix.shape}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Generating 4 replaceability figures")
    print("=" * 70)
    for name, fn in [
        ("Figure 1: replaceability vs greedy step", fig_repl_vs_greedy),
        ("Figure 2: candidate-AUC distribution boxplot", fig_candidate_boxplot),
        ("Figure 4: candidate biotype heatmap", fig_module_heatmap),
        ("Figure 3: substitution-augmented ROC", fig_substitution_roc),
    ]:
        print(f"\n>>> {name}")
        try:
            out, meta = fn()
            print(f"  saved: {out}")
            for k, v in meta.items():
                print(f"  {k}: {v}")
        except Exception as exc:
            print(f"  FAILED: {exc!r}")


if __name__ == "__main__":
    main()
