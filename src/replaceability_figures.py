"""Generate four replaceability-related figures for the manuscript.

(1) repl_vs_greedy.png — replaceability% vs greedy backward elimination step
    for the 15 protein-coding panel members. Shows the convergence between
    two independent criteria.

(2) repl_candidate_boxplot.png — distribution of replacement candidate AUCs
    per PC panel gene. Highlights the spread that the single replaceability
    % number collapses.

(3) substitution_roc_all.png — for GSE76220 and GSE122649, overlay the ROC of
    the critical panel's cohort-present genes against the same panel with its
    vocabulary gaps filled by cross_cohort_substitution.py's surrogates.

(4) repl_module_heatmap.png — for each PC panel gene, the count of valid
    replacement candidates broken down by biotype (protein-coding /
    pseudogene / ncRNA / artifact) — shows that the "module density" varies
    not just in raw count but in candidate quality.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ALS_DIR = Path(__file__).parents[1]
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(ALS_DIR))

# Reuse loaders from the substitution script
from cross_cohort_substitution import (  # noqa: E402
    _ensemble_scores,
    _load_gpl24676_ctd,
    _load_gse76220,
    _load_gse122649,
)
from utils import (  # noqa: E402
    core_panel_symbols,
    critical_panel_symbols,
    zero_shot_scale,
)

from utils import selected_lgbm_params  # noqa: E402
_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_REPL_CSV = SCRIPT_DIR / "gene_replacement_results.csv"
_REPL_CLEAN = SCRIPT_DIR / "gene_replacement_clean_statistics.txt"
_ITER_STAT = SCRIPT_DIR / "iterative_panel_elimination_statistics.txt"


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
    pc_genes = core_panel_symbols(protein_coding_only=True)
    critical_pc = set(critical_panel_symbols(protein_coding_only=True))

    points = []
    for g in pc_genes:
        if g in drop_steps and g in repl:
            points.append(
                {
                    "gene": g,
                    "step": drop_steps[g],
                    "repl": repl[g],
                    "is_critical": g in critical_pc,
                }
            )

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for p in points:
        colour = "#C62828" if p["is_critical"] else "#888"
        marker = "o" if p["is_critical"] else "s"
        ax.scatter(
            p["repl"],
            p["step"],
            s=110,
            c=colour,
            marker=marker,
            edgecolor="black",
            lw=0.6,
            zorder=3,
        )
        # label offset
        ax.annotate(
            p["gene"],
            (p["repl"], p["step"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8.5,
        )

    # Quadrant guides. The retained-panel boundary follows the panel size, so a
    # rerun of the stopping rule moves the line instead of silently mislabelling it.
    k_crit = len(critical_panel_symbols())
    n_dropped = len(pd.read_csv(_PANEL_CSV)) - k_crit
    ax.axhline(
        n_dropped + 0.5,
        color="grey",
        ls="--",
        lw=0.7,
        label=(
            f"Greedy peak k={k_crit} (drop step {n_dropped}/{n_dropped + 1} boundary)"
        ),
    )
    ax.axvline(25, color="grey", ls=":", lw=0.7, label="Replaceability = 25%")

    ax.set_xlabel("Genome-wide replaceability (% of non-artifact candidates valid)")
    ax.set_ylabel(
        "Greedy backward elimination drop step\n"
        "(later = harder to drop = more anchored)"
    )
    # Derived from `points`, never hardcoded: panel membership changes whenever
    # the greedy stopping rule is rerun, and a baked-in gene list goes stale
    # silently (it did once already).
    low = sorted([p for p in points if p["repl"] < 25], key=lambda p: p["repl"])
    top3 = sorted(points, key=lambda p: -p["repl"])[:3]
    n_crit = len(critical_pc)
    ax.set_title(
        "Convergence of two independent criteria — replaceability and "
        "cross-cohort greedy elimination\n"
        f"{len(low)} hardest-to-replace genes "
        f"({', '.join(p['gene'] for p in low)}; <25%): "
        f"{sum(p['is_critical'] for p in low)}/{len(low)} in the "
        f"{n_crit}-PC critical panel; 3 most-replaceable PC genes "
        f"({', '.join(p['gene'] for p in top3)}): "
        f"{sum(not p['is_critical'] for p in top3)}/3 dropped.",
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
    pc_genes = core_panel_symbols(protein_coding_only=True)
    critical_pc = set(critical_panel_symbols(protein_coding_only=True))

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
    bp = ax.boxplot(
        data, labels=labels, patch_artist=True, showfliers=False, widths=0.7
    )
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    # Baseline reference
    # baseline written by gene_replacement.py (donor-grouped within-cohort model)
    baseline = float(
        re.search(
            r"Baseline panel AUC[^:]*:\s+([\d.]+)",
            (SCRIPT_DIR / "gene_replacement_statistics.txt").read_text(),
        ).group(1)
    )
    threshold = baseline - 0.002
    ax.axhline(
        baseline, color="black", ls="--", lw=0.8, label=f"Baseline AUC = {baseline:.4f}"
    )
    ax.axhline(
        threshold,
        color="grey",
        ls=":",
        lw=0.7,
        label=f"Non-inferiority cutoff = {threshold:.4f}",
    )
    ax.set_ylabel("Candidate replacement AUC (5-fold CV)")
    ax.set_title(
        "Distribution of replacement candidate AUCs per PC panel gene "
        "(sorted by genome-wide replaceability)\n"
        f"Red = critical-panel member ({len(critical_pc)} PC); "
        "grey = dropped by greedy elimination.",
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

_SUB_JSON = SCRIPT_DIR / "cross_cohort_substitution_surrogates.json"


def fig_substitution_roc() -> tuple[Path, dict]:
    """Baseline vs surrogate-filled critical panel on GSE76220 and GSE122649.

    The surrogates are read from cross_cohort_substitution.py's output, never
    hardcoded: two earlier hardcoded sets survived reruns that had replaced them.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    params = dict(selected_lgbm_params(), n_jobs=-1, verbose=-1)
    subs = json.loads(_SUB_JSON.read_text())  # {cohort: [[missing, symbol, ensg], ...]}
    panel = pd.read_csv(_PANEL_CSV)
    bases = dict(zip(panel["symbol"], panel["feature"].str.split(".").str[0]))
    crit = critical_panel_symbols()
    _, y_train, X_train_log, feat_train = _load_gpl24676_ctd()
    tr_col = {f.split(".")[0]: j for j, f in enumerate(feat_train)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    meta: dict[str, float] = {}
    for ax, (cohort, loader) in zip(axes, (("GSE76220", _load_gse76220), ("GSE122649", _load_gse122649))):
        X_te, vocab, y_te = loader()
        v = {sym: i for i, sym in enumerate(vocab)}
        present = [(sym, bases[sym]) for sym in crit if sym in v]
        fills = [(m, sym, e) for m, sym, e in subs.get(cohort, []) if sym in v and e in tr_col]
        arms = (
            (f"Baseline {len(present)}/{len(crit)}", present, "#888"),
            ("Filled: " + ", ".join(f"{m}$\\to${sym}" for m, sym, _ in fills),
             present + [(sym, e) for _, sym, e in fills], "#1565C0"),
        )
        for label, genes, colour in arms:
            Ztr, Zte = zero_shot_scale(
                X_train_log[:, [tr_col[b] for _, b in genes]], X_te[:, [v[g] for g, _ in genes]], cohort
            )
            scores = _ensemble_scores(Ztr, y_train, Zte, params, seeds=tuple(range(5)))
            fpr, tpr, _ = roc_curve(y_te, scores)
            auc = roc_auc_score(y_te, scores)
            ax.plot(fpr, tpr, color=colour, lw=2.0, label=f"{label}  AUC = {auc:.3f}")
            meta[f"{cohort} {label.split(':')[0].split(' ')[0]}"] = round(float(auc), 4)
        ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=0.7)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"{cohort} (n={len(y_te)})", fontsize=11)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Critical-panel vocabulary gaps filled with protein-coding surrogates "
        "(the valid protein-coding candidate most co-expressed with the missing gene)",
        fontsize=11,
    )
    plt.tight_layout()
    out = SCRIPT_DIR / "substitution_roc_all.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    return out, meta


# ---------------------------------------------------------------------------
# Figure 4: candidate biotype heatmap
# ---------------------------------------------------------------------------


def fig_module_heatmap() -> tuple[Path, dict]:
    df = pd.read_csv(_REPL_CSV)
    df = df[df["is_replacement"]]
    pc_genes = core_panel_symbols(protein_coding_only=True)
    critical_pc = set(critical_panel_symbols(protein_coding_only=True))

    # Pre-resolve biotypes
    import mygene

    mg = mygene.MyGeneInfo()
    uniq = sorted(df["candidate_ensg"].unique().tolist())
    print(f"Resolving biotypes for {len(uniq)} candidate ENSG ...")
    res = mg.querymany(
        uniq,
        scopes="ensembl.gene",
        fields="type_of_gene,symbol",
        species="human",
        verbose=False,
    )
    biotype_map = {}
    artifact_prefixes = (
        "OR",
        "TAS",
        "PATE",
        "DEFB",
        "KRTAP",
        "VN1R",
        "LCE",
        "SPRR",
        "MS4A",
    )
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

    biotypes = [
        "protein-coding",
        "pseudogene",
        "ncRNA",
        "artifact-PC",
        "other",
        "unresolved",
    ]
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
        [
            f"{g}{'★' if g in critical_pc else ''} ({repl_pct.get(g, 0):.0f}%)"
            for g in pc_sorted
        ],
        fontsize=9,
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if v:
                ax.text(
                    j,
                    i,
                    str(v),
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if v > matrix.max() / 2 else "black",
                )
    plt.colorbar(im, ax=ax, label="N valid replacement candidates")
    ax.set_title(
        "Valid replacement candidates per PC panel gene, broken down by biotype\n"
        "(★ = member of the greedy-tail critical panel; "
        "sorted by overall replaceability)",
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
