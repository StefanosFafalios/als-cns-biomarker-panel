"""Causal structure learning on the 25-gene panel using PC + Fisher-Z CI test.

Replaces prior-knowledge STRING edges with statistically grounded conditional
independence edges learned directly from the 874-sample GSE153960 expression matrix.

Two genes are connected if their expression is conditionally dependent after
controlling for all other panel members (PC skeleton, alpha=0.05 Fisher-Z).

Outputs
-------
  causal_panel_network.png     — network visualisation coloured by theme/direction
  causal_panel_network_stats.txt — edge list + comparison with STRING edges
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
_PREFILTER_X   = SCRIPT_DIR / "lgbm_prefilter_X.npy"
_PREFILTER_NAMES = SCRIPT_DIR / "lgbm_prefilter_names.txt"
_PANEL_CSV     = SCRIPT_DIR / "lgbm_core25_panel.csv"

ALPHA = 0.01  # CI test significance level

_THEME = {
    "MECOM": "Vascular/BBB", "SERTAD1": "Neurodegeneration",
    "FCN3": "Neuroinflammation", "PROS1": "Neuroinflammation",
    "ANGPT2": "Vascular/BBB", "EMP1": "Neuroinflammation",
    "TINAGL1": "Vascular/BBB", "CKMT2": "Neurodegeneration",
    "VWF": "Vascular/BBB", "CLDN5": "Vascular/BBB",
    "NR4A1": "Neuroinflammation", "SOHLH2": "Microglial",
    "HEXB": "Microglial", "MCEE": "Microglial", "SLC37A2": "Microglial",
}
_DIRECTION = {
    "MECOM": "down", "SERTAD1": "up", "FCN3": "up", "PROS1": "up",
    "ANGPT2": "down", "EMP1": "up", "TINAGL1": "down", "CKMT2": "down",
    "VWF": "down", "CLDN5": "down", "NR4A1": "down", "SOHLH2": "up",
    "HEXB": "up", "MCEE": "up", "SLC37A2": "up",
}
_COLOR_THEME = {
    "Vascular/BBB": "#1565C0",
    "Neuroinflammation": "#C62828",
    "Neurodegeneration": "#6A1B9A",
    "Microglial": "#2E7D32",
}
# STRING medium-confidence edges (combined score >= 0.4), matching the STRING
# network figure (string_network.py / Fig. S24).
_STRING_EDGES = {
    ("VWF", "ANGPT2"), ("VWF", "CLDN5"), ("VWF", "PROS1"), ("ANGPT2", "CLDN5"),
}


def _load_panel_expression() -> tuple[np.ndarray, list[str]]:
    """Load expression matrix for the 15 protein-coding panel genes."""
    panel_df = pd.read_csv(_PANEL_CSV)
    protein_coding = panel_df[~panel_df["symbol"].str.startswith(
        ("ENSG", "LOC", "RPL", "RPS", "SNORD", "MAP3K2-DT", "RHOT1P2")
    )]
    symbols = protein_coding["symbol"].tolist()
    ensg_ids = protein_coding["feature"].str.split(".").str[0].tolist()

    names = Path(_PREFILTER_NAMES).read_text().splitlines()
    names_base = [n.split(".")[0] for n in names]

    cols = []
    kept_symbols = []
    for ensg, sym in zip(ensg_ids, symbols):
        if ensg in names_base:
            cols.append(names_base.index(ensg))
            kept_symbols.append(sym)

    print(f"  Panel genes found in prefilter matrix: {len(cols)}/{len(symbols)}")
    X_full = np.load(_PREFILTER_X, mmap_mode="r")
    X_panel = X_full[:, cols].astype(np.float64)
    return X_panel, kept_symbols


def _run_pc(X: np.ndarray, alpha: float) -> np.ndarray:
    """Run PC algorithm and return adjacency matrix (skeleton only)."""
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz

    print(f"  Running PC (Fisher-Z, alpha={alpha}) on {X.shape[1]} genes × {X.shape[0]} samples ...")
    cg = pc(X, alpha=alpha, indep_test=fisherz, stable=True, uc_rule=0, verbose=False)
    # Return the undirected skeleton adjacency (ignore orientation)
    adj = (cg.G.graph != 0).astype(int)
    # Symmetrise — skeleton is undirected
    return ((adj + adj.T) > 0).astype(int)


def _plot(G: nx.Graph, symbols: list[str], string_edges: set[frozenset]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Panel A — causal skeleton network
    ax = axes[0]
    pos = nx.spring_layout(G, seed=42, k=3.0)
    node_colors = [_COLOR_THEME.get(_THEME.get(n, ""), "#9E9E9E") for n in G.nodes()]
    shared = {frozenset(e) for e in G.edges()} & string_edges
    edge_colors = [
        "#FF6F00" if frozenset(e) in shared else "#546E7A"
        for e in G.edges()
    ]
    node_coll = nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=1200, ax=ax, alpha=0.9
    )
    node_coll.set_zorder(3)  # nodes drawn on top of the legend
    lbls = nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)
    for _t in lbls.values():
        _t.set_zorder(4)
    if G.edges():
        edge_coll = nx.draw_networkx_edges(
            G, pos, edge_color=edge_colors, width=2.5, ax=ax, alpha=0.75
        )
        try:
            edge_coll.set_zorder(1)
        except AttributeError:  # directed graphs return a list of patches
            for _e in edge_coll:
                _e.set_zorder(1)
    for theme, color in _COLOR_THEME.items():
        ax.plot([], [], "o", color=color, label=theme, markersize=10)
    ax.plot([], [], "-", color="#FF6F00", label="Also in STRING", linewidth=2)
    ax.plot([], [], "-", color="#546E7A", label="Causal only", linewidth=2)
    leg = ax.legend(loc="lower left", fontsize=9, framealpha=0.85)
    leg.set_zorder(2)  # legend sits behind the network nodes
    ax.set_title(
        f"PC causal skeleton (Fisher-Z, α={ALPHA})\n"
        f"{G.number_of_nodes()} nodes · {G.number_of_edges()} conditional-dependence edges",
        fontsize=11,
    )
    ax.axis("off")

    # Panel B — degree comparison causal vs STRING
    ax = axes[1]
    causal_deg = dict(G.degree())
    str_deg = {n: 0 for n in symbols}
    for u, v in _STRING_EDGES:
        if u in str_deg:
            str_deg[u] += 1
        if v in str_deg:
            str_deg[v] += 1

    genes_sorted = sorted(causal_deg, key=causal_deg.get, reverse=True)
    x = np.arange(len(genes_sorted))
    w = 0.38
    causal_vals = [causal_deg[g] for g in genes_sorted]
    str_vals    = [str_deg.get(g, 0) for g in genes_sorted]
    ax.barh(x + w/2, causal_vals, w, color="#1B5E20", label="PC causal")
    ax.barh(x - w/2, str_vals,    w, color="#E65100", label="STRING (score≥0.4)", alpha=0.75)
    ax.set_yticks(x)
    ax.set_yticklabels(genes_sorted, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Degree", fontsize=10)
    ax.set_title("Degree: PC causal vs STRING", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    plt.suptitle(
        "Data-driven causal network — 25-gene ALS panel (protein-coding)\n"
        "GSE153960 GPL24676 · n=874 samples",
        fontsize=12,
    )
    plt.tight_layout()
    out = SCRIPT_DIR / "causal_panel_network.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out.name}")


def main() -> None:
    print("\n" + "=" * 65)
    print("PC Causal Skeleton — ALS Panel Genes")
    print("=" * 65)

    X, symbols = _load_panel_expression()
    adj = _run_pc(X, ALPHA)

    G = nx.Graph()
    G.add_nodes_from(symbols)
    n = len(symbols)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                G.add_edge(symbols[i], symbols[j])
                edges.append((symbols[i], symbols[j]))

    string_edges = {frozenset(e) for e in _STRING_EDGES}
    shared = {frozenset(e) for e in edges} & string_edges
    causal_only = {frozenset(e) for e in edges} - string_edges
    string_only = string_edges - {frozenset(e) for e in edges}

    print(f"\n  Causal edges (PC skeleton): {len(edges)}")
    print(f"  STRING edges (score>=0.4):     {len(string_edges)}")
    print(f"  Shared:                     {len(shared)}")
    print(f"  Causal-only:                {len(causal_only)}")
    print(f"  STRING-only:                {len(string_only)}")
    print("\n  Edge list:")
    for u, v in sorted(edges):
        tag = " [also STRING]" if frozenset((u, v)) in string_edges else ""
        print(f"    {u} — {v}{tag}")

    print("\n  Degree ranking:")
    for g, d in sorted(G.degree(), key=lambda x: -x[1]):
        print(f"    {g:<15} degree={d}")

    lines = [
        "PC Causal Skeleton — ALS Panel (protein-coding genes)",
        "=" * 65,
        f"Method: PC algorithm, Fisher-Z CI test, alpha={ALPHA}",
        f"Data: GSE153960 GPL24676, n=874 samples",
        f"Genes: {', '.join(symbols)}",
        "",
        f"Causal edges (skeleton): {len(edges)}",
        f"STRING edges (score>=0.4):  {len(string_edges)}",
        f"Shared (agree):          {len(shared)}",
        f"Causal-only:             {len(causal_only)}",
        f"STRING-only (not found): {len(string_only)}",
        "",
        "Edge list:",
    ]
    for u, v in sorted(edges):
        tag = " [also in STRING]" if frozenset((u, v)) in string_edges else ""
        lines.append(f"  {u} — {v}{tag}")
    lines += ["", "Degree ranking:"]
    for g, d in sorted(G.degree(), key=lambda x: -x[1]):
        lines.append(f"  {g:<15} degree={d}")

    stat_out = SCRIPT_DIR / "causal_panel_network_stats.txt"
    stat_out.write_text("\n".join(lines))
    print(f"  Saved → {stat_out.name}")

    _plot(G, symbols, string_edges)

    print("\n" + "=" * 65)
    print("DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
