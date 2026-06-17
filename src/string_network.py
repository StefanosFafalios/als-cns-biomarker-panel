"""
STRING Protein-Protein Interaction Network — ALS Panel (15 protein-coding genes)

Queries STRING DB REST API for high-confidence interactions among the protein-coding
panel members, computes degree centrality, and retrieves STRING functional enrichment.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import requests

SCRIPT_DIR = Path(__file__).parent
STRING_API = "https://string-db.org/api/json"
SPECIES = 9606

_PROTEIN_CODING = [
    "MECOM", "SERTAD1", "FCN3", "PROS1", "ANGPT2", "EMP1",
    "TINAGL1", "CKMT2", "VWF", "CLDN5", "NR4A1", "SOHLH2",
    "HEXB", "MCEE", "SLC37A2",
]

_DIRECTION: dict[str, str] = {
    "MECOM": "down", "SERTAD1": "up", "FCN3": "up", "PROS1": "up",
    "ANGPT2": "down", "EMP1": "up", "TINAGL1": "down", "CKMT2": "down",
    "VWF": "down", "CLDN5": "down", "NR4A1": "down", "SOHLH2": "up",
    "HEXB": "up", "MCEE": "up", "SLC37A2": "up",
}

_THEME: dict[str, str] = {
    "MECOM": "Vascular/BBB", "VWF": "Vascular/BBB", "CLDN5": "Vascular/BBB",
    "ANGPT2": "Vascular/BBB", "TINAGL1": "Vascular/BBB",
    "FCN3": "Neuroinflammation", "PROS1": "Neuroinflammation",
    "NR4A1": "Neuroinflammation", "EMP1": "Neuroinflammation",
    "SERTAD1": "Neurodegeneration", "CKMT2": "Neurodegeneration",
    "HEXB": "Microglial", "SLC37A2": "Microglial",
    "MCEE": "Microglial", "SOHLH2": "Microglial",
}

_COLOR_THEME = {
    "Vascular/BBB": "#1565C0",
    "Neuroinflammation": "#C62828",
    "Neurodegeneration": "#6A1B9A",
    "Microglial": "#2E7D32",
}


def _post(endpoint: str, data: dict) -> list[dict]:
    for attempt in range(3):
        try:
            r = requests.post(f"{STRING_API}/{endpoint}", data=data, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    return []


def _get_string_ids() -> dict[str, str]:
    """Map gene symbols → STRING IDs."""
    data = _post("get_string_ids", {
        "identifiers": "\r".join(_PROTEIN_CODING),
        "species": SPECIES,
        "limit": 1,
        "caller_identity": "als_panel_analysis",
    })
    id_map: dict[str, str] = {}
    for item in data:
        symbol = item.get("preferredName") or item.get("queryItem", "")
        if symbol in _PROTEIN_CODING or item.get("queryItem") in _PROTEIN_CODING:
            key = item.get("queryItem", symbol)
            id_map[key] = item.get("stringId", "")
    return id_map


def _get_network(string_ids: list[str], min_score: int) -> list[dict]:
    return _post("network", {
        "identifiers": "\r".join(string_ids),
        "species": SPECIES,
        "required_score": min_score,
        "caller_identity": "als_panel_analysis",
    })


def _get_enrichment(string_ids: list[str]) -> list[dict]:
    return _post("enrichment", {
        "identifiers": "\r".join(string_ids),
        "species": SPECIES,
        "caller_identity": "als_panel_analysis",
    })


def _build_graph(edges: list[dict], id_to_symbol: dict[str, str]) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(_PROTEIN_CODING)
    for e in edges:
        a = id_to_symbol.get(e.get("stringId_A", ""),
                              e.get("preferredName_A", ""))
        b = id_to_symbol.get(e.get("stringId_B", ""),
                              e.get("preferredName_B", ""))
        if a in _PROTEIN_CODING and b in _PROTEIN_CODING and a != b:
            G.add_edge(a, b, weight=float(e.get("score", 0)))
    return G


def main() -> None:
    print("=" * 65)
    print("STRING Protein-Protein Interaction Network")
    print("=" * 65)

    print("\n[1] Mapping symbols to STRING IDs ...")
    id_map = _get_string_ids()
    string_ids = list(id_map.values())
    id_to_sym = {v: k for k, v in id_map.items()}
    print(f"  Mapped: {len(id_map)}/{len(_PROTEIN_CODING)} genes")

    print("[2] Fetching network (medium confidence ≥ 0.4) ...")
    edges_med = _get_network(string_ids, min_score=400)
    print(f"  Medium-confidence edges: {len(edges_med)}")

    print("[3] Fetching network (high confidence ≥ 0.7) ...")
    edges_high = _get_network(string_ids, min_score=700)
    print(f"  High-confidence edges: {len(edges_high)}")

    print("[4] Fetching STRING functional enrichment ...")
    try:
        enrich = _get_enrichment(string_ids)
        enrich_df = pd.DataFrame(enrich)
    except Exception as e:
        print(f"  Enrichment failed: {e}")
        enrich_df = pd.DataFrame()

    # Build graphs
    G_med = _build_graph(edges_med, id_to_sym)
    G_high = _build_graph(edges_high, id_to_sym)

    # Degree and betweenness centrality on medium-confidence graph
    degree = dict(G_med.degree())
    degree_df = (
        pd.DataFrame({"Gene": list(degree.keys()), "Degree": list(degree.values())})
        .sort_values("Degree", ascending=False)
        .reset_index(drop=True)
    )
    bc = nx.betweenness_centrality(G_med, weight="weight")
    degree_df["Betweenness"] = degree_df["Gene"].map(bc)

    # Save edges
    edge_rows = []
    for e in edges_med:
        a = id_to_sym.get(e.get("stringId_A", ""), e.get("preferredName_A", ""))
        b = id_to_sym.get(e.get("stringId_B", ""), e.get("preferredName_B", ""))
        edge_rows.append({"Gene_A": a, "Gene_B": b, "Score": e.get("score", 0),
                          "EvidenceScore": e.get("escore", 0),
                          "TextMiningScore": e.get("tscore", 0)})
    pd.DataFrame(edge_rows).to_csv(SCRIPT_DIR / "string_network_edges.csv", index=False)
    degree_df.to_csv(SCRIPT_DIR / "string_network_degree.csv", index=False)
    if not enrich_df.empty:
        enrich_df.to_csv(SCRIPT_DIR / "string_enrichment.csv", index=False)

    _plot(G_med, G_high, degree_df, enrich_df)
    _write_statistics(G_med, G_high, degree_df, enrich_df, edge_rows)
    print("\nDone. Outputs: string_network.png  string_network_statistics.txt")


def _plot(
    G_med: nx.Graph,
    G_high: nx.Graph,
    degree_df: pd.DataFrame,
    enrich_df: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    # Panel A — medium-confidence network
    ax = axes[0]
    pos = nx.spring_layout(G_med, seed=42, k=2.5)
    node_colors = [_COLOR_THEME.get(_THEME.get(n, ""), "#9E9E9E") for n in G_med.nodes()]
    weights = [G_med[u][v].get("weight", 0.4) for u, v in G_med.edges()]
    nx.draw_networkx_nodes(G_med, pos, node_color=node_colors, node_size=1100, ax=ax, alpha=0.9)
    nx.draw_networkx_labels(G_med, pos, font_size=10, font_weight="bold", ax=ax)
    if G_med.edges():
        nx.draw_networkx_edges(G_med, pos, width=[w * 4 for w in weights],
                               alpha=0.55, edge_color="#78909C", ax=ax)
    # Legend
    for theme, color in _COLOR_THEME.items():
        ax.plot([], [], "o", color=color, label=theme, markersize=10)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.8)
    ax.set_title(f"STRING network (score ≥ 0.4)\n{G_med.number_of_edges()} edges", fontsize=11)
    ax.axis("off")

    # Panel B — degree ranking
    ax = axes[1]
    colors = [_COLOR_THEME.get(_THEME.get(g, ""), "#9E9E9E") for g in degree_df["Gene"]]
    ax.barh(degree_df["Gene"].values[::-1], degree_df["Degree"].values[::-1],
            color=colors[::-1])
    ax.set_xlabel("Degree (number of interactions, score ≥ 0.4)", fontsize=10)
    ax.set_title("Interaction degree per gene", fontsize=11)
    if degree_df["Degree"].max() > 0:
        mean_deg = degree_df["Degree"].mean()
        ax.axvline(mean_deg, color="black", linestyle="--", alpha=0.5, linewidth=1)
        ax.text(mean_deg + 0.05, 0.5, f"mean={mean_deg:.1f}",
                transform=ax.get_yaxis_transform(), fontsize=9)
    ax.tick_params(axis="y", labelsize=10)

    # Panel C — STRING enrichment (top terms)
    ax = axes[2]
    if not enrich_df.empty and "description" in enrich_df.columns:
        fdr_col = "fdr" if "fdr" in enrich_df.columns else "p_value"
        if fdr_col in enrich_df.columns:
            sig = enrich_df[enrich_df[fdr_col] < 0.05].head(15)
        else:
            sig = enrich_df.head(15)
        if not sig.empty:
            neg_log_p = -np.log10(sig[fdr_col].clip(1e-300).values[::-1])
            ax.barh(sig["description"].values[::-1], neg_log_p, color="#FF8F00")
            ax.axvline(-np.log10(0.05), color="red", linestyle="--",
                       linewidth=1, label="FDR=0.05")
            ax.set_xlabel("-log10(FDR)", fontsize=10)
            ax.set_title("STRING functional enrichment\n(FDR < 0.05)", fontsize=11)
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, "No enrichment\nat FDR < 0.05", ha="center",
                    va="center", transform=ax.transAxes, fontsize=11)
            ax.set_title("STRING functional enrichment", fontsize=11)
    else:
        ax.text(0.5, 0.5, "Enrichment\nnot available", ha="center",
                va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("STRING functional enrichment", fontsize=11)
    ax.tick_params(axis="y", labelsize=10)

    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / "string_network.png", dpi=150, bbox_inches="tight")
    plt.close()


def _write_statistics(
    G_med: nx.Graph,
    G_high: nx.Graph,
    degree_df: pd.DataFrame,
    enrich_df: pd.DataFrame,
    edge_rows: list[dict],
) -> None:
    import numpy as np
    lines = [
        "STRING Protein-Protein Interaction Network — ALS Panel",
        "=" * 65,
        f"Query: {', '.join(_PROTEIN_CODING)}",
        f"Species: Homo sapiens (9606)",
        "",
        f"Medium-confidence network (score ≥ 0.4):",
        f"  Nodes: {G_med.number_of_nodes()}",
        f"  Edges: {G_med.number_of_edges()}",
        f"  Density: {nx.density(G_med):.4f}",
        f"  Connected components: {nx.number_connected_components(G_med)}",
        "",
        f"High-confidence network (score ≥ 0.7):",
        f"  Edges: {G_high.number_of_edges()}",
        "",
        "DEGREE RANKING (medium-confidence)",
        "-" * 65,
        f"{'Gene':<15} {'Degree':>8} {'Betweenness':>14} {'Theme':<20} {'Direction':<10}",
        "-" * 65,
    ]
    for _, row in degree_df.iterrows():
        gene = row["Gene"]
        lines.append(
            f"{gene:<15} {int(row['Degree']):>8} {row['Betweenness']:>14.4f} "
            f"{_THEME.get(gene, ''):<20} {_DIRECTION.get(gene, ''):<10}"
        )

    lines += ["", "HIGH-CONFIDENCE EDGES (score ≥ 0.7)", "-" * 65]
    for e in edge_rows:
        if float(e.get("Score", 0)) >= 0.7:
            lines.append(f"  {e['Gene_A']} — {e['Gene_B']}  score={e['Score']:.3f}")

    if not enrich_df.empty and "description" in enrich_df.columns:
        fdr_col = "fdr" if "fdr" in enrich_df.columns else "p_value"
        sig = enrich_df[enrich_df.get(fdr_col, pd.Series([1.0] * len(enrich_df))) < 0.05]
        lines += ["", "STRING FUNCTIONAL ENRICHMENT (FDR < 0.05)", "-" * 65]
        for _, row in sig.head(30).iterrows():
            cat = row.get("category", "")
            desc = row.get("description", "")
            fdr = row.get(fdr_col, "")
            count = row.get("number_of_genes", "")
            lines.append(f"  [{cat}] {desc} — n={count}, FDR={fdr:.3e}"
                         if isinstance(fdr, float) else f"  [{cat}] {desc}")

    (SCRIPT_DIR / "string_network_statistics.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    import numpy as np
    main()
