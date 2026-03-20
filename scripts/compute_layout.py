#!/usr/bin/env python3
"""Pre-compute ForceAtlas2 layout positions and write x/y back to Neo4j.

Fetches the full graph, computes layout via python-igraph, then writes
positions as node properties. Run after ingestion batches.

Usage:
    python scripts/compute_layout.py [--iterations 100]
"""

from __future__ import annotations

import argparse
import sys

from neo4j import GraphDatabase

from kosa.config import Settings


def compute_layout(uri: str, user: str, password: str, iterations: int = 100) -> int:
    """Fetch graph from Neo4j, compute ForceAtlas2 layout, write positions back."""
    import igraph as ig

    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        driver.verify_connectivity()

        # Fetch all nodes and edges
        with driver.session() as session:
            node_result = session.run(
                "MATCH (n) WHERE n:Paper OR n:Technique OR n:Problem OR n:Dataset "
                "RETURN elementId(n) AS id, labels(n)[0] AS label, "
                "n.significance AS sig"
            )
            nodes = list(node_result)

            edge_result = session.run(
                "MATCH (a)-[r]->(b) "
                "RETURN elementId(a) AS src, elementId(b) AS tgt, "
                "type(r) AS rtype, r.confidence AS conf"
            )
            edges = list(edge_result)

        if not nodes:
            print("No nodes found. Nothing to layout.")
            return 0

        # Build igraph graph
        id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}
        g = ig.Graph(n=len(nodes), directed=True)

        # Add node attributes
        g.vs["neo4j_id"] = [n["id"] for n in nodes]
        g.vs["label"] = [n["label"] for n in nodes]
        g.vs["weight"] = [max(0.5, n["sig"] or 0.5) for n in nodes]

        # Add edges (skip if node not in our set)
        edge_list = []
        edge_weights = []
        for e in edges:
            src_idx = id_to_idx.get(e["src"])
            tgt_idx = id_to_idx.get(e["tgt"])
            if src_idx is not None and tgt_idx is not None:
                edge_list.append((src_idx, tgt_idx))
                edge_weights.append(max(0.1, e["conf"] or 0.5))

        g.add_edges(edge_list)
        g.es["weight"] = edge_weights

        print(f"Graph: {g.vcount()} nodes, {g.ecount()} edges")

        # Compute ForceAtlas2 layout (igraph uses "fruchterman_reingold" as closest analog;
        # for true ForceAtlas2, use the layout_forceatlas2 method if available)
        try:
            layout = g.layout_forceatlas2(
                iterations=iterations,
                gravity=1.0,
                scaling_ratio=2.0,
            )
        except AttributeError:
            # Fallback to Fruchterman-Reingold if ForceAtlas2 not available
            print("ForceAtlas2 not available, falling back to Fruchterman-Reingold")
            layout = g.layout_fruchterman_reingold(
                niter=iterations,
                weights=edge_weights if edge_weights else None,
            )

        coords = layout.coords

        # Write positions back to Neo4j
        with driver.session() as session:
            for i, node in enumerate(nodes):
                x, y = coords[i]
                session.run(
                    "MATCH (n) WHERE elementId(n) = $id SET n.x = $x, n.y = $y",
                    {"id": node["id"], "x": float(x), "y": float(y)},
                )

        print(f"Wrote layout positions for {len(nodes)} nodes.")
        return len(nodes)

    finally:
        driver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ForceAtlas2 layout for Kosa KG")
    parser.add_argument("--iterations", type=int, default=100, help="Layout iterations")
    args = parser.parse_args()

    settings = Settings()
    if not settings.neo4j_password:
        print("ERROR: NEO4J_PASSWORD not set.", file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to {settings.neo4j_uri} ...")
    count = compute_layout(
        settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password, args.iterations
    )
    print(f"Done. {count} nodes positioned.")


if __name__ == "__main__":
    main()
