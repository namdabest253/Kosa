import { useCallback, useRef, useState } from "react";
import Graph from "graphology";
import { getNeighborhood, type GraphData } from "../lib/api";
import { edgeTypeColor } from "../lib/colors";

/**
 * Manages a graphology instance, merging subgraphs fetched from the API.
 * Returns the graph and a function to expand from a node.
 */
export function useGraphData() {
  const graphRef = useRef(new Graph({ multi: true, type: "directed" }));
  const [version, setVersion] = useState(0);

  const mergeGraphData = useCallback((data: GraphData) => {
    const g = graphRef.current;

    // Build a quick lookup of edges so we can find neighbors for positioning
    const edgeIndex = new Map<string, string[]>();
    for (const edge of data.edges) {
      if (!edgeIndex.has(edge.source)) edgeIndex.set(edge.source, []);
      if (!edgeIndex.has(edge.target)) edgeIndex.set(edge.target, []);
      edgeIndex.get(edge.source)!.push(edge.target);
      edgeIndex.get(edge.target)!.push(edge.source);
    }

    for (const node of data.nodes) {
      const attrs = node.attributes;

      // Delete backend size so sigma uses defaultNodeSize
      delete (attrs as Record<string, unknown>).size;

      if (!g.hasNode(node.key)) {
        // Position new nodes near an existing neighbor, or near origin with small jitter
        if ((!attrs.x || attrs.x === 0) && (!attrs.y || attrs.y === 0)) {
          const neighbors = edgeIndex.get(node.key) ?? [];
          let placed = false;

          for (const nb of neighbors) {
            if (g.hasNode(nb)) {
              const nbx = g.getNodeAttribute(nb, "x") as number;
              const nby = g.getNodeAttribute(nb, "y") as number;
              // Place in a small circle around the neighbor
              const angle = Math.random() * 2 * Math.PI;
              const r = 1.5 + Math.random() * 1.5;
              attrs.x = nbx + r * Math.cos(angle);
              attrs.y = nby + r * Math.sin(angle);
              placed = true;
              break;
            }
          }

          if (!placed) {
            attrs.x = (Math.random() - 0.5) * 3;
            attrs.y = (Math.random() - 0.5) * 3;
          }
        }

        g.addNode(node.key, attrs);
      } else {
        g.mergeNodeAttributes(node.key, attrs);
      }
    }

    for (const edge of data.edges) {
      // Delete backend size so sigma uses defaultEdgeSize
      delete (edge.attributes as Record<string, unknown>).size;

      // Color-code edges by type
      const et = edge.attributes.edge_type;
      if (et) {
        (edge.attributes as Record<string, unknown>).color = edgeTypeColor(et);
      }

      if (!g.hasEdge(edge.key)) {
        try {
          g.addEdgeWithKey(edge.key, edge.source, edge.target, edge.attributes);
        } catch {
          // source or target missing — skip
        }
      }
    }

    setVersion((v) => v + 1);
  }, []);

  const expandNode = useCallback(
    async (nodeId: string, depth = 1) => {
      const resp = await getNeighborhood(nodeId, depth);
      mergeGraphData(resp.graph);
      return resp;
    },
    [mergeGraphData],
  );

  const clearGraph = useCallback(() => {
    graphRef.current.clear();
    setVersion((v) => v + 1);
  }, []);

  return {
    graph: graphRef.current,
    version,
    expandNode,
    mergeGraphData,
    clearGraph,
  };
}
