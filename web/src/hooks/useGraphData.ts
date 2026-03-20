import { useCallback, useRef, useState } from "react";
import Graph from "graphology";
import { getNeighborhood, type GraphData } from "../lib/api";

/**
 * Manages a graphology instance, merging subgraphs fetched from the API.
 * Returns the graph and a function to expand from a node.
 */
export function useGraphData() {
  const graphRef = useRef(new Graph({ multi: true, type: "directed" }));
  const [version, setVersion] = useState(0);

  const mergeGraphData = useCallback((data: GraphData) => {
    const g = graphRef.current;

    for (const node of data.nodes) {
      if (!g.hasNode(node.key)) {
        g.addNode(node.key, node.attributes);
      } else {
        g.mergeNodeAttributes(node.key, node.attributes);
      }
    }

    for (const edge of data.edges) {
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
