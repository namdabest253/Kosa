import { useCallback, useState } from "react";
import GraphView from "./components/GraphView";
import SearchBar from "./components/SearchBar";
import NodeDetail from "./components/NodeDetail";
import HypothesisList from "./components/HypothesisList";
import ActivationPanel from "./components/ActivationPanel";
import FilterPanel from "./components/FilterPanel";
import Dashboard from "./components/Dashboard";
import { useGraphData } from "./hooks/useGraphData";
import { getReasoningPath, type SearchResult } from "./lib/api";

type Tab = "graph" | "hypotheses" | "dashboard";

export default function App() {
  const { graph, version, expandNode, mergeGraphData, clearGraph } = useGraphData();
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("graph");
  const [nodeTypes, setNodeTypes] = useState<Record<string, boolean>>({
    Paper: true,
    Technique: true,
    Problem: true,
    Dataset: true,
  });
  const [minConfidence, setMinConfidence] = useState(0);

  const handleSearchSelect = useCallback(
    async (result: SearchResult) => {
      await expandNode(result.id);
      setSelectedNode(result.id);
    },
    [expandNode],
  );

  const handleNodeClick = useCallback(
    (nodeId: string) => {
      setSelectedNode(nodeId);
    },
    [],
  );

  const handleExpand = useCallback(
    async (nodeId: string) => {
      await expandNode(nodeId);
    },
    [expandNode],
  );

  const handleActivate = useCallback(
    (nodeId: string) => {
      setSelectedNode(nodeId);
      // ActivationPanel reads seedNodeId from selectedNode
    },
    [],
  );

  const handleSelectHypothesis = useCallback(
    async (hypothesisId: string) => {
      try {
        const pathData = await getReasoningPath(hypothesisId);
        if (pathData.nodes.length > 0) {
          clearGraph();
          mergeGraphData(pathData);
          setActiveTab("graph");
        }
      } catch (e) {
        console.error("Failed to load reasoning path:", e);
      }
    },
    [clearGraph, mergeGraphData],
  );

  const handleToggleType = useCallback((type: string) => {
    setNodeTypes((prev) => ({ ...prev, [type]: !prev[type] }));
  }, []);

  return (
    <div style={styles.app}>
      {/* Top bar */}
      <header style={styles.header}>
        <div style={styles.logo}>Kosa</div>
        <nav style={styles.nav}>
          {(["graph", "hypotheses", "dashboard"] as Tab[]).map((tab) => (
            <button
              key={tab}
              style={{
                ...styles.tab,
                ...(activeTab === tab ? styles.activeTab : {}),
              }}
              onClick={() => setActiveTab(tab)}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </nav>
        {activeTab === "graph" && (
          <div style={styles.searchWrapper}>
            <SearchBar onSelect={handleSearchSelect} />
          </div>
        )}
      </header>

      {/* Main content */}
      <div style={styles.content}>
        {activeTab === "graph" && (
          <>
            {/* Left sidebar: filters + activation */}
            <div style={styles.sidebar}>
              <FilterPanel
                nodeTypes={nodeTypes}
                onToggleType={handleToggleType}
                minConfidence={minConfidence}
                onConfidenceChange={setMinConfidence}
              />
              <ActivationPanel seedNodeId={selectedNode} />
            </div>

            {/* Center: graph */}
            <div style={styles.graphContainer}>
              {graph.order === 0 ? (
                <div style={styles.emptyState}>
                  Search for a node to start exploring the knowledge graph.
                </div>
              ) : (
                <GraphView
                  graph={graph}
                  version={version}
                  onClickNode={handleNodeClick}
                />
              )}
            </div>

            {/* Right panel: node detail */}
            <NodeDetail
              nodeId={selectedNode}
              onClose={() => setSelectedNode(null)}
              onExpand={handleExpand}
              onActivate={handleActivate}
            />
          </>
        )}

        {activeTab === "hypotheses" && (
          <div style={styles.fullPanel}>
            <HypothesisList onSelectHypothesis={handleSelectHypothesis} />
          </div>
        )}

        {activeTab === "dashboard" && (
          <div style={styles.fullPanel}>
            <Dashboard />
          </div>
        )}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  app: {
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    overflow: "hidden",
  },
  header: {
    display: "flex",
    alignItems: "center",
    padding: "10px 20px",
    borderBottom: "1px solid #e0e0e0",
    background: "#fff",
    gap: 16,
    zIndex: 100,
  },
  logo: {
    fontSize: 20,
    fontWeight: 700,
    color: "#1976D2",
    flexShrink: 0,
  },
  nav: { display: "flex", gap: 4 },
  tab: {
    padding: "6px 14px",
    border: "none",
    borderRadius: 6,
    background: "transparent",
    fontSize: 13,
    fontWeight: 500,
    cursor: "pointer",
    color: "#666",
  },
  activeTab: {
    background: "#E3F2FD",
    color: "#1976D2",
  },
  searchWrapper: { flex: 1, maxWidth: 500 },
  content: {
    display: "flex",
    flex: 1,
    overflow: "hidden",
    position: "relative" as const,
  },
  sidebar: {
    width: 260,
    borderRight: "1px solid #e0e0e0",
    background: "#fff",
    overflowY: "auto",
    flexShrink: 0,
  },
  graphContainer: {
    flex: 1,
    position: "relative" as const,
    background: "#fafafa",
  },
  emptyState: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    height: "100%",
    color: "#999",
    fontSize: 16,
  },
  fullPanel: {
    flex: 1,
    overflow: "hidden",
  },
};
