import { useCallback, useEffect, useState } from "react";
import GraphView from "./components/GraphView";
import SearchBar from "./components/SearchBar";
import NodeDetail from "./components/NodeDetail";
import HypothesisList from "./components/HypothesisList";
import ActivationPanel from "./components/ActivationPanel";
import FilterPanel from "./components/FilterPanel";
import Dashboard from "./components/Dashboard";
import { useGraphData } from "./hooks/useGraphData";
import { getAllNodes, getReasoningPath, type SearchResult } from "./lib/api";
import { ThemeContext, light, dark, type Theme } from "./lib/theme";

type Tab = "graph" | "hypotheses" | "dashboard";

export default function App() {
  const { graph, version, expandNode, mergeGraphData, clearGraph } = useGraphData();
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("graph");
  const [darkMode, setDarkMode] = useState(false);
  const [nodeTypes, setNodeTypes] = useState<Record<string, boolean>>({
    Paper: true,
    Technique: true,
    Problem: true,
    Dataset: true,
  });
  const [minConfidence, setMinConfidence] = useState(0);
  const [edgeTypes, setEdgeTypes] = useState<Record<string, boolean>>({
    CITES: true,
    INTRODUCES: true,
    EVALUATES_ON: true,
    HAS_LIMITATION: true,
    MITIGATES: true,
    IMPROVES_OVER: true,
    USES: true,
    IS_INSTANCE_OF: true,
    CAUSED_BY: true,
    TEMPORALLY_FOLLOWS: true,
    SAME_AS: true,
  });

  const theme = darkMode ? dark : light;

  // Load the full graph on mount
  useEffect(() => {
    getAllNodes()
      .then((data) => mergeGraphData(data))
      .catch((e) => console.error("Failed to load full graph:", e));
  }, [mergeGraphData]);

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
    async (nodeId: string, depth = 1) => {
      await expandNode(nodeId, depth);
    },
    [expandNode],
  );

  const handleActivate = useCallback(
    (nodeId: string) => {
      setSelectedNode(nodeId);
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

  const handleToggleEdgeType = useCallback((type: string) => {
    setEdgeTypes((prev) => ({ ...prev, [type]: !prev[type] }));
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      const inInput = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";

      if ((e.key === "/" || (e.key === "k" && (e.ctrlKey || e.metaKey))) && !inInput) {
        e.preventDefault();
        document.getElementById("kosa-search-input")?.focus();
        return;
      }

      if (e.key === "Escape") {
        setSelectedNode(null);
        (document.activeElement as HTMLElement)?.blur();
        return;
      }

      if (!inInput) {
        const tabs: Tab[] = ["graph", "hypotheses", "dashboard"];
        const idx = parseInt(e.key, 10) - 1;
        const tab = tabs[idx];
        if (tab) {
          setActiveTab(tab);
        }
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const s = mkStyles(theme);

  return (
    <ThemeContext.Provider value={theme}>
      <div style={s.app}>
        <header style={s.header}>
          <div style={s.logo}>Kosa</div>
          <nav style={s.nav}>
            {(["graph", "hypotheses", "dashboard"] as Tab[]).map((tab) => (
              <button
                key={tab}
                style={{
                  ...s.tab,
                  ...(activeTab === tab ? s.activeTab : {}),
                }}
                onClick={() => setActiveTab(tab)}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </nav>
          {activeTab === "graph" && (
            <div style={s.searchWrapper}>
              <SearchBar onSelect={handleSearchSelect} />
            </div>
          )}
          <button
            style={s.themeToggle}
            onClick={() => setDarkMode((d) => !d)}
            title={darkMode ? "Switch to light mode" : "Switch to dark mode"}
          >
            {darkMode ? "Light" : "Dark"}
          </button>
        </header>

        <div style={s.content}>
          {activeTab === "graph" && (
            <>
              <div style={s.sidebar}>
                <FilterPanel
                  nodeTypes={nodeTypes}
                  onToggleType={handleToggleType}
                  edgeTypes={edgeTypes}
                  onToggleEdgeType={handleToggleEdgeType}
                  minConfidence={minConfidence}
                  onConfidenceChange={setMinConfidence}
                />
                <ActivationPanel seedNodeId={selectedNode} />
              </div>

              <div style={s.graphContainer}>
                {graph.order === 0 ? (
                  <div style={s.emptyState}>
                    Search for a node to start exploring the knowledge graph.
                  </div>
                ) : (
                  <GraphView
                    graph={graph}
                    version={version}
                    onClickNode={handleNodeClick}
                    onExpandNode={handleExpand}
                    nodeTypes={nodeTypes}
                    edgeTypes={edgeTypes}
                    minConfidence={minConfidence}
                  />
                )}
              </div>

              <NodeDetail
                nodeId={selectedNode}
                onClose={() => setSelectedNode(null)}
                onExpand={handleExpand}
                onActivate={handleActivate}
              />
            </>
          )}

          {activeTab === "hypotheses" && (
            <div style={s.fullPanel}>
              <HypothesisList onSelectHypothesis={handleSelectHypothesis} />
            </div>
          )}

          {activeTab === "dashboard" && (
            <div style={s.fullPanel}>
              <Dashboard />
            </div>
          )}
        </div>
      </div>
    </ThemeContext.Provider>
  );
}

function mkStyles(t: Theme): Record<string, React.CSSProperties> {
  return {
    app: {
      display: "flex",
      flexDirection: "column",
      height: "100vh",
      overflow: "hidden",
      background: t.bg,
      color: t.text,
    },
    header: {
      display: "flex",
      alignItems: "center",
      padding: "10px 20px",
      borderBottom: `1px solid ${t.border}`,
      background: t.bg,
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
      color: t.textMuted,
    },
    activeTab: {
      background: t.bgAlt,
      color: "#1976D2",
    },
    searchWrapper: { flex: 1, maxWidth: 500 },
    themeToggle: {
      padding: "6px 12px",
      border: `1px solid ${t.border}`,
      borderRadius: 6,
      background: t.bgAlt,
      color: t.text,
      fontSize: 12,
      fontWeight: 500,
      cursor: "pointer",
      flexShrink: 0,
    },
    content: {
      display: "flex",
      flex: 1,
      overflow: "hidden",
      position: "relative" as const,
    },
    sidebar: {
      width: 260,
      borderRight: `1px solid ${t.border}`,
      background: t.bg,
      overflowY: "auto",
      flexShrink: 0,
    },
    graphContainer: {
      flex: 1,
      position: "relative" as const,
      backgroundColor: t.graphBg,
      backgroundImage: `radial-gradient(${t.graphDot} 1.5px, transparent 1.5px)`,
      backgroundSize: "24px 24px",
    },
    emptyState: {
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      height: "100%",
      color: t.textMuted,
      fontSize: 16,
    },
    fullPanel: {
      flex: 1,
      overflow: "hidden",
    },
  };
}
