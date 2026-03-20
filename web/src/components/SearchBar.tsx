import { useState } from "react";
import { searchNodes, type SearchResult } from "../lib/api";
import { nodeColor } from "../lib/colors";

interface Props {
  onSelect: (result: SearchResult) => void;
}

export default function SearchBar({ onSelect }: Props) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [typeFilter, setTypeFilter] = useState<string>("");

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const resp = await searchNodes(query, typeFilter || undefined);
      setResults(resp.results);
    } catch (e) {
      console.error("Search failed:", e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.inputRow}>
        <input
          style={styles.input}
          type="text"
          placeholder="Search nodes..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
        />
        <select
          style={styles.select}
          value={typeFilter}
          onChange={(e) => setTypeFilter(e.target.value)}
        >
          <option value="">All types</option>
          <option value="Paper">Paper</option>
          <option value="Technique">Technique</option>
          <option value="Problem">Problem</option>
          <option value="Dataset">Dataset</option>
        </select>
        <button style={styles.button} onClick={handleSearch} disabled={loading}>
          {loading ? "..." : "Search"}
        </button>
      </div>
      {results.length > 0 && (
        <div style={styles.results}>
          {results.map((r) => (
            <div
              key={r.id}
              style={styles.result}
              onClick={() => {
                onSelect(r);
                setResults([]);
              }}
            >
              <span
                style={{
                  ...styles.badge,
                  backgroundColor: nodeColor(r.node_type),
                }}
              >
                {r.node_type}
              </span>
              <span style={styles.label}>{r.label}</span>
              {r.snippet && <span style={styles.snippet}>{r.snippet}</span>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: { position: "relative", zIndex: 10 },
  inputRow: { display: "flex", gap: 8 },
  input: {
    flex: 1,
    padding: "8px 12px",
    fontSize: 14,
    border: "1px solid #ddd",
    borderRadius: 6,
    outline: "none",
  },
  select: {
    padding: "8px 12px",
    fontSize: 14,
    border: "1px solid #ddd",
    borderRadius: 6,
  },
  button: {
    padding: "8px 16px",
    fontSize: 14,
    border: "none",
    borderRadius: 6,
    background: "#2196F3",
    color: "#fff",
    cursor: "pointer",
  },
  results: {
    position: "absolute",
    top: "100%",
    left: 0,
    right: 0,
    background: "#fff",
    border: "1px solid #ddd",
    borderRadius: 6,
    maxHeight: 400,
    overflowY: "auto",
    marginTop: 4,
    boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
  },
  result: {
    padding: "10px 12px",
    cursor: "pointer",
    borderBottom: "1px solid #f0f0f0",
    display: "flex",
    alignItems: "center",
    gap: 8,
  },
  badge: {
    fontSize: 11,
    padding: "2px 8px",
    borderRadius: 4,
    color: "#fff",
    fontWeight: 600,
    flexShrink: 0,
  },
  label: { fontWeight: 500, fontSize: 14 },
  snippet: { fontSize: 12, color: "#666", marginLeft: 8, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" as const },
};
