import { useState } from "react";
import { searchNodes, type SearchResult } from "../lib/api";
import { nodeColor } from "../lib/colors";
import { useTheme, type Theme } from "../lib/theme";

interface Props {
  onSelect: (result: SearchResult) => void;
}

export default function SearchBar({ onSelect }: Props) {
  const theme = useTheme();
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

  const s = mkStyles(theme);

  return (
    <div style={s.container}>
      <div style={s.inputRow}>
        <input
          id="kosa-search-input"
          style={s.input}
          type="text"
          placeholder="Search nodes... (press / to focus)"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
        />
        <select
          style={s.select}
          value={typeFilter}
          onChange={(e) => setTypeFilter(e.target.value)}
        >
          <option value="">All types</option>
          <option value="Paper">Paper</option>
          <option value="Technique">Technique</option>
          <option value="Problem">Problem</option>
          <option value="Dataset">Dataset</option>
        </select>
        <button style={s.button} onClick={handleSearch} disabled={loading}>
          {loading ? "..." : "Search"}
        </button>
      </div>
      {results.length > 0 && (
        <div style={s.results}>
          {results.map((r) => (
            <div
              key={r.id}
              style={s.result}
              onClick={() => {
                onSelect(r);
                setResults([]);
              }}
            >
              <span
                style={{
                  ...s.badge,
                  backgroundColor: nodeColor(r.node_type),
                }}
              >
                {r.node_type}
              </span>
              <span style={s.label}>{r.label}</span>
              {r.snippet && <span style={s.snippet}>{r.snippet}</span>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function mkStyles(t: Theme): Record<string, React.CSSProperties> {
  return {
    container: { position: "relative", zIndex: 10 },
    inputRow: { display: "flex", gap: 8 },
    input: {
      flex: 1,
      padding: "8px 12px",
      fontSize: 14,
      border: `1px solid ${t.inputBorder}`,
      borderRadius: 6,
      outline: "none",
      background: t.inputBg,
      color: t.text,
    },
    select: {
      padding: "8px 12px",
      fontSize: 14,
      border: `1px solid ${t.inputBorder}`,
      borderRadius: 6,
      background: t.inputBg,
      color: t.text,
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
      background: t.bg,
      border: `1px solid ${t.inputBorder}`,
      borderRadius: 6,
      maxHeight: 400,
      overflowY: "auto",
      marginTop: 4,
      boxShadow: `0 4px 12px ${t.shadow}`,
    },
    result: {
      padding: "10px 12px",
      cursor: "pointer",
      borderBottom: `1px solid ${t.borderLight}`,
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
    label: { fontWeight: 500, fontSize: 14, color: t.text },
    snippet: { fontSize: 12, color: t.textMuted, marginLeft: 8, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" as const },
  };
}
