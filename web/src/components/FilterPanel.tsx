import { nodeColor, edgeTypeColor } from "../lib/colors";
import { useTheme, type Theme } from "../lib/theme";

interface Props {
  nodeTypes: Record<string, boolean>;
  onToggleType: (type: string) => void;
  edgeTypes: Record<string, boolean>;
  onToggleEdgeType: (type: string) => void;
  minConfidence: number;
  onConfidenceChange: (val: number) => void;
}

const NODE_TYPES = ["Paper", "Technique", "Problem", "Dataset"];

const EDGE_TYPES = [
  "CITES",
  "INTRODUCES",
  "EVALUATES_ON",
  "HAS_LIMITATION",
  "MITIGATES",
  "IMPROVES_OVER",
  "USES",
  "IS_INSTANCE_OF",
  "CAUSED_BY",
  "TEMPORALLY_FOLLOWS",
  "SAME_AS",
];

function edgeLabel(type: string): string {
  return type
    .split("_")
    .map((w) => w.charAt(0) + w.slice(1).toLowerCase())
    .join(" ");
}

export default function FilterPanel({
  nodeTypes,
  onToggleType,
  edgeTypes,
  onToggleEdgeType,
  minConfidence,
  onConfidenceChange,
}: Props) {
  const theme = useTheme();
  const s = mkStyles(theme);

  return (
    <div style={s.container}>
      <h4 style={s.title}>Filters</h4>

      <div style={s.section}>
        <div style={s.sectionLabel}>Node Types</div>
        {NODE_TYPES.map((t) => (
          <label key={t} style={s.checkbox}>
            <input
              type="checkbox"
              checked={nodeTypes[t] !== false}
              onChange={() => onToggleType(t)}
            />
            <span
              style={{
                ...s.dot,
                backgroundColor: nodeColor(t),
              }}
            />
            {t}
          </label>
        ))}
      </div>

      <div style={s.section}>
        <div style={s.sectionLabel}>Edge Types</div>
        {EDGE_TYPES.map((t) => (
          <label key={t} style={s.checkbox}>
            <input
              type="checkbox"
              checked={edgeTypes[t] !== false}
              onChange={() => onToggleEdgeType(t)}
            />
            <span
              style={{
                ...s.edgeLine,
                backgroundColor: edgeTypeColor(t),
              }}
            />
            <span style={s.edgeLabel}>{edgeLabel(t)}</span>
          </label>
        ))}
      </div>

      <div style={s.section}>
        <div style={s.sectionLabel}>
          Min Confidence: {minConfidence.toFixed(2)}
        </div>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={minConfidence}
          onChange={(e) => onConfidenceChange(parseFloat(e.target.value))}
          style={s.slider}
        />
      </div>
    </div>
  );
}

function mkStyles(t: Theme): Record<string, React.CSSProperties> {
  return {
    container: { padding: 16 },
    title: { fontSize: 15, fontWeight: 600, marginBottom: 12, color: t.text },
    section: { marginBottom: 16 },
    sectionLabel: { fontSize: 12, fontWeight: 600, color: t.textMuted, marginBottom: 6 },
    checkbox: {
      display: "flex",
      alignItems: "center",
      gap: 6,
      fontSize: 13,
      marginBottom: 4,
      cursor: "pointer",
      color: t.text,
    },
    dot: {
      width: 10,
      height: 10,
      borderRadius: "50%",
      display: "inline-block",
    },
    edgeLine: {
      width: 14,
      height: 3,
      borderRadius: 2,
      display: "inline-block",
    },
    edgeLabel: {
      fontSize: 12,
    },
    slider: { width: "100%" },
  };
}
