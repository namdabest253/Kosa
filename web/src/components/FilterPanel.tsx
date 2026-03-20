import { nodeColor } from "../lib/colors";

interface Props {
  nodeTypes: Record<string, boolean>;
  onToggleType: (type: string) => void;
  minConfidence: number;
  onConfidenceChange: (val: number) => void;
}

const NODE_TYPES = ["Paper", "Technique", "Problem", "Dataset"];

export default function FilterPanel({
  nodeTypes,
  onToggleType,
  minConfidence,
  onConfidenceChange,
}: Props) {
  return (
    <div style={styles.container}>
      <h4 style={styles.title}>Filters</h4>

      <div style={styles.section}>
        <div style={styles.sectionLabel}>Node Types</div>
        {NODE_TYPES.map((t) => (
          <label key={t} style={styles.checkbox}>
            <input
              type="checkbox"
              checked={nodeTypes[t] !== false}
              onChange={() => onToggleType(t)}
            />
            <span
              style={{
                ...styles.dot,
                backgroundColor: nodeColor(t),
              }}
            />
            {t}
          </label>
        ))}
      </div>

      <div style={styles.section}>
        <div style={styles.sectionLabel}>
          Min Confidence: {minConfidence.toFixed(2)}
        </div>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={minConfidence}
          onChange={(e) => onConfidenceChange(parseFloat(e.target.value))}
          style={styles.slider}
        />
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: { padding: 16 },
  title: { fontSize: 15, fontWeight: 600, marginBottom: 12 },
  section: { marginBottom: 16 },
  sectionLabel: { fontSize: 12, fontWeight: 600, color: "#666", marginBottom: 6 },
  checkbox: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    fontSize: 13,
    marginBottom: 4,
    cursor: "pointer",
  },
  dot: {
    width: 10,
    height: 10,
    borderRadius: "50%",
    display: "inline-block",
  },
  slider: { width: "100%" },
};
