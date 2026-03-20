import { useActivation } from "../hooks/useActivation";

interface Props {
  seedNodeId: string | null;
}

export default function ActivationPanel({ seedNodeId }: Props) {
  const { steps, running, simulate, stop } = useActivation();

  const handleSimulate = () => {
    if (seedNodeId) {
      simulate(seedNodeId);
    }
  };

  return (
    <div style={styles.container}>
      <h4 style={styles.title}>Activation Wave</h4>

      {!seedNodeId && (
        <p style={styles.hint}>Select a node to use as seed for activation wave.</p>
      )}

      {seedNodeId && (
        <div style={styles.controls}>
          {!running ? (
            <button style={styles.startBtn} onClick={handleSimulate}>
              Start wave from selected node
            </button>
          ) : (
            <button style={styles.stopBtn} onClick={stop}>
              Stop
            </button>
          )}
        </div>
      )}

      {steps.length > 0 && (
        <div style={styles.steps}>
          {steps.map((s) => (
            <div key={s.step} style={styles.step}>
              <div style={styles.stepHeader}>Step {s.step}</div>
              {s.activations.slice(0, 5).map((a) => (
                <div key={a.id} style={styles.activation}>
                  <span style={styles.score}>{a.score.toFixed(3)}</span>
                  <span style={styles.nodeLabel}>{a.label}</span>
                </div>
              ))}
              {s.activations.length > 5 && (
                <span style={styles.more}>
                  +{s.activations.length - 5} more
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: { padding: 16 },
  title: { fontSize: 15, fontWeight: 600, marginBottom: 12 },
  hint: { fontSize: 13, color: "#999" },
  controls: { marginBottom: 12 },
  startBtn: {
    width: "100%",
    padding: "8px 12px",
    border: "none",
    borderRadius: 6,
    background: "#FF9800",
    color: "#fff",
    cursor: "pointer",
    fontSize: 13,
    fontWeight: 500,
  },
  stopBtn: {
    width: "100%",
    padding: "8px 12px",
    border: "none",
    borderRadius: 6,
    background: "#F44336",
    color: "#fff",
    cursor: "pointer",
    fontSize: 13,
    fontWeight: 500,
  },
  steps: { maxHeight: 400, overflowY: "auto" },
  step: {
    marginBottom: 10,
    padding: 8,
    background: "#f8f8f8",
    borderRadius: 6,
  },
  stepHeader: { fontSize: 12, fontWeight: 600, color: "#666", marginBottom: 4 },
  activation: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "2px 0",
    fontSize: 13,
  },
  score: {
    fontFamily: "monospace",
    fontSize: 12,
    color: "#FF9800",
    fontWeight: 600,
    minWidth: 50,
  },
  nodeLabel: { color: "#333" },
  more: { fontSize: 11, color: "#999", marginTop: 2 },
};
