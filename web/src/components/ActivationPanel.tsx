import { useActivation } from "../hooks/useActivation";
import { useTheme, type Theme } from "../lib/theme";

interface Props {
  seedNodeId: string | null;
}

export default function ActivationPanel({ seedNodeId }: Props) {
  const theme = useTheme();
  const { steps, running, simulate, stop } = useActivation();

  const handleSimulate = () => {
    if (seedNodeId) {
      simulate(seedNodeId);
    }
  };

  const s = mkStyles(theme);

  return (
    <div style={s.container}>
      <h4 style={s.title}>Activation Wave</h4>

      {!seedNodeId && (
        <p style={s.hint}>Select a node to use as seed for activation wave.</p>
      )}

      {seedNodeId && (
        <div style={s.controls}>
          {!running ? (
            <button style={s.startBtn} onClick={handleSimulate}>
              Start wave from selected node
            </button>
          ) : (
            <button style={s.stopBtn} onClick={stop}>
              Stop
            </button>
          )}
        </div>
      )}

      {steps.length > 0 && (
        <div style={s.steps}>
          {steps.map((st) => (
            <div key={st.step} style={s.step}>
              <div style={s.stepHeader}>Step {st.step}</div>
              {st.activations.slice(0, 5).map((a) => (
                <div key={a.id} style={s.activation}>
                  <span style={s.score}>{a.score.toFixed(3)}</span>
                  <span style={s.nodeLabel}>{a.label}</span>
                </div>
              ))}
              {st.activations.length > 5 && (
                <span style={s.more}>
                  +{st.activations.length - 5} more
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function mkStyles(t: Theme): Record<string, React.CSSProperties> {
  return {
    container: { padding: 16 },
    title: { fontSize: 15, fontWeight: 600, marginBottom: 12, color: t.text },
    hint: { fontSize: 13, color: t.textMuted },
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
      background: t.bgAlt,
      borderRadius: 6,
    },
    stepHeader: { fontSize: 12, fontWeight: 600, color: t.textMuted, marginBottom: 4 },
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
    nodeLabel: { color: t.text },
    more: { fontSize: 11, color: t.textMuted, marginTop: 2 },
  };
}
