import { useEffect, useState } from "react";
import {
  getHypotheses,
  submitFeedback,
  type Hypothesis,
} from "../lib/api";
import { useTheme, type Theme } from "../lib/theme";

interface Props {
  onSelectHypothesis: (id: string) => void;
}

export default function HypothesisList({ onSelectHypothesis }: Props) {
  const theme = useTheme();
  const [hypotheses, setHypotheses] = useState<Hypothesis[]>([]);
  const [total, setTotal] = useState(0);
  const [sort, setSort] = useState<"elo" | "recent" | "feedback">("elo");

  useEffect(() => {
    getHypotheses(0, 20, sort)
      .then((resp) => {
        setHypotheses(resp.hypotheses);
        setTotal(resp.total);
      })
      .catch(console.error);
  }, [sort]);

  const handleFeedback = async (id: string, vote: "up" | "down") => {
    try {
      const resp = await submitFeedback(id, vote);
      setHypotheses((prev) =>
        prev.map((h) =>
          h.id === id
            ? { ...h, feedback_up: resp.feedback_up, feedback_down: resp.feedback_down }
            : h,
        ),
      );
    } catch (e) {
      console.error("Feedback failed:", e);
    }
  };

  const s = mkStyles(theme);

  return (
    <div style={s.container}>
      <div style={s.header}>
        <h3 style={s.title}>Hypotheses ({total})</h3>
        <select
          style={s.sortSelect}
          value={sort}
          onChange={(e) => setSort(e.target.value as typeof sort)}
        >
          <option value="elo">By Elo</option>
          <option value="recent">Recent</option>
          <option value="feedback">By feedback</option>
        </select>
      </div>

      {hypotheses.length === 0 && (
        <p style={s.empty}>No hypotheses generated yet.</p>
      )}

      {hypotheses.map((h) => (
        <div key={h.id} style={s.card}>
          <div
            style={s.cardTitle}
            onClick={() => onSelectHypothesis(h.id)}
          >
            {h.title}
          </div>
          <p style={s.desc}>{h.description}</p>
          <div style={s.footer}>
            <span style={s.elo}>Elo: {h.elo_score.toFixed(0)}</span>
            <div style={s.feedback}>
              <button
                style={s.fbBtn}
                onClick={() => handleFeedback(h.id, "up")}
              >
                +{h.feedback_up}
              </button>
              <button
                style={{ ...s.fbBtn, color: "#F44336" }}
                onClick={() => handleFeedback(h.id, "down")}
              >
                -{h.feedback_down}
              </button>
            </div>
          </div>
          {h.reasoning_chain.length > 0 && (
            <div style={s.chain}>
              {h.reasoning_chain.join(" -> ")}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function mkStyles(t: Theme): Record<string, React.CSSProperties> {
  return {
    container: { padding: 16, overflowY: "auto", height: "100%" },
    header: {
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      marginBottom: 16,
    },
    title: { fontSize: 16, fontWeight: 600, color: t.text },
    sortSelect: {
      padding: "4px 8px",
      fontSize: 13,
      borderRadius: 4,
      border: `1px solid ${t.inputBorder}`,
      background: t.inputBg,
      color: t.text,
    },
    empty: { color: t.textMuted, fontSize: 14, textAlign: "center" as const, marginTop: 40 },
    card: {
      padding: 14,
      border: `1px solid ${t.border}`,
      borderRadius: 8,
      marginBottom: 12,
    },
    cardTitle: {
      fontWeight: 600,
      fontSize: 14,
      cursor: "pointer",
      color: "#1976D2",
    },
    desc: { fontSize: 13, color: t.textMuted, marginTop: 6, lineHeight: 1.4 },
    footer: {
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      marginTop: 10,
    },
    elo: { fontSize: 12, color: t.textMuted, fontWeight: 500 },
    feedback: { display: "flex", gap: 8 },
    fbBtn: {
      border: `1px solid ${t.border}`,
      borderRadius: 4,
      padding: "3px 10px",
      fontSize: 12,
      cursor: "pointer",
      background: t.bg,
      color: "#4CAF50",
      fontWeight: 600,
    },
    chain: {
      marginTop: 8,
      fontSize: 11,
      color: t.textMuted,
      fontFamily: "monospace",
      padding: "6px 8px",
      background: t.bgAlt,
      borderRadius: 4,
    },
  };
}
