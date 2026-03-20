import { useEffect, useState } from "react";
import {
  getHypotheses,
  submitFeedback,
  type Hypothesis,
} from "../lib/api";

interface Props {
  onSelectHypothesis: (id: string) => void;
}

export default function HypothesisList({ onSelectHypothesis }: Props) {
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

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h3 style={styles.title}>Hypotheses ({total})</h3>
        <select
          style={styles.sortSelect}
          value={sort}
          onChange={(e) => setSort(e.target.value as typeof sort)}
        >
          <option value="elo">By Elo</option>
          <option value="recent">Recent</option>
          <option value="feedback">By feedback</option>
        </select>
      </div>

      {hypotheses.length === 0 && (
        <p style={styles.empty}>No hypotheses generated yet.</p>
      )}

      {hypotheses.map((h) => (
        <div key={h.id} style={styles.card}>
          <div
            style={styles.cardTitle}
            onClick={() => onSelectHypothesis(h.id)}
          >
            {h.title}
          </div>
          <p style={styles.desc}>{h.description}</p>
          <div style={styles.footer}>
            <span style={styles.elo}>Elo: {h.elo_score.toFixed(0)}</span>
            <div style={styles.feedback}>
              <button
                style={styles.fbBtn}
                onClick={() => handleFeedback(h.id, "up")}
              >
                +{h.feedback_up}
              </button>
              <button
                style={{ ...styles.fbBtn, color: "#F44336" }}
                onClick={() => handleFeedback(h.id, "down")}
              >
                -{h.feedback_down}
              </button>
            </div>
          </div>
          {h.reasoning_chain.length > 0 && (
            <div style={styles.chain}>
              {h.reasoning_chain.join(" -> ")}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: { padding: 16, overflowY: "auto", height: "100%" },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 16,
  },
  title: { fontSize: 16, fontWeight: 600 },
  sortSelect: { padding: "4px 8px", fontSize: 13, borderRadius: 4, border: "1px solid #ddd" },
  empty: { color: "#999", fontSize: 14, textAlign: "center" as const, marginTop: 40 },
  card: {
    padding: 14,
    border: "1px solid #e0e0e0",
    borderRadius: 8,
    marginBottom: 12,
  },
  cardTitle: {
    fontWeight: 600,
    fontSize: 14,
    cursor: "pointer",
    color: "#1976D2",
  },
  desc: { fontSize: 13, color: "#444", marginTop: 6, lineHeight: 1.4 },
  footer: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginTop: 10,
  },
  elo: { fontSize: 12, color: "#666", fontWeight: 500 },
  feedback: { display: "flex", gap: 8 },
  fbBtn: {
    border: "1px solid #ddd",
    borderRadius: 4,
    padding: "3px 10px",
    fontSize: 12,
    cursor: "pointer",
    background: "#fff",
    color: "#4CAF50",
    fontWeight: 600,
  },
  chain: {
    marginTop: 8,
    fontSize: 11,
    color: "#888",
    fontFamily: "monospace",
    padding: "6px 8px",
    background: "#f8f8f8",
    borderRadius: 4,
  },
};
