import { useEffect, useState } from "react";
import { getNodeDetail, type NodeDetail as NodeDetailType } from "../lib/api";
import { nodeColor } from "../lib/colors";

interface Props {
  nodeId: string | null;
  onClose: () => void;
  onExpand: (id: string) => void;
  onActivate: (id: string) => void;
}

export default function NodeDetail({ nodeId, onClose, onExpand, onActivate }: Props) {
  const [detail, setDetail] = useState<NodeDetailType | null>(null);

  useEffect(() => {
    if (!nodeId) {
      setDetail(null);
      return;
    }
    getNodeDetail(nodeId).then(setDetail).catch(console.error);
  }, [nodeId]);

  if (!nodeId || !detail) return null;

  const color = nodeColor(detail.node_type);

  return (
    <div style={styles.panel}>
      <div style={styles.header}>
        <span style={{ ...styles.badge, backgroundColor: color }}>{detail.node_type}</span>
        <button style={styles.close} onClick={onClose}>x</button>
      </div>

      <h3 style={styles.title}>
        {(detail.properties["name"] ?? detail.properties["title"] ?? "?") as string}
      </h3>

      <p style={styles.meta}>{detail.neighbor_count} connections</p>

      <div style={styles.props}>
        {Object.entries(detail.properties).map(([k, v]) => (
          <div key={k} style={styles.prop}>
            <span style={styles.propKey}>{k}</span>
            <span style={styles.propVal}>
              {Array.isArray(v) ? (v as string[]).join(", ") : String(v)}
            </span>
          </div>
        ))}
      </div>

      <div style={styles.actions}>
        <button style={styles.actionBtn} onClick={() => onExpand(nodeId)}>
          Expand neighbors
        </button>
        <button style={{ ...styles.actionBtn, background: "#FF9800" }} onClick={() => onActivate(nodeId)}>
          Activate wave
        </button>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  panel: {
    position: "absolute",
    top: 0,
    right: 0,
    width: 360,
    height: "100%",
    background: "#fff",
    borderLeft: "1px solid #ddd",
    padding: 20,
    overflowY: "auto",
    zIndex: 20,
    boxShadow: "-4px 0 12px rgba(0,0,0,0.05)",
  },
  header: { display: "flex", justifyContent: "space-between", alignItems: "center" },
  badge: {
    fontSize: 12,
    padding: "3px 10px",
    borderRadius: 4,
    color: "#fff",
    fontWeight: 600,
  },
  close: {
    border: "none",
    background: "none",
    fontSize: 18,
    cursor: "pointer",
    color: "#999",
  },
  title: { marginTop: 12, fontSize: 18, lineHeight: 1.3 },
  meta: { color: "#666", fontSize: 13, marginTop: 4 },
  props: { marginTop: 16 },
  prop: {
    display: "flex",
    flexDirection: "column",
    padding: "6px 0",
    borderBottom: "1px solid #f0f0f0",
  },
  propKey: { fontSize: 11, fontWeight: 600, color: "#999", textTransform: "uppercase" as const },
  propVal: { fontSize: 13, marginTop: 2, wordBreak: "break-word" as const },
  actions: { marginTop: 20, display: "flex", gap: 8 },
  actionBtn: {
    flex: 1,
    padding: "8px 12px",
    border: "none",
    borderRadius: 6,
    background: "#2196F3",
    color: "#fff",
    cursor: "pointer",
    fontSize: 13,
    fontWeight: 500,
  },
};
