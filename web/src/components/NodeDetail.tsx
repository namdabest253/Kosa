import { useEffect, useState } from "react";
import { getNodeDetail, type NodeDetail as NodeDetailType } from "../lib/api";
import { nodeColor } from "../lib/colors";
import { useTheme, type Theme } from "../lib/theme";

interface Props {
  nodeId: string | null;
  onClose: () => void;
  onExpand: (id: string) => void;
  onActivate: (id: string) => void;
}

export default function NodeDetail({ nodeId, onClose, onExpand, onActivate }: Props) {
  const theme = useTheme();
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
  const s = mkStyles(theme);

  return (
    <div style={s.panel}>
      <div style={s.header}>
        <span style={{ ...s.badge, backgroundColor: color }}>{detail.node_type}</span>
        <button style={s.close} onClick={onClose}>x</button>
      </div>

      <h3 style={s.title}>
        {(detail.properties["name"] ?? detail.properties["title"] ?? "?") as string}
      </h3>

      <p style={s.meta}>{detail.neighbor_count} connections</p>

      <div style={s.props}>
        {Object.entries(detail.properties).map(([k, v]) => (
          <div key={k} style={s.prop}>
            <span style={s.propKey}>{k}</span>
            <span style={s.propVal}>
              {Array.isArray(v) ? (v as string[]).join(", ") : String(v)}
            </span>
          </div>
        ))}
      </div>

      <div style={s.actions}>
        <button style={s.actionBtn} onClick={() => onExpand(nodeId)}>
          Expand neighbors
        </button>
        <button style={{ ...s.actionBtn, background: "#FF9800" }} onClick={() => onActivate(nodeId)}>
          Activate wave
        </button>
      </div>
    </div>
  );
}

function mkStyles(t: Theme): Record<string, React.CSSProperties> {
  return {
    panel: {
      position: "absolute",
      top: 0,
      right: 0,
      width: 360,
      height: "100%",
      background: t.bg,
      borderLeft: `1px solid ${t.border}`,
      padding: 20,
      overflowY: "auto",
      zIndex: 20,
      boxShadow: `-4px 0 12px ${t.shadow}`,
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
      color: t.textMuted,
    },
    title: { marginTop: 12, fontSize: 18, lineHeight: 1.3, color: t.text },
    meta: { color: t.textMuted, fontSize: 13, marginTop: 4 },
    props: { marginTop: 16 },
    prop: {
      display: "flex",
      flexDirection: "column",
      padding: "6px 0",
      borderBottom: `1px solid ${t.borderLight}`,
    },
    propKey: { fontSize: 11, fontWeight: 600, color: t.textMuted, textTransform: "uppercase" as const },
    propVal: { fontSize: 13, marginTop: 2, wordBreak: "break-word" as const, color: t.text },
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
}
