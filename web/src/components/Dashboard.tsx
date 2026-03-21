import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import { getDashboard, type DashboardData } from "../lib/api";
import { NODE_COLORS } from "../lib/colors";
import { useTheme, type Theme } from "../lib/theme";

export default function Dashboard() {
  const theme = useTheme();
  const [data, setData] = useState<DashboardData | null>(null);

  useEffect(() => {
    getDashboard().then(setData).catch(console.error);
  }, []);

  const s = mkStyles(theme);

  if (!data) return <div style={s.loading}>Loading dashboard...</div>;

  const nodeData = Object.entries(data.graph.node_counts).map(([type, count]) => ({
    name: type,
    count,
    fill: NODE_COLORS[type] ?? "#999",
  }));

  const confData = Object.entries(data.quality.confidence_distribution).map(
    ([band, count]) => ({ name: band, count }),
  );

  const yearData = Object.entries(data.quality.year_distribution)
    .map(([year, count]) => ({ year: Number(year), count }))
    .sort((a, b) => a.year - b.year);

  return (
    <div style={s.container}>
      <h2 style={s.heading}>Dashboard</h2>

      <div style={s.grid}>
        <div style={s.stat}>
          <div style={s.statValue}>{data.graph.total_nodes.toLocaleString()}</div>
          <div style={s.statLabel}>Total Nodes</div>
        </div>
        <div style={s.stat}>
          <div style={s.statValue}>{data.graph.total_edges.toLocaleString()}</div>
          <div style={s.statLabel}>Total Edges</div>
        </div>
        <div style={s.stat}>
          <div style={s.statValue}>{data.hypotheses.total}</div>
          <div style={s.statLabel}>Hypotheses</div>
        </div>
        <div style={s.stat}>
          <div style={s.statValue}>
            {data.hypotheses.avg_elo?.toFixed(0) ?? "-"}
          </div>
          <div style={s.statLabel}>Avg Elo</div>
        </div>
      </div>

      <div style={s.chartRow}>
        <div style={s.chartCard}>
          <h4 style={s.chartTitle}>Nodes by Type</h4>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={nodeData}
                dataKey="count"
                nameKey="name"
                outerRadius={80}
                label={({ name, count }: { name: string; count: number }) => `${name}: ${count}`}
              >
                {nodeData.map((entry) => (
                  <Cell key={entry.name} fill={entry.fill} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div style={s.chartCard}>
          <h4 style={s.chartTitle}>Edge Confidence</h4>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={confData}>
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#2196F3" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {yearData.length > 0 && (
        <div style={s.chartCard}>
          <h4 style={s.chartTitle}>Papers by Year</h4>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={yearData}>
              <XAxis dataKey="year" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#4CAF50" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {(data.hypotheses.total_feedback_up > 0 || data.hypotheses.total_feedback_down > 0) && (
        <div style={s.chartCard}>
          <h4 style={s.chartTitle}>Feedback Summary</h4>
          <div style={s.feedbackRow}>
            <span style={{ color: "#4CAF50", fontWeight: 600, fontSize: 20 }}>
              +{data.hypotheses.total_feedback_up}
            </span>
            <span style={{ color: theme.textMuted, fontSize: 14 }}>/</span>
            <span style={{ color: "#F44336", fontWeight: 600, fontSize: 20 }}>
              -{data.hypotheses.total_feedback_down}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

function mkStyles(t: Theme): Record<string, React.CSSProperties> {
  return {
    container: { padding: 24, overflowY: "auto", height: "100%" },
    loading: { padding: 40, textAlign: "center" as const, color: t.textMuted },
    heading: { fontSize: 22, fontWeight: 600, marginBottom: 20, color: t.text },
    grid: {
      display: "grid",
      gridTemplateColumns: "repeat(4, 1fr)",
      gap: 16,
      marginBottom: 24,
    },
    stat: {
      padding: 16,
      background: t.cardBg,
      borderRadius: 8,
      textAlign: "center" as const,
    },
    statValue: { fontSize: 28, fontWeight: 700, color: "#1976D2" },
    statLabel: { fontSize: 12, color: t.textMuted, marginTop: 4 },
    chartRow: { display: "flex", gap: 16, marginBottom: 16 },
    chartCard: {
      flex: 1,
      padding: 16,
      border: `1px solid ${t.border}`,
      borderRadius: 8,
      marginBottom: 16,
    },
    chartTitle: { fontSize: 14, fontWeight: 600, marginBottom: 12, color: t.text },
    feedbackRow: {
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      gap: 12,
      padding: 20,
    },
  };
}
