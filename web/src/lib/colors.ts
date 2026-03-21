/** Node type to color mapping. */

export const NODE_COLORS: Record<string, string> = {
  Paper: "#2196F3",
  Technique: "#4CAF50",
  Problem: "#F44336",
  Dataset: "#FF9800",
};

export const NODE_COLOR_DEFAULT = "#999999";

export function nodeColor(type: string): string {
  return NODE_COLORS[type] ?? NODE_COLOR_DEFAULT;
}

/** Edge type to color mapping. */
export const EDGE_COLORS: Record<string, string> = {
  CITES: "#78909C",          // blue-grey
  INTRODUCES: "#42A5F5",     // light blue
  EVALUATES_ON: "#FFA726",   // orange
  HAS_LIMITATION: "#EF5350", // red
  MITIGATES: "#66BB6A",      // green
  IMPROVES_OVER: "#AB47BC",  // purple
  USES: "#26A69A",           // teal
  IS_INSTANCE_OF: "#8D6E63", // brown
  CAUSED_BY: "#EC407A",      // pink
  TEMPORALLY_FOLLOWS: "#7E57C2", // deep purple
  SAME_AS: "#BDBDBD",        // grey
};

export const EDGE_COLOR_DEFAULT = "#999999";

export function edgeTypeColor(type: string): string {
  return EDGE_COLORS[type] ?? EDGE_COLOR_DEFAULT;
}

/** Confidence to edge color (legacy, used when not coloring by type). */
export function edgeColor(confidence: number): string {
  if (confidence > 0.8) return "#555555";
  if (confidence > 0.5) return "#999999";
  return "#cccccc";
}

/** Activation heat (0-1) to color gradient (blue → yellow → red). */
export function heatColor(score: number): string {
  const clamped = Math.max(0, Math.min(1, score));
  if (clamped < 0.5) {
    // Blue to yellow
    const t = clamped * 2;
    const r = Math.round(255 * t);
    const g = Math.round(255 * t);
    const b = Math.round(255 * (1 - t));
    return `rgb(${r},${g},${b})`;
  }
  // Yellow to red
  const t = (clamped - 0.5) * 2;
  const r = 255;
  const g = Math.round(255 * (1 - t));
  return `rgb(${r},${g},0)`;
}
