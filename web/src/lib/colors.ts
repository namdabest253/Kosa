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

/** Confidence to edge color. */
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
