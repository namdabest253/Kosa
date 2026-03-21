import { createContext, useContext } from "react";

export interface Theme {
  bg: string;
  bgAlt: string;
  text: string;
  textMuted: string;
  border: string;
  borderLight: string;
  inputBg: string;
  inputBorder: string;
  shadow: string;
  cardBg: string;
  graphBg: string;
  graphDot: string;
  nodeDefault: string;
  edgeDefault: string;
  labelColor: string;
  dimColor: string;
  highlightEdge: string;
}

export const light: Theme = {
  bg: "#fff",
  bgAlt: "#f8f8f8",
  text: "#333",
  textMuted: "#666",
  border: "#e0e0e0",
  borderLight: "#f0f0f0",
  inputBg: "#fff",
  inputBorder: "#ddd",
  shadow: "rgba(0,0,0,0.05)",
  cardBg: "#f8f9fa",
  graphBg: "#fafafa",
  graphDot: "#d0d0d0",
  nodeDefault: "#999",
  edgeDefault: "rgba(180,180,180,0.4)",
  labelColor: "#333",
  dimColor: "#e0e0e0",
  highlightEdge: "#666",
};

export const dark: Theme = {
  bg: "#1e1e1e",
  bgAlt: "#2a2a2a",
  text: "#e0e0e0",
  textMuted: "#aaa",
  border: "#3a3a3a",
  borderLight: "#333",
  inputBg: "#2a2a2a",
  inputBorder: "#444",
  shadow: "rgba(0,0,0,0.3)",
  cardBg: "#252525",
  graphBg: "#1a1a1a",
  graphDot: "#333",
  nodeDefault: "#bbb",
  edgeDefault: "rgba(120,120,120,0.4)",
  labelColor: "#e0e0e0",
  dimColor: "#444",
  highlightEdge: "#999",
};

export const ThemeContext = createContext<Theme>(light);

export function useTheme(): Theme {
  return useContext(ThemeContext);
}
