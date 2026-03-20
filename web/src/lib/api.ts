/** Thin API client for the Kosa backend. */

const BASE = "/api/v1";

async function fetchJSON<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

async function postJSON<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

// --- Graph ---

export interface GraphStats {
  total_nodes: number;
  total_edges: number;
  node_counts: { type: string; count: number }[];
  edge_counts: { type: string; count: number }[];
}

export interface SearchResult {
  id: string;
  label: string;
  node_type: string;
  score: number;
  snippet: string;
}

export interface GraphNode {
  key: string;
  attributes: {
    label: string;
    node_type: string;
    x: number;
    y: number;
    size: number;
    color: string;
    properties: Record<string, unknown>;
  };
}

export interface GraphEdge {
  key: string;
  source: string;
  target: string;
  attributes: {
    edge_type: string;
    confidence: number;
    color: string;
    size: number;
    properties: Record<string, unknown>;
  };
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface NeighborhoodResponse {
  graph: GraphData;
  center_id: string;
  depth: number;
  truncated: boolean;
  total_neighbor_count: number;
}

export interface NodeDetail {
  id: string;
  node_type: string;
  properties: Record<string, unknown>;
  neighbor_count: number;
}

export const getGraphStats = () => fetchJSON<GraphStats>(`${BASE}/graph/stats`);

export const searchNodes = (q: string, type?: string, limit = 20) => {
  const params = new URLSearchParams({ q, limit: String(limit) });
  if (type) params.set("type", type);
  return fetchJSON<{ results: SearchResult[]; query: string; total: number }>(
    `${BASE}/graph/search?${params}`,
  );
};

export const getNeighborhood = (id: string, depth = 1, limit = 50) =>
  fetchJSON<NeighborhoodResponse>(
    `${BASE}/graph/neighborhood/${encodeURIComponent(id)}?depth=${depth}&limit=${limit}`,
  );

export const getNodeDetail = (id: string) =>
  fetchJSON<NodeDetail>(`${BASE}/graph/node/${encodeURIComponent(id)}`);

// --- Hypotheses ---

export interface Hypothesis {
  id: string;
  title: string;
  description: string;
  elo_score: number;
  reasoning_chain: string[];
  feedback_up: number;
  feedback_down: number;
  created_at: string;
}

export const getHypotheses = (skip = 0, limit = 20, sort = "elo") =>
  fetchJSON<{ hypotheses: Hypothesis[]; total: number }>(
    `${BASE}/hypotheses?skip=${skip}&limit=${limit}&sort=${sort}`,
  );

export const submitFeedback = (id: string, vote: "up" | "down") =>
  postJSON<{ hypothesis_id: string; feedback_up: number; feedback_down: number }>(
    `${BASE}/hypotheses/${encodeURIComponent(id)}/feedback`,
    { vote },
  );

export const getReasoningPath = (id: string) =>
  fetchJSON<GraphData>(`${BASE}/hypotheses/${encodeURIComponent(id)}/path`);

// --- Stats ---

export interface DashboardData {
  graph: {
    node_counts: Record<string, number>;
    edge_counts: Record<string, number>;
    total_nodes: number;
    total_edges: number;
  };
  quality: {
    confidence_distribution: Record<string, number>;
    year_distribution: Record<string, number>;
    venue_tier_distribution: Record<string, number>;
  };
  hypotheses: {
    total: number;
    total_feedback_up: number;
    total_feedback_down: number;
    avg_elo: number | null;
  };
}

export const getDashboard = () => fetchJSON<DashboardData>(`${BASE}/stats/dashboard`);
