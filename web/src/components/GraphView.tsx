import {
  SigmaContainer,
  useRegisterEvents,
  useSigma,
} from "@react-sigma/core";
import "@react-sigma/core/lib/style.css";
import EdgeCurveProgram from "@sigma/edge-curve";
import type Graph from "graphology";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTheme } from "../lib/theme";

interface Props {
  graph: Graph;
  version: number;
  onClickNode: (nodeId: string) => void;
  onExpandNode: (nodeId: string, depth: number) => void;
  highlightedNodes?: Set<string>;
  activationScores?: Map<string, number>;
  nodeTypes?: Record<string, boolean>;
  edgeTypes?: Record<string, boolean>;
  minConfidence?: number;
}

// --- Depth control overlay ---

function DepthControl({
  depth,
  onChange,
}: {
  depth: number;
  onChange: (d: number) => void;
}) {
  const theme = useTheme();
  return (
    <div
      style={{
        position: "absolute",
        bottom: 16,
        left: 16,
        zIndex: 10,
        display: "flex",
        alignItems: "center",
        gap: 8,
        background: theme.bg,
        border: `1px solid ${theme.border}`,
        borderRadius: 8,
        padding: "6px 12px",
        fontSize: 13,
        color: theme.text,
        boxShadow: `0 2px 8px ${theme.shadow}`,
      }}
    >
      <span style={{ fontWeight: 600, color: theme.textMuted }}>Depth</span>
      {[1, 2, 3, 4, 5].map((d) => (
        <button
          key={d}
          onClick={() => onChange(d)}
          style={{
            width: 28,
            height: 28,
            border:
              d === depth
                ? "2px solid #1976D2"
                : `1px solid ${theme.border}`,
            borderRadius: 6,
            background: d === depth ? "#1976D2" : theme.bgAlt,
            color: d === depth ? "#fff" : theme.text,
            fontSize: 13,
            fontWeight: 600,
            cursor: "pointer",
          }}
        >
          {d}
        </button>
      ))}
    </div>
  );
}

// --- Zoom tracker (inside SigmaContainer, renders nothing) ---

function ZoomTracker({
  onZoomChange,
}: {
  onZoomChange: (zoom: number) => void;
}) {
  const sigma = useSigma();

  useEffect(() => {
    const cam = sigma.getCamera();
    const update = () => onZoomChange(1 / cam.ratio);
    update();
    cam.on("updated", update);
    return () => {
      cam.off("updated", update);
    };
  }, [sigma, onZoomChange]);

  return null;
}

// --- Drag controller using native mousemove ---
// Uses native mousemove (not sigma's moveBody which skips nodes),
// spring-based pull for connected nodes, and max edge length constraint.

const DRAG_THRESHOLD_SQ = 9;
const SETTLE_EPSILON = 0.15;
const MAX_EDGE_LENGTH = 3;
const EDGE_SPRING_STRENGTH = 0.2;
const DRAG_REPULSION = 0.12; // repulsion between non-dragged nodes during drag
const DRAG_REPULSION_RADIUS = 3.5;

interface DragNode {
  origX: number;
  origY: number;
  curDX: number;
  curDY: number;
  lerpSpeed: number;
  pullFactor: number;
}

interface DragState {
  node: string | null;
  active: boolean;
  moved: boolean;
  released: boolean;
  mouseStartScreen: { x: number; y: number };
  draggedOrigX: number;
  draggedOrigY: number;
  targetDX: number;
  targetDY: number;
  nodes: Map<string, DragNode>;
  animFrame: number;
}

function DragController({
  onDragChange,
}: {
  onDragChange: (active: boolean) => void;
}) {
  const sigma = useSigma();
  const registerEvents = useRegisterEvents();
  const stateRef = useRef<DragState>({
    node: null,
    active: false,
    moved: false,
    released: false,
    mouseStartScreen: { x: 0, y: 0 },
    draggedOrigX: 0,
    draggedOrigY: 0,
    targetDX: 0,
    targetDY: 0,
    nodes: new Map(),
    animFrame: 0,
  });

  useEffect(() => {
    const s = stateRef.current;
    const container = sigma.getContainer();

    // Block scroll-wheel zoom while dragging
    const blockWheel = (e: WheelEvent) => {
      if (s.active) {
        e.preventDefault();
        e.stopPropagation();
      }
    };
    container.addEventListener("wheel", blockWheel, { passive: false });

    // Native mousemove — fires regardless of what's under cursor
    const onMouseMove = (e: MouseEvent) => {
      if (!s.active || !s.node) return;

      const rect = container.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      if (!s.moved) {
        const mdx = x - s.mouseStartScreen.x;
        const mdy = y - s.mouseStartScreen.y;
        if (mdx * mdx + mdy * mdy < DRAG_THRESHOLD_SQ) return;
        s.moved = true;
        s.animFrame = requestAnimationFrame(animate);
      }

      const mouseGraph = sigma.viewportToGraph({ x, y });
      s.targetDX = mouseGraph.x - s.draggedOrigX;
      s.targetDY = mouseGraph.y - s.draggedOrigY;
    };

    const onMouseUp = () => {
      if (!s.active) return;
      sigma.getCamera().enable();
      onDragChange(false);
      if (!s.moved) {
        s.node = null;
        s.active = false;
        return;
      }
      s.released = true;
      s.node = null;
    };

    container.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);

    function animate() {
      const graph = sigma.getGraph();
      let allSettled = true;

      // Phase 1: lerp each node toward its drag-pull target
      for (const [, cn] of s.nodes) {
        const targetX = cn.origX + s.targetDX * cn.pullFactor;
        const targetY = cn.origY + s.targetDY * cn.pullFactor;
        const curX = cn.origX + cn.curDX;
        const curY = cn.origY + cn.curDY;
        const gapX = targetX - curX;
        const gapY = targetY - curY;

        if (
          Math.abs(gapX) < SETTLE_EPSILON &&
          Math.abs(gapY) < SETTLE_EPSILON
        ) {
          cn.curDX = s.targetDX * cn.pullFactor;
          cn.curDY = s.targetDY * cn.pullFactor;
        } else {
          cn.curDX += gapX * cn.lerpSpeed;
          cn.curDY += gapY * cn.lerpSpeed;
          allSettled = false;
        }
      }

      // Phase 2: repulsion between non-dragged nodes (keeps shape circular)
      const nodeEntries = Array.from(s.nodes.entries());
      for (let i = 0; i < nodeEntries.length; i++) {
        const [idA, a] = nodeEntries[i]!;
        if (idA === s.node) continue;
        const ax = a.origX + a.curDX;
        const ay = a.origY + a.curDY;

        for (let j = i + 1; j < nodeEntries.length; j++) {
          const [idB, b] = nodeEntries[j]!;
          if (idB === s.node) continue;
          const bx = b.origX + b.curDX;
          const by = b.origY + b.curDY;

          const dx = ax - bx;
          const dy = ay - by;
          const distSq = dx * dx + dy * dy;
          const rSq = DRAG_REPULSION_RADIUS * DRAG_REPULSION_RADIUS;

          if (distSq < rSq && distSq > 0.01) {
            const dist = Math.sqrt(distSq);
            const t = dist / DRAG_REPULSION_RADIUS;
            const force = DRAG_REPULSION * (1 - t) * (1 - t);
            const fx = (dx / dist) * force;
            const fy = (dy / dist) * force;
            a.curDX += fx;
            a.curDY += fy;
            b.curDX -= fx;
            b.curDY -= fy;
            allSettled = false;
          }
        }
      }

      // Phase 3: max edge length constraint — spring pulls if edge too long
      graph.forEachEdge((_edge, _attrs, src, tgt) => {
        const a = s.nodes.get(src);
        const b = s.nodes.get(tgt);
        if (!a || !b) return;

        const ax = a.origX + a.curDX;
        const ay = a.origY + a.curDY;
        const bx = b.origX + b.curDX;
        const by = b.origY + b.curDY;

        const dx = bx - ax;
        const dy = by - ay;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist > MAX_EDGE_LENGTH) {
          const overflow = dist - MAX_EDGE_LENGTH;
          const fx = (dx / dist) * overflow * EDGE_SPRING_STRENGTH;
          const fy = (dy / dist) * overflow * EDGE_SPRING_STRENGTH;

          if (src !== s.node) {
            a.curDX += fx;
            a.curDY += fy;
          }
          if (tgt !== s.node) {
            b.curDX -= fx;
            b.curDY -= fy;
          }
          allSettled = false;
        }
      });

      // Phase 4: write positions to graph
      for (const [id, cn] of s.nodes) {
        graph.setNodeAttribute(id, "x", cn.origX + cn.curDX);
        graph.setNodeAttribute(id, "y", cn.origY + cn.curDY);
      }

      if (s.released && allSettled) {
        s.active = false;
        return;
      }

      s.animFrame = requestAnimationFrame(animate);
    }

    registerEvents({
      downNode: ({ node, event }) => {
        event.preventSigmaDefault();
        const graph = sigma.getGraph();

        if (s.animFrame) {
          cancelAnimationFrame(s.animFrame);
          s.animFrame = 0;
        }

        const nx = graph.getNodeAttribute(node, "x") as number;
        const ny = graph.getNodeAttribute(node, "y") as number;

        s.node = node;
        s.active = true;
        s.moved = false;
        s.released = false;
        s.mouseStartScreen = { x: event.x, y: event.y };
        s.draggedOrigX = nx;
        s.draggedOrigY = ny;
        s.targetDX = 0;
        s.targetDY = 0;

        // Degree ratio determines how much the dragged node pulls the graph.
        // Leaf (ratio ~0.1): pull² = 0.01 → barely moves the graph.
        // Hub (ratio ~1.0): pull² = 1.0 → drags the whole graph.
        const draggedDegree = graph.degree(node);
        let maxDegree = 1;
        graph.forEachNode((n) => {
          const d = graph.degree(n);
          if (d > maxDegree) maxDegree = d;
        });
        const degreeRatio = draggedDegree / maxDegree;
        // Quadratic scaling: leaf=very weak, hub=strong
        const influence = degreeRatio * degreeRatio;

        // Pull per hop (scaled by influence):
        //   hop 0: 1.0 (dragged node always follows mouse)
        //   hop 1: 0.7 * influence
        //   hop 2: 0.3 * influence
        //   hop 3+: 0.1 * influence
        // Lerp (fluid motion):
        //   hop 1: 0.10, hop 2: 0.06, hop 3+: 0.03
        const PULL = [1.0, 0.7, 0.3, 0.1];
        const LERP = [1.0, 0.10, 0.06, 0.03];

        s.nodes.clear();
        const queue: Array<{ id: string; hops: number }> = [
          { id: node, hops: 0 },
        ];
        const visited = new Set<string>([node]);

        while (queue.length > 0) {
          const { id, hops } = queue.shift()!;
          const x = graph.getNodeAttribute(id, "x") as number;
          const y = graph.getNodeAttribute(id, "y") as number;

          const hopIdx = Math.min(hops, 3);
          const pullFactor =
            id === node ? 1.0 : (PULL[hopIdx] ?? 0.05) * influence;
          const lerpSpeed =
            id === node ? 1.0 : LERP[hopIdx] ?? 0.02;

          s.nodes.set(id, {
            origX: x,
            origY: y,
            curDX: 0,
            curDY: 0,
            lerpSpeed,
            pullFactor,
          });

          for (const nb of graph.neighbors(id)) {
            if (!visited.has(nb)) {
              visited.add(nb);
              queue.push({ id: nb, hops: hops + 1 });
            }
          }
        }

        sigma.getCamera().disable();
        onDragChange(true);
      },
    });

    return () => {
      if (s.animFrame) cancelAnimationFrame(s.animFrame);
      container.removeEventListener("wheel", blockWheel);
      container.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
  }, [sigma, registerEvents, onDragChange]);

  return null;
}

// --- Initial layout: circle ---

function InitialLayout({ version }: { version: number }) {
  const sigma = useSigma();

  useEffect(() => {
    const graph = sigma.getGraph();
    if (graph.order === 0) return;

    const nodes = graph.nodes();
    const n = nodes.length;
    const radius = Math.max(2, Math.sqrt(n) * 0.8);

    nodes.forEach((id, i) => {
      const angle = (2 * Math.PI * i) / n;
      const x = graph.getNodeAttribute(id, "x") as number;
      const y = graph.getNodeAttribute(id, "y") as number;
      if (Math.abs(x) < 0.1 && Math.abs(y) < 0.1) {
        graph.setNodeAttribute(id, "x", radius * Math.cos(angle));
        graph.setNodeAttribute(id, "y", radius * Math.sin(angle));
      }
    });

    const cam = sigma.getCamera();
    cam.animate({ x: 0.5, y: 0.5, ratio: 1, angle: 0 }, { duration: 300 });
  }, [sigma, version]);

  return null;
}

// --- Continuous force simulation: repulsion + edge attraction ---
// Runs on a timer after graph changes, gradually settling nodes
// into a natural layout where dense clusters form circles.

const SIM_REPULSION = 0.15; // push force between all node pairs
const SIM_REPULSION_RADIUS = 4; // distance within which repulsion acts
const SIM_EDGE_ATTRACTION = 0.02; // pull along edges (keeps connected nodes together)
const SIM_IDEAL_EDGE_LEN = 2; // edges shorter than this don't attract
const SIM_DAMPING = 0.85; // velocity damping per frame
const SIM_MAX_VELOCITY = 0.3; // cap per-frame movement
const SIM_FRAMES = 120; // run for this many frames after each graph change

function ForceSimulation({
  version,
  dragActive,
}: {
  version: number;
  dragActive: boolean;
}) {
  const sigma = useSigma();
  const velRef = useRef<Map<string, { vx: number; vy: number }>>(new Map());

  useEffect(() => {
    const graph = sigma.getGraph();
    if (graph.order === 0) return;

    // Reset velocities
    velRef.current.clear();
    graph.forEachNode((id) => {
      velRef.current.set(id, { vx: 0, vy: 0 });
    });

    let remaining = SIM_FRAMES;
    let animFrame = 0;

    function step() {
      if (remaining <= 0 || dragActive) return;
      remaining--;

      const nodes = graph.nodes();
      const vel = velRef.current;

      // Ensure all nodes have velocity entries
      for (const id of nodes) {
        if (!vel.has(id)) vel.set(id, { vx: 0, vy: 0 });
      }

      // Repulsion: all pairs within radius
      for (let i = 0; i < nodes.length; i++) {
        const idA = nodes[i]!;
        const ax = graph.getNodeAttribute(idA, "x") as number;
        const ay = graph.getNodeAttribute(idA, "y") as number;
        const va = vel.get(idA)!;

        for (let j = i + 1; j < nodes.length; j++) {
          const idB = nodes[j]!;
          const bx = graph.getNodeAttribute(idB, "x") as number;
          const by = graph.getNodeAttribute(idB, "y") as number;

          const dx = ax - bx;
          const dy = ay - by;
          const distSq = dx * dx + dy * dy;

          if (distSq < SIM_REPULSION_RADIUS * SIM_REPULSION_RADIUS && distSq > 0.01) {
            const dist = Math.sqrt(distSq);
            const t = dist / SIM_REPULSION_RADIUS;
            const force = SIM_REPULSION * (1 - t) * (1 - t);
            const fx = (dx / dist) * force;
            const fy = (dy / dist) * force;

            va.vx += fx;
            va.vy += fy;
            const vb = vel.get(idB)!;
            vb.vx -= fx;
            vb.vy -= fy;
          }
        }
      }

      // Edge attraction: pull connected nodes toward ideal distance
      graph.forEachEdge((_edge, _attrs, src, tgt) => {
        const ax = graph.getNodeAttribute(src, "x") as number;
        const ay = graph.getNodeAttribute(src, "y") as number;
        const bx = graph.getNodeAttribute(tgt, "x") as number;
        const by = graph.getNodeAttribute(tgt, "y") as number;

        const dx = bx - ax;
        const dy = by - ay;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist > SIM_IDEAL_EDGE_LEN && dist > 0.01) {
          const pull = (dist - SIM_IDEAL_EDGE_LEN) * SIM_EDGE_ATTRACTION;
          const fx = (dx / dist) * pull;
          const fy = (dy / dist) * pull;

          const va = vel.get(src);
          const vb = vel.get(tgt);
          if (va) { va.vx += fx; va.vy += fy; }
          if (vb) { vb.vx -= fx; vb.vy -= fy; }
        }
      });

      // Apply velocities with damping and cap
      for (const id of nodes) {
        const v = vel.get(id)!;
        v.vx *= SIM_DAMPING;
        v.vy *= SIM_DAMPING;

        // Cap velocity
        const speed = Math.sqrt(v.vx * v.vx + v.vy * v.vy);
        if (speed > SIM_MAX_VELOCITY) {
          v.vx = (v.vx / speed) * SIM_MAX_VELOCITY;
          v.vy = (v.vy / speed) * SIM_MAX_VELOCITY;
        }

        const x = graph.getNodeAttribute(id, "x") as number;
        const y = graph.getNodeAttribute(id, "y") as number;
        graph.setNodeAttribute(id, "x", x + v.vx);
        graph.setNodeAttribute(id, "y", y + v.vy);
      }

      animFrame = requestAnimationFrame(step);
    }

    // Small delay so InitialLayout runs first
    const timeout = setTimeout(() => {
      animFrame = requestAnimationFrame(step);
    }, 50);

    return () => {
      clearTimeout(timeout);
      if (animFrame) cancelAnimationFrame(animFrame);
    };
  }, [sigma, version, dragActive]);

  return null;
}

// --- Hover highlight (color only, no size change) ---

function HoverHighlighter() {
  const sigma = useSigma();
  const registerEvents = useRegisterEvents();
  const theme = useTheme();
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  useEffect(() => {
    registerEvents({
      enterNode: ({ node }) => setHoveredNode(node),
      leaveNode: () => setHoveredNode(null),
    });
  }, [registerEvents]);

  useEffect(() => {
    if (!hoveredNode) {
      sigma.setSetting("nodeReducer", null);
      sigma.setSetting("edgeReducer", null);
      sigma.refresh();
      return;
    }

    const graph = sigma.getGraph();
    const neighbors = new Set(graph.neighbors(hoveredNode));
    neighbors.add(hoveredNode);

    sigma.setSetting("nodeReducer", (node, data) => {
      if (neighbors.has(node)) {
        return { ...data };
      }
      return { ...data, color: theme.dimColor, label: null };
    });

    sigma.setSetting("edgeReducer", (edge, data) => {
      const src = graph.source(edge);
      const tgt = graph.target(edge);
      if (src === hoveredNode || tgt === hoveredNode) {
        return { ...data, color: theme.highlightEdge };
      }
      return { ...data, hidden: true };
    });

    sigma.refresh();
  }, [sigma, hoveredNode, theme]);

  return null;
}

// --- Click events ---

function GraphEvents({ onClickNode }: { onClickNode: (id: string) => void }) {
  const registerEvents = useRegisterEvents();

  useEffect(() => {
    registerEvents({
      clickNode: ({ node }) => onClickNode(node),
    });
  }, [registerEvents, onClickNode]);

  return null;
}

// --- Filter ---

function FilterApplier({
  nodeTypes,
  edgeTypes,
  minConfidence,
}: {
  nodeTypes: Record<string, boolean>;
  edgeTypes: Record<string, boolean>;
  minConfidence: number;
}) {
  const sigma = useSigma();

  useEffect(() => {
    const hiddenNodes = new Set<string>();

    sigma.setSetting("nodeReducer", (node, data) => {
      const nt = data.node_type as string | undefined;
      if (nt && nodeTypes[nt] === false) {
        hiddenNodes.add(node);
        return { ...data, hidden: true };
      }
      hiddenNodes.delete(node);
      return data;
    });

    sigma.setSetting("edgeReducer", (edge, data) => {
      const graph = sigma.getGraph();
      const src = graph.source(edge);
      const tgt = graph.target(edge);
      if (hiddenNodes.has(src) || hiddenNodes.has(tgt)) {
        return { ...data, hidden: true };
      }
      const et = data.edge_type as string | undefined;
      if (et && edgeTypes[et] === false) {
        return { ...data, hidden: true };
      }
      const conf = (data.confidence as number | undefined) ?? 1.0;
      if (conf < minConfidence) {
        return { ...data, hidden: true };
      }
      return data;
    });

    sigma.refresh();
  }, [sigma, nodeTypes, edgeTypes, minConfidence]);

  return null;
}

// --- Double-click to expand ---

function DoubleClickExpander({
  depth,
  onExpand,
}: {
  depth: number;
  onExpand: (nodeId: string, depth: number) => void;
}) {
  const registerEvents = useRegisterEvents();

  useEffect(() => {
    registerEvents({
      doubleClickNode: ({ node, event }) => {
        event.preventSigmaDefault();
        onExpand(node, depth);
      },
    });
  }, [registerEvents, depth, onExpand]);

  return null;
}

// --- No-op hover draw (disables sigma's built-in hover circle) ---

// eslint-disable-next-line @typescript-eslint/no-empty-function
const noopHoverDraw = () => {};

// --- Main ---

export default function GraphView({
  graph,
  version,
  onClickNode,
  onExpandNode,
  nodeTypes = {},
  edgeTypes = {},
  minConfidence = 0,
}: Props) {
  const theme = useTheme();
  const [depth, setDepth] = useState(1);
  const [zoom, setZoom] = useState(1);
  const [dragActive, setDragActive] = useState(false);
  const [simKey, setSimKey] = useState(0);
  const hasFilters =
    Object.values(nodeTypes).some((v) => v === false) ||
    Object.values(edgeTypes).some((v) => v === false) ||
    minConfidence > 0;

  const handleZoomChange = useCallback((z: number) => setZoom(z), []);
  const handleDragChange = useCallback((active: boolean) => {
    setDragActive(active);
    // When drag ends, bump simKey to re-trigger force sim (restores shape)
    if (!active) setSimKey((k) => k + 1);
  }, []);

  const sigmaSettings = useMemo(
    () => ({
      defaultNodeColor: theme.nodeDefault,
      defaultNodeSize: 6,
      defaultEdgeColor: theme.edgeDefault,
      defaultEdgeSize: 0.3,
      defaultEdgeType: "curved" as const,
      edgeProgramClasses: {
        curved: EdgeCurveProgram,
      },
      // Disable sigma's built-in hover enlargement
      defaultDrawNodeHover: noopHoverDraw,
      labelRenderedSizeThreshold: 5,
      labelFont: "Inter, sans-serif",
      labelSize: 13,
      labelWeight: "500",
      labelColor: { color: theme.labelColor },
      edgeLabelSize: 10,
      minCameraRatio: 0.02,
      maxCameraRatio: 10,
      enableEdgeEvents: false,
      zoomDuration: 200,
      inertiaDuration: 0,
      inertiaRatio: 0,
      itemSizesReference: "screen" as const,
      autoRescale: false,
      autoCenter: false,
      hideEdgesOnMove: false,
      labelDensity: 2,
    }),
    [theme],
  );

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        position: "relative",
        // @ts-expect-error CSS custom property
        "--sigma-background-color": "transparent",
      }}
    >
      <SigmaContainer
        graph={graph}
        style={{ width: "100%", height: "100%" }}
        settings={sigmaSettings}
      >
        <InitialLayout version={version} />
        <ForceSimulation version={version + simKey} dragActive={dragActive} />
        <GraphEvents onClickNode={onClickNode} />
        <DragController onDragChange={handleDragChange} />
        <DoubleClickExpander depth={depth} onExpand={onExpandNode} />
        {!hasFilters && <HoverHighlighter />}
        {hasFilters && (
          <FilterApplier nodeTypes={nodeTypes} edgeTypes={edgeTypes} minConfidence={minConfidence} />
        )}
        <ZoomTracker onZoomChange={handleZoomChange} />
      </SigmaContainer>
      <DepthControl depth={depth} onChange={setDepth} />
      <div
        style={{
          position: "absolute",
          bottom: 16,
          right: 16,
          zIndex: 10,
          background: theme.bg,
          border: `1px solid ${theme.border}`,
          borderRadius: 8,
          padding: "5px 10px",
          fontSize: 12,
          fontWeight: 500,
          color: theme.textMuted,
          boxShadow: `0 2px 8px ${theme.shadow}`,
          fontVariantNumeric: "tabular-nums",
          pointerEvents: "none",
        }}
      >
        {zoom.toFixed(1)}x
      </div>
    </div>
  );
}
