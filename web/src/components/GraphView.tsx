import {
  SigmaContainer,
  useRegisterEvents,
  useCamera,
} from "@react-sigma/core";
import "@react-sigma/core/lib/style.css";
import type Graph from "graphology";
import { useEffect } from "react";

interface Props {
  graph: Graph;
  version: number;
  onClickNode: (nodeId: string) => void;
  highlightedNodes?: Set<string>;
  activationScores?: Map<string, number>;
}

function GraphEvents({ onClickNode }: { onClickNode: (id: string) => void }) {
  const registerEvents = useRegisterEvents();
  const camera = useCamera();

  useEffect(() => {
    registerEvents({
      clickNode: ({ node }) => onClickNode(node),
      doubleClickNode: ({ node }) => {
        // Center + zoom on double click
        const pos = camera.getState();
        camera.animate(
          { ...pos, x: 0.5, y: 0.5, ratio: pos.ratio * 0.5 },
          { duration: 300 },
        );
        onClickNode(node);
      },
    });
  }, [registerEvents, onClickNode, camera]);

  return null;
}

export default function GraphView({
  graph,
  onClickNode,
}: Props) {
  return (
    <div style={{ width: "100%", height: "100%" }}>
      <SigmaContainer
        graph={graph}
        style={{ width: "100%", height: "100%" }}
        settings={{
          defaultNodeColor: "#999",
          defaultEdgeColor: "#ccc",
          labelRenderedSizeThreshold: 8,
          labelFont: "sans-serif",
          labelSize: 12,
          edgeLabelFont: "sans-serif",
          edgeLabelSize: 10,
          minCameraRatio: 0.02,
          maxCameraRatio: 10,
          enableEdgeEvents: false,
        }}
      >
        <GraphEvents onClickNode={onClickNode} />
      </SigmaContainer>
    </div>
  );
}
