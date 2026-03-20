import { useCallback, useRef, useState } from "react";

export interface ActivationStep {
  step: number;
  activations: { id: string; score: number; label: string }[];
}

/**
 * WebSocket hook for activation wave simulation.
 * Streams step-by-step propagation from a seed node.
 */
export function useActivation() {
  const [steps, setSteps] = useState<ActivationStep[]>([]);
  const [running, setRunning] = useState(false);
  const [scores, setScores] = useState<Map<string, number>>(new Map());
  const wsRef = useRef<WebSocket | null>(null);

  const simulate = useCallback(
    (seedId: string, numSteps = 5, decay = 0.7) => {
      // Close previous connection
      if (wsRef.current) {
        wsRef.current.close();
      }

      setSteps([]);
      setScores(new Map());
      setRunning(true);

      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const ws = new WebSocket(`${protocol}//${window.location.host}/api/v1/activation/simulate`);
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(JSON.stringify({ seed_id: seedId, steps: numSteps, decay }));
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data as string) as
          | ActivationStep
          | { step: -1; done: true }
          | { error: string };

        if ("error" in data) {
          console.error("Activation error:", data.error);
          setRunning(false);
          return;
        }

        if ("done" in data) {
          setRunning(false);
          return;
        }

        setSteps((prev) => [...prev, data]);
        setScores((prev) => {
          const next = new Map(prev);
          for (const a of data.activations) {
            const existing = next.get(a.id) ?? 0;
            if (a.score > existing) {
              next.set(a.id, a.score);
            }
          }
          return next;
        });
      };

      ws.onclose = () => setRunning(false);
      ws.onerror = () => setRunning(false);
    },
    [],
  );

  const stop = useCallback(() => {
    wsRef.current?.close();
    setRunning(false);
  }, []);

  return { steps, scores, running, simulate, stop };
}
