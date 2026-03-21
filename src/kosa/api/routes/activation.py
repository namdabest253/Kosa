"""Activation wave simulation endpoint.

Implements typed random walk with edge-type-specific transition weights.
Streams step-by-step propagation via WebSocket for animated visualization.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from kosa.api.deps import get_driver
from kosa.api.ratelimit import RateLimiter

router = APIRouter()

_ws_limiter = RateLimiter(max_requests=5, window_seconds=60)

# Edge type transition weights for typed random walk
EDGE_TYPE_WEIGHTS: dict[str, float] = {
    "CITES": 0.3,  # Ground truth but less semantically interesting
    "INTRODUCES": 0.9,
    "EVALUATES_ON": 0.5,
    "HAS_LIMITATION": 0.8,
    "MITIGATES": 0.9,
    "IMPROVES_OVER": 0.85,
    "USES": 0.7,
    "IS_INSTANCE_OF": 0.6,
    "CAUSED_BY": 0.7,
    "TEMPORALLY_FOLLOWS": 0.5,
    "SAME_AS": 0.1,  # Entity resolution edges — don't walk these
}


@router.websocket("/simulate")
async def simulate_activation(ws: WebSocket):
    """WebSocket endpoint for activation wave simulation.

    Client sends: {"seed_id": "...", "steps": 5, "decay": 0.7}
    Server streams: {"step": N, "activations": [{"id": "...", "score": 0.X, "label": "..."}]}
    """
    # Rate-limit before accepting the connection
    client_ip = "unknown"
    if ws.client:
        client_ip = ws.client.host
    forwarded = ws.headers.get("x-forwarded-for")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()

    if not _ws_limiter.check_ip(client_ip):
        await ws.close(code=1008, reason="Rate limit exceeded")
        return

    await ws.accept()

    try:
        data = await ws.receive_json()
        seed_id = data.get("seed_id")
        steps = min(data.get("steps", 5), 10)
        decay = data.get("decay", 0.7)

        if not seed_id:
            await ws.send_json({"error": "seed_id required"})
            return

        driver = await get_driver()

        # Track activation scores: node_id → score
        activations: dict[str, float] = {seed_id: 1.0}
        visited: set[str] = set()
        labels: dict[str, str] = {}

        # Get seed label
        async with driver.session() as session:
            r = await session.run(
                "MATCH (n) WHERE elementId(n) = $id RETURN n.name AS name, n.title AS title",
                {"id": seed_id},
            )
            rec = await r.single()
            if rec:
                labels[seed_id] = rec["name"] or rec["title"] or "?"

        # Send initial state
        await ws.send_json(
            {
                "step": 0,
                "activations": [{"id": seed_id, "score": 1.0, "label": labels.get(seed_id, "?")}],
            }
        )

        # Propagate wave step by step
        for step in range(1, steps + 1):
            frontier = {nid: score for nid, score in activations.items() if nid not in visited}
            if not frontier:
                break

            new_activations: dict[str, float] = {}

            async with driver.session() as session:
                for nid, parent_score in frontier.items():
                    visited.add(nid)

                    result = await session.run(
                        "MATCH (n)-[r]-(m) WHERE elementId(n) = $id "
                        "RETURN elementId(m) AS mid, type(r) AS rtype, "
                        "r.confidence AS conf, m.name AS name, m.title AS title, "
                        "m.significance AS sig "
                        "ORDER BY r.confidence DESC LIMIT 50",
                        {"id": nid},
                    )

                    async for rec in result:
                        mid = rec["mid"]
                        rtype = rec["rtype"]
                        conf = rec["conf"] or 0.5
                        sig = rec["sig"] or 0.5

                        # Typed walk: multiply parent score × decay × edge weight × confidence × significance
                        edge_weight = EDGE_TYPE_WEIGHTS.get(rtype, 0.5)
                        score = parent_score * decay * edge_weight * conf * sig

                        if mid not in activations or score > activations.get(mid, 0):
                            new_activations[mid] = max(new_activations.get(mid, 0), score)
                            labels[mid] = rec["name"] or rec["title"] or "?"

            # Merge new activations
            for mid, score in new_activations.items():
                if score > activations.get(mid, 0):
                    activations[mid] = score

            # Send step results (only newly activated or updated nodes)
            step_results = sorted(
                [
                    {"id": mid, "score": round(score, 4), "label": labels.get(mid, "?")}
                    for mid, score in new_activations.items()
                    if score > 0.01  # Threshold: ignore negligible activations
                ],
                key=lambda x: x["score"],
                reverse=True,
            )[:50]  # Cap results per step

            await ws.send_json({"step": step, "activations": step_results})
            await asyncio.sleep(0.3)  # Pacing for smooth animation

        await ws.send_json({"step": -1, "done": True})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass
