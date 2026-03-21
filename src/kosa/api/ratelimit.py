"""Simple in-memory sliding-window rate limiter for FastAPI."""

from __future__ import annotations

import time
from collections import defaultdict

from fastapi import HTTPException, Request


class RateLimiter:
    """Sliding-window rate limiter keyed by client IP.

    Usage as a FastAPI dependency::

        limiter = RateLimiter(max_requests=10, window_seconds=60)

        @router.post("/endpoint")
        async def endpoint(request: Request, _=Depends(limiter)):
            ...
    """

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)

    def _client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _prune(self, key: str, now: float) -> None:
        cutoff = now - self.window_seconds
        hits = self._hits[key]
        # Remove expired timestamps
        while hits and hits[0] < cutoff:
            hits.pop(0)

    async def __call__(self, request: Request) -> None:
        now = time.monotonic()
        key = self._client_ip(request)
        self._prune(key, now)

        if len(self._hits[key]) >= self.max_requests:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit exceeded. Max {self.max_requests}"
                    f" requests per {self.window_seconds}s."
                ),
            )
        self._hits[key].append(now)

    def check_ip(self, ip: str) -> bool:
        """Non-raising check for WebSocket use. Returns True if allowed."""
        now = time.monotonic()
        self._prune(ip, now)
        if len(self._hits[ip]) >= self.max_requests:
            return False
        self._hits[ip].append(now)
        return True
