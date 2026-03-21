"""Tests for the rate limiter."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from kosa.api.ratelimit import RateLimiter


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_allows_under_limit(self):
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        request = MagicMock()
        request.client.host = "127.0.0.1"
        request.headers = {}

        # Should allow 3 requests
        for _ in range(3):
            await limiter(request)

    @pytest.mark.asyncio
    async def test_blocks_over_limit(self):
        from fastapi import HTTPException

        limiter = RateLimiter(max_requests=2, window_seconds=60)
        request = MagicMock()
        request.client.host = "127.0.0.1"
        request.headers = {}

        await limiter(request)
        await limiter(request)

        with pytest.raises(HTTPException) as exc_info:
            await limiter(request)
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_different_ips_independent(self):
        limiter = RateLimiter(max_requests=1, window_seconds=60)

        req1 = MagicMock()
        req1.client.host = "1.1.1.1"
        req1.headers = {}

        req2 = MagicMock()
        req2.client.host = "2.2.2.2"
        req2.headers = {}

        await limiter(req1)
        await limiter(req2)  # Should not raise

    def test_check_ip_allows_and_blocks(self):
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        assert limiter.check_ip("10.0.0.1") is True
        assert limiter.check_ip("10.0.0.1") is True
        assert limiter.check_ip("10.0.0.1") is False

    @pytest.mark.asyncio
    async def test_forwarded_ip(self):
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        request = MagicMock()
        request.client.host = "127.0.0.1"
        request.headers = {"x-forwarded-for": "203.0.113.50, 70.41.3.18"}

        await limiter(request)
        # IP should be 203.0.113.50, not 127.0.0.1
        assert len(limiter._hits["203.0.113.50"]) == 1
        assert len(limiter._hits.get("127.0.0.1", [])) == 0

    def test_window_expiry(self):
        limiter = RateLimiter(max_requests=1, window_seconds=1)
        assert limiter.check_ip("10.0.0.2") is True
        assert limiter.check_ip("10.0.0.2") is False
        # Manually expire the window
        limiter._hits["10.0.0.2"][0] = time.monotonic() - 2
        assert limiter.check_ip("10.0.0.2") is True
