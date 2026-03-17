"""Shared test fixtures."""

import pytest

from kosa.config import Settings


@pytest.fixture
def test_settings():
    """Settings with test defaults (no real API keys or DB connections)."""
    return Settings(
        openai_api_key="test-key",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="test-password",
    )
