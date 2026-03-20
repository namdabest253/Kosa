"""Dependency injection for FastAPI routes."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from neo4j import AsyncGraphDatabase, AsyncSession

from kosa.config import settings

_driver = None


async def get_driver():
    """Lazily create and return the async Neo4j driver."""
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
    return _driver


async def get_session() -> AsyncGenerator[AsyncSession]:
    """Yield an async Neo4j session, closed after request."""
    driver = await get_driver()
    async with driver.session() as session:
        yield session


async def close_driver():
    """Close the Neo4j driver (called on shutdown)."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None


Neo4jSession = Annotated[AsyncSession, Depends(get_session)]
