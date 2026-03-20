"""FastAPI application for the Kosa knowledge graph web interface."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from kosa.api.deps import close_driver, get_driver
from kosa.api.routes import activation, graph, hypotheses, stats
from kosa.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: verify Neo4j connection. Shutdown: close driver."""
    driver = await get_driver()
    await driver.verify_connectivity()
    yield
    await close_driver()


app = FastAPI(
    title="Kosa Knowledge Graph API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(graph.router, prefix="/api/v1/graph", tags=["graph"])
app.include_router(hypotheses.router, prefix="/api/v1/hypotheses", tags=["hypotheses"])
app.include_router(activation.router, prefix="/api/v1/activation", tags=["activation"])
app.include_router(stats.router, prefix="/api/v1/stats", tags=["stats"])

# Serve frontend static files in production
_web_dist = Path(__file__).resolve().parent.parent.parent.parent / "web" / "dist"
if _web_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_web_dist), html=True), name="frontend")


@app.get("/api/health")
async def health():
    return {"status": "ok"}
