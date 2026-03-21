"""Streaming ingestion API endpoints: trigger and monitor ingestion jobs."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from kosa.ingestion.streaming import (
    IngestionJob,
    get_job,
    read_watermark,
    register_job,
    run_streaming_ingestion,
)

router = APIRouter()


class TriggerRequest(BaseModel):
    arxiv_ids: list[str] | None = None
    date_from: str | None = None
    date_to: str | None = None


class TriggerResponse(BaseModel):
    job_id: str
    status: str


@router.post("/trigger", response_model=TriggerResponse)
async def trigger_ingestion(
    request: TriggerRequest,
    background_tasks: BackgroundTasks,
):
    """Trigger a streaming ingestion job.

    Accepts either a list of arXiv IDs or a date range.
    The job runs in the background; poll /status/{job_id} for progress.
    """
    if not request.arxiv_ids and not (request.date_from and request.date_to):
        raise HTTPException(
            status_code=422,
            detail="Must provide either arxiv_ids or date_from + date_to",
        )

    job_id = str(uuid.uuid4())[:8]
    job = IngestionJob(
        job_id=job_id,
        arxiv_ids=request.arxiv_ids or [],
        date_from=request.date_from,
        date_to=request.date_to,
    )
    register_job(job)

    background_tasks.add_task(run_streaming_ingestion, job)

    return TriggerResponse(job_id=job_id, status="pending")


@router.get("/status/{job_id}")
async def get_ingestion_status(job_id: str):
    """Check progress of a streaming ingestion job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job.to_dict()


@router.get("/watermark")
async def get_watermark():
    """Return the current streaming ingestion watermark (last processed date)."""
    last_date = read_watermark()
    return {"last_date": last_date}
