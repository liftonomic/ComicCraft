"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="", tags=["health"])


@router.get("/health")
async def health() -> dict:
    """Return a lightweight liveness signal."""
    return {"status": "ok"}