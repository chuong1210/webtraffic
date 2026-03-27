"""
Stream routes – start/stop stream + status.
"""

from __future__ import annotations
from fastapi import APIRouter

from app.models.detection_model import StreamStartRequest, SuccessResponse
from app.services.stream_service import stream_service

router = APIRouter(prefix="/api/v1/stream", tags=["stream"])


@router.post("/start", response_model=SuccessResponse, status_code=202)
async def start_stream(body: StreamStartRequest):
    """Start video capture in background; returns immediately (no blocking). YouTube resolve runs async."""
    stream_service.start(body.url)
    return SuccessResponse(success=True, message="Stream đang kết nối (YouTube có thể mất vài giây)...")


@router.post("/stop", response_model=SuccessResponse)
async def stop_stream():
    """Stop the active stream."""
    stream_service.stop()
    return SuccessResponse(success=True, message="Stream stopped")


def _device_info():
    """Return CUDA/CPU device info for display in UI."""
    try:
        import torch
        cuda = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda else None
        return {"cuda_available": cuda, "device_name": name}
    except Exception:
        return {"cuda_available": False, "device_name": None}


@router.get("/device")
async def stream_device():
    """Return whether backend is using GPU (CUDA) or CPU. For UI status pill."""
    return _device_info()


@router.get("/status")
async def stream_status():
    s = stream_service.stats
    return {
        "active": s.stream_active,
        "fps": s.fps,
        "frame_count": s.frame_count,
        "error": s.stream_error or None,
    }


@router.get("/frame")
async def stream_frame():
    """
    Return latest encoded frame + detections + stats.
    If no frame is available yet, returns an empty payload with current stats.
    """
    payload = stream_service.get_latest()
    if payload is None:
        # Keep response shape stable for frontend
        s = stream_service.stats
        return {
            "frame": None,
            "detections": [],
            "stats": s.model_dump(),
        }
    return payload
