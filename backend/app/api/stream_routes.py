"""
Stream routes – start/stop stream + status.
"""

from __future__ import annotations
import asyncio
import base64
import concurrent.futures
import cv2
from typing import Annotated
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from app.models.detection_model import StreamStartRequest, SuccessResponse
from app.services.stream_service import stream_service

# Thread pool for thumbnail grabs (non-blocking)
_thumb_executor = concurrent.futures.ThreadPoolExecutor(max_workers=6, thread_name_prefix="thumb")

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


@router.get("/thumbnail")
async def stream_thumbnail(url: Annotated[str, Query(description="RTSP or video URL")]):
    """
    Grab a single frame from any RTSP/video URL and return as base64 JPEG.
    Used by the camera wall to show live previews without starting the main stream.
    Runs in a thread pool so it doesn't block the event loop.
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_thumb_executor, _grab_thumbnail, url)
    return JSONResponse(result)


def _grab_thumbnail(url: str) -> dict:
    """Synchronous: open stream, grab 1 frame, close immediately."""
    try:
        import os
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "rtsp_transport;tcp|buffer_size;4096000|max_delay;500000|stimeout;5000000|fflags;nobuffer|flags;low_delay"
        )
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)

        if not cap.isOpened():
            return {"ok": False, "frame": None, "error": "Cannot open stream"}

        # Skip 10 frames — HEVC needs reference frames to decode cleanly
        # (avoids "PPS out of range" artifacts in first few frames)
        frame: cv2.typing.MatLike | None = None
        for i in range(15):
            ret, f = cap.read()
            if ret and f is not None and i >= 8:
                frame = f
                break

        cap.release()

        if frame is None:
            return {"ok": False, "frame": None, "error": "No frame"}

        # Resize to thumbnail (320px wide) to keep response small
        h, w = frame.shape[:2]
        thumb_h = int(320 * h / w) if w > 0 else 180
        thumb = cv2.resize(frame, (320, thumb_h), interpolation=cv2.INTER_AREA)

        _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
        b64 = base64.b64encode(buf.tobytes()).decode()
        return {"ok": True, "frame": b64, "error": None}

    except Exception as e:
        return {"ok": False, "frame": None, "error": str(e)}


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
