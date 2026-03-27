"""
Model routes – upload, list, load, delete YOLO .pt files.
"""

from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path

from app.core.config import settings
from app.models.detection_model import (
    ModelInfo, ModelLoadRequest, SuccessResponse, ErrorResponse
)
from app.services.model_service import model_service

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("", response_model=list[ModelInfo])
async def list_models():
    """Return all available .pt models."""
    return model_service.list_models()


@router.post("/upload", response_model=SuccessResponse)
async def upload_model(file: UploadFile = File(...)):
    """Upload a YOLOv8 .pt file; store in models_storage and load as active model."""
    if not file.filename or not file.filename.endswith(".pt"):
        raise HTTPException(status_code=400, detail="Only .pt files are accepted")

    # Store in dedicated models folder (models_storage) per architecture spec
    save_path = settings.MODELS_DIR / file.filename
    content = await file.read()
    save_path.write_bytes(content)

    try:
        model_service.load(save_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return SuccessResponse(success=True, message=f"Model '{file.filename}' loaded ✓")


@router.post("/load", response_model=SuccessResponse)
async def load_model(body: ModelLoadRequest):
    """Load a previously uploaded model by name. Use 'default' for YOLOv8n."""
    try:
        if body.name == "default":
            # Load pretrained YOLOv8n (auto-downloads from ultralytics hub)
            from app.ml.yolo_model import yolo_model
            yolo_model.load_pretrained("yolov8n.pt")
            return SuccessResponse(success=True, message="Loaded default YOLOv8n")
        path = model_service._resolve(body.name)
        model_service.load(path)
        return SuccessResponse(success=True, message=f"Loaded '{body.name}'")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/{name}", response_model=SuccessResponse)
async def delete_model(name: str):
    """Delete a model file from storage."""
    try:
        model_service.delete(name)
        return SuccessResponse(success=True, message=f"Deleted '{name}'")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
