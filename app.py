"""
Traffic Monitoring Demo – Flask Backend
YOLOv8 Vehicle Detection, Counting & ROI
Production-ready version for university thesis demo
"""

import os
import cv2
import json
import time
import logging
import threading
import numpy as np
from collections import deque
from flask import Flask, Response, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ─── Logging Setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("traffic_monitor")

# ─── App Setup ───────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # 300 MB

os.makedirs("uploads", exist_ok=True)

# ─── Global State ─────────────────────────────────────────────────────────────
model = None
model_name = None

cap = None
stream_thread = None
stream_running = False
frame_queue = deque(maxlen=2)          # thread-safe, only keep last 2 frames
frame_lock = threading.Lock()

roi_points: list = []                  # [(x, y), ...]
roi_active: bool = False

# Configurable settings
settings = {
    "conf_threshold": 0.35,
    "line_position": 0.55,             # fraction of frame height
    "max_fps": 30,
    "detect_classes": [0, 1, 2, 3, 5, 7],  # COCO ids
}

counting_line_y: int | None = None

# Stats
stats = {
    "total": 0,
    "classes": {},
    "fps": 0.0,
    "stream_active": False,
    "model_loaded": False,
    "model_name": "",
    "roi_active": False,
    "frame_count": 0,
    "conf_threshold": 0.35,
    "line_position": 0.55,
}

# Simple centroid tracker
counted_ids: set = set()
tracked_centroids: dict = {}           # id → (cx, cy, cls)
next_track_id: int = 0

# Timeline: sliding window of per-second counts  (last 60 seconds)
timeline_lock = threading.Lock()
timeline: deque = deque(maxlen=60)     # each entry = count that second
_timeline_last_total = 0
_timeline_last_ts = time.time()

# ─── Constants ───────────────────────────────────────────────────────────────
VEHICLE_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

CLASS_COLORS = {
    "person":     (255, 178,  50),
    "bicycle":    ( 50, 205,  50),
    "car":        (  0, 165, 255),
    "motorcycle": (255,   0, 255),
    "bus":        (  0, 255, 255),
    "truck":      (255,  50,  50),
}


# ─── Utility Functions ────────────────────────────────────────────────────────

def point_in_polygon(px: float, py: float, polygon: list) -> bool:
    """Ray-casting algorithm – point inside polygon check."""
    if len(polygon) < 3:
        return True
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-9) + xi
        ):
            inside = not inside
        j = i
    return inside


def draw_roi_overlay(frame: np.ndarray, points: list, active: bool) -> np.ndarray:
    """Draw transparent ROI polygon over frame."""
    if len(points) < 2:
        return frame
    overlay = frame.copy()
    pts = np.array(points, dtype=np.int32)
    color = (0, 255, 120) if active else (0, 200, 255)
    if len(points) >= 3:
        cv2.fillPoly(overlay, [pts], color)
    cv2.polylines(overlay, [pts], isClosed=(len(points) >= 3), color=color, thickness=2)
    for p in points:
        cv2.circle(overlay, p, 5, color, -1)
    alpha = 0.25 if active else 0.15
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def draw_counting_line(frame: np.ndarray, y: int) -> None:
    """Draw the vehicle counting line."""
    h, w = frame.shape[:2]
    cv2.line(frame, (0, y), (w, y), (0, 120, 255), 2)
    cv2.putText(
        frame, "Counting Line",
        (10, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 1
    )


def draw_hud(frame: np.ndarray, frame_stats: dict) -> None:
    """Draw HUD overlay with FPS and per-class counts."""
    h, w = frame.shape[:2]
    rows = len(frame_stats)
    box_h = 30 + rows * 22 + 26
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (230, box_h), (15, 15, 20), -1)
    frame[:] = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    cv2.putText(
        frame, f"FPS: {stats['fps']:.1f}   Frame: {stats['frame_count']}",
        (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 200), 1,
    )
    cv2.putText(
        frame, f"Total: {stats['total']}",
        (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 245, 160), 1,
    )
    y0 = 66
    for cls_name, cnt in frame_stats.items():
        color = CLASS_COLORS.get(cls_name, (200, 200, 200))
        cv2.putText(
            frame, f"  {cls_name}: {cnt}",
            (14, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
        )
        y0 += 22


# ─── Detection Pipeline ───────────────────────────────────────────────────────

def process_frame(frame: np.ndarray) -> np.ndarray:
    """Run YOLOv8 inference + ROI filter + line-crossing counting."""
    global model, counted_ids, next_track_id, counting_line_y

    if model is None:
        return frame

    conf = settings["conf_threshold"]
    results = model(frame, verbose=False, conf=conf)[0]

    frame_stats: dict = {}

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in VEHICLE_CLASSES:
            continue

        cls_name = VEHICLE_CLASSES[cls_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # ROI filter
        if roi_active and len(roi_points) >= 3:
            if not point_in_polygon(cx, cy, roi_points):
                continue

        color = CLASS_COLORS.get(cls_name, (200, 200, 200))

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Rounded label background
        label = f"{cls_name} {confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1
        )

        frame_stats[cls_name] = frame_stats.get(cls_name, 0) + 1

        # Line-crossing counter (simple centroid proximity tracker)
        if counting_line_y is not None:
            matched_id = None
            for tid, (tx, ty, tcls) in list(tracked_centroids.items()):
                if tcls == cls_name and abs(cx - tx) < 70 and abs(cy - ty) < 70:
                    matched_id = tid
                    break

            if matched_id is None:
                matched_id = next_track_id
                next_track_id += 1

            prev_y = tracked_centroids.get(matched_id, (cx, cy, cls_name))[1]
            tracked_centroids[matched_id] = (cx, cy, cls_name)

            if matched_id not in counted_ids:
                crossed = (prev_y < counting_line_y <= cy) or (cy <= counting_line_y < prev_y)
                if crossed:
                    counted_ids.add(matched_id)
                    stats["total"] += 1
                    stats["classes"][cls_name] = stats["classes"].get(cls_name, 0) + 1
                    logger.debug("Counted %s (id=%d), total=%d", cls_name, matched_id, stats["total"])

    # Draw overlays
    frame = draw_roi_overlay(frame, roi_points, roi_active)
    if counting_line_y is not None:
        draw_counting_line(frame, counting_line_y)
    draw_hud(frame, frame_stats)

    return frame


# ─── Stream Worker ────────────────────────────────────────────────────────────

def stream_worker(rtsp_url: str) -> None:
    """Background thread: capture → detect → queue frame."""
    global cap, stream_running, stats

    logger.info("Opening stream: %s", rtsp_url)
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        logger.error("Cannot open stream: %s", rtsp_url)
        stats["stream_active"] = False
        stream_running = False
        return

    stats["stream_active"] = True
    fps_counter = 0
    t0 = time.time()
    frame_interval = 1.0 / settings["max_fps"]

    logger.info("Stream started – url=%s", rtsp_url)

    while stream_running:
        t_frame = time.time()
        ret, frame = cap.read()

        if not ret:
            # Loop for local video files
            if not rtsp_url.lower().startswith("rtsp://"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                logger.debug("Looping video file")
                continue
            logger.warning("Stream ended unexpectedly")
            break

        stats["frame_count"] += 1
        fps_counter += 1
        elapsed = time.time() - t0
        if elapsed >= 1.0:
            stats["fps"] = round(fps_counter / elapsed, 1)
            fps_counter = 0
            t0 = time.time()
            _update_timeline()

        try:
            processed = process_frame(frame.copy())
        except Exception as exc:
            logger.error("Frame processing error: %s", exc)
            processed = frame

        with frame_lock:
            if frame_queue.maxlen:
                frame_queue.append(processed.copy())

        # Throttle to max_fps
        sleep_time = frame_interval - (time.time() - t_frame)
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    stats["stream_active"] = False
    stream_running = False
    logger.info("Stream stopped")


def _update_timeline():
    """Append current-second vehicle delta to timeline."""
    global _timeline_last_total, _timeline_last_ts
    with timeline_lock:
        delta = stats["total"] - _timeline_last_total
        timeline.append({"t": int(time.time()), "v": max(0, delta)})
        _timeline_last_total = stats["total"]


def generate_frames():
    """MJPEG generator – yield JPEG frames from queue."""
    while True:
        frame = None
        with frame_lock:
            if frame_queue:
                frame = frame_queue[-1]

        if frame is None:
            time.sleep(0.04)
            continue

        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )
        time.sleep(1.0 / settings["max_fps"])


def _reset_counters():
    """Reset all counting state."""
    global counted_ids, tracked_centroids, next_track_id, _timeline_last_total
    stats["total"] = 0
    stats["classes"] = {}
    stats["frame_count"] = 0
    stats["fps"] = 0.0
    counted_ids = set()
    tracked_centroids = {}
    next_track_id = 0
    with timeline_lock:
        timeline.clear()
        _timeline_last_total = 0


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── Model Management ─────────────────────────────────────────────────────────

@app.route("/api/models", methods=["GET"])
def list_models():
    """List all uploaded .pt models."""
    folder = app.config["UPLOAD_FOLDER"]
    files = [
        {
            "name": f,
            "size_mb": round(os.path.getsize(os.path.join(folder, f)) / 1e6, 2),
            "active": f == model_name,
        }
        for f in os.listdir(folder)
        if f.endswith(".pt")
    ]
    files.sort(key=lambda x: x["name"])
    return jsonify({"models": files})


@app.route("/api/upload-model", methods=["POST"])
def upload_model():
    global model, model_name

    if "model" not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400

    f = request.files["model"]
    if not f.filename or not f.filename.endswith(".pt"):
        return jsonify({"success": False, "error": "Only .pt files are accepted"}), 400

    filename = secure_filename(f.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(path)
    logger.info("Model file saved: %s", path)

    try:
        from ultralytics import YOLO
        model = YOLO(path)
        model_name = filename
        stats["model_loaded"] = True
        stats["model_name"] = filename
        logger.info("Model loaded successfully: %s", filename)
        return jsonify({"success": True, "message": f"Model '{filename}' loaded ✓"})
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/models/load", methods=["POST"])
def load_model():
    """Load a previously uploaded model by name."""
    global model, model_name

    data = request.get_json() or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"success": False, "error": "No model name provided"}), 400

    path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(name))
    if not os.path.isfile(path):
        return jsonify({"success": False, "error": "Model not found"}), 404

    try:
        from ultralytics import YOLO
        model = YOLO(path)
        model_name = name
        stats["model_loaded"] = True
        stats["model_name"] = name
        logger.info("Switched to model: %s", name)
        return jsonify({"success": True, "message": f"Loaded '{name}'"})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/models/<name>", methods=["DELETE"])
def delete_model(name):
    """Delete a model file."""
    global model, model_name

    safe = secure_filename(name)
    path = os.path.join(app.config["UPLOAD_FOLDER"], safe)
    if not os.path.isfile(path):
        return jsonify({"success": False, "error": "Model not found"}), 404

    # Unload if active
    if safe == model_name:
        model = None
        model_name = None
        stats["model_loaded"] = False
        stats["model_name"] = ""

    os.remove(path)
    logger.info("Deleted model: %s", safe)
    return jsonify({"success": True, "message": f"Deleted '{safe}'"})


# ── Settings ─────────────────────────────────────────────────────────────────

@app.route("/api/settings", methods=["GET"])
def get_settings():
    return jsonify({
        "conf_threshold": settings["conf_threshold"],
        "line_position": settings["line_position"],
        "max_fps": settings["max_fps"],
    })


@app.route("/api/settings", methods=["POST"])
def update_settings():
    global counting_line_y

    data = request.get_json() or {}

    if "conf_threshold" in data:
        val = float(data["conf_threshold"])
        settings["conf_threshold"] = max(0.1, min(0.95, val))
        stats["conf_threshold"] = settings["conf_threshold"]
        logger.info("Confidence threshold → %.2f", settings["conf_threshold"])

    if "line_position" in data:
        val = float(data["line_position"])
        settings["line_position"] = max(0.1, min(0.95, val))
        stats["line_position"] = settings["line_position"]
        # Recompute line pixel position from last known frame height
        if cap and cap.isOpened():
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if h > 0:
                counting_line_y = int(h * settings["line_position"])
        logger.info("Line position → %.2f", settings["line_position"])

    if "max_fps" in data:
        settings["max_fps"] = max(1, min(60, int(data["max_fps"])))

    return jsonify({"success": True, "settings": settings})


# ── Stream ────────────────────────────────────────────────────────────────────

@app.route("/api/stream/start", methods=["POST"])
def start_stream():
    global stream_thread, stream_running, counting_line_y

    data = request.get_json() or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"success": False, "error": "No URL provided"}), 400

    # Stop existing stream
    if stream_running:
        stream_running = False
        if stream_thread:
            stream_thread.join(timeout=4)
        with frame_lock:
            frame_queue.clear()

    _reset_counters()

    # Probe stream for dimensions
    probe = cv2.VideoCapture(url)
    if not probe.isOpened():
        logger.error("Cannot open URL: %s", url)
        return jsonify({"success": False, "error": "Cannot open stream. Check the URL."}), 400

    ret, probe_frame = probe.read()
    if ret:
        h = probe_frame.shape[0]
        counting_line_y = int(h * settings["line_position"])
        logger.info("Frame height=%d → counting_line_y=%d", h, counting_line_y)
    probe.release()

    stream_running = True
    stream_thread = threading.Thread(target=stream_worker, args=(url,), daemon=True)
    stream_thread.start()

    return jsonify({"success": True, "message": "Stream started"})


@app.route("/api/stream/stop", methods=["POST"])
def stop_stream():
    global stream_running
    stream_running = False
    with frame_lock:
        frame_queue.clear()
    stats["stream_active"] = False
    return jsonify({"success": True})


@app.route("/api/stream/status", methods=["GET"])
def stream_status():
    return jsonify({
        "active": stats["stream_active"],
        "fps": stats["fps"],
        "frame_count": stats["frame_count"],
    })


@app.route("/api/video-feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ── ROI ───────────────────────────────────────────────────────────────────────

@app.route("/api/roi", methods=["POST"])
def set_roi():
    global roi_points, roi_active

    data = request.get_json() or {}
    points = data.get("points", [])
    active = data.get("active", True)

    roi_points = [(int(p["x"]), int(p["y"])) for p in points]
    roi_active = active and len(roi_points) >= 3
    stats["roi_active"] = roi_active

    logger.info("ROI set: %d points, active=%s", len(roi_points), roi_active)
    return jsonify({"success": True, "points": len(roi_points), "active": roi_active})


@app.route("/api/roi/clear", methods=["POST"])
def clear_roi():
    global roi_points, roi_active
    roi_points = []
    roi_active = False
    stats["roi_active"] = False
    return jsonify({"success": True})


# ── Stats & Timeline ──────────────────────────────────────────────────────────

@app.route("/api/stats")
def get_stats():
    return jsonify(stats)


@app.route("/api/timeline")
def get_timeline():
    with timeline_lock:
        return jsonify({"timeline": list(timeline)})


@app.route("/api/reset-count", methods=["POST"])
def reset_count():
    _reset_counters()
    logger.info("Counters reset")
    return jsonify({"success": True})


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting Traffic Monitor on http://0.0.0.0:5000")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
