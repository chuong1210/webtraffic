"""Video utility functions."""

import re
import subprocess
import sys

import cv2


# Patterns for YouTube URLs (watch, short, embed)
YOUTUBE_PATTERN = re.compile(
    r"^(https?://)?(www\.)?(youtube\.com/(watch\?v=|embed/|v/)|youtu\.be/)[\w-]+",
    re.IGNORECASE,
)

# Video ID: 10-15 chars, alphanumeric + hyphen + underscore (YouTube IDs are typically 11)
VIDEO_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{10,15}$")


def _extract_youtube_video_id(url: str) -> str | None:
    """Extract video ID from YouTube URL, or None if not found / invalid."""
    url = url.strip()
    # watch?v=ID, embed/ID, v/ID
    m = re.search(r"(?:watch\?v=|embed/|v/)([a-zA-Z0-9_-]{10,15})", url, re.IGNORECASE)
    if m:
        return m.group(1)
    # youtu.be/ID
    m = re.search(r"youtu\.be/([a-zA-Z0-9_-]{10,15})", url, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def is_youtube_url(url: str) -> bool:
    """Return True if the URL is a YouTube watch/short/embed link."""
    if not url or not url.strip():
        return False
    return bool(YOUTUBE_PATTERN.match(url.strip()))


def validate_youtube_url(url: str) -> None:
    """
    Raise ValueError if URL looks like YouTube but video ID is missing or invalid.
    Call before resolve_youtube_url to avoid blocking on yt-dlp with bad URLs.
    """
    if not is_youtube_url(url):
        return
    video_id = _extract_youtube_video_id(url)
    if not video_id or not VIDEO_ID_PATTERN.match(video_id):
        raise ValueError(
            "Link YouTube không hợp lệ. Cần video ID đầy đủ (ví dụ: .../watch?v=XXXXXXXXXXX)"
        )


def resolve_youtube_url(youtube_url: str) -> str | None:
    """
    Resolve a YouTube URL to a direct stream URL using yt-dlp.
    Returns the URL string, or None if resolution fails (yt-dlp missing or video unavailable).
    """
    url = youtube_url.strip()
    if not is_youtube_url(url):
        return None
    try:
        result = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "-g", "--no-check-certificate", url],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0 or not result.stdout:
            return None
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None


def probe_video(url: str) -> dict | None:
    """
    Open video source and return basic properties.
    Returns None if source cannot be opened.
    """
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        cap.release()
        return None

    info = {
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":    cap.get(cv2.CAP_PROP_FPS) or 30,
    }
    cap.release()
    return info


def is_rtsp(url: str) -> bool:
    return url.lower().startswith("rtsp://")
