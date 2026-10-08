"""
Warehouse Box Detection System — FastAPI Backend v3.1
=====================================================
Supports:
  - Single-camera live detection (Webcam & IP phone streams like DroidCam/IP Webcam)
  - Multi-camera pallet counting sessions (Vision Arch mode)
  - Direct IP Camera stream ingestion (DroidCam / IP Webcam / RTSP / HTTP)
  - Photo upload detection
  - Model hot-swapping
  - Frame blur filtering
  - Session management with temporal tracking & fusion
"""

import io
import os
import time
import base64
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
from collections import Counter

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

from counting_engine import (
    SessionManager, PalletSession, DetectedBox,
    is_frame_blurry, SessionState,
)

# ──────────────────────────────────────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Warehouse Box Detection System", version="3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────────────────────────────────────

def get_available_models() -> List[str]:
    return sorted(f.name for f in Path(".").glob("*.pt"))

MODEL_PATH = "best.pt" if os.path.exists("best.pt") else "yolov8n.pt"
print(f"[STARTUP] Loading YOLO model from {MODEL_PATH} ...")
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names if hasattr(model, "names") else {0: "box"}
print(f"[STARTUP] Model loaded — classes: {CLASS_NAMES}")

# ──────────────────────────────────────────────────────────────────────────────
# Session Manager (global singleton)
# ──────────────────────────────────────────────────────────────────────────────

session_mgr = SessionManager()

# ──────────────────────────────────────────────────────────────────────────────
# IP Stream Background Stream Reader & Snapshot Grabber
# ──────────────────────────────────────────────────────────────────────────────

import urllib.request

def normalize_and_get_snapshot_urls(url: str) -> List[str]:
    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://") and not url.startswith("rtsp://"):
        url = "http://" + url
    
    candidates = []
    if ":4747" in url:
        base = url.split(":4747")[0] + ":4747"
        candidates = [
            f"{base}/cam/1/frame.jpg",    # DroidCam instant JPEG snapshot (fastest)
            f"{base}/video",               # DroidCam MJPEG stream
            f"{base}/mjpegfeed",           # DroidCam alternate MJPEG
            url
        ]
    elif ":8080" in url:
        base = url.split(":8080")[0] + ":8080"
        candidates = [
            f"{base}/shot.jpg",            # IP Webcam instant JPEG snapshot
            f"{base}/video",               # IP Webcam stream
            url
        ]
    else:
        candidates = [url]
        if not url.endswith(("/video", ".mjpg", ".jpg", ".jpeg")):
            candidates.append(url.rstrip("/") + "/video")
            candidates.append(url.rstrip("/") + "/cam/1/frame.jpg")

    return list(dict.fromkeys(candidates))


class IPStreamReader:
    """Reads frames from an IP camera stream (DroidCam, IP Webcam, RTSP, MJPEG) in a background thread."""
    def __init__(self, url: str):
        self.url = url
        self.cap: Optional[cv2.VideoCapture] = None
        self.last_frame: Optional[np.ndarray] = None
        self.last_time = 0.0
        self.lock = threading.Lock()
        self.running = True
        self.error: Optional[str] = None
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.running:
            # First try snapshot endpoint if DroidCam/IP Webcam
            snapshot_urls = [u for u in normalize_and_get_snapshot_urls(self.url) if u.endswith((".jpg", ".jpeg")) or "/cam/" in u or "/shot" in u]
            got_snapshot = False
            for s_url in snapshot_urls:
                try:
                    req = urllib.request.Request(s_url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=1.0) as resp:
                        if resp.status == 200:
                            raw = resp.read()
                            arr = np.frombuffer(raw, dtype=np.uint8)
                            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if img is not None:
                                with self.lock:
                                    self.last_frame = img
                                    self.last_time = time.time()
                                    self.error = None
                                got_snapshot = True
                                time.sleep(0.04) # ~25 FPS
                                break
                except Exception:
                    pass

            if got_snapshot:
                continue

            # Fallback to cv2.VideoCapture for video/mjpeg streams
            if self.cap is None or not self.cap.isOpened():
                try:
                    self.cap = cv2.VideoCapture(self.url)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception as e:
                    self.error = str(e)
                    time.sleep(0.8)
                    continue

            if not self.cap.isOpened():
                self.error = "Cannot open stream"
                time.sleep(0.8)
                continue

            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.last_frame = frame
                    self.last_time = time.time()
                    self.error = None
                time.sleep(0.02)
            else:
                time.sleep(0.05)
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None

    def read(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.last_frame is not None and (time.time() - self.last_time < 5.0):
                return self.last_frame.copy()
        return None

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

# Global registry of active IP stream readers
ip_stream_pool: Dict[str, IPStreamReader] = {}
pool_lock = threading.Lock()

def get_or_create_ip_stream(url: str) -> IPStreamReader:
    url = url.strip()
    with pool_lock:
        if url not in ip_stream_pool or not ip_stream_pool[url].running:
            ip_stream_pool[url] = IPStreamReader(url)
        return ip_stream_pool[url]

def fetch_ip_frame(url: str, timeout_sec: float = 2.0) -> Optional[np.ndarray]:
    url_candidates = normalize_and_get_snapshot_urls(url)
    
    # 1. Try fast HTTP snapshot endpoints
    for c_url in url_candidates:
        if c_url.endswith((".jpg", ".jpeg")) or "/cam/" in c_url or "/shot" in c_url:
            try:
                req = urllib.request.Request(c_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=min(1.0, timeout_sec)) as resp:
                    if resp.status == 200:
                        raw = resp.read()
                        arr = np.frombuffer(raw, dtype=np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if img is not None:
                            return img
            except Exception:
                pass

    # 2. Try pooled background reader
    reader = get_or_create_ip_stream(url)
    t_end = time.time() + timeout_sec
    while time.time() < t_end:
        frame = reader.read()
        if frame is not None:
            return frame
        time.sleep(0.04)

    return None

# ──────────────────────────────────────────────────────────────────────────────
# Pydantic Request Schemas
# ──────────────────────────────────────────────────────────────────────────────

class FramePayload(BaseModel):
    image: str                          # base64 (data-URI or raw)
    confidence: Optional[float] = 0.35

class MultiFramePayload(BaseModel):
    image: str
    camera_role: str                    # "top" | "front" | "side"
    session_id: str
    confidence: Optional[float] = 0.35
    blur_filter: Optional[bool] = True  # skip blurry frames?

class MultiIPFramePayload(BaseModel):
    stream_url: str                     # e.g. "http://192.168.1.102:4747/video"
    camera_role: str                    # "top" | "front" | "side"
    session_id: str
    confidence: Optional[float] = 0.35
    blur_filter: Optional[bool] = True
    return_image: Optional[bool] = True # return base64 frame for browser canvas

class SingleIPFramePayload(BaseModel):
    stream_url: str
    confidence: Optional[float] = 0.35

class TestIPPayload(BaseModel):
    stream_url: str

class ModelSwitchPayload(BaseModel):
    model_name: str

class SessionCreatePayload(BaseModel):
    auto_detect: Optional[bool] = True  # auto entry/exit detection?


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _decode_base64_image(data: str) -> np.ndarray:
    """Decode a base64 (or data-URI) string into an OpenCV BGR ndarray."""
    if "," in data:
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data)
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image from base64 payload")
    return img


def _encode_base64_image(img: np.ndarray, quality: int = 70) -> str:
    """Encode OpenCV BGR ndarray to JPEG data-URI."""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


def _run_yolo(img: np.ndarray, conf: float) -> List[DetectedBox]:
    """Run YOLO inference and return list of DetectedBox objects."""
    conf = max(0.05, min(0.99, conf))
    results = model.predict(source=img, conf=conf, verbose=False)
    out: List[DetectedBox] = []
    if results and results[0].boxes is not None:
        for b in results[0].boxes:
            xyxy = b.xyxy[0].tolist()
            out.append(DetectedBox(
                x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3],
                confidence=float(b.conf[0]),
                class_id=int(b.cls[0]),
                class_name=CLASS_NAMES.get(int(b.cls[0]), f"cls_{int(b.cls[0])}"),
            ))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — UI Pages
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def index():
    p = Path("Upload_for_Detection.html")
    return FileResponse(p) if p.exists() else JSONResponse(status_code=404, content={"msg": "HTML not found"})


@app.get("/example_input.jpg")
async def example_image():
    p = Path("example_input.jpg")
    return FileResponse(p) if p.exists() else JSONResponse(status_code=404, content={"msg": "not found"})


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — Model Info & Hot-Swap
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/model_info")
async def model_info():
    return {
        "active_model": MODEL_PATH,
        "classes": CLASS_NAMES,
        "class_list": list(CLASS_NAMES.values()) if isinstance(CLASS_NAMES, dict) else CLASS_NAMES,
        "available_models": get_available_models(),
    }


@app.post("/api/switch_model")
async def switch_model(payload: ModelSwitchPayload):
    global model, MODEL_PATH, CLASS_NAMES
    if not os.path.exists(payload.model_name):
        raise HTTPException(404, f"Model '{payload.model_name}' not found")
    try:
        model = YOLO(payload.model_name)
        MODEL_PATH = payload.model_name
        CLASS_NAMES = model.names if hasattr(model, "names") else {0: "box"}
        return {"success": True, "active_model": MODEL_PATH, "classes": CLASS_NAMES}
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — IP Camera Utilities
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/ip_cam/test")
async def test_ip_cam(payload: TestIPPayload):
    """Test connection to an IP camera stream (DroidCam / IP Webcam / HTTP)."""
    try:
        frame = fetch_ip_frame(payload.stream_url, timeout_sec=3.0)
        if frame is None:
            return JSONResponse(status_code=400, content={
                "success": False,
                "error": f"Could not connect to {payload.stream_url}. Verify the phone and laptop are on the same Wi-Fi and DroidCam is active."
            })
        
        # Make a small preview thumbnail
        h, w = frame.shape[:2]
        thumb = cv2.resize(frame, (min(320, w), int(h * (min(320, w) / w))))
        b64 = _encode_base64_image(thumb, quality=60)
        return {
            "success": True,
            "width": w,
            "height": h,
            "preview": b64,
            "message": "Stream connected successfully!"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/detect_ip_frame")
async def detect_ip_frame(payload: SingleIPFramePayload):
    """Fetch frame from IP stream, run single-camera detection, return boxes + base64 image."""
    try:
        t0 = time.time()
        img = fetch_ip_frame(payload.stream_url, timeout_sec=2.0)
        if img is None:
            return JSONResponse(status_code=400, content={"success": False, "error": "Stream frame timeout"})

        detections = _run_yolo(img, payload.confidence or 0.35)
        ms = round((time.time() - t0) * 1000, 1)
        cc = Counter(d.class_name for d in detections)
        b64 = _encode_base64_image(img, quality=65)

        return {
            "success": True,
            "inference_ms": ms,
            "total_boxes": len(detections),
            "class_counts": dict(cc),
            "boxes": [d.to_dict() for d in detections],
            "image": b64,
            "image_width": img.shape[1],
            "image_height": img.shape[0],
        }
    except Exception as e:
        return JSONResponse(500, {"success": False, "error": str(e)})


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — Single-Camera Live Detection (Browser Canvas Base64)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/detect_frame")
async def detect_frame(payload: FramePayload):
    """Single-camera, stateless per-frame detection (no session tracking)."""
    try:
        t0 = time.time()
        img = _decode_base64_image(payload.image)
        detections = _run_yolo(img, payload.confidence or 0.35)
        ms = round((time.time() - t0) * 1000, 1)

        cc = Counter(d.class_name for d in detections)
        return {
            "success": True,
            "inference_ms": ms,
            "total_boxes": len(detections),
            "class_counts": dict(cc),
            "boxes": [d.to_dict() for d in detections],
            "image_width": img.shape[1],
            "image_height": img.shape[0],
        }
    except Exception as e:
        return JSONResponse(500, {"success": False, "error": str(e)})


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — Multi-Camera Vision Arch Sessions
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/session/create")
async def session_create(payload: SessionCreatePayload):
    """Create a new pallet counting session."""
    s = session_mgr.create(auto_detect=payload.auto_detect)
    return {"success": True, "session": s.full_status()}


@app.get("/api/session/active")
async def session_active():
    """Get the currently active session."""
    s = session_mgr.active
    if not s:
        return {"success": True, "session": None}
    return {"success": True, "session": s.full_status()}


@app.get("/api/session/{session_id}")
async def session_status(session_id: str):
    """Get status of a specific session."""
    s = session_mgr.get(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return {"success": True, "session": s.full_status()}


@app.post("/api/session/{session_id}/end")
async def session_end(session_id: str):
    """Manually end a session and get the final fused count."""
    s = session_mgr.end(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return {"success": True, "session": s.full_status()}


@app.get("/api/session/history/all")
async def session_history():
    """Get all past sessions."""
    return {"success": True, "sessions": session_mgr.history()}


@app.post("/api/detect_multi_frame")
async def detect_multi_frame(payload: MultiFramePayload):
    """
    Accepts browser-captured frame (base64) + camera_role + session_id.
    Runs YOLO, feeds into the session's tracking engine, returns running state.
    """
    session = session_mgr.get(payload.session_id)
    if not session:
        raise HTTPException(404, f"Session '{payload.session_id}' not found")
    if session.state == SessionState.COMPLETED:
        return {
            "success": True,
            "message": "Session already completed",
            "session": session.full_status(),
        }

    try:
        t0 = time.time()
        img = _decode_base64_image(payload.image)

        # Optional blur filtering
        blurry = False
        blur_score = 0.0
        if payload.blur_filter:
            blurry, blur_score = is_frame_blurry(img, threshold=60.0)

        # Run YOLO (only if not blurry)
        detections = _run_yolo(img, payload.confidence or 0.35) if not blurry else []
        ms = round((time.time() - t0) * 1000, 1)

        # Feed into session counting engine
        session_result = session.process_frame(
            camera_role=payload.camera_role,
            detections=detections,
            is_blurry=blurry,
            blur_score=blur_score,
        )

        return {
            "success": True,
            "inference_ms": ms,
            "blurry": blurry,
            "blur_score": round(blur_score, 1),
            "raw_detections": len(detections),
            "boxes": [d.to_dict() for d in detections],
            "image_width": img.shape[1],
            "image_height": img.shape[0],
            **session_result,
        }

    except Exception as e:
        return JSONResponse(500, {"success": False, "error": str(e)})


@app.post("/api/detect_multi_ip_frame")
async def detect_multi_ip_frame(payload: MultiIPFramePayload):
    """
    Direct IP Stream endpoint (for DroidCam / IP Webcam / RTSP streams).
    Reads frame directly from phone stream URL in Python backend, runs YOLO,
    feeds into session tracking engine, and returns detection results + frame image.
    """
    session = session_mgr.get(payload.session_id)
    if not session:
        raise HTTPException(404, f"Session '{payload.session_id}' not found")
    if session.state == SessionState.COMPLETED:
        return {
            "success": True,
            "message": "Session already completed",
            "session": session.full_status(),
        }

    try:
        t0 = time.time()
        img = fetch_ip_frame(payload.stream_url, timeout_sec=2.0)
        if img is None:
            return JSONResponse(status_code=400, content={"success": False, "error": f"Failed to grab frame from {payload.stream_url}"})

        # Blur filtering
        blurry = False
        blur_score = 0.0
        if payload.blur_filter:
            blurry, blur_score = is_frame_blurry(img, threshold=60.0)

        detections = _run_yolo(img, payload.confidence or 0.35) if not blurry else []
        ms = round((time.time() - t0) * 1000, 1)

        # Feed session tracking engine
        session_result = session.process_frame(
            camera_role=payload.camera_role,
            detections=detections,
            is_blurry=blurry,
            blur_score=blur_score,
        )

        resp = {
            "success": True,
            "inference_ms": ms,
            "blurry": blurry,
            "blur_score": round(blur_score, 1),
            "raw_detections": len(detections),
            "boxes": [d.to_dict() for d in detections],
            "image_width": img.shape[1],
            "image_height": img.shape[0],
            **session_result,
        }

        if payload.return_image:
            resp["image"] = _encode_base64_image(img, quality=65)

        return resp

    except Exception as e:
        return JSONResponse(500, {"success": False, "error": str(e)})


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — Photo Upload Detection
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/detect_file")
async def detect_file(file: UploadFile = File(...), confidence: float = Form(0.35)):
    try:
        t0 = time.time()
        raw = await file.read()
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "Could not decode image")

        detections = _run_yolo(img, confidence)
        ms = round((time.time() - t0) * 1000, 1)
        cc = Counter(d.class_name for d in detections)

        results = model.predict(source=img, conf=max(0.05, min(0.99, confidence)), verbose=False)
        annotated = results[0].plot() if results else img
        _, buf = cv2.imencode(".jpg", annotated)
        b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

        return {
            "success": True,
            "inference_ms": ms,
            "total_boxes": len(detections),
            "class_counts": dict(cc),
            "boxes": [d.to_dict() for d in detections],
            "annotated_image": b64,
            "image_width": img.shape[1],
            "image_height": img.shape[0],
        }
    except Exception as e:
        return JSONResponse(500, {"success": False, "error": str(e)})


# ══════════════════════════════════════════════════════════════════════════════
#  Legacy Endpoints (backwards compat)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/YOLO_Box_Prediction_Website/")
def legacy_website(file: UploadFile):
    try:
        tmp = Path("temp_upload.jpg")
        with open(tmp, "wb") as f:
            f.write(file.file.read())
        res = model(str(tmp))
        im = Image.fromarray(res[0].plot()[..., ::-1])
        out = Path("example_output.jpg")
        im.save(out)
        return FileResponse(out)
    except Exception as e:
        return {"message": str(e)}


@app.post("/YOLO_Box_Prediction_Service/")
async def legacy_service(file: UploadFile):
    try:
        tmp = Path("temp_service_upload.jpg")
        with open(tmp, "wb") as f:
            f.write(await file.read())
        res = model(str(tmp))
        im = Image.fromarray(res[0].plot()[..., ::-1])
        out = Path("example_output.jpg")
        im.save(out)
        return {
            "file_metadata": {"file_name": str(out), "file_size": os.path.getsize(out)},
            "file_data": FileResponse(out),
        }
    except Exception as e:
        return {"message": str(e)}