"""
Multi-Camera Pallet Box Counting Engine
========================================
Handles:
  - Temporal box tracking across video frames (IOU-based)
  - Pallet entry/exit detection (auto start/stop counting)
  - Per-camera accumulation and deduplication
  - Multi-camera evidence fusion for one final box count
  - Frame quality filtering (blur detection)
"""

import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import cv2


# ──────────────────────────────────────────────────────────────────────────────
# Enums & Data Classes
# ──────────────────────────────────────────────────────────────────────────────

class SessionState(str, Enum):
    WAITING   = "waiting"      # Waiting for pallet to enter the arch
    COUNTING  = "counting"     # Pallet detected — actively counting
    COMPLETED = "completed"    # Pallet exited — final count available


@dataclass
class DetectedBox:
    """A single bounding box detected in one frame."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def to_dict(self) -> dict:
        return {
            "x1": round(self.x1, 2), "y1": round(self.y1, 2),
            "x2": round(self.x2, 2), "y2": round(self.y2, 2),
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id, "class_name": self.class_name,
        }


@dataclass
class TrackedBox:
    """A box tracked across multiple consecutive frames."""
    track_id: str
    sightings: List[Dict] = field(default_factory=list)

    @property
    def total_sightings(self) -> int:
        return len(self.sightings)

    @property
    def avg_confidence(self) -> float:
        if not self.sightings:
            return 0.0
        return sum(s["box"].confidence for s in self.sightings) / len(self.sightings)

    @property
    def last_box(self) -> Optional[DetectedBox]:
        return self.sightings[-1]["box"] if self.sightings else None

    @property
    def last_frame(self) -> int:
        return self.sightings[-1]["frame_idx"] if self.sightings else -1

    def add_sighting(self, frame_idx: int, box: DetectedBox, timestamp: float):
        self.sightings.append({"frame_idx": frame_idx, "box": box, "timestamp": timestamp})


# ──────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────────────────────────────────────

def compute_iou(a: DetectedBox, b: DetectedBox) -> float:
    """Intersection-over-Union between two bounding boxes."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def is_frame_blurry(frame: np.ndarray, threshold: float = 80.0) -> Tuple[bool, float]:
    """
    Laplacian-variance blur detector.
    Returns (is_blurry, variance_score).  Lower score = blurrier.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return var < threshold, float(var)


# ──────────────────────────────────────────────────────────────────────────────
# Per-Camera Tracker
# ──────────────────────────────────────────────────────────────────────────────

class CameraTracker:
    """
    IOU-based multi-object tracker for a single camera stream.
    Matches new detections to existing tracks each frame, creates new tracks
    for unmatched detections, and retires tracks not seen recently.
    """

    def __init__(self, camera_role: str,
                 iou_threshold: float = 0.25,
                 max_lost_frames: int = 12):
        self.camera_role = camera_role
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames

        self.tracked_boxes: Dict[str, TrackedBox] = {}
        self.frame_count: int = 0
        self.frame_history: List[Dict] = []      # per-frame raw counts
        self._next_id: int = 0

    # ---- internal helpers ----
    def _new_id(self) -> str:
        self._next_id += 1
        return f"{self.camera_role}_{self._next_id}"

    def _active_tracks(self) -> Dict[str, TrackedBox]:
        return {
            tid: tb for tid, tb in self.tracked_boxes.items()
            if self.frame_count - tb.last_frame <= self.max_lost_frames
        }

    # ---- main update ----
    def update(self, detections: List[DetectedBox], timestamp: float) -> Dict:
        """Feed one frame of detections.  Returns a frame-level summary dict."""
        self.frame_count += 1
        active = self._active_tracks()

        matched_det: set = set()
        matched_trk: set = set()

        # Greedy IOU matching (detection → nearest active track)
        for d_idx, det in enumerate(detections):
            best_iou, best_tid = 0.0, None
            for tid, tb in active.items():
                if tid in matched_trk:
                    continue
                iou = compute_iou(det, tb.last_box)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou, best_tid = iou, tid
            if best_tid:
                self.tracked_boxes[best_tid].add_sighting(self.frame_count, det, timestamp)
                matched_det.add(d_idx)
                matched_trk.add(best_tid)

        # Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_det:
                tb = TrackedBox(track_id=self._new_id())
                tb.add_sighting(self.frame_count, det, timestamp)
                self.tracked_boxes[tb.track_id] = tb

        record = {
            "frame_idx": self.frame_count,
            "timestamp": timestamp,
            "raw_count": len(detections),
            "active_tracks": len(self._active_tracks()),
        }
        self.frame_history.append(record)
        return record

    # ---- aggregate statistics ----
    def reliable_count(self, min_sightings: int = 3) -> int:
        """Boxes seen in at least *min_sightings* frames."""
        return sum(1 for tb in self.tracked_boxes.values()
                   if tb.total_sightings >= min_sightings)

    def peak_count(self) -> int:
        if not self.frame_history:
            return 0
        return max(fh["raw_count"] for fh in self.frame_history)

    def median_count(self) -> int:
        nz = [fh["raw_count"] for fh in self.frame_history if fh["raw_count"] > 0]
        return int(np.median(nz)) if nz else 0

    def mode_count(self) -> int:
        nz = [fh["raw_count"] for fh in self.frame_history if fh["raw_count"] > 0]
        if not nz:
            return 0
        return Counter(nz).most_common(1)[0][0]

    def summary(self) -> Dict:
        return {
            "camera_role": self.camera_role,
            "total_frames": self.frame_count,
            "unique_tracks": len(self.tracked_boxes),
            "reliable_count": self.reliable_count(),
            "peak_count": self.peak_count(),
            "median_count": self.median_count(),
            "mode_count": self.mode_count(),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Pallet Session (orchestrates one pallet pass-through)
# ──────────────────────────────────────────────────────────────────────────────

class PalletSession:
    """
    Encapsulates one counting episode — from pallet entry to pallet exit.

    Lifecycle:
        WAITING  →  (boxes appear for N frames)  →  COUNTING
        COUNTING →  (no boxes for M frames)      →  COMPLETED
                 →  (manual end)                  →  COMPLETED
    """

    ENTRY_FRAMES  = 5    # consecutive "present" frames to start
    EXIT_FRAMES   = 15   # consecutive "absent" frames to auto-finish

    def __init__(self, session_id: Optional[str] = None, auto_detect: bool = True):
        self.session_id: str = session_id or uuid.uuid4().hex[:8]
        self.auto_detect: bool = auto_detect
        self.state: SessionState = SessionState.WAITING if auto_detect else SessionState.COUNTING
        self.created_at: float = time.time()
        self.started_at: Optional[float] = time.time() if not auto_detect else None
        self.completed_at: Optional[float] = None

        self.camera_trackers: Dict[str, CameraTracker] = {}

        self._consec_present: int = 0
        self._consec_absent: int = 0

        self.total_frames: int = 0
        self.skipped_blur: int = 0

        self.final_count: Optional[int] = None
        self.fusion_details: Optional[Dict] = None

    # ---- tracker access ----
    def _tracker(self, role: str) -> CameraTracker:
        if role not in self.camera_trackers:
            self.camera_trackers[role] = CameraTracker(role)
        return self.camera_trackers[role]

    # ---- main frame ingestion ----
    def process_frame(self, camera_role: str,
                      detections: List[DetectedBox],
                      is_blurry: bool = False,
                      blur_score: float = 0.0) -> Dict:
        """
        Ingest one frame from one camera.
        Returns a status dict with running counts and session state.
        """
        self.total_frames += 1
        ts = time.time()

        # Skip blurry frames
        if is_blurry:
            self.skipped_blur += 1
            return self._status_dict(camera_role, skipped=True,
                                     skip_reason="blur", blur_score=blur_score)

        tracker = self._tracker(camera_role)
        frame_info = tracker.update(detections, ts)

        # ---- auto entry / exit detection ----
        has_boxes = len(detections) > 0

        if self.auto_detect and self.state == SessionState.WAITING:
            if has_boxes:
                self._consec_present += 1
                self._consec_absent = 0
                if self._consec_present >= self.ENTRY_FRAMES:
                    self.state = SessionState.COUNTING
                    self.started_at = ts
            else:
                self._consec_present = 0

        elif self.state == SessionState.COUNTING and self.auto_detect:
            if has_boxes:
                self._consec_absent = 0
            else:
                self._consec_absent += 1
                if self._consec_absent >= self.EXIT_FRAMES:
                    self._finalize()

        return self._status_dict(camera_role, frame_info=frame_info)

    # ---- finalize & fuse ----
    def force_complete(self):
        if self.state != SessionState.COMPLETED:
            self._finalize()

    def _finalize(self):
        self.state = SessionState.COMPLETED
        self.completed_at = time.time()
        cam_summaries = {r: t.summary() for r, t in self.camera_trackers.items()}
        self.fusion_details = self._fuse(cam_summaries)
        self.final_count = self.fusion_details["fused_count"]

    def _fuse(self, cam_summaries: Dict) -> Dict:
        """
        Multi-camera fusion strategy.

        Single camera  → mode count (most frequent per-frame count)
        Top + Front    → weighted combination / max with agreement bonus
        Fallback       → max reliable count across cameras
        """
        if not cam_summaries:
            return {"fused_count": 0, "strategy": "no_data", "per_camera": {}}

        per_cam: Dict[str, Dict] = {}
        for role, s in cam_summaries.items():
            mode   = s.get("mode_count", 0)
            median = s.get("median_count", 0)
            reliable = s.get("reliable_count", 0)
            peak   = s.get("peak_count", 0)
            best   = mode or median or reliable
            per_cam[role] = {
                "mode": mode, "median": median,
                "reliable": reliable, "peak": peak,
                "best_estimate": best,
                "total_frames": s.get("total_frames", 0),
            }

        roles = set(per_cam.keys())

        if len(roles) == 1:
            role = list(roles)[0]
            fused = per_cam[role]["best_estimate"]
            strategy = f"single_camera_{role}"

        elif "top" in roles and "front" in roles:
            te = per_cam["top"]["best_estimate"]
            fe = per_cam["front"]["best_estimate"]

            if te > 0 and fe > 0:
                fused = max(te, fe)
                ratio = min(te, fe) / max(te, fe) if max(te, fe) else 0
                strategy = "multi_cam_agreement" if ratio > 0.8 else "multi_cam_max"
            else:
                fused = te or fe
                strategy = "multi_cam_top_only" if te else "multi_cam_front_only"

            if "side" in roles:
                se = per_cam["side"]["best_estimate"]
                if se > fused:
                    fused = se
                    strategy += "+side_override"

        else:
            fused = max(c["best_estimate"] for c in per_cam.values())
            strategy = "max_across_cameras"

        return {"fused_count": fused, "strategy": strategy, "per_camera": per_cam}

    # ---- status helpers ----
    def _status_dict(self, camera_role: str, *,
                     frame_info: Optional[Dict] = None,
                     skipped: bool = False,
                     skip_reason: str = "",
                     blur_score: float = 0.0) -> Dict:
        running = {}
        for r, t in self.camera_trackers.items():
            running[r] = {
                "reliable": t.reliable_count(),
                "median": t.median_count(),
                "mode": t.mode_count(),
                "peak": t.peak_count(),
                "frames": t.frame_count,
            }
        elapsed = time.time() - (self.started_at or self.created_at)
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "auto_detect": self.auto_detect,
            "skipped": skipped,
            "skip_reason": skip_reason,
            "blur_score": round(blur_score, 1),
            "camera_role": camera_role,
            "frame_info": frame_info,
            "running_counts": running,
            "elapsed_s": round(elapsed, 1),
            "total_frames": self.total_frames,
            "skipped_blur": self.skipped_blur,
            "cameras_active": list(self.camera_trackers.keys()),
            "final_count": self.final_count,
            "fusion_details": self.fusion_details,
        }

    def full_status(self) -> Dict:
        running = {r: t.summary() for r, t in self.camera_trackers.items()}
        elapsed = time.time() - (self.started_at or self.created_at)
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "auto_detect": self.auto_detect,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "cameras": list(self.camera_trackers.keys()),
            "total_frames": self.total_frames,
            "skipped_blur": self.skipped_blur,
            "elapsed_s": round(elapsed, 1),
            "per_camera": running,
            "final_count": self.final_count,
            "fusion_details": self.fusion_details,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Session Manager (holds all sessions)
# ──────────────────────────────────────────────────────────────────────────────

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, PalletSession] = {}
        self.active_id: Optional[str] = None

    def create(self, auto_detect: bool = True) -> PalletSession:
        s = PalletSession(auto_detect=auto_detect)
        self.sessions[s.session_id] = s
        self.active_id = s.session_id
        return s

    @property
    def active(self) -> Optional[PalletSession]:
        return self.sessions.get(self.active_id) if self.active_id else None

    def get(self, sid: str) -> Optional[PalletSession]:
        return self.sessions.get(sid)

    def end(self, sid: str) -> Optional[PalletSession]:
        s = self.sessions.get(sid)
        if s:
            s.force_complete()
            if self.active_id == sid:
                self.active_id = None
        return s

    def history(self) -> List[Dict]:
        return [s.full_status() for s in self.sessions.values()]
