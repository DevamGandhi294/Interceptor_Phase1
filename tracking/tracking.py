"""
tracking/tracking.py  —  Hybrid YOLO + DaSiamRPN + Kalman state machine
========================================================================
Single-target. Your design:

  YOLO detects -> user permits (SPACE) -> DaSiamRPN locks on
  DaSiamRPN tracks EVERY frame (smooth, fast, survives YOLO gaps)
  Kalman smooths DaSiam output + predicts when DaSiam fails
  Every N frames YOLO re-checks; if DaSiam drifted (low IoU) -> re-init DaSiam
  DaSiam fails -> Kalman COASTS -> times out -> LOST
  LOST: YOLO keeps detecting (boxes shown) but waits for user SPACE to re-lock

States: SEARCHING -> TRACKING -> COASTING -> LOST -> (SEARCHING)

Public API (used by main.py):
    ht = HybridTracker()
    ht.lock(frame, box)              # called on user SPACE
    out = ht.update(frame, detections)  # every frame
    ht.reset()
  out = dict:
    {'state', 'box'(x1y1x2y2 or None), 'center'(cx,cy or None),
     'name'(tracker backend), 'drifted'(bool last recheck)}
"""

import cv2

from tracking.tracker_factory import make_tracker
from tracking.kalman import KalmanTracker, iou
from config import (
    YOLO_RECHECK_EVERY,
    DRIFT_IOU_MIN,
    REACQUIRE_SECONDS,
)


def _xywh_to_xyxy(b):
    x, y, w, h = b
    return (int(x), int(y), int(x + w), int(y + h))


def _xyxy_to_xywh(b):
    x1, y1, x2, y2 = b[:4]
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))


class HybridTracker:
    # states
    SEARCHING = "SEARCHING"
    TRACKING  = "TRACKING"
    COASTING  = "COASTING"
    LOST      = "LOST"

    def __init__(self, log_fn=None):
        self.log = log_fn if log_fn else print
        self.state = self.SEARCHING
        self.tracker = None
        self.name = "-"
        self.kalman = KalmanTracker()
        self._frame_since_check = 0
        self.drifted = False

    # ── lifecycle ────────────────────────────────────────────────────────────
    def lock(self, frame, box):
        """User-permitted lock. box = (x1,y1,x2,y2,...) from a YOLO detection."""
        xyxy = box[:4]
        self.tracker, self.name = make_tracker(self.log)
        ok = self.tracker.init(frame, _xyxy_to_xywh(xyxy))
        if not ok:
            self.log("[HYB] tracker init failed")
            self.state = self.SEARCHING
            return False
        self.kalman.init(xyxy)
        self.state = self.TRACKING
        self._frame_since_check = 0
        self.drifted = False
        self.log(f"[HYB] Locked via {self.name}")
        return True

    def reset(self):
        self.state = self.SEARCHING
        self.tracker = None
        self.kalman.reset()
        self._frame_since_check = 0
        self.drifted = False

    # ── per-frame update ──────────────────────────────────────────────────────
    def update(self, frame, detections):
        if self.state in (self.SEARCHING, self.LOST):
            # YOLO still runs in main; we just stay idle awaiting user SPACE.
            # If detections exist we report LOST (re-lock available), else SEARCHING.
            self.state = self.LOST if detections else self.SEARCHING
            return self._out(None)

        # ── TRACKING / COASTING ────────────────────────────────────────────────
        ok, box_xywh = self.tracker.update(frame)
        box_xyxy = _xywh_to_xyxy(box_xywh) if ok else None

        if ok and box_xyxy is not None:
            # smooth with Kalman
            self.kalman.update(box_xyxy)
            self.state = self.TRACKING

            # periodic YOLO drift re-check
            self._frame_since_check += 1
            if self._frame_since_check >= YOLO_RECHECK_EVERY and detections:
                self._frame_since_check = 0
                best = max(detections, key=lambda d: iou(d, box_xyxy))
                if iou(best, box_xyxy) < DRIFT_IOU_MIN:
                    # DaSiam drifted -> re-init from YOLO box
                    self.drifted = True
                    self.tracker, self.name = make_tracker(self.log)
                    self.tracker.init(frame, _xyxy_to_xywh(best[:4]))
                    self.kalman.init(best[:4])
                    box_xyxy = best[:4]
                    self.log("[HYB] DaSiam drifted -> re-init from YOLO")
                else:
                    self.drifted = False

            return self._out(box_xyxy)

        # tracker failed this frame -> COAST on Kalman prediction
        pred = self.kalman.predict()
        self.kalman.mark_lost()
        if pred is not None and not self.kalman.is_lost:
            self.state = self.COASTING
            return self._out(pred)

        # Kalman timed out -> fully lost, await user permission
        self.state = self.LOST if detections else self.SEARCHING
        self.tracker = None
        self.log("[HYB] Target lost -> awaiting user re-lock (SPACE)")
        return self._out(None)

    # ── output helper ─────────────────────────────────────────────────────────
    def _out(self, box):
        center = None
        if box is not None:
            x1, y1, x2, y2 = box[:4]
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
        return {
            "state":  self.state,
            "box":    tuple(box[:4]) if box is not None else None,
            "center": center,
            "name":   self.name,
            "drifted": self.drifted,
        }