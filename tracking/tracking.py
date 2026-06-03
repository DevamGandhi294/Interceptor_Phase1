"""
tracking/tracking.py  —  Pure tracker (DaSiamRPN), NO YOLO re-verify
=====================================================================
Design (your latest):
  YOLO finds object ONCE -> user SPACE -> DaSiamRPN locks on
  DaSiamRPN tracks EVERY frame, 100% on its own. No YOLO re-check.
  Kalman ONLY fills in when DaSiam itself reports failure (coasting).
  DaSiam + Kalman both fail -> LOST -> YOLO detects again, wait for SPACE.

This is the smoothest single-target setup: nothing interrupts the tracker.

States: SEARCHING -> TRACKING -> COASTING -> LOST -> (SEARCHING)

API (main.py):
    ht = HybridTracker()
    ht.lock(frame, box)                 # user SPACE on a YOLO detection
    out = ht.update(frame, detections)  # every frame
    ht.reset()
  out = {'state','box','center','name'}
"""

import cv2

from tracking.tracker_factory import make_tracker
from tracking.kalman import KalmanTracker


def _xywh_to_xyxy(b):
    x, y, w, h = b
    return (int(x), int(y), int(x + w), int(y + h))


def _xyxy_to_xywh(b):
    x1, y1, x2, y2 = b[:4]
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))


class HybridTracker:
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

    # ── lock / reset ───────────────────────────────────────────────────────────
    def lock(self, frame, box):
        """User-permitted lock. box=(x1,y1,x2,y2,...) from a YOLO detection."""
        xyxy = box[:4]
        self.tracker, self.name = make_tracker(self.log)
        if not self.tracker.init(frame, _xyxy_to_xywh(xyxy)):
            self.log("[TRK] init failed")
            self.state = self.SEARCHING
            return False
        self.kalman.init(xyxy)
        self.state = self.TRACKING
        self.log(f"[TRK] Locked via {self.name}")
        return True

    def reset(self):
        self.state = self.SEARCHING
        self.tracker = None
        self.kalman.reset()

    # ── per-frame update ──────────────────────────────────────────────────────
    def update(self, frame, detections):
        # Idle states: tracker not running. Report LOST if YOLO sees something
        # (so user can re-lock), else SEARCHING.
        if self.state in (self.SEARCHING, self.LOST):
            self.state = self.LOST if detections else self.SEARCHING
            return self._out(None)

        # TRACKING / COASTING: run the tracker, NOTHING interrupts it.
        ok, box_xywh = self.tracker.update(frame)

        if ok:
            box_xyxy = _xywh_to_xyxy(box_xywh)
            
            # Kalman predicts where drone SHOULD be (velocity feed-forward)
            self.kalman.update(box_xyxy)
            pred = self.kalman.predict()
            
            # If DaSiam result drifted far from Kalman's velocity prediction,
            # the search window probably lost the fast target — re-seed DaSiam
            # at the predicted position.
            if pred is not None:
                px1, py1, px2, py2 = pred
                pcx, pcy = (px1+px2)//2, (py1+py2)//2
                bcx, bcy = (box_xyxy[0]+box_xyxy[2])//2, (box_xyxy[1]+box_xyxy[3])//2
                jump = ((pcx-bcx)**2 + (pcy-bcy)**2) ** 0.5
                if jump > FAST_RESEED_PX:
                    # re-init tracker centered on prediction
                    w = box_xyxy[2]-box_xyxy[0]
                    h = box_xyxy[3]-box_xyxy[1]
                    new = (pcx-w//2, pcy-h//2, pcx+w//2, pcy+h//2)
                    self.tracker, self.name = make_tracker(self.log)
                    self.tracker.init(frame, _xyxy_to_xywh(new))
                    box_xyxy = new
            
            self.state = self.TRACKING
            return self._out(box_xyxy)

        # Tracker failed this frame -> coast on Kalman prediction.
        pred = self.kalman.predict()
        self.kalman.mark_lost()
        if pred is not None and not self.kalman.is_lost:
            self.state = self.COASTING
            return self._out(pred)

        # Both failed -> fully lost, await user re-lock.
        self.tracker = None
        self.state = self.LOST if detections else self.SEARCHING
        self.log("[TRK] Lost -> awaiting re-lock (SPACE)")
        return self._out(None)

    # ── output ─────────────────────────────────────────────────────────────────
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
        }