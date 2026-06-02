"""
tracking/kalman.py  —  8-state Kalman tracker + matching helpers
================================================================
8-state filter: [cx, cy, w, h, vcx, vcy, vw, vh]
Measurement:    [cx, cy, w, h]   (full bounding box)

Tracks box SIZE too (not just position), which helps Z/area control later.

Public API (used by main.py):
    t = KalmanTracker()
    t.init(box)                       # box = (x1,y1,x2,y2,...) — lock on
    pred = t.predict()                # (x1,y1,x2,y2) or None
    t.update(box)                     # feed a matched detection
    t.mark_lost(); t.is_lost
    t.enter_reacquire(); t.reacquire_expired; t.reacquire_remaining
    t.try_reacquire(detections)       # find best size/IoU match after loss

Matching helpers (used by main.py): iou, center_dist, best_match, box_center
"""

import time

import cv2
import numpy as np

from config import REACQUIRE_SECONDS


# ── Geometry / matching helpers ───────────────────────────────────────────────
def box_center(box):
    return (box[0] + box[2]) // 2, (box[1] + box[3]) // 2


def center_dist(b1, b2):
    cx1, cy1 = box_center(b1)
    cx2, cy2 = box_center(b2)
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5


def iou(b1, b2):
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def best_match(detections, predicted_box, max_dist=120):
    """Pick detection closest to predicted box (IoU + distance score)."""
    if predicted_box is None:
        return None
    best, best_score = None, -1
    for det in detections:
        d = center_dist(det, predicted_box)
        if d > max_dist:
            continue
        score = iou(det, predicted_box) + (1 - d / max_dist) * 0.3
        if score > best_score:
            best_score = score
            best = det
    return best


# ── 8-state Kalman tracker ────────────────────────────────────────────────────
class KalmanTracker:
    MAX_LOST     = 45
    REMATCH_DIST = 120

    def __init__(self):
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        self.kf.processNoiseCov   = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.processNoiseCov[4:, 4:] *= 10
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 5.0
        self.kf.errorCovPost        = np.eye(8, dtype=np.float32) * 10.0

        self.initialized        = False
        self.lost_frames        = 0
        self.reacquiring        = False
        self.reacquire_deadline = None
        self.last_known_box     = None
        self.last_known_size    = None

    # -- lifecycle --
    def init(self, box):
        cx, cy, w, h = self._to_cwh(box)
        state = np.array([cx, cy, w, h, 0, 0, 0, 0],
                         dtype=np.float32).reshape(8, 1)
        self.kf.statePre  = state.copy()
        self.kf.statePost = state.copy()
        self.initialized        = True
        self.lost_frames        = 0
        self.reacquiring        = False
        self.reacquire_deadline = None
        self.last_known_box     = box[:4]
        self.last_known_size    = (w, h)

    def predict(self):
        if not self.initialized:
            return None
        return self._to_xyxy(self.kf.predict())

    def update(self, box):
        cx, cy, w, h = self._to_cwh(box)
        self.kf.correct(np.array([cx, cy, w, h],
                        dtype=np.float32).reshape(4, 1))
        self.lost_frames        = 0
        self.reacquiring        = False
        self.reacquire_deadline = None
        self.last_known_box     = box[:4]
        self.last_known_size    = (w, h)

    def mark_lost(self):
        self.lost_frames += 1

    @property
    def is_lost(self):
        return self.lost_frames > self.MAX_LOST

    def reset(self):
        self.initialized        = False
        self.lost_frames        = 0
        self.reacquiring        = False
        self.reacquire_deadline = None

    # -- re-acquisition --
    def enter_reacquire(self):
        self.reacquiring        = True
        self.reacquire_deadline = time.time() + REACQUIRE_SECONDS
        print(f"[KF] Target lost — re-acquiring for {REACQUIRE_SECONDS}s")

    @property
    def reacquire_expired(self):
        return (self.reacquiring and self.reacquire_deadline is not None
                and time.time() > self.reacquire_deadline)

    @property
    def reacquire_remaining(self):
        if not self.reacquiring or self.reacquire_deadline is None:
            return 0.0
        return max(0.0, self.reacquire_deadline - time.time())

    def try_reacquire(self, detections):
        """Best detection by size similarity + IoU to last-known box."""
        if not self.last_known_size or not detections:
            return None
        lw, lh = self.last_known_size
        best, best_score = None, -1
        for det in detections:
            x1, y1, x2, y2, conf, _ = det
            dw, dh = x2 - x1, y2 - y1
            w_ratio = min(lw, dw) / max(lw, dw) if max(lw, dw) > 0 else 0
            h_ratio = min(lh, dh) / max(lh, dh) if max(lh, dh) > 0 else 0
            size_score = (w_ratio + h_ratio) / 2
            iou_score  = iou(det, self.last_known_box) if self.last_known_box else 0
            score      = size_score * 0.6 + iou_score * 0.3 + conf * 0.1
            if score > best_score and size_score > 0.5:
                best_score = score
                best = det
        return best

    # -- conversions --
    def _to_cwh(self, box):
        b = np.array(box).flatten()
        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2,
                float(b[2] - b[0]), float(b[3] - b[1]))

    def _to_xyxy(self, state):
        s = state.flatten()
        cx, cy, w, h = float(s[0]), float(s[1]), float(s[2]), float(s[3])
        return (int(cx - w / 2), int(cy - h / 2),
                int(cx + w / 2), int(cy + h / 2))