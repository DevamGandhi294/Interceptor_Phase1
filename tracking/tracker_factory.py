"""
tracking/tracker_factory.py  —  Tracker selection + guaranteed fallback
=======================================================================
  - TemplateTracker : matchTemplate fallback (works on ANY OpenCV)
  - make_tracker()  : returns best available (DaSiamRPN > MIL > MOSSE > Template)

DaSiamRPN needs opencv-contrib-python + 3 ONNX files (paths in config).
"""

import os
import cv2
import numpy as np

from config import (
    DASIAMRPN_MODEL,
    DASIAMRPN_KERNEL_CLS1,
    DASIAMRPN_KERNEL_R1,
)


def _log(msg):
    print(msg)


class TemplateTracker:
    """matchTemplate inside a 3x search window. No contrib needed."""
    SCALE = 3.0
    CONF  = 0.30

    def __init__(self):
        self.template = None
        self.bbox     = None

    def init(self, frame, bbox):
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]
        x = max(0, min(x, fw - w))
        y = max(0, min(y, fh - h))
        self.template = frame[y:y+h, x:x+w].copy()
        self.bbox = (x, y, w, h)
        return True

    def update(self, frame):
        if self.template is None:
            return False, self.bbox
        x, y, w, h = self.bbox
        fh, fw = frame.shape[:2]
        th, tw = self.template.shape[:2]
        sw, sh = int(tw * self.SCALE), int(th * self.SCALE)
        sx = max(0, x - (sw - tw) // 2)
        sy = max(0, y - (sh - th) // 2)
        ex = min(fw, sx + sw)
        ey = min(fh, sy + sh)
        search = frame[sy:ey, sx:ex]
        if search.shape[0] < th or search.shape[1] < tw:
            return True, self.bbox
        res = cv2.matchTemplate(search, self.template, cv2.TM_CCOEFF_NORMED)
        _, score, _, best = cv2.minMaxLoc(res)
        nx = max(0, min(sx + best[0], fw - tw))
        ny = max(0, min(sy + best[1], fh - th))
        self.bbox = (nx, ny, tw, th)
        patch = frame[ny:ny+th, nx:nx+tw]
        if patch.shape[:2] == self.template.shape[:2]:
            alpha = 0.92 if score >= self.CONF else 0.98
            self.template = cv2.addWeighted(self.template, alpha, patch, 1 - alpha, 0)
        return True, self.bbox


def make_tracker(log_fn=None):
    """Returns (tracker_instance, name_str). DaSiamRPN preferred."""
    log = log_fn if log_fn else _log

    # 1. DaSiamRPN (needs contrib + 3 onnx files)
    if (os.path.isfile(DASIAMRPN_MODEL)       and os.path.getsize(DASIAMRPN_MODEL)       > 1000 and
        os.path.isfile(DASIAMRPN_KERNEL_CLS1) and os.path.getsize(DASIAMRPN_KERNEL_CLS1) > 1000 and
        os.path.isfile(DASIAMRPN_KERNEL_R1)   and os.path.getsize(DASIAMRPN_KERNEL_R1)   > 1000):
        try:
            p = cv2.TrackerDaSiamRPN_Params()
            p.model       = DASIAMRPN_MODEL
            p.kernel_cls1 = DASIAMRPN_KERNEL_CLS1
            p.kernel_r1   = DASIAMRPN_KERNEL_R1
            log("[TRACKER] DaSiamRPN ready")
            return cv2.TrackerDaSiamRPN_create(p), "DaSiamRPN"
        except Exception as e:
            log(f"[TRACKER] DaSiamRPN failed: {e}")

    # 2. MIL
    try:
        log("[TRACKER] Falling back to MIL")
        return cv2.TrackerMIL_create(), "MIL"
    except Exception:
        pass

    # 3. Legacy MOSSE
    try:
        log("[TRACKER] Falling back to MOSSE")
        return cv2.legacy.TrackerMOSSE_create(), "MOSSE"
    except Exception:
        pass

    # 4. Template (always works)
    log("[TRACKER] Using TemplateTracker (pip install opencv-contrib-python for DaSiamRPN)")
    return TemplateTracker(), "Template"