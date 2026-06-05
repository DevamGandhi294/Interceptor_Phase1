"""
detection.py  —  YOLO (ONNX) drone detector
============================================
Wraps ONNX load/export + inference into a Detector class so main.py does:

    from detection import Detector
    det = Detector()
    detections = det.run(frame)        # [(x1,y1,x2,y2,score,cls_id), ...]

Phase 1: single full-frame tile (short range). Tiling removed (was dead code).
Sky-mask helpers kept as optional pre-filter but OFF by default for office use.
"""

import os
import shutil

import cv2
import numpy as np
import onnxruntime as ort
from config import USE_NPU, YOLO_QNN_MODEL, QNN_BACKEND_LIB

# Config — move these to config.py later if you want them centralised.
CONFIDENCE      = 0.50
INPUT_SIZE      = 640
MODEL_PT        = "best1.pt"
MODEL_ONNX      = "best1.onnx"
NMS_IOU         = 0.50

CLASSES   = ["UAV_DRONE", "SHAHED_DRONE"]
CLASS_IDS = set(range(len(CLASSES)))


# ── ONNX export helpers ───────────────────────────────────────────────────────
def _export_onnx():
    """Export best1.pt -> best1.onnx at INPUT_SIZE. Requires ultralytics."""
    from ultralytics import YOLO
    print(f"[DET] Exporting ONNX at {INPUT_SIZE}px...")
    model = YOLO(MODEL_PT)
    model.export(format="onnx", simplify=True, imgsz=INPUT_SIZE, end2end=False)

    pt_dir   = os.path.dirname(os.path.abspath(MODEL_PT))
    pt_stem  = os.path.splitext(os.path.basename(MODEL_PT))[0]
    exported = os.path.join(pt_dir, pt_stem + ".onnx")

    if os.path.exists(exported) and os.path.abspath(exported) != os.path.abspath(MODEL_ONNX):
        shutil.move(exported, MODEL_ONNX)
    elif not os.path.exists(MODEL_ONNX):
        cands = [f for f in os.listdir(".") if f.endswith(".onnx")]
        if cands:
            shutil.move(cands[0], MODEL_ONNX)
        else:
            raise FileNotFoundError("Export ran but no .onnx produced.")
    print(f"[DET] Saved {MODEL_ONNX}")


def _onnx_input_size(path):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    shape = s.get_inputs()[0].shape   # [N, C, H, W]
    return shape[2], shape[3]


def _ensure_onnx():
    """Make sure a correctly-sized ONNX exists; export/re-export if needed."""
    need = not os.path.exists(MODEL_ONNX)
    if not need:
        try:
            h, w = _onnx_input_size(MODEL_ONNX)
            if h != INPUT_SIZE or w != INPUT_SIZE:
                print(f"[DET] ONNX is {h}x{w}, need {INPUT_SIZE} — re-exporting")
                os.remove(MODEL_ONNX)
                need = True
        except Exception:
            os.remove(MODEL_ONNX)
            need = True
    if need:
        _export_onnx()


# ── Optional sky mask (kept, off by default) ──────────────────────────────────
def get_sky_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    fh, fw = frame.shape[:2]
    blue = cv2.inRange(hsv, np.array([85, 30, 120], np.uint8),
                            np.array([145, 255, 255], np.uint8))
    grey = cv2.inRange(hsv, np.array([0, 0, 180], np.uint8),
                            np.array([180, 50, 255], np.uint8))
    hazy = cv2.inRange(hsv, np.array([85, 10, 160], np.uint8),
                            np.array([145, 80, 255], np.uint8))
    mask = cv2.bitwise_or(cv2.bitwise_or(blue, grey), hazy)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if np.count_nonzero(mask) / (fh * fw) < 0.05:
        mask = np.zeros((fh, fw), np.uint8)
        mask[:int(fh * 0.45), :] = 255
    return mask


# ── Detector ──────────────────────────────────────────────────────────────────
class Detector:
    def __init__(self, conf=CONFIDENCE, input_size=INPUT_SIZE):
        self.conf       = conf
        self.input_size = input_size

        _ensure_onnx()

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session    = ort.InferenceSession(
            MODEL_ONNX, sess_options=opts,
            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        print(f"[DET] Model loaded: {MODEL_ONNX}  input={self.input_size}px")
        print(f"[DET] Classes: {CLASSES}")

    # -- preprocessing --
    def _preprocess(self, img):
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, 0)

    # -- one inference pass over the full frame --
    def run(self, frame):
        """Return list of (x1,y1,x2,y2,score,cls_id) in full-frame coords."""
        fh, fw = frame.shape[:2]
        scale_x = fw / self.input_size
        scale_y = fh / self.input_size

        tensor  = self._preprocess(frame)
        outputs = self.session.run(None, {self.input_name: tensor})[0]
        outputs = np.squeeze(outputs).T          # (num_anchors, 4+num_classes)

        dets = []
        for det in outputs:
            scores = det[4:]
            cls_id = int(scores.argmax())
            score  = float(scores[cls_id])
            if cls_id not in CLASS_IDS or score < self.conf:
                continue
            cx, cy, w, h = det[:4]
            x1 = int((cx - w / 2) * scale_x)
            y1 = int((cy - h / 2) * scale_y)
            x2 = int((cx + w / 2) * scale_x)
            y2 = int((cy + h / 2) * scale_y)
            dets.append((x1, y1, x2, y2, score, cls_id))

        if not dets:
            return []

        # clamp + drop degenerate
        dets = [(max(0, x1), max(0, y1), min(fw, x2), min(fh, y2), s, c)
                for x1, y1, x2, y2, s, c in dets
                if x2 > x1 + 2 and y2 > y1 + 2]

        boxes  = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2, _, _ in dets]
        scores = [s for *_, s, _ in dets]
        idxs   = cv2.dnn.NMSBoxes(boxes, scores, self.conf, NMS_IOU)
        return [dets[i] for i in idxs] if len(idxs) > 0 else []