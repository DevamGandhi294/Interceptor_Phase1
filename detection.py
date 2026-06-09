"""
detection.py  —  YOLO drone detector (NPU/CPU) with LETTERBOX
=============================================================
Ports the proven letterbox + un-pad logic from the working human
detection code. Letterbox keeps aspect ratio (no distortion) which
raises detection confidence and gives correct box positions.

Output handled: single (1,5,8400) = [cx,cy,w,h,score]  (1 class)
Vectorized parsing (no per-anchor Python loop) for speed.
"""

import os
import shutil

import cv2
import numpy as np
import onnxruntime as ort

from config import USE_NPU, YOLO_QNN_MODEL, QNN_BACKEND_LIB

CONFIDENCE = 0.25       # lowered; tune up once detection confirmed
INPUT_SIZE = 640
MODEL_PT   = "best1.pt"
MODEL_ONNX = "best1.onnx"
NMS_IOU    = 0.45

CLASSES   = ["UAV_DRONE", "SHAHED_DRONE"]
CLASS_IDS = set(range(len(CLASSES)))


def _export_onnx():
    from ultralytics import YOLO
    print(f"[DET] Exporting ONNX at {INPUT_SIZE}px...")
    model = YOLO(MODEL_PT)
    model.export(format="onnx", simplify=True, imgsz=INPUT_SIZE, end2end=False)
    pt_dir   = os.path.dirname(os.path.abspath(MODEL_PT))
    pt_stem  = os.path.splitext(os.path.basename(MODEL_PT))[0]
    exported = os.path.join(pt_dir, pt_stem + ".onnx")
    if os.path.exists(exported) and os.path.abspath(exported) != os.path.abspath(MODEL_ONNX):
        shutil.move(exported, MODEL_ONNX)
    print(f"[DET] Saved {MODEL_ONNX}")


def _ensure_onnx():
    if not os.path.exists(MODEL_ONNX):
        _export_onnx()


class Detector:
    def __init__(self, conf=CONFIDENCE, input_size=INPUT_SIZE,
                 backend_path=None, model_path=None):
        self.conf       = conf
        self.input_size = input_size
        self.on_npu     = False
        self._backend   = backend_path or QNN_BACKEND_LIB

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = self._build_session(opts, model_path)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.in_dtype   = np.uint8 if "uint8" in inp.type else np.float32
        ishape = inp.shape
        self.is_nhwc = (len(ishape) == 4 and ishape[-1] == 3)
        print(f"[DET] input={ishape} dtype={self.in_dtype.__name__} "
              f"nhwc={self.is_nhwc} npu={self.on_npu}")
        print(f"[DET] Classes: {CLASSES}")

    def _build_session(self, opts, model_path):
        if USE_NPU:
            qnn_model = model_path or YOLO_QNN_MODEL
            if os.path.exists(qnn_model):
                try:
                    sess = ort.InferenceSession(
                        qnn_model, sess_options=opts,
                        providers=["QNNExecutionProvider", "CPUExecutionProvider"],
                        provider_options=[{"backend_path": self._backend}, {}])
                    if "QNNExecutionProvider" in sess.get_providers():
                        self.on_npu = True
                        print(f"[DET] YOLO on NPU (QNN): {qnn_model}")
                        return sess
                except Exception:
                    import traceback
                    print("[DET] QNN FAILED:")
                    traceback.print_exc()
            else:
                print(f"[DET] {qnn_model} not found — CPU fallback")

        _ensure_onnx()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        sess = ort.InferenceSession(
            MODEL_ONNX, sess_options=opts,
            providers=["CPUExecutionProvider"])
        print(f"[DET] YOLO on CPU: {MODEL_ONNX}")
        return sess

    # ── letterbox preprocess (keeps aspect ratio) ───────────────────────────────
    def _letterbox(self, frame):
        fh, fw = frame.shape[:2]
        s   = min(self.input_size / fw, self.input_size / fh)
        nw, nh = int(fw * s), int(fh * s)
        px, py = (self.input_size - nw) // 2, (self.input_size - nh) // 2
        resized = cv2.resize(frame, (nw, nh))
        canvas  = np.full((self.input_size, self.input_size, 3), 114, np.uint8)
        canvas[py:py+nh, px:px+nw] = resized
        return canvas, s, px, py

    def _to_tensor(self, canvas_bgr):
        img = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
        if self.in_dtype == np.uint8:
            t = img.astype(np.uint8)
        else:
            t = img.astype(np.float32) / 255.0
        if self.is_nhwc:
            return np.expand_dims(t, 0)
        return np.expand_dims(np.transpose(t, (2, 0, 1)), 0)

    # ── inference + vectorized parse + un-letterbox ─────────────────────────────
    def run(self, frame):
        fh, fw = frame.shape[:2]
        canvas, s, px, py = self._letterbox(frame)
        tensor = self._to_tensor(canvas)

        out = self.session.run(None, {self.input_name: tensor})[0]
        out = np.squeeze(out)
        if out.shape[0] == 5:
            out = out.T                      # (8400, 5)

        scores = out[:, 4].astype(np.float32)
        keep = scores >= self.conf
        if not np.any(keep):
            return []

        rows = out[keep]
        sc = rows[:, 4]
        cx, cy = rows[:, 0], rows[:, 1]
        w,  h  = rows[:, 2], rows[:, 3]

        # boxes are in letterboxed 640-space (cxcywh) -> xyxy in canvas
        x1 = cx - w / 2; y1 = cy - h / 2
        x2 = cx + w / 2; y2 = cy + h / 2

        # un-letterbox: remove pad, divide by scale -> original frame coords
        x1 = (x1 - px) / s; x2 = (x2 - px) / s
        y1 = (y1 - py) / s; y2 = (y2 - py) / s

        x1 = np.clip(x1, 0, fw); x2 = np.clip(x2, 0, fw)
        y1 = np.clip(y1, 0, fh); y2 = np.clip(y2, 0, fh)

        boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        idxs = cv2.dnn.NMSBoxes(boxes_xywh, sc.tolist(), self.conf, NMS_IOU)
        if len(idxs) == 0:
            return []

        dets = []
        for i in np.array(idxs).flatten():
            if (x2[i] - x1[i]) < 2 or (y2[i] - y1[i]) < 2:
                continue
            dets.append((int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i]),
                         float(sc[i]), 0))
        return dets
