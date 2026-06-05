"""
io/camera.py  —  Camera / video source management
===================================================
Contains:
  - _browse_video_file() : tkinter file dialog
  - open_source()        : opens camera index or video file
  - CameraThread         : background capture thread
"""

import os
import platform
import threading
import time
from config import WIDTH, HEIGHT, TARGET_FPS, CAMERA_INDEX, CAMERA_IS_MIPI

import cv2

from config import WIDTH, HEIGHT, TARGET_FPS, CAMERA_INDEX


def _browse_video_file():
    """Open a tkinter file dialog and return the selected path (or None)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.ts *.m4v"),
                ("All files",   "*.*"),
            ],
        )
        root.destroy()
        return path if path else None
    except Exception as e:
        print(f"[FILE] tkinter dialog unavailable ({e})")
        return None


def open_source(source=None, log_fn=None):
    """
    Open a capture source.

    source:
      None      → default camera (CAMERA_INDEX from config)
      int       → specific camera index
      str       → video file path
      "browse"  → open file dialog

    Returns (cv2.VideoCapture, label_str, is_file_bool)
    """
    log = log_fn if log_fn is not None else print
    IS_WIN = platform.system() == "Windows"
    IS_LIN = platform.system() == "Linux"

    def _cfg_cam(cap, label, idx):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        for _ in range(5): cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log(f"[CAM] {label}: {w}x{h} @ {TARGET_FPS}fps  index={idx}")
        return cap

    if source == "browse":
        source = _browse_video_file()
        if not source:
            raise RuntimeError("No file selected in browse dialog")

    if isinstance(source, str):
        if not os.path.isfile(source):
            raise RuntimeError(f"Video file not found: {source}")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {source}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_v = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        label = os.path.basename(source)
        log(f"[FILE] {label}: {w}x{h} @ {fps_v:.1f}fps  frames={total}")
        return cap, label, True

    cam_idx = CAMERA_INDEX if source is None else int(source)

    # ── Windows: DirectShow ───────────────────────────────────────────────────
    if IS_WIN:
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            return _cfg_cam(cap, "DirectShow", cam_idx), f"Camera[{cam_idx}]", False
        log("[CAM] DirectShow failed — trying default")

    # ── Radxa MIPI CSI camera: GStreamer (only if CAMERA_IS_MIPI) ──────────────
    if IS_LIN and CAMERA_IS_MIPI:
        pipeline = (
            f"qtiqmmfsrc ! video/x-raw,width={WIDTH},height={HEIGHT},"
            f"framerate={TARGET_FPS}/1 ! videoconvert ! "
            f"appsink drop=true max-buffers=1"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            for _ in range(5): cap.read()
            log(f"[CAM] MIPI-GStreamer: {WIDTH}x{HEIGHT} @ {TARGET_FPS}fps")
            return cap, "Camera[MIPI]", False
        log("[CAM] MIPI GStreamer failed — trying V4L2")

    # ── Linux USB: V4L2 ────────────────────────────────────────────────────────
    if IS_LIN:
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            return _cfg_cam(cap, "V4L2", cam_idx), f"Camera[{cam_idx}]", False
        log("[CAM] V4L2 failed — trying default")

    # ── Generic default ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(cam_idx)
    if cap.isOpened():
        return _cfg_cam(cap, "Default", cam_idx), f"Camera[{cam_idx}]", False

    # ── Scan other indices ─────────────────────────────────────────────────────
    for idx in [0, 1, 2]:
        if idx == cam_idx:
            continue
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            log(f"[CAM] Opened index {idx} instead of {cam_idx}")
            return _cfg_cam(cap, f"Default[{idx}]", idx), f"Camera[{idx}]", False

    raise RuntimeError(f"Cannot open camera index {cam_idx}")


class CameraThread:
    """
    Runs cv2.VideoCapture in a background thread.
    read() always returns the most recent frame with no blocking.
    """

    def __init__(self, cap: cv2.VideoCapture, is_file: bool = False):
        self.cap      = cap
        self.is_file  = is_file
        self.frame    = None
        self.eof      = False
        self.lock     = threading.Lock()
        self.running  = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        fps   = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        delay = 1.0 / max(fps, 1.0) if self.is_file else 0.0

        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                if self.is_file:
                    with self.lock:
                        self.eof = True
                    while self.running and self.eof:
                        time.sleep(0.05)
                    continue
                time.sleep(0.002)
                continue

            f = frame[:, :, :3] if (frame.ndim == 3 and frame.shape[2] == 4) else frame
            with self.lock:
                self.frame = f
                self.eof   = False

            if delay:
                time.sleep(delay)

    def read(self):
        """Returns (ok: bool, frame: ndarray | None)."""
        with self.lock:
            return (self.frame is not None), (self.frame.copy() if self.frame is not None else None)

    def is_eof(self) -> bool:
        with self.lock:
            return self.eof

    def reset_video(self):
        """Seek to beginning (video files only)."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        with self.lock:
            self.eof   = False
            self.frame = None

    def release(self):
        self.running = False
        with self.lock:
            self.eof = False
        self._thread.join(timeout=2.0)
        try:
            self.cap.release()
        except Exception:
            pass