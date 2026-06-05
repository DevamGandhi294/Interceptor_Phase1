"""
config.py  —  Shared constants for the Aim Tracker system
=========================================================
All magic numbers live here. Platform detection at top so every
path/port resolves correctly on Windows (laptop) and Linux (Radxa).
No duplicate definitions — each constant set ONCE.
"""

import platform as _platform

# ── Platform detection (must be first) ────────────────────────────────────────
_IS_WIN = _platform.system() == "Windows"
_IS_LIN = _platform.system() == "Linux"

# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA
# ══════════════════════════════════════════════════════════════════════════════
WIDTH        = 640
HEIGHT       = 480
TARGET_FPS   = 30
CAMERA_INDEX = 0              # USB webcam usually 0 on both Windows and Radxa
CAMERA_IS_MIPI = False        # True on Radxa if using MIPI CSI cam (GStreamer)

# ══════════════════════════════════════════════════════════════════════════════
#  TRACKING
# ══════════════════════════════════════════════════════════════════════════════
DEAD_ZONE_W  = 60
DEAD_ZONE_H  = 60
TRACK_HALF_W = 80
TRACK_HALF_H = 80

REACQUIRE_SECONDS  = 1.0      # Kalman coast / re-acquire window (was 5.0)
YOLO_RECHECK_EVERY = 15       # frames between YOLO drift re-checks
DRIFT_IOU_MIN      = 0.30     # below this IoU at re-check -> re-init DaSiam
FAST_RESEED_PX     = 60       # DaSiam vs Kalman gap that triggers re-seed

# ── DaSiamRPN model paths (CPU ONNX) ──────────────────────────────────────────
DASIAMRPN_MODEL       = "models/dasiamrpn_model.onnx"
DASIAMRPN_KERNEL_CLS1 = "models/dasiamrpn_kernel_cls1.onnx"
DASIAMRPN_KERNEL_R1   = "models/dasiamrpn_kernel_r1.onnx"

# ══════════════════════════════════════════════════════════════════════════════
#  NPU / RADXA
# ══════════════════════════════════════════════════════════════════════════════
# Use NPU (QNN) on Radxa/Linux; CPU on Windows laptop.
USE_NPU         = _IS_LIN
# QNN-converted YOLO model (created on Radxa via AI Hub / QAIRT).
# Falls back to the CPU .onnx automatically if this doesn't exist.
YOLO_QNN_MODEL  = "models/best1_qnn.onnx"
QNN_BACKEND_LIB = "libQnnHtp.so"     # Hexagon Tensor Processor backend

# ══════════════════════════════════════════════════════════════════════════════
#  SERIAL / FLIGHT CONTROLLER (MSP)
# ══════════════════════════════════════════════════════════════════════════════
PORT = "COM3" if _IS_WIN else "/dev/ttyACM0"   # FC over MSP
BAUD = 115200

MOTOR_STOP = 1000
MOTOR_RUN  = 1200

# Motor index mapping (SpeedyBee F405 V3)
M1 = 0   # Front-Left
M2 = 1   # Front-Right
M3 = 2   # Rear-Left
M4 = 3   # Rear-Right

# ══════════════════════════════════════════════════════════════════════════════
#  CRSF RECEIVER (ELRS)
# ══════════════════════════════════════════════════════════════════════════════
# NOTE: /dev/ttyAMA0 is Pi-style. On Radxa Dragon Q6A confirm with
#       `ls /dev/tty*` — it may be /dev/ttyHS0 or similar. Update here.
CRSF_PORT = "COM4" if _IS_WIN else "/dev/ttyAMA0"
CRSF_BAUD = 420000

# ══════════════════════════════════════════════════════════════════════════════
#  RC OVERRIDE
# ══════════════════════════════════════════════════════════════════════════════
RC_CENTER           = 1500
RC_OFFSET           = 60
RC_MIN_OFFSET       = 15
RC_MAX_OFFSET       = 100
RC_MAX_PX_X         = WIDTH  // 2
RC_MAX_PX_Y         = HEIGHT // 2
RC_FOLLOW_CHANNEL   = 8
RC_FOLLOW_THRESHOLD = 1700
RC_AUTO_THROTTLE    = 1500

# Phase 1 safety: FALSE = open-loop (compute + log RC, send NOTHING to FC).
# TRUE = closed-loop (RC override IS sent). Keep FALSE on the tripod.
ENABLE_FC_OUTPUT = False

# ── Auto-follow trigger ───────────────────────────────────────────────────────
AUTO_TRIGGER_CH  = 8          # CH8 = SD switch on Boxer
AUTO_TRIGGER_THR = 1700       # CH8 above this = AUTO ON

# ── Auto-mode RC outputs ──────────────────────────────────────────────────────
AUTO_PITCH_PWM  = 1600
YAW_MAX_OFFSET  = 350
YAW_TIMEOUT_S   = 0.5
THR_MAX_OFFSET  = 290
THR_MIN_SAFE    = 900
THR_MAX_SAFE    = 2000

# ── Target area Z-axis control ────────────────────────────────────────────────
TARGET_BOX_AREA  = 6000
KP_Z             = 0.08
Z_DEADZONE       = 500
Z_MAX_CORRECTION = 300

# ══════════════════════════════════════════════════════════════════════════════
#  HUD COLOURS (BGR)
# ══════════════════════════════════════════════════════════════════════════════
AIM_COLOR       = (0,   255,   0)
LOCK_COLOR      = (0,    60, 255)
CMD_COLOR       = (0,   220, 255)
FPS_COLOR       = (180, 255,   0)
DEAD_ZONE_COLOR = (60,   60,  60)
WARN_COLOR      = (0,    80, 255)
FC_COLOR        = (200, 200,  50)
ARMED_COLOR     = (0,   255,   0)
DISARMED_COLOR  = (0,    60, 255)

# ══════════════════════════════════════════════════════════════════════════════
#  SMALL OBJECT DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
SOD_MIN_AREA    =    4
SOD_MAX_AREA    = 5000
SOD_MIN_DIM     =    2
SOD_MAX_DIM     =  120
SOD_MAX_ASPECT  =    8.0
SOD_H_TOL       =   18
SOD_S_TOL       =   60
SOD_V_TOL       =   60
SOD_BORDER_PX   =   24
SOD_BG_RESAMPLE =   60

# ── CLAHE ─────────────────────────────────────────────────────────────────────
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID  = (8, 8)

# ══════════════════════════════════════════════════════════════════════════════
#  KALMAN FILTER
# ══════════════════════════════════════════════════════════════════════════════
KALMAN_PROCESS_NOISE  = 0.03
KALMAN_MEASURE_NOISE  = 0.5
KALMAN_LOST_TIMEOUT_S = 1.0   # was 3.0 — faster give-up
KALMAN_RELOCK_EVERY   = 8

# ── DriftGuard ────────────────────────────────────────────────────────────────
DRIFT_BOUNDARY_PX        = 40
DRIFT_BOUNDARY_TIMEOUT_S = 5.0
DRIFT_SIZE_SPIKE_RATIO   = 2.5
DRIFT_SIZE_HISTORY       = 20
DRIFT_ASPECT_MAX         = 4.0

# ── LockConfirmation ──────────────────────────────────────────────────────────
CONFIRM_FRAMES     = 8
CONFIRM_MAX_JUMP   = 30
CONFIRM_SIZE_RATIO = 2.5
CONFIRM_SCORE_MIN  = 0.12

# ── AutoRetarget ──────────────────────────────────────────────────────────────
RETARGET_TIMEOUT_S = 12.0

# ── AimQualityScorer ──────────────────────────────────────────────────────────
AIM_QUALITY_HISTORY = 8

# ══════════════════════════════════════════════════════════════════════════════
#  RECORDING / LOGGING
# ══════════════════════════════════════════════════════════════════════════════
BLACKBOX_DIR     = "~/Downloads/blackbox" if _IS_WIN else "blackbox"
BLACKBOX_FPS     = 30
BLACKBOX_MAX_MIN = 10

FLIGHT_LOG_DIR      = "logs"
FLIGHT_LOG_INTERVAL = 1

HEADLESS = _IS_LIN