"""
config.py  —  Shared constants for the Aim Tracker system
All magic numbers live here. No logic, no imports.
"""

# ── Camera ────────────────────────────────────────────────────────────────────
WIDTH        = 640
HEIGHT       = 480
TARGET_FPS   = 30
CAMERA_INDEX = 1

# ── Tracking ROI ──────────────────────────────────────────────────────────────
DEAD_ZONE_W  = 60
DEAD_ZONE_H  = 60
TRACK_HALF_W = 80    # half-width  of ROI box on click
TRACK_HALF_H = 80    # half-height of ROI box on click

# ── DaSiamRPN model paths ─────────────────────────────────────────────────────
DASIAMRPN_MODEL       = "models/dasiamrpn_model.onnx"
DASIAMRPN_KERNEL_CLS1 = "models/dasiamrpn_kernel_cls1.onnx"
DASIAMRPN_KERNEL_R1   = "models/dasiamrpn_kernel_r1.onnx"

# ── Serial / FC ───────────────────────────────────────────────────────────────
PORT       = '/dev/ttyTHS1'
BAUD       = 115200
MOTOR_STOP = 1000
MOTOR_RUN  = 1200

# Motor index mapping (SpeedyBee F405 V3)
M1 = 0   # Front-Left
M2 = 1   # Front-Right
M3 = 2   # Rear-Left
M4 = 3   # Rear-Right

# ── RC Override ───────────────────────────────────────────────────────────────
RC_CENTER           = 1500
RC_OFFSET           = 60
RC_MIN_OFFSET       = 15
RC_MAX_OFFSET       = 100
RC_MAX_PX_X         = WIDTH  // 2
RC_MAX_PX_Y         = HEIGHT // 2
RC_FOLLOW_CHANNEL   = 8
RC_FOLLOW_THRESHOLD = 1700
RC_AUTO_THROTTLE    = 1500

# ── HUD colours (BGR) ─────────────────────────────────────────────────────────
AIM_COLOR       = (0,   255,   0)
LOCK_COLOR      = (0,    60, 255)
CMD_COLOR       = (0,   220, 255)
FPS_COLOR       = (180, 255,   0)
DEAD_ZONE_COLOR = (60,   60,  60)
WARN_COLOR      = (0,    80, 255)
FC_COLOR        = (200, 200,  50)
ARMED_COLOR     = (0,   255,   0)
DISARMED_COLOR  = (0,    60, 255)

# ── SmallObjectDetector thresholds ────────────────────────────────────────────
SOD_MIN_AREA   =    4
SOD_MAX_AREA   = 5000
SOD_MIN_DIM    =    2
SOD_MAX_DIM    =  120
SOD_MAX_ASPECT =    8.0
SOD_H_TOL      =   18    # hue ±18 (out of 180)
SOD_S_TOL      =   60    # saturation ±60
SOD_V_TOL      =   60    # value ±60
SOD_BORDER_PX  =   24    # background sampling strip width
SOD_BG_RESAMPLE=   60    # resample background every N frames

# ── CLAHE ─────────────────────────────────────────────────────────────────────
CLAHE_CLIP_LIMIT   = 3.0
CLAHE_TILE_GRID    = (8, 8)

# ── Kalman filter noise ───────────────────────────────────────────────────────
KALMAN_PROCESS_NOISE  = 0.03
KALMAN_MEASURE_NOISE  = 0.5
KALMAN_LOST_TIMEOUT_S = 3.0
KALMAN_RELOCK_EVERY   = 8

# ── DriftGuard thresholds ─────────────────────────────────────────────────────
DRIFT_BOUNDARY_PX        = 40
DRIFT_BOUNDARY_TIMEOUT_S = 5.0
DRIFT_SIZE_SPIKE_RATIO   = 2.5
DRIFT_SIZE_HISTORY       = 20
DRIFT_ASPECT_MAX         = 4.0

# ── LockConfirmation ──────────────────────────────────────────────────────────
CONFIRM_FRAMES      = 8
CONFIRM_MAX_JUMP    = 30     # pixels
CONFIRM_SIZE_RATIO  = 2.5
CONFIRM_SCORE_MIN   = 0.12

# ── AutoRetarget ──────────────────────────────────────────────────────────────
RETARGET_TIMEOUT_S  = 12.0

# ── AimQualityScorer ──────────────────────────────────────────────────────────
AIM_QUALITY_HISTORY = 8

# ══════════════════════════════════════════════════════════════════════════════
#  DRONE CONNECTIVITY  (Raspberry Pi 4B / INAV FC + ELRS)
# ══════════════════════════════════════════════════════════════════════════════

# ── CRSF receiver (ELRS) ──────────────────────────────────────────────────────
CRSF_PORT = '/dev/ttyAMA0'
CRSF_BAUD = 420000

# ── Auto-follow trigger ───────────────────────────────────────────────────────
AUTO_TRIGGER_CH  = 8       # CH8 = SD switch on Boxer
AUTO_TRIGGER_THR = 1700    # CH8 value above this = AUTO ON

# ── Auto-mode RC outputs ──────────────────────────────────────────────────────
AUTO_PITCH_PWM  = 1600    # fixed pitch lean when auto is active
YAW_MAX_OFFSET  = 350     # max ± PWM deviation from 1500 for yaw
YAW_TIMEOUT_S   = 0.5     # drop back to pilot if target lost > this long
THR_MAX_OFFSET  = 290     # max throttle deviation from base
THR_MIN_SAFE    = 900     # hard floor for throttle in auto mode
THR_MAX_SAFE    = 2000    # hard ceiling for throttle in auto mode

# ── Target area Z-axis control ────────────────────────────────────────────────
TARGET_BOX_AREA  = 6000
KP_Z             = 0.08
Z_DEADZONE       = 500
Z_MAX_CORRECTION = 300

# ── BlackBox recording ────────────────────────────────────────────────────────
BLACKBOX_DIR     = "~/Downloads/blackbox"
BLACKBOX_FPS     = 30
BLACKBOX_MAX_MIN = 10

# ── FlightLogger ──────────────────────────────────────────────────────────────
FLIGHT_LOG_DIR      = "logs"
FLIGHT_LOG_INTERVAL = 1    # log every N frames

ENABLE_FC_OUTPUT = False
 
# ── Platform-aware ports (override the hardcoded ones above) ──────────────────
import platform as _platform
_IS_WIN = _platform.system() == "Windows"
 
# FC serial (MSP).  Windows COMx  vs  Linux /dev/ttyACMx
PORT = "COM3"        if _IS_WIN else "/dev/ttyACM0"
BAUD = 115200
 
# ELRS CRSF receiver.  Windows COMx  vs  Linux /dev/ttyAMA0
CRSF_PORT = "COM4"   if _IS_WIN else "/dev/ttyAMA0"
CRSF_BAUD = 420000
 
# Camera index — Windows laptop webcam is usually 0
CAMERA_INDEX = 0     if _IS_WIN else 1

REACQUIRE_SECONDS = 5.0

# Lower = catches drift faster but more jumpy. Higher = smoother but drifts more.
YOLO_RECHECK_EVERY = 15
 
# If YOLO box vs DaSiam box IoU falls below this at re-check -> re-init DaSiam.
DRIFT_IOU_MIN = 0.30