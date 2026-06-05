"""
main.py  —  Drone Aim Tracker  (Phase 1 integration)
====================================================
Pipeline:  camera -> detection -> kalman -> command_sender(open-loop)
                  -> gui -> blackbox -> flight_logger

Phase 1 safety: command_sender only SENDS RC to the FC if
config.ENABLE_FC_OUTPUT is True. Keep it False on the tripod.

HEADLESS (config.HEADLESS): when True (drone in flight, no monitor),
all GUI windows are skipped but detection/tracking/recording/FC keep
running. Lock is triggered by CH8 on the radio instead of SPACE.

Controls (windowed mode):
    SPACE = lock nearest    R = reset    V = video file    C = webcam    Q = quit
Headless: CH8 high = lock,  Ctrl-C = quit
"""

import sys
import time
import traceback

import cv2
import numpy as np

from config import WIDTH, HEIGHT, HEADLESS
from camera import open_source, CameraThread
from detection import Detector, CLASSES
from tracking.kalman import KalmanTracker, best_match
from gui import draw_hud, draw_zoom_inset
from drone.blackbox import BlackBox
from drone.flight_logger import FlightLogger

# command_sender pulls in CRSF + FC readers; wrap in try so vision-only
# testing still works if no FC/ELRS hardware is attached.
try:
    from drone.command_sender import CommandSender
    HAVE_SENDER = True
except Exception as e:
    print(f"[MAIN] command_sender unavailable ({e}) — vision-only mode")
    HAVE_SENDER = False


# ── FPS counter ───────────────────────────────────────────────────────────────
class FPSCounter:
    def __init__(self, avg_over=30):
        self.ts = []
        self.avg_over = avg_over

    def tick(self):
        self.ts.append(time.perf_counter())
        if len(self.ts) > self.avg_over:
            self.ts.pop(0)

    def fps(self):
        if len(self.ts) < 2:
            return 0.0
        el = self.ts[-1] - self.ts[0]
        return (len(self.ts) - 1) / el if el > 0 else 0.0


def reset_tracking():
    """fresh (lock_active, delta, predicted_box, matched_box)."""
    return False, (0, 0), None, None


def main():
    WINDOW = "Drone Tracker"

    if not HEADLESS:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, WIDTH, HEIGHT)
        splash = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        cv2.putText(splash, "Loading...", (WIDTH // 2 - 70, HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        cv2.imshow(WINDOW, splash)
        cv2.waitKey(1)
    else:
        print("[MAIN] HEADLESS mode — no GUI window. CH8=lock, Ctrl-C=quit")

    # ── modules ────────────────────────────────────────────────────────────────
    detector = Detector()
    tracker  = KalmanTracker()
    fps_c    = FPSCounter()
    blackbox = BlackBox()
    flog     = FlightLogger()

    sender = None
    if HAVE_SENDER:
        try:
            sender = CommandSender(verbose=False)
            print("[MAIN] CommandSender active (FC output gated by config)")
        except Exception as e:
            print(f"[MAIN] CommandSender init failed ({e}) — vision-only")

    # ── camera ───────────────────────────────────────────────────────────────────
    cap, source_label, is_file = open_source()
    cam = CameraThread(cap, is_file=is_file)
    time.sleep(0.3)   # let first frame arrive

    # ── state ────────────────────────────────────────────────────────────────────
    lock_active = False
    delta = (0, 0)
    predicted_box = None
    matched_box = None

    if not HEADLESS:
        print("\nControls: SPACE=lock  R=reset  V=video  C=cam  Q=quit\n")

    while True:
        ok, frame = cam.read()
        if not ok or frame is None:
            if is_file and cam.is_eof():
                cam.reset_video()
            time.sleep(0.005)
            continue

        fh, fw = frame.shape[:2]
        aim_x, aim_y = fw // 2, fh // 2

        # ── detection ──────────────────────────────────────────────────────────
        detections = detector.run(frame)

        # ── tracking ───────────────────────────────────────────────────────────
        box_area = 0
        if lock_active:
            if tracker.reacquiring:
                if tracker.reacquire_expired:
                    lock_active, delta, predicted_box, matched_box = reset_tracking()
                    tracker.reset()
                    print("[MAIN] Re-acquire failed — lock released")
                else:
                    reacq = tracker.try_reacquire(detections)
                    if reacq:
                        tracker.init(reacq)
                        matched_box = reacq
                        predicted_box = tracker.predict()
                        bx1, by1, bx2, by2 = reacq[:4]
                        delta = ((bx1 + bx2) // 2 - aim_x, (by1 + by2) // 2 - aim_y)
                        print("[MAIN] Re-acquired")
                    else:
                        matched_box = predicted_box = None
                        delta = (0, 0)

            elif tracker.initialized:
                predicted_box = tracker.predict()
                match = best_match(detections, predicted_box,
                                   max_dist=tracker.REMATCH_DIST)
                if match:
                    tracker.update(match)
                    matched_box = match
                    predicted_box = tracker.predict()
                    bx1, by1, bx2, by2 = match[:4]
                    delta = ((bx1 + bx2) // 2 - aim_x, (by1 + by2) // 2 - aim_y)
                    box_area = (bx2 - bx1) * (by2 - by1)
                else:
                    tracker.mark_lost()
                    matched_box = None
                    if predicted_box:
                        bx1, by1, bx2, by2 = predicted_box
                        delta = ((bx1 + bx2) // 2 - aim_x, (by1 + by2) // 2 - aim_y)
                    if tracker.is_lost:
                        tracker.enter_reacquire()

        fps_c.tick()
        fps = fps_c.fps()
        dx, dy = delta

        # ── command_sender (open-loop unless ENABLE_FC_OUTPUT) ───────────────────
        fc_state = {}
        follow = False
        throttle = 1000
        if sender is not None:
            direction = "LOST" if (lock_active and matched_box is None) else "ON"
            sender.send_direction(direction, lock_active, dx, dy, box_area)
            fc_state = sender.get_fc_state()
            follow   = sender.is_follow_active()
            throttle = sender.get_throttle()

        # ── attitude for HUD ─────────────────────────────────────────────────────
        attitude = None
        if fc_state:
            attitude = {
                'yaw':   fc_state.get('yaw',   0.0),
                'pitch': fc_state.get('pitch', 0.0),
                'roll':  fc_state.get('roll',  0.0),
            }

        # ── tracker status string ────────────────────────────────────────────────
        if not lock_active:
            tstatus = "IDLE"
        elif tracker.reacquiring:
            tstatus = "REACQ"
        elif matched_box is None:
            tstatus = "COAST"
        else:
            tstatus = "LOCKED"

        # ── blackbox (CLEAN frame + telemetry) — ALWAYS runs, even headless ──────
        blackbox.write(frame, {
            "armed":   fc_state.get("armed", False),
            "follow":  follow,
            "locked":  lock_active,
            "tracker": tstatus,
            "volt":    fc_state.get("voltage", 0.0),
            "ch3":     throttle,
            "motors":  fc_state.get("motors", [0, 0, 0, 0]),
            "roll":    fc_state.get("roll", 0.0),
            "pitch":   fc_state.get("pitch", 0.0),
            "yaw":     fc_state.get("yaw", 0.0),
            "dx":      dx, "dy": dy,
            "fps":     fps,
            "area":    box_area,
        })

        # ── flight logger (CSV) — ALWAYS runs ────────────────────────────────────
        flog.tick(
            fps=fps, locked=lock_active,
            direction=("LOST" if tstatus == "COAST" else "ON TARGET"),
            dx=dx, dy=dy, tracker=tstatus, box_area=box_area,
            fc_state=fc_state, follow=follow, throttle=throttle,
        )

        # ── live window HUD (windowed mode only) ─────────────────────────────────
        key = 255
        if not HEADLESS:
            display = frame.copy()
            if lock_active:
                inset = matched_box or predicted_box
                if inset:
                    display = draw_zoom_inset(display, inset)
            display = draw_hud(display, {
                'aim': (aim_x, aim_y),
                'detections': detections,
                'fps': fps,
                'lock_active': lock_active,
                'predicted_box': predicted_box,
                'matched_box': matched_box,
                'delta': delta,
                'tracker': tracker,
                'attitude': attitude,
                'source_label': source_label,
            })
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(1) & 0xFF

        # ── lock trigger: SPACE (windowed) OR CH8 auto-request (radio) ───────────
        want_lock = (key == ord(' '))
        if sender is not None and sender.poll_and_clear_auto_lock():
            want_lock = True

        if want_lock and not lock_active:
            if detections:
                aimed = min(detections, key=lambda d:
                            ((d[0] + d[2]) // 2 - aim_x) ** 2 +
                            ((d[1] + d[3]) // 2 - aim_y) ** 2)
                tracker.init(aimed)
                lock_active = True
                matched_box = aimed
                predicted_box = None
                delta = (0, 0)
                cx, cy = (aimed[0] + aimed[2]) // 2, (aimed[1] + aimed[3]) // 2
                print(f"[MAIN] Locked {CLASSES[aimed[5]]} @({cx},{cy}) "
                      f"conf={aimed[4]:.2f}")
            else:
                print("[MAIN] No detection to lock")

        # ── other keys (windowed mode only) ──────────────────────────────────────
        if key == ord('q'):
            break
        elif key == ord('r'):
            lock_active, delta, predicted_box, matched_box = reset_tracking()
            tracker.reset()
            print("[MAIN] Lock reset")
        elif key == ord('v'):
            try:
                cam.release()
                cap, source_label, is_file = open_source("browse")
                cam = CameraThread(cap, is_file=is_file)
                time.sleep(0.3)
                lock_active, delta, predicted_box, matched_box = reset_tracking()
                tracker.reset()
            except Exception as e:
                print(f"[MAIN] video open failed: {e}")
        elif key == ord('c'):
            try:
                cam.release()
                cap, source_label, is_file = open_source()
                cam = CameraThread(cap, is_file=is_file)
                time.sleep(0.3)
                lock_active, delta, predicted_box, matched_box = reset_tracking()
                tracker.reset()
            except Exception as e:
                print(f"[MAIN] cam open failed: {e}")

    # ── cleanup ──────────────────────────────────────────────────────────────────
    cam.release()
    blackbox.stop()
    flog.close()
    if sender is not None:
        sender.close()
    if not HEADLESS:
        cv2.destroyAllWindows()
    print("[MAIN] Clean exit")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[MAIN] Ctrl-C — shutting down")
    except Exception:
        traceback.print_exc()
        try:
            with open("crash_log.txt", "w") as f:
                traceback.print_exc(file=f)
        except Exception:
            pass
        sys.exit(1)