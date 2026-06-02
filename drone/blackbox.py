"""
Drone/blackbox.py  —  Flight recorder: raw + HUD video + JSON telemetry
=======================================================================
Per flight segment (rotates every BLACKBOX_MAX_MIN minutes) writes:
    flight_<ts>.mp4       ->  HUD overlay video
    flight_<ts>_raw.mp4   ->  clean camera frames (no overlay)
    flight_<ts>_log.json  ->  per-frame telemetry + summary + keyframes
    frames/<event>.jpg    ->  key-event snapshots (ARM, LOCK, LOST, ...)

Usage:
    bb = BlackBox()
    bb.write(frame, overlay_data)   # frame = CLEAN frame; HUD drawn internally
    bb.stop()

NOTE: pass the CLEAN camera frame to write(). The HUD is drawn on a copy
for the HUD stream; the raw stream gets the untouched frame.

Frame size is taken from the FIRST frame (not config) so it stays correct
when the camera resolution differs between Windows laptop and Radxa.
"""

import cv2
import time
import os
import json
import threading

from config import BLACKBOX_DIR, BLACKBOX_FPS, BLACKBOX_MAX_MIN, WIDTH, HEIGHT


class BlackBox:
    """
    Records two MP4 streams (raw + HUD) plus a synced JSON telemetry log.
    Saves key-event JPEG snapshots (arm, lock, low battery, etc.).
    """

    def __init__(self,
                 output_dir:  str = BLACKBOX_DIR,
                 fps:         int = BLACKBOX_FPS,
                 width:       int = WIDTH,
                 height:      int = HEIGHT,
                 max_minutes: int = BLACKBOX_MAX_MIN):
        self.output_dir   = os.path.expanduser(output_dir)
        self.fps          = fps
        # width/height are hints only; real size locked on first frame.
        self.width        = width
        self.height       = height
        self._size_locked = False
        self.max_frames   = fps * 60 * max_minutes

        self._writer       = None     # HUD stream
        self._raw_writer   = None     # raw stream
        self._lock         = threading.Lock()
        self._frame_count  = 0
        self._current_file = None
        self._raw_file     = None
        self._log_file     = None
        self._log_data     = []
        self._start_time   = time.time()
        self._key_frames   = []
        self._last_state   = {}

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "frames"), exist_ok=True)
        print(f"[BLACKBOX] Output dir: {self.output_dir}")

    # -- writer creation --------------------------------------------------------
    def _open_writer(self, path, size):
        """Try mp4v first; fall back to XVID/.avi if codec unavailable.
        Returns (writer, final_path)."""
        import cv2
        import os
        for fourcc_str, ext in (("mp4v", ".mp4"), ("XVID", ".avi"), ("MJPG", ".avi")):
            p = os.path.splitext(path)[0] + ext
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            wr = cv2.VideoWriter(p, fourcc, self.fps, size)
            if wr.isOpened():
                if fourcc_str != "mp4v":
                    print(f"[BLACKBOX] mp4v unavailable, using {fourcc_str} -> {p}")
                return wr, p
            wr.release()
        print(f"[BLACKBOX] ERROR: no working codec for {path}")
        return None, path
    
    
    def _new_writer(self):
        import os
        import time
        ts       = time.strftime("%Y%m%d_%H%M%S")
        hud_path = os.path.join(self.output_dir, f"flight_{ts}.mp4")
        raw_path = os.path.join(self.output_dir, f"flight_{ts}_raw.mp4")
        size     = (self.width, self.height)
    
        w, hud_final     = self._open_writer(hud_path, size)
        self._raw_writer, raw_final = self._open_writer(raw_path, size)
    
        self._current_file = hud_final
        self._raw_file     = raw_final
        self._log_file     = os.path.splitext(hud_final)[0] + "_log.json"
        self._frame_count  = 0
        self._log_data     = []
        self._start_time   = time.time()
        self._key_frames   = []
        print(f"[BLACKBOX] Recording HUD: {hud_final}")
        print(f"[BLACKBOX] Recording RAW: {raw_final}")
        print(f"[BLACKBOX] Log:           {self._log_file}")
        return w
    

    def _fit(self, frame):
        """Resize frame to the locked writer size if it differs."""
        h, w = frame.shape[:2]
        if (w, h) != (self.width, self.height):
            return cv2.resize(frame, (self.width, self.height))
        return frame

    # -- main write -------------------------------------------------------------
    def write(self, frame, overlay_data: dict = None):
        """Write one frame. `frame` must be the CLEAN camera frame."""
        if frame is None:
            return
        d = overlay_data or {}

        # Lock real size from the first frame.
        if not self._size_locked:
            self.height, self.width = frame.shape[:2]
            self._size_locked = True
            print(f"[BLACKBOX] Frame size locked: {self.width}x{self.height}")

        raw = self._fit(frame)
        f   = raw.copy()

        armed   = d.get("armed",   False)
        follow  = d.get("follow",  False)
        volt    = d.get("volt",    0.0)
        ch3     = d.get("ch3",     1500)
        dist    = d.get("dist",    0.0)
        tracker = d.get("tracker", "IDLE")
        motors  = (list(d.get("motors", [0] * 4)) + [0, 0, 0, 0])[:4]
        roll    = d.get("roll",    0.0)
        pitch   = d.get("pitch",   0.0)
        yaw     = d.get("yaw",     0.0)
        dx      = d.get("dx",      0)
        dy      = d.get("dy",      0)
        fps_val = d.get("fps",     0.0)
        area    = d.get("area",    0)
        locked  = d.get("locked",  False)
        ch1     = d.get("ch1",     1500)
        ch2     = d.get("ch2",     1500)
        ch4     = d.get("ch4",     1500)

        ts_str  = time.strftime("%Y-%m-%d %H:%M:%S")
        elapsed = round(time.time() - self._start_time, 2)

        # -- HUD overlay (drawn on copy `f`, raw stays clean) -------------------
        cv2.rectangle(f, (0, 0), (self.width, 22), (0, 0, 0), -1)
        arm_col = (0, 220, 80) if armed  else (80,  80,  220)
        fol_col = (0, 220, 80) if follow else (140, 140, 140)
        cv2.putText(f, ts_str,                           (4,   15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.putText(f, "ARMED" if armed else "DISARMED", (200, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, arm_col, 1)
        cv2.putText(f, "AUTO"  if follow else "MANUAL",  (295, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, fol_col, 1)
        cv2.putText(f, f"V:{volt:.1f}V",                 (365, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.putText(f, f"THR:{ch3}",                     (425, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.putText(f, f"FPS:{fps_val:.0f}",             (500, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.putText(f, tracker,                          (560, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,   200, 200), 1)

        cv2.rectangle(f, (0, self.height-22), (self.width, self.height), (0, 0, 0), -1)
        cv2.putText(f, f"M1:{motors[0]} M2:{motors[1]} M3:{motors[2]} M4:{motors[3]}",
                    (4, self.height-8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        cv2.putText(f, f"R:{roll:+.1f} P:{pitch:+.1f} Y:{yaw:+.1f}  dx:{dx:+d} dy:{dy:+d}  D:{dist:.0f}cm",
                    (280, self.height-8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        cv2.putText(f, f"t:{elapsed:.1f}s",
                    (self.width-70, self.height-8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 100), 1)

        with self._lock:
            if self._writer is None:
                self._writer = self._new_writer()

            # raw first, then HUD
            if self._raw_writer is not None:
                self._raw_writer.write(raw)
            self._writer.write(f)
            self._frame_count += 1

            entry = {
                "t":       elapsed,
                "frame":   self._frame_count,
                "ts":      ts_str,
                "armed":   armed,
                "follow":  follow,
                "locked":  locked,
                "tracker": tracker,
                "volt":    round(volt,  2),
                "ch1":     ch1,
                "ch2":     ch2,
                "ch3":     ch3,
                "ch4":     ch4,
                "motors":  motors,
                "roll":    round(roll,  1),
                "pitch":   round(pitch, 1),
                "yaw":     round(yaw,   1),
                "dx":      dx,
                "dy":      dy,
                "dist_cm": round(dist,  1),
                "area":    area,
                "fps":     round(fps_val, 1),
            }
            self._log_data.append(entry)

            # -- Key-event detection -------------------------------------------
            prev    = self._last_state
            is_key  = False
            reasons = []

            if not prev.get("armed")  and armed:      reasons.append("ARMED");         is_key = True
            if     prev.get("armed")  and not armed:  reasons.append("DISARMED");      is_key = True
            if not prev.get("follow") and follow:     reasons.append("AUTO_ON");       is_key = True
            if     prev.get("follow") and not follow: reasons.append("AUTO_OFF");      is_key = True
            if not prev.get("locked") and locked:     reasons.append("TARGET_LOCKED"); is_key = True
            if     prev.get("locked") and not locked: reasons.append("TARGET_LOST");   is_key = True
            if "LOST" in tracker and "LOST" not in prev.get("tracker", ""):
                                                      reasons.append("TRACKER_LOST");  is_key = True
            if volt > 0 and volt < 11.0 and prev.get("volt", 99) >= 11.0:
                                                      reasons.append("LOW_BATTERY");   is_key = True
            if self._frame_count % (self.fps * 5) == 0:
                                                      reasons.append("PERIODIC");      is_key = True

            if is_key:
                kf_name = f"frame_{elapsed:.1f}s_{'_'.join(reasons)}.jpg"
                kf_path = os.path.join(self.output_dir, "frames", kf_name)
                cv2.imwrite(kf_path, f)
                entry["keyframe"]         = kf_name
                entry["keyframe_reasons"] = reasons
                self._key_frames.append({"t": elapsed, "file": kf_name, "reasons": reasons})
                print(f"[BLACKBOX] Key frame: {kf_name}")

            self._last_state = entry

            # Flush JSON every 5 seconds
            if self._frame_count % (self.fps * 5) == 0:
                self._flush_log()

            # Rotate file every max_frames
            if self._frame_count >= self.max_frames:
                self._writer.release()
                if self._raw_writer is not None:
                    self._raw_writer.release()
                self._flush_log()
                self._writer = self._new_writer()

    # -- logging ----------------------------------------------------------------
    def _flush_log(self):
        if not self._log_file or not self._log_data:
            return
        output = {
            "summary":   self._make_summary(),
            "keyframes": self._key_frames,
            "frames":    self._log_data,
        }
        try:
            with open(self._log_file, 'w') as fh:
                json.dump(output, fh, indent=2)
        except Exception as e:
            print(f"[BLACKBOX] Log flush error: {e}")

    def _make_summary(self) -> dict:
        if not self._log_data:
            return {}
        armed_frames  = [e for e in self._log_data if e["armed"]]
        follow_frames = [e for e in self._log_data if e["follow"]]
        locked_frames = [e for e in self._log_data if e["locked"]]
        volts = [e["volt"]    for e in self._log_data if e["volt"]    > 0]
        dists = [e["dist_cm"] for e in self._log_data if e["dist_cm"] > 0]
        return {
            "total_frames":       len(self._log_data),
            "duration_s":         round(self._log_data[-1]["t"], 1),
            "armed_frames":       len(armed_frames),
            "follow_frames":      len(follow_frames),
            "locked_frames":      len(locked_frames),
            "volt_min":           round(min(volts), 2) if volts else 0,
            "volt_max":           round(max(volts), 2) if volts else 0,
            "dist_avg_cm":        round(sum(dists) / len(dists), 1) if dists else 0,
            "dist_min_cm":        round(min(dists), 1) if dists else 0,
            "dist_max_cm":        round(max(dists), 1) if dists else 0,
            "tracker_lost_count": sum(1 for e in self._log_data if "LOST" in e.get("tracker", "")),
            "keyframe_count":     len(self._key_frames),
        }

    # -- shutdown ---------------------------------------------------------------
    def stop(self):
        with self._lock:
            if self._writer:
                self._writer.release()
                self._writer = None
            if self._raw_writer:
                self._raw_writer.release()
                self._raw_writer = None
            self._flush_log()
        print(f"[BLACKBOX] HUD:    {self._current_file}")
        print(f"[BLACKBOX] RAW:    {self._raw_file}")
        print(f"[BLACKBOX] Log:    {self._log_file}")
        print(f"[BLACKBOX] Frames: {self.output_dir}/frames/")

    def current_file(self):
        return self._current_file