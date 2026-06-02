"""
drone/flight_logger.py  —  Per-flight .txt log for the drone follow system
===========================================================================
Writes one text file per session to FLIGHT_LOG_DIR (default: logs/).
Header block + CSV body (one line per frame) + summary footer on close().

Usage:
    from drone.flight_logger import FlightLogger

    log = FlightLogger()

    # inside the main loop:
    log.tick(
        fps=fps_display, locked=locked, direction=direction,
        dx=dx, dy=dy, tracker=tracker_status, box_area=box_area,
        dist_cm=dist_cm, fc_state=sender.get_fc_state(),
        follow=sender.is_follow_active(), throttle=sender.get_throttle(),
    )

    log.close()    # call on exit
"""

import os
import time
from datetime import datetime

from config import FLIGHT_LOG_DIR, FLIGHT_LOG_INTERVAL


_HEADER = """\
================================================================================
  DRONE FOLLOW SYSTEM  —  Flight Log
  Start : {start}
  File  : {filename}
================================================================================
  Columns (CSV after this block):
    time_s   | seconds since flight start
    fps      | display FPS
    locked   | 1/0 target locked
    dir      | direction command
    dx       | pixel error X
    dy       | pixel error Y
    tracker  | tracker status string
    area_px2 | bounding-box area (pixels²)
    dist_cm  | estimated distance (cm)
    armed    | FC armed 1/0
    follow   | auto-follow active 1/0
    throttle | CH3 µs
    voltage  | battery voltage (V)
    roll     | FC roll (°)
    pitch    | FC pitch (°)
    yaw      | FC yaw (°)
    m1..m4   | motor µs
--------------------------------------------------------------------------------
time_s,fps,locked,dir,dx,dy,tracker,area_px2,dist_cm,armed,follow,throttle,voltage,roll,pitch,yaw,m1,m2,m3,m4
"""

_FOOTER = """\
--------------------------------------------------------------------------------
  FLIGHT SUMMARY
  End        : {end}
  Duration   : {duration}
  Frames     : {frames}
  Avg FPS    : {avg_fps:.1f}
  Lock time  : {lock_pct:.1f}%  ({lock_frames} / {frames} frames)
  Min dist   : {min_dist:.0f} cm
  Max dist   : {max_dist:.0f} cm
  Min volt   : {min_volt:.2f} V
  Max volt   : {max_volt:.2f} V
================================================================================
"""


class FlightLogger:
    """Writes one .txt flight log per session."""

    def __init__(self, log_dir: str = FLIGHT_LOG_DIR):
        os.makedirs(log_dir, exist_ok=True)

        self._start_wall = time.time()
        self._start_dt   = datetime.now()
        fname            = f"flight_{self._start_dt.strftime('%Y%m%d_%H%M%S')}.txt"
        self._path       = os.path.join(log_dir, fname)
        self._fh         = open(self._path, "w", buffering=1)  # line-buffered

        self._fh.write(_HEADER.format(
            start    = self._start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            filename = fname,
        ))

        # Accumulated stats
        self._frame_count = 0
        self._lock_frames = 0
        self._fps_sum     = 0.0
        self._min_dist    = float("inf")
        self._max_dist    = 0.0
        self._min_volt    = float("inf")
        self._max_volt    = 0.0
        self._tick_count  = 0

        print(f"[FLOG] Flight log started → {self._path}")

    # ── Public API ────────────────────────────────────────────────────────────

    def tick(self,
             fps:       float = 0.0,
             locked:    bool  = False,
             direction: str   = "ON TARGET",
             dx:        int   = 0,
             dy:        int   = 0,
             tracker:   str   = "IDLE",
             box_area:  int   = 0,
             dist_cm:   float = 0.0,
             fc_state:  dict  = None,
             follow:    bool  = False,
             throttle:  int   = 1000):
        """Call once per frame from the main loop."""
        self._tick_count += 1
        if self._tick_count % FLIGHT_LOG_INTERVAL != 0:
            return

        fc      = fc_state or {}
        armed   = int(fc.get("armed",   False))
        voltage = fc.get("voltage",  0.0)
        roll    = fc.get("roll",     0.0)
        pitch   = fc.get("pitch",    0.0)
        yaw     = fc.get("yaw",      0.0)
        motors  = fc.get("motors",   [0, 0, 0, 0])
        m1, m2, m3, m4 = (motors + [0, 0, 0, 0])[:4]

        t_s       = time.time() - self._start_wall
        dir_clean = direction.replace(",", "/")

        line = (
            f"{t_s:.3f},{fps:.1f},{int(locked)},{dir_clean},"
            f"{dx},{dy},{tracker},{box_area},{dist_cm:.1f},"
            f"{armed},{int(follow)},{throttle},{voltage:.2f},"
            f"{roll:.2f},{pitch:.2f},{yaw:.2f},"
            f"{m1},{m2},{m3},{m4}\n"
        )
        self._fh.write(line)

        # Accumulate stats
        self._frame_count += 1
        self._fps_sum     += fps
        if locked:
            self._lock_frames += 1
        if dist_cm > 0:
            self._min_dist = min(self._min_dist, dist_cm)
            self._max_dist = max(self._max_dist, dist_cm)
        if voltage > 0:
            self._min_volt = min(self._min_volt, voltage)
            self._max_volt = max(self._max_volt, voltage)

    def close(self):
        """Write summary footer and close the file."""
        if self._fh.closed:
            return

        end_dt   = datetime.now()
        duration = end_dt - self._start_dt
        total_s  = duration.total_seconds()
        h        = int(total_s // 3600)
        m        = int((total_s % 3600) // 60)
        s        = int(total_s % 60)

        n   = max(self._frame_count, 1)
        avg = self._fps_sum / n

        self._fh.write(_FOOTER.format(
            end         = end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            duration    = f"{h:02d}:{m:02d}:{s:02d}",
            frames      = self._frame_count,
            avg_fps     = avg,
            lock_pct    = 100.0 * self._lock_frames / n,
            lock_frames = self._lock_frames,
            min_dist    = self._min_dist if self._min_dist < float("inf") else 0,
            max_dist    = self._max_dist,
            min_volt    = self._min_volt if self._min_volt < float("inf") else 0,
            max_volt    = self._max_volt,
        ))

        self._fh.close()
        print(f"[FLOG] Flight log closed → {self._path}")