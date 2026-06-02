"""
drone/command_sender.py  —  ELRS CRSF → MSP bridge (INAV) with Stage-1 auto
=============================================================================
CH8 high (or GUI button) enables auto mode when a target is locked:
  - Auto pitch    = AUTO_PITCH_PWM (fixed forward lean)
  - Auto yaw      = 1500 + (dx / HALF_WIDTH)  * YAW_MAX_OFFSET  [clamped]
  - Auto throttle = base + (-dy / HALF_HEIGHT) * THR_MAX_OFFSET  [clamped]
                    where base = pilot throttle at the moment of lock
"""

import serial
import struct
import threading
import time

from config import (
    ENABLE_FC_OUTPUT, PORT, BAUD, WIDTH, HEIGHT, RC_CENTER,
    AUTO_TRIGGER_CH, AUTO_TRIGGER_THR, AUTO_PITCH_PWM,
    YAW_MAX_OFFSET, YAW_TIMEOUT_S,
    THR_MAX_OFFSET, THR_MIN_SAFE, THR_MAX_SAFE,
)
from drone.crsf_reader import CRSFReader
from drone.fc_reader   import FCReader

MSP_SET_RAW_RC = 200


def _build_msp(cmd: int, data: bytes = b'') -> bytearray:
    size     = len(data)
    checksum = size ^ cmd
    for b in data:
        checksum ^= b
    buf = bytearray(b'$M<')
    buf.append(size)
    buf.append(cmd)
    buf.extend(data)
    buf.append(checksum)
    return buf


class CommandSender:
    """
    Runs a 50 Hz background loop that:
      1. Reads live RC from CRSFReader (Boxer controller → ELRS RX)
      2. Optionally overrides pitch/yaw/throttle with tracker-derived values
      3. Forwards 8-channel MSP SET_RAW_RC to the INAV flight controller

    Public API used by main.py and DroneGUI:
        sender.send_direction(direction, locked, dx, dy, box_area)
        sender.is_follow_active()  → bool
        sender.get_fc_state()      → dict
        sender.get_throttle()      → int µs
        sender.get_sent_channels() → list[int]  (8 values)
        sender.get_crsf_channels() → list[int]  (16 values)
        sender.request_auto_lock()
        sender.poll_and_clear_auto_lock() → bool
        sender.set_gui_auto(on)
        sender.close()
    """

    def __init__(self, rate_hz: int = 50, verbose: bool = False):
        self._verbose = verbose
        self._period  = 1.0 / rate_hz

        # FC serial
        self._ser_lock = threading.Lock()
        self._ser      = None
        self.available = False
        try:
            self._ser      = serial.Serial(PORT, BAUD, timeout=0.05)
            self.available = True
            print(f"[SENDER] FC port opened: {PORT} @ {BAUD}")
        except Exception as e:
            print(f"[SENDER] FC port unavailable: {e}")

        # ELRS RX
        self._crsf = CRSFReader()
        self._crsf.start()

        # FC state polling
        self._fcr = None
        if self.available:
            self._fcr = FCReader(self._ser, self._ser_lock, verbose=verbose)
            self._fcr.start()

        # Tracker input
        self._dx_latest          = 0
        self._dy_latest          = 0
        self._dx_timestamp       = 0.0
        self._tracker_lost       = True
        self._tracker_was_locked = False

        # Auto-lock request
        self._auto_lock_requested = False
        self._lock_req_lock       = threading.Lock()

        # GUI override
        self._gui_auto_request = False

        # CH8 edge detection
        self._prev_ch8_high = False

        # Base throttle captured at lock moment
        self._base_throttle = RC_CENTER
        self._base_valid    = False

        # Last 8 channels sent to FC
        self._last_sent = [RC_CENTER, RC_CENTER, 1000, RC_CENTER,
                           RC_CENTER, RC_CENTER, RC_CENTER, RC_CENTER]
        self._sent_lock = threading.Lock()

        # Background forwarding loop
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="CmdSender")
        self._thread.start()

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def send_direction(self, direction: str, locked: bool,
                       dx: int, dy: int, box_area: int = 0):
        """Called from main loop every frame with latest tracker state."""
        now = time.time()

        # Lock rising edge: snapshot pilot throttle as base
        if locked and not self._tracker_was_locked:
            ch = self._crsf.channels()
            if len(ch) >= 3:
                self._base_throttle = ch[2]
                self._base_valid    = True
                print(f"[SENDER] Base throttle captured: {self._base_throttle}")

        # Lock falling edge: invalidate base
        if not locked and self._tracker_was_locked:
            self._base_valid = False
            print("[SENDER] Base released (lock lost)")

        self._tracker_was_locked = locked

        if locked and direction != "LOST":
            self._dx_latest    = dx
            self._dy_latest    = dy
            self._dx_timestamp = now
            self._tracker_lost = False
        else:
            self._tracker_lost = True

    def is_follow_active(self) -> bool:
        ch      = self._crsf.channels()
        idx     = AUTO_TRIGGER_CH - 1
        ch_auto = ch[idx] if len(ch) > idx else 1000
        fresh   = (time.time() - self._dx_timestamp) <= YAW_TIMEOUT_S
        return ((ch_auto > AUTO_TRIGGER_THR or self._gui_auto_request)
                and fresh
                and not self._tracker_lost
                and self._base_valid)

    def request_auto_lock(self):
        with self._lock_req_lock:
            self._auto_lock_requested = True

    def poll_and_clear_auto_lock(self) -> bool:
        with self._lock_req_lock:
            if self._auto_lock_requested:
                self._auto_lock_requested = False
                return True
        return False

    def set_gui_auto(self, on: bool):
        was = self._gui_auto_request
        self._gui_auto_request = bool(on)
        if self._gui_auto_request and not was:
            self.request_auto_lock()

    def get_throttle(self) -> int:
        with self._sent_lock:
            return self._last_sent[2]

    def get_sent_channels(self) -> list:
        with self._sent_lock:
            return list(self._last_sent)

    def get_crsf_channels(self) -> list:
        return self._crsf.channels()

    def get_fc_state(self) -> dict:
        if self._fcr is None:
            return {
                "armed": False, "voltage": 0.0, "motors": [0] * 4,
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                "rc_channels": [], "altitude_m": 0.0, "altitude_cm": 0,
                "variometer": 0, "mah_drawn": 0, "rssi": 0,
                "arm_flags": 0, "last_update": 0.0,
                "gyro_ok": False, "acc_ok": False, "baro_ok": False,
            }
        return self._fcr.get_state()

    def close(self):
        self._running = False
        time.sleep(0.1)
        if self._fcr is not None:
            self._fcr.stop()
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
        print("[SENDER] Closed")

    # ── AUTO CALCULATIONS ─────────────────────────────────────────────────────

    def _dx_to_yaw_pwm(self, dx: int) -> int:
        half_w = WIDTH / 2.0
        ratio  = max(-1.0, min(1.0, dx / half_w))
        return int(RC_CENTER + ratio * YAW_MAX_OFFSET)

    def _dy_to_throttle_pwm(self, dy: int, base: int) -> int:
        half_h = HEIGHT / 2.0
        ratio  = max(-1.0, min(1.0, (-dy) / half_h))
        thr    = base + ratio * THR_MAX_OFFSET
        return int(max(THR_MIN_SAFE, min(THR_MAX_SAFE, thr)))

    # ── RC FORWARDING LOOP (50 Hz) ────────────────────────────────────────────

    def _loop(self):
        next_tick = time.time()
        while self._running:
            now = time.time()
            if now < next_tick:
                time.sleep(max(0.0, next_tick - now))
            next_tick += self._period

            ch = self._crsf.channels()
            if len(ch) < 8:
                continue

            roll, pilot_pitch, pilot_throttle, pilot_yaw = ch[0], ch[1], ch[2], ch[3]
            ch5, ch6, ch7, ch8 = ch[4], ch[5], ch[6], ch[7]

            # CH8 rising edge → request auto-lock
            ch8_high = ch8 > AUTO_TRIGGER_THR
            if ch8_high and not self._prev_ch8_high:
                self.request_auto_lock()
            self._prev_ch8_high = ch8_high

            # Decide auto vs manual
            want_auto = ch8_high or self._gui_auto_request
            fresh     = (time.time() - self._dx_timestamp) <= YAW_TIMEOUT_S
            auto_on   = (want_auto and fresh
                         and not self._tracker_lost
                         and self._base_valid)

            if auto_on:
                final_yaw      = self._dx_to_yaw_pwm(self._dx_latest)
                final_pitch    = AUTO_PITCH_PWM
                final_throttle = self._dy_to_throttle_pwm(
                    self._dy_latest, self._base_throttle)
            else:
                final_yaw      = pilot_yaw
                final_pitch    = pilot_pitch
                final_throttle = pilot_throttle

            out = [roll, final_pitch, final_throttle, final_yaw,
                   ch5, ch6, ch7, ch8]

            with self._sent_lock:
                self._last_sent = list(out)

            if not self.available or self._ser is None or not ENABLE_FC_OUTPUT:
                continue

            data = struct.pack('<8H', *out)
            pkt  = _build_msp(MSP_SET_RAW_RC, data)
            try:
                with self._ser_lock:
                    self._ser.write(pkt)
            except Exception as e:
                if self._verbose:
                    print(f"[SENDER] write error: {e}")