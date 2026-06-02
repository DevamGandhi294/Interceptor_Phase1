"""
drone/fc_reader.py  —  INAV FC MSP telemetry poller
=====================================================
Polls the flight controller over MSP at ~80 Hz round-robin across
all telemetry commands.  All state is readable via get_state().

Thread-safe — runs in its own daemon thread started by CommandSender.
"""

import threading
import time
import struct

MSP_STATUS        = 101
MSP_RAW_IMU       = 102
MSP_MOTOR         = 104
MSP_RC            = 105
MSP_ATTITUDE      = 108
MSP_ALTITUDE      = 109
MSP_ANALOG        = 110
MSP_SENSOR_STATUS = 151


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


def _request(cmd: int) -> bytearray:
    return _build_msp(cmd)


def _read_msp(ser):
    """Read one MSP response from ser.  Returns (cmd, payload) or (None, None)."""
    try:
        while True:
            b = ser.read(1)
            if not b:
                return None, None
            if b == b'$':
                break
        header = ser.read(2)
        if len(header) < 2 or header[0:1] != b'M':
            return None, None
        size_b = ser.read(1)
        if not size_b:
            return None, None
        size  = size_b[0]
        cmd_b = ser.read(1)
        if not cmd_b:
            return None, None
        cmd     = cmd_b[0]
        payload = ser.read(size) if size > 0 else b''
        ser.read(1)   # checksum (not verified)
        return cmd, payload
    except Exception:
        return None, None


class FCReader:
    """
    Polls INAV FC over MSP and exposes a thread-safe state dict.

    Usage:
        fcr = FCReader(ser, ser_lock, verbose=False)
        fcr.start()
        state = fcr.get_state()   # dict with armed, voltage, motors, etc.
        fcr.stop()
    """

    def __init__(self, ser, lock, verbose: bool = False):
        self._ser     = ser
        self._lock    = lock
        self._verbose = verbose
        self._running = False
        self._thread  = None
        self._state   = {
            "armed":       False,
            "roll":        0.0,
            "pitch":       0.0,
            "yaw":         0.0,
            "voltage":     0.0,
            "mah_drawn":   0,
            "rssi":        0,
            "motors":      [1000] * 4,
            "rc_channels": [],
            "altitude_cm": 0,
            "altitude_m":  0.0,
            "variometer":  0,
            "acc_x":       0,
            "acc_y":       0,
            "acc_z":       0,
            "gyro_x":      0,
            "gyro_y":      0,
            "gyro_z":      0,
            "gyro_ok":     False,
            "acc_ok":      False,
            "baro_ok":     False,
            "mag_ok":      False,
            "gps_ok":      False,
            "arm_flags":   0,
            "last_update": 0.0,
        }

    def start(self):
        self._running = True
        self._thread  = threading.Thread(
            target=self._poll_loop, daemon=True, name="FCReader")
        self._thread.start()
        print("[FCReader] Started — MSP polling active")

    def stop(self):
        self._running = False

    def is_armed(self) -> bool:
        return self._state["armed"]

    def get_state(self) -> dict:
        return dict(self._state)

    def _poll_loop(self):
        cmds = [150, MSP_ANALOG, MSP_ATTITUDE, MSP_ALTITUDE,
                MSP_RC, MSP_MOTOR, MSP_RAW_IMU, MSP_SENSOR_STATUS]
        idx = 0
        while self._running:
            cmd = cmds[idx % len(cmds)]
            idx += 1
            try:
                with self._lock:
                    self._ser.write(_request(cmd))
                    cmd_r, payload = _read_msp(self._ser)
                if cmd_r is not None and payload is not None:
                    self._parse(cmd_r, payload)
            except Exception as e:
                if self._verbose:
                    print(f"[FCReader] Error: {e}")
            time.sleep(0.012)

    def _parse(self, cmd: int, p: bytes):
        s = self._state
        if cmd == 150 and len(p) >= 7:
            s["armed"] = bool(p[6] & 0x01)
            if len(p) >= 11:
                s["arm_flags"] = struct.unpack_from('<I', p, 6)[0]
        elif cmd == MSP_ANALOG and len(p) >= 7:
            s["voltage"]   = p[0] / 10.0
            s["mah_drawn"] = struct.unpack_from('<H', p, 1)[0]
            s["rssi"]      = struct.unpack_from('<H', p, 3)[0]
        elif cmd == MSP_ATTITUDE and len(p) >= 6:
            s["roll"]  = struct.unpack_from('<h', p, 0)[0] / 10.0
            s["pitch"] = struct.unpack_from('<h', p, 2)[0] / 10.0
            s["yaw"]   = struct.unpack_from('<h', p, 4)[0] / 1.0
        elif cmd == MSP_ALTITUDE and len(p) >= 6:
            s["altitude_cm"] = struct.unpack_from('<i', p, 0)[0]
            s["altitude_m"]  = s["altitude_cm"] / 100.0
            s["variometer"]  = struct.unpack_from('<h', p, 4)[0]
        elif cmd == MSP_RC and len(p) >= 2:
            n = len(p) // 2
            s["rc_channels"] = [
                struct.unpack_from('<H', p, i*2)[0] for i in range(n)]
        elif cmd == MSP_MOTOR and len(p) >= 8:
            s["motors"] = [
                struct.unpack_from('<H', p, i*2)[0] for i in range(4)]
        elif cmd == MSP_RAW_IMU and len(p) >= 18:
            s["gyro_z"] = struct.unpack_from('<h', p, 10)[0]
        elif cmd == MSP_SENSOR_STATUS and len(p) >= 7:
            s["gyro_ok"] = bool(p[1])
            s["acc_ok"]  = bool(p[2])
            s["baro_ok"] = bool(p[3])
        s["last_update"] = time.time()