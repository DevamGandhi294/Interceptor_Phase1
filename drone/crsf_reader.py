"""
drone/crsf_reader.py  —  ELRS CRSF serial receiver
====================================================
Parses CRSF RC frames from the ELRS receiver connected to
CRSF_PORT (default /dev/ttyAMA0 at 420000 baud).

Runs in a background daemon thread.  Thread-safe read via channels().
"""

import threading
import time
import serial

from config import CRSF_PORT, CRSF_BAUD

CRSF_SYNC = 0xC8
CRSF_RC   = 0x16


def _parse_crsf(payload) -> list:
    """Decode 22-byte CRSF RC payload → 16 channels in [1000..2000] µs."""
    bits = int.from_bytes(bytes(payload[:22]), 'little')
    ch   = []
    for i in range(16):
        v = (bits >> (i * 11)) & 0x7FF
        ch.append(max(1000, min(2000,
                  int((v - 172) / (1811 - 172) * 1000 + 1000))))
    return ch


class CRSFReader:
    """
    Background CRSF reader.

    Usage:
        crsf = CRSFReader()
        crsf.start()
        ch = crsf.channels()   # list of 16 values, 1000–2000 µs
    """

    def __init__(self):
        self._lock     = threading.Lock()
        self._ch       = [1500] * 16
        self._ch[2]    = 1000   # throttle starts low
        self.connected = False

    def start(self):
        threading.Thread(target=self._loop, daemon=True,
                         name="CRSF").start()

    def channels(self) -> list:
        with self._lock:
            return list(self._ch)

    def _loop(self):
        while True:
            try:
                s = serial.Serial(CRSF_PORT, CRSF_BAUD, timeout=0.002)
                self.connected = True
                print(f"[CRSF] Connected: {CRSF_PORT} @ {CRSF_BAUD}")
                buf  = bytearray()
                last = time.time()
                while True:
                    d = s.read(64)
                    if d:
                        buf.extend(d)
                        last = time.time()
                    while len(buf) >= 3:
                        if buf[0] != CRSF_SYNC:
                            buf.pop(0)
                            continue
                        ln = buf[1]
                        if len(buf) < ln + 2:
                            break
                        if buf[2] == CRSF_RC and ln >= 24:
                            ch = _parse_crsf(buf[3:3+22])
                            with self._lock:
                                self._ch = ch
                        buf = buf[ln+2:]
                    if time.time() - last > 1.5:
                        self.connected = False
                s.close()
            except Exception as e:
                self.connected = False
                print(f"[CRSF] Error: {e}")
                time.sleep(1.0)