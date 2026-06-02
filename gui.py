"""
gui.py  —  HUD / overlay drawing for the Aim Tracker (Phase 1)
==============================================================
Pure drawing. No detection, no tracking, no camera logic.
main.py builds a `state` dict each frame and calls draw_hud().

HUD elements (Phase 1):
  - dx/dy + FPS + lock state   (core)
  - zoom inset of target
  - FC attitude (yaw / pitch / roll)

Adding a new HUD element later = add one key to `state` + one block here.
main.py signature never changes.
"""

import cv2

from config import REACQUIRE_SECONDS
from detection import CLASSES

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
C_LOCK     = (0, 255, 0)
C_COAST    = (0, 165, 255)
C_REACQ    = (0, 0, 255)
C_IDLE     = (0, 255, 255)
C_INFO     = (0, 200, 255)
C_GREY     = (180, 180, 180)
C_DIM      = (150, 150, 150)
C_FC       = (200, 200, 50)
FONT       = cv2.FONT_HERSHEY_SIMPLEX


# ── Small primitives ──────────────────────────────────────────────────────────
def draw_crosshair(frame, cx, cy, size=20, color=C_IDLE, thickness=1):
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, thickness)


def draw_zoom_inset(frame, target_box, inset_size=120, padding=10):
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = target_box[:4]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    bw, bh = max(x2 - x1, 10), max(y2 - y1, 10)
    cx1 = max(0, cx - bw // 2); cy1 = max(0, cy - bh // 2)
    cx2 = min(fw, cx + bw // 2); cy2 = min(fh, cy + bh // 2)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.shape[0] < 4 or crop.shape[1] < 4:
        return frame
    zoom = cv2.resize(crop, (inset_size, inset_size), interpolation=cv2.INTER_LINEAR)
    zc = inset_size // 2
    cv2.line(zoom, (zc - 10, zc), (zc + 10, zc), C_LOCK, 1)
    cv2.line(zoom, (zc, zc - 10), (zc, zc + 10), C_LOCK, 1)
    ix = fw - inset_size - padding
    frame[padding:padding + inset_size, ix:ix + inset_size] = zoom
    cv2.rectangle(frame, (ix, padding), (ix + inset_size, padding + inset_size),
                  C_DIM, 1)
    return frame


def _draw_dashed_box(frame, box, color=(255, 100, 0), dash=8):
    px1, py1, px2, py2 = box
    for x in range(px1, px2, dash * 2):
        cv2.line(frame, (x, py1), (min(x + dash, px2), py1), color, 1)
        cv2.line(frame, (x, py2), (min(x + dash, px2), py2), color, 1)
    for y in range(py1, py2, dash * 2):
        cv2.line(frame, (px1, y), (px1, min(y + dash, py2)), color, 1)
        cv2.line(frame, (px2, y), (px2, min(y + dash, py2)), color, 1)


# ── FC attitude block (Phase 1 addition) ──────────────────────────────────────
def _draw_fc_attitude(frame, attitude):
    """
    attitude: dict {'yaw':float,'pitch':float,'roll':float} or None.
    Drawn bottom-left so it never overlaps the zoom inset (top-right).
    """
    fh = frame.shape[0]
    if attitude is None:
        cv2.putText(frame, "FC: no telemetry", (10, fh - 78),
                    FONT, 0.5, C_REACQ, 1)
        return
    yaw   = attitude.get('yaw',   0.0)
    pitch = attitude.get('pitch', 0.0)
    roll  = attitude.get('roll',  0.0)
    cv2.putText(frame, f"Yaw  : {yaw:+6.1f}", (10, fh - 78), FONT, 0.55, C_FC, 1)
    cv2.putText(frame, f"Pitch: {pitch:+6.1f}", (10, fh - 60), FONT, 0.55, C_FC, 1)
    cv2.putText(frame, f"Roll : {roll:+6.1f}", (10, fh - 42), FONT, 0.55, C_FC, 1)


# ── Main entry point ──────────────────────────────────────────────────────────
def draw_hud(frame, state):
    """
    Draw the full HUD. `state` keys (all optional except aim):
      detections     : list of (x1,y1,x2,y2,conf,cls_id)
      fps            : float
      aim            : (aim_x, aim_y)        REQUIRED
      lock_active    : bool
      predicted_box  : (x1,y1,x2,y2) or None
      matched_box    : (x1,y1,x2,y2,conf,cls_id) or None
      delta          : (dx, dy)
      tracker        : KalmanTracker (for lost_frames / reacquiring / MAX_LOST)
      attitude       : {'yaw','pitch','roll'} or None
      source_label   : str
    Returns the annotated frame.
    """
    fh, fw = frame.shape[:2]
    aim_x, aim_y = state['aim']

    detections   = state.get('detections', [])
    fps          = state.get('fps', 0.0)
    lock_active  = state.get('lock_active', False)
    predicted    = state.get('predicted_box')
    matched      = state.get('matched_box')
    delta        = state.get('delta', (0, 0))
    tracker      = state.get('tracker')
    attitude     = state.get('attitude')
    source_label = state.get('source_label', 'webcam')

    # All detections (grey)
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        cv2.rectangle(frame, (x1, y1), (x2, y2), C_GREY, 1)
        cv2.putText(frame, f"{CLASSES[cls_id]} {conf:.2f}", (x1, y1 - 4),
                    FONT, 0.4, C_GREY, 1)

    if not lock_active:
        if detections:
            nearest = min(detections, key=lambda d:
                          ((d[0] + d[2]) // 2 - aim_x) ** 2 +
                          ((d[1] + d[3]) // 2 - aim_y) ** 2)
            x1, y1, x2, y2 = nearest[:4]
            cv2.rectangle(frame, (x1, y1), (x2, y2), C_IDLE, 2)
            cv2.putText(frame, "SPACE to lock", (x1, y1 - 8),
                        FONT, 0.5, C_IDLE, 1)
    else:
        reacq = bool(tracker and tracker.reacquiring)
        if reacq:
            color = C_REACQ
        elif not matched:
            color = C_COAST
        else:
            color = C_LOCK

        if predicted and not reacq:
            _draw_dashed_box(frame, predicted)

        draw_box = matched or predicted
        if draw_box:
            bx1, by1, bx2, by2 = draw_box[:4]
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
            ocx, ocy = (bx1 + bx2) // 2, (by1 + by2) // 2
            cv2.line(frame, (aim_x, aim_y), (ocx, ocy), color, 1)
            cv2.circle(frame, (ocx, ocy), 5, color, -1)
            if matched:
                cls_id = matched[5] if len(matched) > 5 else 0
                label = f"LOCKED {CLASSES[cls_id]}"
            elif reacq:
                label = f"RE-ACQ {tracker.reacquire_remaining:.1f}s"
            else:
                label = f"COAST {tracker.lost_frames}/{tracker.MAX_LOST}"
            cv2.putText(frame, label, (bx1, by1 - 22), FONT, 0.5, color, 1)

        dx, dy = delta
        cv2.putText(frame, f"dX:{dx:+d}px", (10, fh - 24), FONT, 0.65, C_LOCK, 2)
        cv2.putText(frame, f"dY:{dy:+d}px", (10, fh - 4), FONT, 0.65, C_LOCK, 2)

        # Track health bar
        if tracker:
            bw = 150
            filled = max(0, int(bw * (1 - tracker.lost_frames / tracker.MAX_LOST)))
            cv2.rectangle(frame, (fw - 160, 10), (fw - 10, 28), (50, 50, 50), -1)
            cv2.rectangle(frame, (fw - 160, 10), (fw - 160 + filled, 28), C_LOCK, -1)
            if reacq:
                ratio = tracker.reacquire_remaining / REACQUIRE_SECONDS
                rfill = int(bw * ratio)
                cv2.rectangle(frame, (fw - 160, 32), (fw - 10, 48), (50, 50, 50), -1)
                cv2.rectangle(frame, (fw - 160, 32), (fw - 160 + rfill, 48), C_REACQ, -1)

    # Crosshair colour by state
    cross = (C_LOCK if (lock_active and not (tracker and tracker.reacquiring))
             else C_REACQ if (lock_active and tracker and tracker.reacquiring)
             else C_IDLE)
    draw_crosshair(frame, aim_x, aim_y, color=cross)

    # FC attitude (bottom-left)
    _draw_fc_attitude(frame, attitude)

    # Top-left info
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), FONT, 0.7, C_INFO, 2)
    cv2.putText(frame, f"Drones: {len(detections)}", (10, 55), FONT, 0.7, C_INFO, 2)
    cv2.putText(frame, "SPACE=lock R=reset V=video C=cam Q=quit",
                (10, 85), FONT, 0.4, C_DIM, 1)

    # Source label bottom-right
    txt = f"SRC: {source_label}"
    (tw, _), _ = cv2.getTextSize(txt, FONT, 0.4, 1)
    cv2.putText(frame, txt, (fw - tw - 8, fh - 8), FONT, 0.4, (100, 200, 100), 1)

    return frame