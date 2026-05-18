"""
VisionGate X — Helmet Detection (binary labels + any-frame helmet for video final)

Rules:
- On screen / per rider: "HELMET DETECTED" only when helmet wins; else "NO HELMET"
  (unknown / uncertain -> NO HELMET).

- End of video: declare HELMET DETECTED if there is solid evidence anywhere in the clip:
  (see FINAL_* settings below).

Run: python modules/helmet.py
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from modules.detector import BBox, DetectionResult


# ============================================================
# SETTINGS
# ============================================================

VIDEO_PATH = "data/sample_videos/bike2.mp4"
MAX_FRAMES = 500
SHOW_WINDOW = True
PRINT_FRAME_LOGS = True

# Frame: slightly easier to call "helmet" when shapes/colours match
HELMET_SCORE_THR = 0.42
NO_HELMET_SCORE_THR = 0.50
HEL_WIN_MARGIN = 0.01  # H must be > N by this much for helmet

# Video-level final (OR logic — any can trigger HELMET)
FINAL_HELMET_IF_ANY_INTERNAL_HELMET = True   # any frame classified helmet -> video HELMET
FINAL_HELMET_IF_MAX_H = 0.52                  # or best helmet score in whole video >= this
FINAL_HELMET_IF_FRACTION = 0.06               # or helmet frames / total frames >= this

SAVE_HEAD_DEBUG = False
HEAD_DIR = "outputs/helmet_heads"

USE_LOOSE_CROP = True


# ============================================================
# BBOX / CROPS  (fix H=0,N=0 from empty crops)
# ============================================================

def _clip_box(box: BBox, W: int, H: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(W - 1, int(box.x1)))
    y1 = max(0, min(H - 1, int(box.y1)))
    x2 = max(0, min(W, int(box.x2)))
    y2 = max(0, min(H, int(box.y2)))
    return x1, y1, x2, y2


def _bbox_area(b: BBox) -> int:
    return max(0, b.x2 - b.x1) * max(0, b.y2 - b.y1)


def _head_region_tight(frame: np.ndarray, person: BBox) -> np.ndarray | None:
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = _clip_box(person, W, H)
    w = x2 - x1
    h = y2 - y1
    if w <= 8 or h <= 8:
        return None

    head_h = max(int(h * 0.48), 16)
    top = y1 + int(h * 0.00)
    bottom = min(y2, top + head_h)
    cx = (x1 + x2) // 2
    hw = max(int(w * 0.38), 10)
    left = max(x1, cx - hw)
    right = min(x2, cx + hw)
    if right <= left or bottom <= top:
        return None
    return frame[top:bottom, left:right]


def _head_region_loose(frame: np.ndarray, person: BBox) -> np.ndarray | None:
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = _clip_box(person, W, H)
    w = x2 - x1
    h = y2 - y1
    if w <= 8 or h <= 8:
        return None

    head_h = max(int(h * 0.62), 24)
    top = y1
    bottom = min(y2, top + head_h)
    cx = (x1 + x2) // 2
    hw = max(int(w * 0.48), 12)
    left = max(x1, cx - hw)
    right = min(x2, cx + hw)
    if right <= left or bottom <= top:
        return None
    return frame[top:bottom, left:right]


def _head_region_top_half(frame: np.ndarray, person: BBox) -> np.ndarray | None:
    """Last resort: top 55% of person box, full width."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = _clip_box(person, W, H)
    w = x2 - x1
    h = y2 - y1
    if w <= 8 or h <= 8:
        return None
    mh = max(int(h * 0.55), 20)
    crop = frame[y1:y1 + mh, x1:x2]
    if crop is None or crop.size == 0:
        return None
    return crop


def _get_head_crop(frame: np.ndarray, person: BBox, frame_idx: int, pid: int) -> np.ndarray | None:
    for crop_fn in (_head_region_tight, _head_region_loose, _head_region_top_half):
        c = crop_fn(frame, person)
        if c is None or c.size == 0:
            continue
        if c.shape[0] >= 12 and c.shape[1] >= 12:
            if SAVE_HEAD_DEBUG:
                os.makedirs(HEAD_DIR, exist_ok=True)
                tag = "tight" if crop_fn is _head_region_tight else (
                    "loose" if crop_fn is _head_region_loose else "tophalf"
                )
                cv2.imwrite(
                    os.path.join(HEAD_DIR, f"f{frame_idx:04d}_p{pid}_{tag}.jpg"),
                    c,
                )
            return c
    return None


# ============================================================
# SCORING
# ============================================================

def _skin_ratio(bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 14, 25], np.uint8), np.array([32, 235, 255], np.uint8))
    m2 = cv2.inRange(hsv, np.array([160, 14, 25], np.uint8), np.array([180, 235, 255], np.uint8))
    skin = cv2.bitwise_or(m1, m2)
    return float(np.count_nonzero(skin)) / float(max(1, skin.size))


def _edge_density(bgr: np.ndarray) -> float:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    e = cv2.Canny(g, 50, 140)
    return float(np.count_nonzero(e)) / float(max(1, e.size))


def _uniformity(bgr: np.ndarray) -> float:
    s = cv2.resize(bgr, (48, 48), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].reshape(-1)
    hist, _ = np.histogram(h, bins=18, range=(0, 180))
    p = hist.astype(np.float32) / max(1.0, float(hist.sum()))
    nz = p[p > 0]
    ent = -np.sum(nz * np.log2(nz))
    return 1.0 - float(ent / np.log2(18.0))


def _dark_ratio(bgr: np.ndarray) -> float:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(np.count_nonzero(g < 80)) / float(max(1, g.size))


def _helmet_scores(bgr: np.ndarray) -> tuple[float, float]:
    skin = _skin_ratio(bgr)
    edge = _edge_density(bgr)
    uni = _uniformity(bgr)
    dark = _dark_ratio(bgr)

    helmet = (
        0.46 * (1.0 - skin)
        + 0.24 * min(1.0, edge * 5.0)
        + 0.17 * uni
        + 0.13 * min(1.0, dark * 2.0)
    )
    no_helmet = (
        0.54 * skin
        + 0.20 * (1.0 - uni)
        + 0.14 * max(0.0, 0.20 - edge) * 5.0
        + 0.12 * max(0.0, 0.30 - dark) * 3.0
    )
    return (
        float(max(0.0, min(1.0, helmet))),
        float(max(0.0, min(1.0, no_helmet))),
    )


def _classify_scores(hs: float, ns: float) -> str:
    if hs >= HELMET_SCORE_THR and hs > ns + HEL_WIN_MARGIN:
        return "helmet"
    if ns >= NO_HELMET_SCORE_THR and ns > hs + HEL_WIN_MARGIN:
        return "no_helmet"
    return "unknown"


# ============================================================
# OPTIONAL YOLO HELMET
# ============================================================

class _CustomHelmetYOLO:
    def __init__(self):
        from ultralytics import YOLO
        self._model = YOLO(config.HELMET_MODEL_PATH)

    def predict(self, crop: np.ndarray) -> tuple[str, float, float]:
        if crop is None or crop.size == 0:
            return "unknown", 0.0, 0.0
        r0 = self._model.predict(source=crop, conf=0.25, verbose=False)
        bh, bn = 0.0, 0.0
        for r in r0:
            if r.boxes is None:
                continue
            for box in r.boxes:
                name = r.names[int(box.cls[0])].lower()
                cf = float(box.conf[0]) if box.conf is not None else 0.0
                if "no" in name or "without" in name:
                    bn = max(bn, cf)
                elif "helmet" in name:
                    bh = max(bh, cf)
        if bh >= 0.40 and bh > bn:
            return "helmet", bh, bn
        if bn >= 0.40 and bn > bh:
            return "no_helmet", bh, bn
        return "unknown", bh, bn


# ============================================================
# API
# ============================================================

class HelmetChecker:
    def __init__(self):
        self._yolo = None
        if getattr(config, "USE_CUSTOM_HELMET_MODEL", False) and os.path.exists(
            str(getattr(config, "HELMET_MODEL_PATH", ""))
        ):
            self._yolo = _CustomHelmetYOLO()

    def check_frame(
        self,
        frame: np.ndarray,
        detection: DetectionResult,
        frame_idx: int = 0,
    ) -> tuple[dict[int, str], dict[int, tuple[float, float]], dict[int, str], dict[int, str]]:
        """
        Returns:
          internal   : helmet | no_helmet | unknown
          scores     : (H, N)
          display    : HELMET DETECTED | NO HELMET
        """
        pairs = detection.rider_vehicle_pairs()
        paired = set()
        for p, _ in pairs:
            try:
                paired.add(detection.persons.index(p))
            except ValueError:
                pass

        internal: dict[int, str] = {}
        scores: dict[int, tuple[float, float]] = {}
        display: dict[int, str] = {}

        for i, person in enumerate(detection.persons):
            if i not in paired:
                internal[i] = "unknown"
                scores[i] = (0.0, 0.0)
                display[i] = "NO HELMET"
                continue

            crop = _get_head_crop(frame, person, frame_idx, i)

            if self._yolo is not None and crop is not None:
                state, hs, ns = self._yolo.predict(crop)
            elif crop is not None:
                hs, ns = _helmet_scores(crop)
                state = _classify_scores(hs, ns)
            else:
                state, hs, ns = "unknown", 0.0, 0.0

            internal[i] = state
            scores[i] = (hs, ns)
            display[i] = "HELMET DETECTED" if state == "helmet" else "NO HELMET"

        return internal, scores, display

    def riders_without_helmet(
        self,
        detection: DetectionResult,
        display_statuses: dict[int, str],
    ) -> list[tuple[BBox, BBox]]:
        out: list[tuple[BBox, BBox]] = []
        for person, veh in detection.rider_vehicle_pairs():
            try:
                idx = detection.persons.index(person)
            except ValueError:
                continue
            if display_statuses.get(idx) == "NO HELMET":
                out.append((person, veh))
        return out


# ============================================================
# DRAW + MAIN
# ============================================================

def _col(txt: str) -> tuple[int, int, int]:
    return (0, 255, 0) if txt == "HELMET DETECTED" else (0, 0, 255)


if __name__ == "__main__":
    from modules.detector import VehicleDetector

    detector = VehicleDetector()
    checker = HelmetChecker()

    # stats for main rider
    tot: dict[int, int] = defaultdict(int)
    hel: dict[int, int] = defaultdict(int)
    max_h: dict[int, float] = defaultdict(float)
    area_sum: dict[int, float] = defaultdict(float)
    seen: dict[int, int] = defaultdict(int)

    frame_idx = 0

    for frame, result in detector.process_video(VIDEO_PATH, max_frames=MAX_FRAMES):
        frame_idx += 1
        internal, scores, display = checker.check_frame(frame, result, frame_idx)

        for i, _p in enumerate(result.persons):
            if i not in internal:
                continue
            tot[i] += 1
            seen[i] += 1
            area_sum[i] += float(_bbox_area(result.persons[i]))
            hs, ns = scores[i]
            max_h[i] = max(max_h[i], hs)
            if internal[i] == "helmet":
                hel[i] += 1

        if PRINT_FRAME_LOGS:
            parts = []
            for i, txt in display.items():
                hs, ns = scores.get(i, (0.0, 0.0))
                parts.append(f"p{i}:{txt} (H={hs:.2f},N={ns:.2f})")
            if parts:
                print(f"Frame {frame_idx}: " + " | ".join(parts))

        if SHOW_WINDOW:
            vis = frame.copy()
            for i, person in enumerate(result.persons):
                t = display.get(i, "NO HELMET")
                cv2.rectangle(vis, (person.x1, person.y1), (person.x2, person.y2), _col(t), 2)
                cv2.putText(
                    vis,
                    t,
                    (person.x1, max(22, person.y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    _col(t),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow("Helmet", vis)
            if cv2.waitKey(1) == ord("q"):
                break

    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    print("\n========== FINAL (VIDEO) ==========")
    if not tot:
        print("No paired riders in this video.")
        raise SystemExit(0)

    # pick main rider: largest average area
    best_pid = max(tot.keys(), key=lambda pid: (area_sum[pid] / max(1, seen[pid])))

    T = max(1, tot[best_pid])
    frac = hel[best_pid] / float(T)

    v_final = "NO HELMET"
    if FINAL_HELMET_IF_ANY_INTERNAL_HELMET and hel[best_pid] >= 1:
        v_final = "HELMET DETECTED"
    elif max_h[best_pid] >= FINAL_HELMET_IF_MAX_H:
        v_final = "HELMET DETECTED"
    elif frac >= FINAL_HELMET_IF_FRACTION:
        v_final = "HELMET DETECTED"

    print(f"Main rider index: {best_pid}")
    print(f"Frames: {T}, helmet frames: {hel[best_pid]}, max H: {max_h[best_pid]:.3f}, helmet fraction: {frac:.2%}")
    print(f"Final video result: {v_final}")