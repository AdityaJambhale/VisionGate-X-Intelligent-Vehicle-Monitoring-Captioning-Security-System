"""
VisionGate X - ANPR (Robust + Faster + Multi-format)
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import os
import re
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from modules.detector import BBox


# ============================================================
# CONFIG
# ============================================================

FAST_MODE = True                 # True => faster, False => higher recall
MAX_FRAMES = 350
DEBUG_SAVE = False

FRAME_SKIP = 2 if FAST_MODE else 1
MIN_CONF = 0.08 if FAST_MODE else 0.05
EARLY_ACCEPT_CONF = 0.72 if FAST_MODE else 0.82
TRACK_TTL_SEC = 1.5              # track memory lifetime


# ============================================================
# PLATE RULES
# ============================================================

# Format A: LLDDL{1..3}DDDD
RE_A = re.compile(r"^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$")
# Format B: LLDDDDL{1..2}
RE_B = re.compile(r"^[A-Z]{2}\d{4}[A-Z]{1,2}$")

VALID_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN",
    "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD",
    "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ",
    "SK", "TN", "TR", "TS", "UK", "UP", "WB",
}

# Extra prefixes from your dataset/videos.
PROJECT_VALID_PREFIXES = {
    "AB",
}

LETTER_TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "Z": "2",
    "A": "4",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
}
DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
}


def _alnum_upper(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper().strip())


def _to_letter(ch: str) -> tuple[str, float]:
    if ch.isalpha():
        return ch, 0.0
    if ch in DIGIT_TO_LETTER:
        return DIGIT_TO_LETTER[ch], 0.14
    return ch, 0.6


def _to_digit(ch: str) -> tuple[str, float]:
    if ch.isdigit():
        return ch, 0.0
    if ch in LETTER_TO_DIGIT:
        return LETTER_TO_DIGIT[ch], 0.14
    return ch, 0.6


def _state_penalty(plate: str) -> float:
    prefix = plate[:2]
    if prefix in VALID_STATE_CODES or prefix in PROJECT_VALID_PREFIXES:
        return -0.30
    return 0.55


def _fix_prefix_confusion(plate: str) -> str:
    """
    Fix common OCR confusion in the first 2 letters.
    """
    if len(plate) < 2:
        return plate

    p0, p1 = plate[0], plate[1]

    # OB / 0B commonly misread when true prefix is AB.
    if p0 in {"O", "0"} and p1 == "B":
        return "AB" + plate[2:]

    return plate


def _normalize_a(raw: str) -> tuple[str, float]:
    s = _alnum_upper(raw)
    if len(s) < 8:
        return "", 99.0
    if len(s) > 13:
        s = s[:13]

    best_p = ""
    best_pen = 99.0

    for m in (1, 2, 3):
        n = 8 + m
        if len(s) < n:
            continue
        for st in range(0, len(s) - n + 1):
            t = s[st:st + n]
            out = []
            pen = 0.0

            for ch in t[0:2]:
                x, p = _to_letter(ch)
                out.append(x)
                pen += p
            for ch in t[2:4]:
                x, p = _to_digit(ch)
                out.append(x)
                pen += p
            for ch in t[4:4 + m]:
                x, p = _to_letter(ch)
                out.append(x)
                pen += p
            for ch in t[4 + m:8 + m]:
                x, p = _to_digit(ch)
                out.append(x)
                pen += p

            plate = "".join(out)
            if not RE_A.fullmatch(plate):
                continue

            pen += _state_penalty(plate)
            if len(plate) == 10:
                pen -= 0.06
            if plate[-4:].isdigit():
                pen -= 0.02

            if pen < best_pen:
                best_pen = pen
                best_p = plate

    return best_p, best_pen


def _normalize_b(raw: str) -> tuple[str, float]:
    s = _alnum_upper(raw)
    if len(s) < 7:
        return "", 99.0
    if len(s) > 11:
        s = s[:11]

    best_p = ""
    best_pen = 99.0

    for m in (1, 2):
        n = 6 + m
        if len(s) < n:
            continue
        for st in range(0, len(s) - n + 1):
            t = s[st:st + n]
            out = []
            pen = 0.0

            for ch in t[0:2]:
                x, p = _to_letter(ch)
                out.append(x)
                pen += p
            for ch in t[2:6]:
                x, p = _to_digit(ch)
                out.append(x)
                pen += p
            for ch in t[6:6 + m]:
                x, p = _to_letter(ch)
                out.append(x)
                pen += p

            plate = "".join(out)
            if not RE_B.fullmatch(plate):
                continue

            pen += _state_penalty(plate)
            if len(plate) == 8:
                pen -= 0.04

            if pen < best_pen:
                best_pen = pen
                best_p = plate

    return best_p, best_pen


def _normalize_best(raw: str) -> tuple[str, float]:
    pa, pena = _normalize_a(raw)
    pb, penb = _normalize_b(raw)

    if pa and pb:
        plate, pen = (pa, pena) if pena <= penb else (pb, penb)
    elif pa:
        plate, pen = pa, pena
    elif pb:
        plate, pen = pb, penb
    else:
        return "", 99.0

    fixed = _fix_prefix_confusion(plate)
    if fixed != plate:
        pen -= 0.10

    return fixed, pen


def _valid_plate(p: str) -> bool:
    return bool(RE_A.fullmatch(p) or RE_B.fullmatch(p))


# ============================================================
# TRACK MEMORY
# ============================================================

@dataclass
class TrackMem:
    bbox: BBox
    last_seen: float
    candidates: deque = field(default_factory=lambda: deque(maxlen=200))   # (plate, conf, pen)
    raw: deque = field(default_factory=lambda: deque(maxlen=300))          # (raw, conf)


def _iou(a: BBox, b: BBox) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    aa = max(1, (a.x2 - a.x1) * (a.y2 - a.y1))
    bb = max(1, (b.x2 - b.x1) * (b.y2 - b.y1))
    return inter / float(aa + bb - inter)


# ============================================================
# ROI + PREPROCESS
# ============================================================

def _extract_plate_rois(frame: np.ndarray, veh: BBox) -> list[np.ndarray]:
    h = veh.y2 - veh.y1
    w = veh.x2 - veh.x1
    H, W = frame.shape[:2]

    windows = [
        (0.12, 0.88, 0.32, 0.90),
        (0.18, 0.82, 0.40, 0.86),
        (0.24, 0.76, 0.50, 0.80),
    ]
    if FAST_MODE:
        windows = windows[:2]

    rois: list[np.ndarray] = []

    for rx1, rx2, ry1, ry2 in windows:
        x1 = max(0, int(veh.x1 + w * rx1))
        x2 = min(W, int(veh.x1 + w * rx2))
        y1 = max(0, int(veh.y1 + h * ry1))
        y2 = min(H, int(veh.y1 + h * ry2))
        if x2 > x1 and y2 > y1:
            crop = frame[y1:y2, x1:x2]
            if crop is not None and crop.size > 0:
                rois.append(crop)

    bx1 = max(0, int(veh.x1 + w * 0.15))
    bx2 = min(W, int(veh.x1 + w * 0.85))
    by1 = max(0, int(veh.y1 + h * 0.35))
    by2 = min(H, int(veh.y1 + h * 0.92))
    base = frame[by1:by2, bx1:bx2]
    if base is not None and base.size > 0:
        g = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        e = cv2.Canny(g, 60, 160)
        cnts, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        proposals = []
        for c in cnts:
            x, y, cw, ch = cv2.boundingRect(c)
            ar = cw / float(max(ch, 1))
            area = cw * ch
            if 2.0 <= ar <= 7.5 and area > 500:
                proposals.append((area, x, y, cw, ch))
        proposals.sort(reverse=True, key=lambda t: t[0])

        for _, x, y, cw, ch in proposals[:3]:
            pad_x = int(cw * 0.12)
            pad_y = int(ch * 0.35)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(base.shape[1], x + cw + pad_x)
            y2 = min(base.shape[0], y + ch + pad_y)
            c = base[y1:y2, x1:x2]
            if c is not None and c.size > 0:
                rois.append(c)

    return rois


def _preprocess_variants(crop: np.ndarray) -> list[np.ndarray]:
    if crop is None or crop.size == 0:
        return []

    h, w = crop.shape[:2]

    if w < 520:
        scale = 620.0 / max(w, 1)
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 35, 35)

    ksharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharp = cv2.filter2D(gray, -1, ksharp)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(sharp)

    th_bin = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    th_inv = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    th_adp = cv2.adaptiveThreshold(
        clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9
    )

    k = np.ones((2, 2), np.uint8)
    th_bin = cv2.morphologyEx(th_bin, cv2.MORPH_CLOSE, k, iterations=1)
    th_inv = cv2.morphologyEx(th_inv, cv2.MORPH_CLOSE, k, iterations=1)
    th_adp = cv2.morphologyEx(th_adp, cv2.MORPH_CLOSE, k, iterations=1)

    if DEBUG_SAVE:
        os.makedirs("outputs", exist_ok=True)
        cv2.imwrite("outputs/dbg_gray.jpg", gray)
        cv2.imwrite("outputs/dbg_bin.jpg", th_bin)
        cv2.imwrite("outputs/dbg_inv.jpg", th_inv)
        cv2.imwrite("outputs/dbg_adp.jpg", th_adp)

    if FAST_MODE:
        return [clahe, th_bin]
    return [clahe, th_bin, th_inv, th_adp]


# ============================================================
# ANPR
# ============================================================

class ANPRReader:
    def __init__(self):
        self._reader = None
        self._frame_idx = 0
        self._next_tid = 1
        self._tracks: dict[int, TrackMem] = {}

        # Persistent memories so final decision survives track cleanup.
        self._global_candidates = deque(maxlen=2000)  # (plate, conf, pen)
        self._global_raw = deque(maxlen=3000)         # (raw, conf)

    @property
    def reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(
                config.OCR_LANGUAGES,
                gpu=False,
                verbose=False
            )
        return self._reader

    def _assign_track(self, veh: BBox) -> int:
        best_tid = -1
        best_iou = 0.0
        for tid, mem in self._tracks.items():
            i = _iou(veh, mem.bbox)
            if i > best_iou:
                best_iou = i
                best_tid = tid

        if best_tid != -1 and best_iou >= 0.25:
            self._tracks[best_tid].bbox = veh
            self._tracks[best_tid].last_seen = time.time()
            return best_tid

        tid = self._next_tid
        self._next_tid += 1
        self._tracks[tid] = TrackMem(bbox=veh, last_seen=time.time())
        return tid

    def _cleanup_tracks(self):
        now = time.time()
        dead = [tid for tid, m in self._tracks.items() if now - m.last_seen > TRACK_TTL_SEC]
        for tid in dead:
            del self._tracks[tid]

    def _ocr_vehicle(self, frame: np.ndarray, veh: BBox, tid: int) -> tuple[str, float]:
        mem = self._tracks[tid]

        best_plate = ""
        best_conf = 0.0
        best_pen = 99.0
        best_score = -999.0

        rois = _extract_plate_rois(frame, veh)
        if not rois:
            return "", 0.0

        for roi in rois:
            for img in _preprocess_variants(roi):
                out = self.reader.readtext(
                    img,
                    detail=1,
                    paragraph=False,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    width_ths=0.7,
                    height_ths=0.7,
                    text_threshold=0.5,
                    low_text=0.2,
                    link_threshold=0.3,
                )

                for (_, txt, conf) in out:
                    conf = float(conf)
                    raw = _alnum_upper(txt)
                    if not raw:
                        continue

                    if len(raw) >= 7:
                        mem.raw.append((raw, conf))
                        self._global_raw.append((raw, conf))

                    if conf < MIN_CONF:
                        continue

                    plate, pen = _normalize_best(raw)
                    if not plate:
                        continue

                    score = conf - pen
                    if score > best_score:
                        best_score = score
                        best_plate = plate
                        best_conf = conf
                        best_pen = pen

                    if best_conf >= EARLY_ACCEPT_CONF and best_pen < 0.38:
                        mem.candidates.append((best_plate, best_conf, best_pen))
                        self._global_candidates.append((best_plate, best_conf, best_pen))
                        return best_plate, best_conf

        if best_plate:
            mem.candidates.append((best_plate, best_conf, best_pen))
            self._global_candidates.append((best_plate, best_conf, best_pen))
            return best_plate, best_conf

        return "", 0.0

    def read_plates_in_frame(self, frame: np.ndarray, vehicles: list[BBox]) -> dict[int, tuple[str, float]]:
        self._frame_idx += 1
        do_ocr = (self._frame_idx % FRAME_SKIP == 0)
        self._cleanup_tracks()

        out: dict[int, tuple[str, float]] = {}

        for i, veh in enumerate(vehicles):
            if veh.label not in {"car", "motorcycle", "bus", "truck"}:
                out[i] = ("", 0.0)
                continue

            tid = self._assign_track(veh)

            if not do_ocr:
                out[i] = ("", 0.0)
                continue

            plate, conf = self._ocr_vehicle(frame, veh, tid)
            out[i] = (plate, conf)

        return out

    def _final_from_track(self, mem: TrackMem) -> str:
        if mem.candidates:
            wmap = defaultdict(float)
            for p, c, pen in mem.candidates:
                w = max(0.01, c) * (1.28 - min(1.0, pen))
                if _valid_plate(p):
                    w += 0.1
                if p[:2] in VALID_STATE_CODES or p[:2] in PROJECT_VALID_PREFIXES:
                    w += 0.2
                wmap[p] += w

            top = sorted(wmap.items(), key=lambda kv: kv[1], reverse=True)[:8]
            if top:
                if len(top) == 1 or top[0][1] >= 1.30 * top[1][1]:
                    return top[0][0]

                by_len = defaultdict(list)
                for p, w in top:
                    by_len[len(p)].append((p, w))
                best_len = max(by_len.items(), key=lambda kv: sum(x[1] for x in kv[1]))[0]
                group = by_len[best_len]

                fused = []
                for j in range(best_len):
                    bucket = defaultdict(float)
                    for p, w in group:
                        bucket[p[j]] += w
                    fused.append(max(bucket.items(), key=lambda kv: kv[1])[0])

                cand = "".join(fused)
                if _valid_plate(cand):
                    return cand

                for p, _w in top:
                    if _valid_plate(p):
                        return p
                return top[0][0]

        raw_map = defaultdict(float)
        for raw, conf in mem.raw:
            p, pen = _normalize_best(raw)
            if p:
                raw_map[p] += max(0.01, conf) * (1.08 - min(0.95, pen))

        if raw_map:
            return max(raw_map.items(), key=lambda kv: kv[1])[0]

        return ""

    def final_plate(self) -> str:
        # Primary global vote
        if self._global_candidates:
            wmap = defaultdict(float)
            for p, c, pen in self._global_candidates:
                w = max(0.01, c) * (1.30 - min(1.0, pen))
                if _valid_plate(p):
                    w += 0.20
                if p[:2] in VALID_STATE_CODES or p[:2] in PROJECT_VALID_PREFIXES:
                    w += 0.35
                wmap[p] += w

            if wmap:
                return max(wmap.items(), key=lambda kv: kv[1])[0]

        # Global raw fallback
        if self._global_raw:
            raw_map = defaultdict(float)
            for raw, conf in self._global_raw:
                p, pen = _normalize_best(raw)
                if p:
                    raw_map[p] += max(0.01, conf) * (1.08 - min(0.95, pen))
            if raw_map:
                return max(raw_map.items(), key=lambda kv: kv[1])[0]

        # Fallback from active tracks
        best_plate = ""
        best_score = -1.0
        for _tid, mem in self._tracks.items():
            p = self._final_from_track(mem)
            if not p:
                continue
            score = len(mem.candidates) + 0.2 * len(mem.raw)
            if p[:2] in VALID_STATE_CODES or p[:2] in PROJECT_VALID_PREFIXES:
                score += 1.0
            if score > best_score:
                best_score = score
                best_plate = p
        return best_plate


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    from modules.detector import VehicleDetector

    detector = VehicleDetector()
    anpr = ANPRReader()

    video_path = "data/sample_videos/bike1.mp4"

    for frame, result in detector.process_video(video_path, max_frames=MAX_FRAMES):
        plate_results = anpr.read_plates_in_frame(frame, result.vehicles)
        plate_texts = {}

        for idx, (plate, conf) in plate_results.items():
            if plate:
                print(f"Detected: {plate} | Conf: {conf:.2f}")
                plate_texts[idx] = plate

        annotated = detector.draw_boxes(frame, result, plate_texts=plate_texts)
        cv2.imshow("VisionGate X ANPR", annotated)

        if cv2.waitKey(1 if FAST_MODE else 20) == ord("q"):
            break

    cv2.destroyAllWindows()

    print("\n===== FINAL DETECTED PLATE =====")
    final = anpr.final_plate()
    print(final if final else "No plate detected")