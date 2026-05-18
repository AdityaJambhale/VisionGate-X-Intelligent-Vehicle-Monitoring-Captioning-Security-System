"""
VisionGate X — Live campus captioning (YOLO + Helmet heuristic + EasyOCR plate)

Caption example:
  Blue motorcycle, rider without helmet, number plate AB12CD3456 — entered campus at 03:45:12 PM

Requires: ultralytics, opencv-python, easyocr, numpy
Optional DB: sqlite3 (stdlib)
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from ultralytics import YOLO

try:
    import easyocr
except ImportError as e:
    raise SystemExit("Install EasyOCR: pip install easyocr") from e


# ═══════════════════════════════════════════════════════════════
# CONFIG — edit paths
# ═══════════════════════════════════════════════════════════════

VIDEO_PATH = os.path.join("data", "sample_videos", "bike4.mp4")
MODEL_PATH = os.path.join("models", "yolov8n.pt")

CONFIDENCE = 0.35
FRAME_SKIP_OCR = 4          # run plate OCR every N processed frames (speed)
FRAME_SKIP_HELMET = 2       # helmet score every N processed frames
PROCESS_EVERY_N_FRAMES = 1  # 1 = all frames; 2 = skip every other frame for speed

SHOW_WINDOW = True
USE_SQLITE = True
SQLITE_DB = os.path.join("outputs", "campus_events.db")

# COCO ids
PERSON_CLASS = 0
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Loose plate pattern (Indian-style compact); your ANPR normalizer can be stricter later
PLATE_RE = re.compile(r"^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$|^[A-Z]{2}\d{4}[A-Z]{1,2}$")


# ═══════════════════════════════════════════════════════════════
# Geometry helpers
# ═══════════════════════════════════════════════════════════════

def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(1, (ax2 - ax1) * (ay2 - ay1))
    ba = max(1, (bx2 - bx1) * (by2 - by1))
    union = aa + ba - inter
    return inter / union


def head_crop(frame: np.ndarray, xyxy: tuple[int, int, int, int]) -> np.ndarray | None:
    x1, y1, x2, y2 = xyxy
    H, W = frame.shape[:2]
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H, y2))
    w, h = x2 - x1, y2 - y1
    if w < 10 or h < 10:
        return None
    hh = max(int(h * 0.42), 12)
    top = y1
    bot = min(y2, y1 + hh)
    cx = (x1 + x2) // 2
    hw = max(int(w * 0.38), 8)
    left = max(x1, cx - hw)
    right = min(x2, cx + hw)
    if right <= left or bot <= top:
        return None
    c = frame[top:bot, left:right]
    return c if c.size > 0 else None


def plate_roi_motorcycle(frame: np.ndarray, xyxy: tuple[int, int, int, int]) -> np.ndarray | None:
    """Lower-central band of motorcycle box (front plate)."""
    x1, y1, x2, y2 = xyxy
    H, W = frame.shape[:2]
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H, y2))
    w, h = x2 - x1, y2 - y1
    if w < 20 or h < 20:
        return None
    rx1 = int(x1 + w * 0.22)
    rx2 = int(x1 + w * 0.78)
    ry1 = int(y1 + h * 0.38)
    ry2 = int(y1 + h * 0.68)
    roi = frame[ry1:ry2, rx1:rx2]
    return roi if roi is not None and roi.size > 0 else None


def dominant_vehicle_color_name(bgr: np.ndarray) -> str:
    if bgr is None or bgr.size == 0:
        return "unknown colour"
    s = cv2.resize(bgr, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    mask = (sat > 40) & (val > 40)
    if np.count_nonzero(mask) < 30:
        # dark / grey
        v_mean = float(np.mean(val))
        if v_mean < 60:
            return "black"
        return "grey or white"

    hp = h[mask]

    def frac(cond: np.ndarray) -> float:
        return float(np.mean(cond))

    # H in OpenCV: 0–180. Ranges are approximate.
    if frac((hp < 15) | (hp > 160)) > 0.38:
        return "red"
    if frac((hp >= 35) & (hp <= 85)) > 0.35:
        return "green"
    if frac((hp >= 90) & (hp <= 125)) > 0.32:
        return "blue"
    if frac((hp >= 20) & (hp <= 35)) > 0.35:
        return "yellow or gold"
    if frac(hp < 25) > 0.4 and float(np.mean(val[mask])) > 180:
        return "white"
    if float(np.mean(val[mask])) < 80:
        return "black"

    return "coloured"


# ═══════════════════════════════════════════════════════════════
# Helmet heuristic (fast)
# ═══════════════════════════════════════════════════════════════

def skin_ratio(bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 14, 30], np.uint8), np.array([30, 230, 255], np.uint8))
    m2 = cv2.inRange(hsv, np.array([160, 14, 30], np.uint8), np.array([180, 230, 255], np.uint8))
    m = cv2.bitwise_or(m1, m2)
    return float(np.count_nonzero(m)) / float(max(1, m.size))


def edge_density(bgr: np.ndarray) -> float:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    e = cv2.Canny(g, 55, 140)
    return float(np.count_nonzero(e)) / float(max(1, e.size))


def helmet_scores(bgr: np.ndarray) -> tuple[float, float]:
    s = skin_ratio(bgr)
    edg = edge_density(bgr)
    small = cv2.resize(bgr, (40, 40), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].reshape(-1)
    hist, _ = np.histogram(h, bins=18, range=(0, 180))
    p = hist.astype(np.float32) / max(1.0, float(hist.sum()))
    nz = p[p > 0]
    ent = -np.sum(nz * np.log2(nz))
    uni = 1.0 - float(ent / np.log2(18.0))

    hs = 0.48 * (1.0 - s) + 0.28 * min(1.0, edg * 5.0) + 0.24 * uni
    ns = 0.55 * s + 0.25 * (1.0 - uni)
    hs = float(max(0, min(1, hs)))
    ns = float(max(0, min(1, ns)))
    return hs, ns


def helmet_label_from_crop(crop: np.ndarray | None) -> str:
    if crop is None or crop.size == 0:
        return "helmet status unknown"
    hs, ns = helmet_scores(crop)
    if hs >= 0.45 and hs > ns:
        return "helmet on"
    if ns >= 0.48 and ns > hs:
        return "no helmet"
    # default strict campus rule: unknown → treat as no helmet for messaging
    if hs > ns:
        return "helmet on"
    return "no helmet"


# ═══════════════════════════════════════════════════════════════
# Plate OCR + clean
# ═══════════════════════════════════════════════════════════════

def alnum_upper(t: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", t.upper())


def best_plate_string(texts: list[str]) -> str:
    best = ""
    best_s = -1
    for t in texts:
        s = alnum_upper(t)
        for mlen in (9, 10, 11, 8):
            if len(s) >= mlen:
                for st in range(0, len(s) - mlen + 1):
                    cand = s[st : st + mlen]
                    if PLATE_RE.match(cand):
                        score = mlen + (0.5 if cand[:2].isalpha() else 0)
                        if score > best_s:
                            best_s = score
                            best = cand
    return best


@dataclass
class PlateTracker:
    last_text: str = ""
    last_time: float = 0.0

    def update(self, reader: "easyocr.Reader", roi_bgr: np.ndarray | None, now: float) -> str:
        if roi_bgr is None or roi_bgr.size == 0:
            return self.last_text

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # upscale small plate
        h, w = thr.shape[:2]
        if w < 220:
            thr = cv2.resize(thr, (int(w * 1.8), int(h * 1.8)), interpolation=cv2.INTER_CUBIC)

        out = reader.readtext(thr, detail=0, paragraph=False)
        joined = out if isinstance(out, list) else [str(out)]
        cand = best_plate_string(joined)
        if cand:
            self.last_text = cand
            self.last_time = now
        return self.last_text


def ensure_db(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS campus_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            caption TEXT,
            plate TEXT,
            vehicle TEXT,
            colour TEXT,
            helmet TEXT
        )
        """
    )
    con.commit()
    con.close()


def log_event(db_path: str, caption: str, plate: str, vehicle: str, colour: str, helmet: str) -> None:
    con = sqlite3.connect(db_path)
    con.execute(
        "INSERT INTO campus_events (ts, caption, plate, vehicle, colour, helmet) VALUES (?,?,?,?,?,?)",
        (datetime.now().isoformat(timespec="seconds"), caption, plate, vehicle, colour, helmet),
    )
    con.commit()
    con.close()


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    print("Loading YOLO…")
    model = YOLO(MODEL_PATH)

    print("Loading EasyOCR (first run may download models)…")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {VIDEO_PATH}")

    if USE_SQLITE:
        ensure_db(SQLITE_DB)

    plate_tracker = PlateTracker()
    last_caption_full = ""
    last_log_t = 0.0

    processed = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed += 1
        if processed % PROCESS_EVERY_N_FRAMES != 0:
            continue

        results = model.predict(frame, conf=CONFIDENCE, verbose=False)

        people: list[tuple[tuple[int, int, int, int], float]] = []
        vehicles: list[tuple[str, tuple[int, int, int, int], float]] = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                xyxy = (x1, y1, x2, y2)
                if cls == PERSON_CLASS:
                    people.append((xyxy, conf))
                elif cls in VEHICLE_CLASSES:
                    vehicles.append((VEHICLE_CLASSES[cls], xyxy, conf))

        # Pair motorcycle with best overlapping person
        moto = next(((bb, cf) for label, bb, cf in vehicles if label == "motorcycle"), None)

        rider_bb = None
        rider_conf = 0.0
        if moto is not None:
            mbb, _ = moto
            best_iou = 0.0
            for pbb, pcf in people:
                j = iou(mbb, pbb)
                if j > best_iou:
                    best_iou = j
                    rider_bb = pbb
                    rider_conf = pcf

        # Helmet + colour + plate
        now = time.time()
        colour_txt = "unknown colour"
        hel_txt = "no helmet"
        plate_txt = plate_tracker.last_text

        if moto is not None:
            mbb, mcf = moto
            crop_v = frame[mbb[1] : mbb[3], mbb[0] : mbb[2]]
            colour_txt = dominant_vehicle_color_name(crop_v)

            if rider_bb is not None and processed % FRAME_SKIP_HELMET == 0:
                hc = head_crop(frame, rider_bb)
                hel_txt = helmet_label_from_crop(hc)

            if processed % FRAME_SKIP_OCR == 0:
                proi = plate_roi_motorcycle(frame, mbb)
                plate_txt = plate_tracker.update(reader, proi, now)

            # Draw
            cv2.rectangle(frame, (mbb[0], mbb[1]), (mbb[2], mbb[3]), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"motorcycle {mcf:.2f}",
                (mbb[0], max(20, mbb[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            if rider_bb is not None:
                col = (255, 120, 0) if hel_txt == "no helmet" else (255, 0, 0)
                cv2.rectangle(frame, (rider_bb[0], rider_bb[1]), (rider_bb[2], rider_bb[3]), col, 2)
                cv2.putText(
                    frame,
                    f"rider {rider_conf:.2f}",
                    (rider_bb[0], max(20, rider_bb[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    col,
                    2,
                )

        # Build caption
        clock = datetime.now().strftime("%I:%M:%S %p")
        if moto is None:
            caption = f"Monitoring campus entrance… [{clock}]"
        else:
            vname = "motorcycle"
            ctitle = colour_txt.title()
            if hel_txt == "helmet on":
                hhuman = "with rider wearing a helmet"
            elif hel_txt == "no helmet":
                hhuman = "with rider without a helmet"
            else:
                hhuman = "with rider (helmet unclear)"

            pl = plate_txt or "number plate not read yet"
            caption = (
                f"{ctitle} {vname} {hhuman}, number plate {pl} — "
                f"entered campus at {clock}"
            )

        last_caption_full = caption

        # Footer panel
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (0, frame.shape[0] - 110),
            (frame.shape[1], frame.shape[0]),
            (20, 24, 32),
            -1,
        )
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        y0 = frame.shape[0] - 88
        for i, line in enumerate(_wrap_caption(caption, max_chars=72)):
            cv2.putText(
                frame,
                line,
                (16, y0 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (230, 240, 255),
                2,
                cv2.LINE_AA,
            )

        if SHOW_WINDOW:
            cv2.imshow("VisionGate X — Live campus caption", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Log ~1 Hz when we have a motorcycle
        if USE_SQLITE and moto is not None and (now - last_log_t) > 1.0:
            log_event(SQLITE_DB, caption, plate_txt or "", "motorcycle", colour_txt, hel_txt)
            last_log_t = now

    cap.release()
    cv2.destroyAllWindows()
    print("\nDone.")
    print("Last caption:", last_caption_full)
    print(f"Processed frames (after skip): {processed}, time {time.time() - t0:.1f}s")


def _wrap_caption(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_chars:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines[:3]


# ─────────────────────────────────────────────────────────────
# Pipeline API (imported by modules/__init__.py → pipeline.py)
# ─────────────────────────────────────────────────────────────


class CaptionGenerator:
    """
    Short template captions for each vehicle row.
    Used by pipeline.Pipeline — not the same as live_campus_caption.py runner.
    """

    def caption_vehicle(
        self,
        vehicle,
        plate: str = "",
        helmet_status: str = "unknown",
    ) -> str:
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import config as app_config

        label = getattr(vehicle, "label", "vehicle")
        veh_lc = label.lower()
        noun = veh_lc.title() if veh_lc else "Vehicle"
        gate = getattr(app_config, "CHECKPOINT_LABEL", "the checkpoint")

        pl_raw = (plate or "").strip()
        pl_disp = pl_raw.upper() if pl_raw else None

        # helmet_status passed in is DB-normalised: helmet | no_helmet (unknown folded server-side).
        hs = (helmet_status or "no_helmet").lower()

        if hs == "helmet":
            if pl_disp:
                return (
                    f"{gate} · Live ingress: {noun} carrying plate «{pl_disp}». "
                    "Helmet seen in-frame at least once — rider marked compliant for this passage."
                )
            return (
                f"{gate} · Live ingress: {noun} (registration not captured). "
                "Helmet seen in-frame — rider marked compliant for this passage."
            )
        if pl_disp:
            return (
                f"{gate} · Priority flag: {noun} tagged «{pl_disp}». "
                "Helmet absent or unverified end-to-end — treated as non-compliant and logged."
            )
        return (
            f"{gate} · Priority flag: {noun} with no reliable plate read. "
            "Helmet absent or unverified — treated as non-compliant and logged."
        )

    def caption_violation(
        self,
        plate: str,
        violation_desc: str,
        fine_inr: int,
    ) -> str:
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import config as app_config

        gate = getattr(app_config, "CHECKPOINT_LABEL", "the checkpoint")
        pl = (plate or "").strip().upper() or "UNKNOWN"
        return (
            f"{gate} · E-challan draft queued for plate {pl}: {violation_desc}. "
            f"Suggested penalty ₹{int(fine_inr):,}; PDF issued for examiner workflow."
        )


if __name__ == "__main__":
    main()