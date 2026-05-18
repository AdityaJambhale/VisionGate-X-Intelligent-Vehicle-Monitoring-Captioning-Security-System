"""
VisionGate X — Live campus caption (integrated)

Uses YOUR pipeline modules only:
  - modules.detector.VehicleDetector  (YOLOv8, same as rest of app)
  - modules.helmet.HelmetChecker      (same helmet logic as helmet.py)
  - modules.anpr.ANPRReader           (same plate OCR + voting as anpr.py)

Caption example:
  Blue motorcycle — rider: HELMET DETECTED — plate JK01BB9740 — campus entry 03:45:12 PM

Run from project root (visiongate_x):
  python live_campus_caption.py
  python modules/live_campus_caption.py

Edit VIDEO_PATH below.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime

import cv2
import numpy as np

# Project root on path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modules.anpr import ANPRReader, _valid_plate
from modules.detector import BBox, DetectionResult, VehicleDetector
from modules.helmet import HelmetChecker


# ─────────────────────────────────────────
# USER SETTINGS
# ─────────────────────────────────────────

VIDEO_PATH = os.path.join(_ROOT, "data", "sample_videos", "bike1.mp4")
MAX_FRAMES = 0  # 0 = full video
SHOW_WINDOW = True
PRINT_LIVE_CAPTIONS = True
VERBOSE_DEBUG = False
PRINT_EVERY_N_FRAMES = 15
USE_SQLITE = True
SQLITE_DB = os.path.join(_ROOT, "outputs", "campus_events.db")

# Gate session (bike in frame): commit after MIN / samples / MAX without waiting for bike to leave.
CAPTION_COMMIT_MIN_SEC = 0.75
CAPTION_COMMIT_MAX_SEC = 2.0
PLATE_SAMPLES_BEFORE_COMMIT = 5
SESSION_IDLE_FRAMES = 12


# ─────────────────────────────────────────
# Colour label (visual only; independent of YOLO)
# ─────────────────────────────────────────

def dominant_vehicle_color_name(bgr: np.ndarray) -> str:
    if bgr is None or bgr.size == 0:
        return "unknown-colour"
    s = cv2.resize(bgr, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    mask = (sat > 40) & (val > 40)
    if np.count_nonzero(mask) < 30:
        v_mean = float(np.mean(val))
        return "black" if v_mean < 60 else "grey-or-white"
    hp = h[mask]

    def frac(cond: np.ndarray) -> float:
        return float(np.mean(cond))

    if frac((hp < 15) | (hp > 160)) > 0.38:
        return "red"
    if frac((hp >= 35) & (hp <= 85)) > 0.35:
        return "green"
    if frac((hp >= 90) & (hp <= 125)) > 0.32:
        return "blue"
    if frac((hp >= 20) & (hp <= 35)) > 0.35:
        return "yellow-or-gold"
    if frac(hp < 25) > 0.4 and float(np.mean(val[mask])) > 180:
        return "white"
    if float(np.mean(val[mask])) < 80:
        return "black"
    return "coloured"


def _pick_primary_motorcycle(detection: DetectionResult) -> tuple[int, BBox] | None:
    motos = [(i, v) for i, v in enumerate(detection.vehicles) if v.label == "motorcycle"]
    if not motos:
        return None
    return max(motos, key=lambda iv: iv[1].width * iv[1].height)


def _rider_idx_for_vehicle(detection: DetectionResult, moto: BBox) -> int | None:
    for person, veh in detection.rider_vehicle_pairs():
        if (
            abs(veh.x1 - moto.x1) < 8
            and abs(veh.y1 - moto.y1) < 8
            and abs(veh.x2 - moto.x2) < 8
            and abs(veh.y2 - moto.y2) < 8
        ):
            try:
                return detection.persons.index(person)
            except ValueError:
                return None
    return None


def _wrap_lines(text: str, max_chars: int) -> list[str]:
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
    return lines[:4]


def _helmet_for_draw(internal: dict[int, str], i: int) -> str:
    s = internal.get(i, "unknown")
    if s in ("helmet", "no_helmet", "unknown"):
        return s
    return "unknown"


def _colour_display_name(raw: str) -> str:
    if not raw:
        return "Unknown-colour"
    return raw.replace("-", " ").strip().title()


def _helmet_with_without(display: str) -> str:
    return "with a helmet" if display == "HELMET DETECTED" else "without a helmet"


def build_gate_caption(
    *,
    has_bike: bool,
    colour_key: str,
    plate_show: str,
    h_ui: str,
    clock: str,
) -> str:
    if not has_bike:
        return f"Gate idle — no bike in view at {clock}."

    colour_pretty = _colour_display_name(colour_key)
    helmet_phrase = _helmet_with_without(h_ui)
    plate_txt = (plate_show or "").strip()

    if plate_txt:
        return (
            f"{colour_pretty} bike entered the gate with number plate {plate_txt} "
            f"{helmet_phrase} at {clock}."
        )
    return (
        f"{colour_pretty} bike entered the gate; number plate still reading — "
        f"{helmet_phrase} at {clock}."
    )


def _majority_helmet(values: list[str]) -> str:
    if not values:
        return "NO HELMET"
    return Counter(values).most_common(1)[0][0]


def _majority_colour(values: list[str]) -> str:
    if not values:
        return "unknown-colour"
    return Counter(values).most_common(1)[0][0]


def _pick_stable_plate(anpr: ANPRReader, samples: list[str]) -> str:
    fp = (anpr.final_plate() or "").strip()
    if fp:
        return fp
    if not samples:
        return ""
    c = Counter(samples)
    valid_ps = [p for p in c if _valid_plate(p)]
    if valid_ps:
        return max(valid_ps, key=lambda p: c[p])
    return c.most_common(1)[0][0]


def ensure_db(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS campus_captions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            caption TEXT,
            plate TEXT,
            helmet_ui TEXT,
            colour TEXT
        )
        """
    )
    con.commit()
    con.close()


def log_db(path: str, caption: str, plate: str, helmet_ui: str, colour: str) -> None:
    con = sqlite3.connect(path)
    con.execute(
        "INSERT INTO campus_captions (ts, caption, plate, helmet_ui, colour) VALUES (?,?,?,?,?)",
        (datetime.now().isoformat(timespec="seconds"), caption, plate, helmet_ui, colour),
    )
    con.commit()
    con.close()


def main() -> None:
    if not os.path.isfile(VIDEO_PATH):
        print(f"Video not found: {VIDEO_PATH}")
        sys.exit(1)

    if USE_SQLITE:
        ensure_db(SQLITE_DB)

    print("Loading VehicleDetector (YOLOv8)…")
    detector = VehicleDetector()

    print("Loading HelmetChecker (same as modules/helmet.py)…")
    helmet_chk = HelmetChecker()

    print("Loading ANPRReader (same OCR pipeline as modules/anpr.py)…")
    anpr = ANPRReader()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Cannot open video.")
        sys.exit(1)

    frame_no = 0
    last_caption = ""
    last_log_t = 0.0
    last_live_fp: object = None
    last_db_fp: object = None

    idle_streak = 0
    session_active = False
    session_start = 0.0
    samples_plate: list[str] = []
    samples_colour: list[str] = []
    samples_helmet: list[str] = []
    committed = False
    committed_snapshot: tuple[str, str, str, str] | None = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        if MAX_FRAMES and frame_no > MAX_FRAMES:
            break

        detection = detector.detect_frame(frame, frame_number=frame_no)

        internal, scores, display = helmet_chk.check_frame(frame, detection, frame_idx=frame_no)

        plate_by_veh = anpr.read_plates_in_frame(frame, detection.vehicles)
        global_plate = anpr.final_plate()

        moto_pick = _pick_primary_motorcycle(detection)
        clock = datetime.now().strftime("%I:%M:%S %p")
        just_committed = False
        sqlite_payload: tuple[str, str, str] | None = None
        plate_show = ""
        colour = ""
        h_ui = ""

        if moto_pick is None:
            idle_streak += 1
            if idle_streak >= SESSION_IDLE_FRAMES:
                session_active = False
                committed = False
                committed_snapshot = None
                samples_plate.clear()
                samples_colour.clear()
                samples_helmet.clear()
            caption = build_gate_caption(
                has_bike=False,
                colour_key="",
                plate_show="",
                h_ui="",
                clock=clock,
            )
        else:
            idle_streak = 0
            if not session_active:
                session_active = True
                session_start = time.monotonic()
                samples_plate.clear()
                samples_colour.clear()
                samples_helmet.clear()
                committed = False
                committed_snapshot = None

            moto_idx, moto = moto_pick
            rider_i = _rider_idx_for_vehicle(detection, moto)

            crop_v = frame[moto.y1 : moto.y2, moto.x1 : moto.x2]
            colour = dominant_vehicle_color_name(crop_v)

            plate_frame, _conf = plate_by_veh.get(moto_idx, ("", 0.0))
            plate_show = plate_frame or global_plate or ""

            if rider_i is not None:
                h_ui = display.get(rider_i, "NO HELMET")
                in_st = internal.get(rider_i, "unknown")
            else:
                h_ui = "NO HELMET"
                in_st = "unknown"

            samples_colour.append(colour)
            samples_helmet.append(h_ui)
            if plate_show.strip():
                samples_plate.append(plate_show.strip())

            elapsed = time.monotonic() - session_start
            live_fp = ("bike", colour, plate_show.strip(), h_ui)

            if committed and committed_snapshot is not None:
                p_s, c_s, hu_s, ck_s = committed_snapshot
                caption = build_gate_caption(
                    has_bike=True,
                    colour_key=c_s,
                    plate_show=p_s,
                    h_ui=hu_s,
                    clock=ck_s,
                )
            else:
                can_commit = elapsed >= CAPTION_COMMIT_MIN_SEC and (
                    bool((anpr.final_plate() or "").strip())
                    or len(samples_plate) >= PLATE_SAMPLES_BEFORE_COMMIT
                    or elapsed >= CAPTION_COMMIT_MAX_SEC
                )
                if can_commit:
                    plate_f = _pick_stable_plate(anpr, samples_plate)
                    colour_f = _majority_colour(samples_colour)
                    helmet_f = _majority_helmet(samples_helmet)
                    ck_commit = datetime.now().strftime("%I:%M:%S %p")
                    committed_snapshot = (plate_f, colour_f, helmet_f, ck_commit)
                    committed = True
                    just_committed = True
                    sqlite_payload = (plate_f, helmet_f, colour_f)
                    caption = build_gate_caption(
                        has_bike=True,
                        colour_key=colour_f,
                        plate_show=plate_f,
                        h_ui=helmet_f,
                        clock=ck_commit,
                    )
                    last_live_fp = live_fp
                else:
                    if elapsed < CAPTION_COMMIT_MIN_SEC:
                        rem = CAPTION_COMMIT_MIN_SEC - elapsed
                        caption = (
                            f"Analysing vehicle at gate — hold ~{rem:.1f}s for consolidated readout…"
                        )
                    else:
                        caption = (
                            "Analysing vehicle at gate — consolidating plate, colour, and helmet…"
                        )

            if VERBOSE_DEBUG and (frame_no == 1 or frame_no % PRINT_EVERY_N_FRAMES == 0):
                hs, ns = scores.get(rider_i or 0, (0.0, 0.0))
                print(
                    f"[f{frame_no}] plate={plate_show!r} global={global_plate!r} | "
                    f"helmet_ui={h_ui!r} internal={in_st!r} | H={hs:.2f} N={ns:.2f}"
                )

        if PRINT_LIVE_CAPTIONS and just_committed:
            print(caption, flush=True)

        last_caption = caption

        helmet_draw: dict[int, str] = {}
        for i, _p in enumerate(detection.persons):
            helmet_draw[i] = _helmet_for_draw(internal, i)

        plate_draw: dict[int, str] = {}
        for vi, (txt, _c) in plate_by_veh.items():
            if txt:
                plate_draw[vi] = txt
        if moto_pick and global_plate and moto_pick[0] not in plate_draw:
            plate_draw[moto_pick[0]] = global_plate

        vis = detector.draw_boxes(frame, detection, helmet_statuses=helmet_draw, plate_texts=plate_draw)

        overlay = vis.copy()
        h, w = overlay.shape[:2]
        cv2.rectangle(overlay, (0, h - 120), (w, h), (18, 20, 28), -1)
        cv2.addWeighted(overlay, 0.62, vis, 0.38, 0, vis)

        y0 = h - 102
        for k, line in enumerate(_wrap_lines(caption, max_chars=76)):
            cv2.putText(
                vis,
                line,
                (16, y0 + k * 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (235, 240, 255),
                2,
                cv2.LINE_AA,
            )

        if SHOW_WINDOW:
            cv2.imshow("VisionGate X — Live caption (helmet + ANPR)", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if USE_SQLITE and just_committed and sqlite_payload is not None:
            now = time.time()
            if sqlite_payload != last_db_fp or (now - last_log_t) >= 1.0:
                p_db, hu_db, c_db = sqlite_payload
                log_db(SQLITE_DB, caption, p_db, hu_db, c_db)
                last_db_fp = sqlite_payload
                last_log_t = now

    cap.release()

    if session_active and samples_colour and not committed:
        elapsed = time.monotonic() - session_start
        can_eof = elapsed >= CAPTION_COMMIT_MIN_SEC and (
            bool((anpr.final_plate() or "").strip())
            or len(samples_plate) >= PLATE_SAMPLES_BEFORE_COMMIT
            or elapsed >= CAPTION_COMMIT_MAX_SEC
        )
        if can_eof:
            plate_f = _pick_stable_plate(anpr, samples_plate)
            colour_f = _majority_colour(samples_colour)
            helmet_f = _majority_helmet(samples_helmet)
            ck_commit = datetime.now().strftime("%I:%M:%S %p")
            caption = build_gate_caption(
                has_bike=True,
                colour_key=colour_f,
                plate_show=plate_f,
                h_ui=helmet_f,
                clock=ck_commit,
            )
            last_caption = caption
            if PRINT_LIVE_CAPTIONS:
                print(caption, flush=True)
            if USE_SQLITE:
                now = time.time()
                payload = (plate_f, helmet_f, colour_f)
                if payload != last_db_fp or (now - last_log_t) >= 1.0:
                    log_db(SQLITE_DB, caption, plate_f, helmet_f, colour_f)
                    last_db_fp = payload
                    last_log_t = now

    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    print("\n--- Session done ---")
    print("Last caption:", last_caption)
    print("ANPR global plate (best vote):", anpr.final_plate() or "(none)")


if __name__ == "__main__":
    main()
