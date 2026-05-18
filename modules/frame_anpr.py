import os
import re
from collections import Counter

import cv2
import easyocr

from ultralytics import YOLO


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

FRAMES_DIR = "data/extracted_frames"

PLATE_MODEL_PATH = "models/license_plate_detector.pt"

VALID_PLATE_REGEX = r"[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}"


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────

plate_model = YOLO(
    PLATE_MODEL_PATH
)

ocr_reader = easyocr.Reader(
    ['en'],
    gpu=False
)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def clean_text(text):

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text.upper()
    )

    return text


def is_valid_plate(text):

    return bool(
        re.match(
            VALID_PLATE_REGEX,
            text
        )
    )


# ─────────────────────────────────────────────
# OCR ON SINGLE IMAGE
# ─────────────────────────────────────────────

def process_frame(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return None

    results = plate_model(
        image,
        verbose=False
    )

    best_plate = None

    best_conf = 0

    for r in results:

        for box in r.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            plate_crop = image[
                y1:y2,
                x1:x2
            ]

            if plate_crop.size == 0:
                continue

            # Preprocess
            gray = cv2.cvtColor(
                plate_crop,
                cv2.COLOR_BGR2GRAY
            )

            gray = cv2.resize(
                gray,
                None,
                fx=3,
                fy=3,
                interpolation=cv2.INTER_CUBIC
            )

            thresh = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]

            ocr_results = ocr_reader.readtext(
                thresh,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )

            for (_, text, conf) in ocr_results:

                cleaned = clean_text(text)

                if len(cleaned) < 6:
                    continue

                if conf > best_conf:

                    best_plate = cleaned
                    best_conf = conf

    return best_plate


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

all_detected = []

frame_files = sorted(
    os.listdir(FRAMES_DIR)
)

for frame_file in frame_files:

    frame_path = os.path.join(
        FRAMES_DIR,
        frame_file
    )

    plate = process_frame(
        frame_path
    )

    if plate:

        print(
            f"{frame_file} → {plate}"
        )

        all_detected.append(
            plate
        )

print("\n===== FINAL DETECTED PLATE =====")

if all_detected:

    valid = [
        p for p in all_detected
        if len(p) >= 6
    ]

    if valid:

        counter = Counter(valid)

        final_plate = counter.most_common(1)[0][0]

        print(final_plate)

    else:

        print("No reliable plate")

else:

    print("No plate detected")