"""
VisionGate X — Central configuration
All paths, thresholds, fine amounts, and credentials live here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "models", "yolov8n.pt")
DB_PATH      = os.path.join(BASE_DIR, "visiongate.db")
CHALLAN_DIR  = os.path.join(BASE_DIR, "outputs", "challans")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "outputs", "snapshots")

# Create output dirs if they don't exist
os.makedirs(CHALLAN_DIR,  exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ── YOLOv8 / Detection ────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.40   # minimum detection confidence
NMS_IOU_THRESHOLD    = 0.45

# COCO class IDs we care about
YOLO_CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

VEHICLE_CLASS_NAMES = {"car", "motorcycle", "bus", "truck", "bicycle"}
RIDER_CLASS_NAMES   = {"person"}

# Proximity threshold (pixels): how close a person must be to a motorcycle
# to be considered its rider
RIDER_VEHICLE_PROXIMITY = 120   # px

# ── Helmet detection ──────────────────────────────────────────────────────────
# Heuristic: if a person bounding-box top-quarter has a distinct blob → helmet
# Set to True once a custom helmet model is added
USE_CUSTOM_HELMET_MODEL = False
HELMET_MODEL_PATH       = os.path.join(BASE_DIR, "models", "helmet_model", "best.pt")

# ── OCR / ANPR ────────────────────────────────────────────────────────────────
OCR_LANGUAGES         = ["en"]
PLATE_MIN_CONFIDENCE  = 0.3
PLATE_REGION_EXPAND   = 25    # px to expand detected plate crop

# ── Violation rules ───────────────────────────────────────────────────────────
VIOLATION_TYPES = {
    "NO_HELMET": {
        "description": "Riding without helmet",
        "fine_inr": 500,
    },
    "WRONG_SIDE": {
        "description": "Driving on wrong side",
        "fine_inr": 1000,
    },
}

# ── Notifications ─────────────────────────────────────────────────────────────
TWILIO_SID   = os.getenv("TWILIO_SID",   "")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN", "")
TWILIO_FROM  = os.getenv("TWILIO_FROM",  "")

SMTP_HOST    = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT    = int(os.getenv("SMTP_PORT", 587))
SMTP_USER    = os.getenv("SMTP_USER", "")
SMTP_PASS    = os.getenv("SMTP_PASS", "")

NOTIFY_EMAIL_TO = os.getenv("NOTIFY_EMAIL_TO", "")

# ── App UI ────────────────────────────────────────────────────────────────────
APP_TITLE   = "VisionGate X"
APP_ICON    = "🚦"
APP_VERSION = "1.0.0"
CHECKPOINT_LABEL = os.getenv("VISIONGATE_GATE_NAME", "Gate A")

# Video processing
FRAME_SKIP = 1   # process every Nth frame (1 = every frame, 2 = every other)
MAX_FRAMES_PREVIEW = 500   # limit for dashboard preview rendering

# Vehicle track persistence (one DB row per passage, not every frame)
TRACK_IOU_MATCH_THRESHOLD = 0.22   # min IoU to match bbox to existing track
TRACK_MAX_MISSED_FRAMES = 14       # after this many frames unseen → finalize track


def finalize_track_helmet_status(track_value: str) -> str:
    """
    Persisted helmet status for DB + captions:
    - If any frame in the track saw 'helmet' → helmet
    - Otherwise (no_helmet or only unknown) → no_helmet (unknown counts as not wearing)
    """
    if (track_value or "").lower() == "helmet":
        return "helmet"
    return "no_helmet"
