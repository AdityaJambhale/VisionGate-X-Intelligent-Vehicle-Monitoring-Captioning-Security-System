# VisionGate X

**Intelligent Vehicle Monitoring, Captioning & Automated E-Challan Security System**

An AI-powered real-time traffic monitoring platform that detects vehicles, checks helmet compliance, reads number plates, generates descriptive captions, and issues automated e-challans — all from CCTV footage or uploaded video.

---

## Project structure

```
visiongate_x/
├── app.py                  # Streamlit dashboard (entry point)
├── pipeline.py             # Core end-to-end pipeline
├── config.py               # All settings & constants
├── requirements.txt
├── .env.example            # Copy → .env and fill credentials
│
├── modules/
│   ├── __init__.py
│   ├── detector.py         # YOLOv8 vehicle + person detection
│   ├── helmet.py           # Helmet compliance checking
│   ├── anpr.py             # Licence plate OCR (EasyOCR)
│   ├── caption.py          # Human-readable event captions
│   ├── violation.py        # Traffic rules engine
│   ├── challan.py          # PDF challan + SMS/email notifications
│   └── database.py         # SQLite CRUD layer
│
├── utils/
│   ├── helpers.py          # Logging, snapshots, video info
│   └── image_utils.py      # Frame conversion/resize helpers
│
├── data/
│   ├── sample_videos/      # Put test videos here
│   └── test_images/
│
├── models/
│   └── yolov8n.pt          # Auto-downloaded on first run
│
└── outputs/
    ├── challans/           # Generated PDF challans
    └── snapshots/          # Violation frame snapshots
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your SMTP / Twilio credentials (optional for local demo)
```

### 3. Launch the dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### 4. Process a video (CLI)

```bash
python pipeline.py path/to/video.mp4
```

Optional flags:
```
--max-frames 300     process only first 300 frames
--no-snaps           skip saving snapshot images
```

---

## How it works

```
Video input
    │
    ▼
YOLOv8 detection  ──→  vehicles (car, motorcycle, bus…)
                   ──→  persons (riders)
    │
    ├── Helmet checker     heuristic skin-tone analysis on head crop
    ├── ANPR (EasyOCR)     crop bottom-third of vehicle → OCR → normalise
    ├── Caption generator  template-based human-readable event description
    │
    ▼
Violation engine    checks rules (helmet, future: speed, wrong-side)
    │
    ├── Challan generator  ReportLab PDF + optional SMS/email
    └── SQLite database    detections / violations / challans tables
    │
    ▼
Streamlit dashboard  live frames │ logs │ violations │ challans │ analytics
```

---

## Modules at a glance

| Module | Class | Key method |
|--------|-------|-----------|
| `detector.py` | `VehicleDetector` | `detect_frame(frame)` → `DetectionResult` |
| `helmet.py` | `HelmetChecker` | `check_frame(frame, detection)` → `{idx: status}` |
| `anpr.py` | `ANPRReader` | `read_plates_in_frame(frame, vehicles)` → `{idx: (plate, conf)}` |
| `caption.py` | `CaptionGenerator` | `caption_vehicle(...)` → `str` |
| `violation.py` | `ViolationEngine` | `evaluate(detection, helmets, plates)` → `[ViolationEvent]` |
| `challan.py` | `ChallanGenerator` | `generate(violation_event)` → `(challan_id, pdf_path)` |
| `database.py` | `Database` | CRUD for detections / violations / challans |
| `pipeline.py` | `Pipeline` | `process_video(path)` generator |

---

## Configuration (config.py)

| Setting | Default | Description |
|---------|---------|-------------|
| `CONFIDENCE_THRESHOLD` | 0.40 | Minimum YOLOv8 detection confidence |
| `FRAME_SKIP` | 2 | Process every Nth frame |
| `RIDER_VEHICLE_PROXIMITY` | 120 px | Max distance for rider→vehicle pairing |
| `HELMET_FINE_AMOUNT` | ₹500 | Fine for no-helmet violation |
| `USE_CUSTOM_HELMET_MODEL` | False | Enable custom YOLO helmet model |

---

## Upgrading helmet detection

1. Train a YOLOv8 model on a helmet dataset (e.g. from Roboflow).
2. Place the exported `best.pt` in `models/helmet_model/`.
3. Set `USE_CUSTOM_HELMET_MODEL = True` in `config.py`.

The `HelmetChecker` class will automatically use the custom model.

---

## Future roadmap

- [ ] Live CCTV RTSP stream support
- [ ] DeepSORT vehicle tracking across frames
- [ ] Speed estimation from frame timestamps
- [ ] Wrong-side detection
- [ ] Government RTO API integration for real challan issuance
- [ ] Automated payment gateway
- [ ] Mobile app (Flutter/React Native)
- [ ] Cloud deployment (AWS / GCP)
- [ ] WhatsApp notification via Twilio WhatsApp API

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Detection | YOLOv8 (Ultralytics) |
| OCR | EasyOCR |
| Computer Vision | OpenCV |
| Dashboard | Streamlit |
| Database | SQLite |
| PDF Generation | ReportLab |
| Notifications | SMTP email + Twilio SMS |
| Language | Python 3.10+ |
