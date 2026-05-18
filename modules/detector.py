"""
VisionGate X — Vehicle Detector
Runs YOLOv8 inference on individual frames or video files.
Returns structured DetectionResult objects for downstream modules.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    confidence: float

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def to_dict(self) -> dict:
        return {
            "x1": self.x1, "y1": self.y1,
            "x2": self.x2, "y2": self.y2,
            "label": self.label,
            "confidence": round(self.confidence, 3),
        }

    def iou(self, other: "BBox") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = self.width * self.height + other.width * other.height - inter
        return inter / union if union > 0 else 0.0

    def proximity(self, other: "BBox") -> float:
        """Euclidean distance between centers."""
        cx1, cy1 = self.center
        cx2, cy2 = other.center
        return float(np.hypot(cx1 - cx2, cy1 - cy2))


@dataclass
class DetectionResult:
    frame_number: int
    vehicles: list[BBox] = field(default_factory=list)
    persons:  list[BBox] = field(default_factory=list)
    all_boxes: list[BBox] = field(default_factory=list)

    def rider_vehicle_pairs(self) -> list[tuple[BBox, BBox]]:
        """
        Pair each person with their nearest motorcycle/bicycle
        if within RIDER_VEHICLE_PROXIMITY pixels.
        """
        pairs = []
        motos = [v for v in self.vehicles if v.label in {"motorcycle", "bicycle"}]
        for person in self.persons:
            if not motos:
                break
            nearest = min(motos, key=lambda m: person.proximity(m))
            if person.proximity(nearest) <= config.RIDER_VEHICLE_PROXIMITY:
                pairs.append((person, nearest))
        return pairs


class VehicleDetector:
    """
    Wraps Ultralytics YOLOv8 for VisionGate X.
    Lazy-loads the model on first use.
    """

    def __init__(self, model_path: str = config.MODEL_PATH):
        self.model_path = model_path
        self._model = None

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load(self):
        from ultralytics import YOLO
        # YOLOv8n will auto-download on first run if not present
        self._model = YOLO(self.model_path)

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    # ── Core inference ─────────────────────────────────────────────────────────

    def detect_frame(self, frame: np.ndarray, frame_number: int = 0) -> DetectionResult:
        """
        Run inference on a single BGR frame (OpenCV format).
        Returns a DetectionResult with parsed bounding boxes.
        """
        results = self.model.predict(
            source=frame,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.NMS_IOU_THRESHOLD,
            verbose=False,
        )

        detection = DetectionResult(frame_number=frame_number)

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                label  = config.YOLO_CLASS_NAMES.get(cls_id)
                if label is None:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2,
                            label=label, confidence=conf)
                detection.all_boxes.append(bbox)
                if label in config.VEHICLE_CLASS_NAMES:
                    detection.vehicles.append(bbox)
                elif label in config.RIDER_CLASS_NAMES:
                    detection.persons.append(bbox)

        return detection

    # ── Annotated frame ────────────────────────────────────────────────────────

    def draw_boxes(
        self,
        frame: np.ndarray,
        detection: DetectionResult,
        helmet_statuses: dict[int, str] | None = None,
        plate_texts: dict[int, str] | None = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes on a copy of the frame.
        helmet_statuses: {person_box_index → 'helmet'/'no_helmet'/'unknown'}
        plate_texts:     {vehicle_box_index → plate_string}
        """
        frame = frame.copy()
        COLOR = {
            "car":        (0, 200, 255),
            "motorcycle": (0, 140, 255),
            "bus":        (255, 150, 0),
            "truck":      (255, 100, 0),
            "bicycle":    (180, 255, 0),
            "person":     (0, 255, 180),
        }
        HELMET_COLOR = {
            "helmet":    (0, 220, 60),
            "no_helmet": (0, 0, 230),
            "unknown":   (150, 150, 150),
        }

        for i, bbox in enumerate(detection.vehicles):
            color = COLOR.get(bbox.label, (200, 200, 200))
            cv2.rectangle(frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 2)
            plate_txt = (plate_texts or {}).get(i, "")
            label_str = f"{bbox.label} {bbox.confidence:.2f}"
            if plate_txt:
                label_str += f" | {plate_txt}"
            cv2.putText(frame, label_str, (bbox.x1, bbox.y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        for i, bbox in enumerate(detection.persons):
            status = (helmet_statuses or {}).get(i, "unknown")
            color  = HELMET_COLOR[status]
            cv2.rectangle(frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 2)
            cv2.putText(frame, status.replace("_", " ").upper(),
                        (bbox.x1, bbox.y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)

        return frame

    # ── Video processing ───────────────────────────────────────────────────────

    def process_video(
        self,
        video_path: str,
        frame_callback=None,
        max_frames: int | None = None,
    ):
        """
        Iterate over video frames, yielding (frame, DetectionResult) tuples.
        Optionally calls frame_callback(frame, result) on each processed frame.
        Respects config.FRAME_SKIP.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        frame_idx   = 0
        proc_count  = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if max_frames and proc_count >= max_frames:
                    break

                if frame_idx % config.FRAME_SKIP == 0:
                    result = self.detect_frame(frame, frame_number=frame_idx)
                    if frame_callback:
                        frame_callback(frame, result)
                    yield frame, result
                    proc_count += 1

                frame_idx += 1
        finally:
            cap.release()



if __name__ == "__main__":

    detector = VehicleDetector()

    video_path = "data/sample_videos/bike.mp4"

    for frame, result in detector.process_video(
        video_path,
        max_frames=100
    ):

        annotated = detector.draw_boxes(frame, result)

        cv2.imshow("VisionGate X Detection", annotated)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break

    cv2.destroyAllWindows()