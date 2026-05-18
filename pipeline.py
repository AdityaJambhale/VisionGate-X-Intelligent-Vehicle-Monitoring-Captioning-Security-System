"""
VisionGate X — Core Pipeline
Orchestrates all modules: detection → helmet → ANPR → violation → challan → DB.
Persists one consolidated detection per vehicle *track* (not every frame).
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field

import numpy as np

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from modules import (
    VehicleDetector,
    DetectionResult,
    HelmetChecker,
    ANPRReader,
    CaptionGenerator,
    ViolationEngine,
    ViolationEvent,
    ChallanGenerator,
    Database,
)
from modules.detector import BBox
from utils.helpers import save_snapshot, get_logger
from utils.image_utils import draw_overlay_text

logger = get_logger("pipeline")


@dataclass
class _VehicleTrack:
    """Single vehicle identity across frames until finalized."""

    bbox: BBox
    label: str
    last_seen_frame: int
    missed: int = 0
    best_plate: str = ""
    best_plate_conf: float = 0.0
    helmet: str = "unknown"
    max_vehicle_conf: float = 0.0
    snapshot_path: str = ""
    persisted: bool = False
    last_detection_id: int = 0


@dataclass
class FrameResult:
    """Everything produced for a single processed frame."""

    frame_number: int
    annotated_frame: np.ndarray | None = None
    detection: DetectionResult | None = None
    helmet_statuses: dict = field(default_factory=dict)
    plate_map: dict = field(default_factory=dict)
    captions: list[str] = field(default_factory=list)
    violations: list[ViolationEvent] = field(default_factory=list)
    challan_ids: list[str] = field(default_factory=list)
    # Rows written to SQLite this frame (for dashboard live feed)
    new_db_events: list[dict] = field(default_factory=list)


class Pipeline:
    """
    End-to-end VisionGate X pipeline.

    Usage:
        pipe = Pipeline()
        for result in pipe.process_video("traffic.mp4"):
            print(result.captions)
    """

    def __init__(self, db: Database | None = None):
        self.detector = VehicleDetector()
        self.helmet = HelmetChecker()
        self.anpr = ANPRReader()
        self.caption = CaptionGenerator()
        self.violation = ViolationEngine()
        self.challan = ChallanGenerator()
        self.db = db or Database()
        self._tracks: list[_VehicleTrack] = []
        self._tracking_session_active = False
        self._video_tail_events: list[dict] = []

    # ── Tracking helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _helmet_for_vehicle(
        detection: DetectionResult,
        helmet_statuses: dict[int, str],
        vehicle: BBox,
    ) -> str:
        h_status = "unknown"
        for person, veh in detection.rider_vehicle_pairs():
            if veh is vehicle:
                try:
                    pidx = detection.persons.index(person)
                    h_status = helmet_statuses.get(pidx, "unknown")
                except ValueError:
                    pass
                break
        return h_status

    @staticmethod
    def _merge_plate_track(tr: _VehicleTrack, text: str, conf: float) -> None:
        text = (text or "").strip()
        if not text:
            return
        conf = float(conf)
        if conf > tr.best_plate_conf or not tr.best_plate:
            tr.best_plate = text
            tr.best_plate_conf = conf

    @staticmethod
    def _merge_helmet_track(tr: _VehicleTrack, h: str) -> None:
        """
        Across the track: one positive helmet frame locks helmet for the passage.
        Unknown frames do not downgrade a prior no_helmet (handled at persist).
        """
        h = (h or "unknown").lower()
        if h == "helmet":
            tr.helmet = "helmet"
        elif h == "no_helmet":
            if tr.helmet != "helmet":
                tr.helmet = "no_helmet"
        # unknown: keep current (starts as unknown until we see helmet or no_helmet)

    def _persist_track_to_db(
        self,
        tr: _VehicleTrack,
        source: str,
        result: FrameResult,
    ) -> int:
        stored_helmet = config.finalize_track_helmet_status(tr.helmet)
        caption_text = self.caption.caption_vehicle(tr.bbox, tr.best_plate, stored_helmet)
        det_id = self.db.insert_detection(
            source=source,
            frame_number=tr.last_seen_frame,
            vehicle_type=tr.label,
            plate_number=tr.best_plate,
            helmet_status=stored_helmet,
            confidence=max(tr.max_vehicle_conf, tr.bbox.confidence),
            caption=caption_text,
            bbox_json=[tr.bbox.to_dict()],
            snapshot_path=tr.snapshot_path or "",
        )
        tr.persisted = True
        tr.last_detection_id = det_id
        result.new_db_events.append(
            {
                "plate": tr.best_plate or "—",
                "vehicle": tr.label,
                "helmet": stored_helmet,
                "caption": caption_text,
            }
        )
        return det_id

    def _reset_tracking(self) -> None:
        self._tracks = []

    def _flush_stale_tracks(self, source: str, frame_number: int, result: FrameResult) -> None:
        """Finalize tracks that have left the scene (high miss count)."""
        kept: list[_VehicleTrack] = []
        for tr in self._tracks:
            if tr.missed <= config.TRACK_MAX_MISSED_FRAMES:
                kept.append(tr)
                continue
            if not tr.persisted:
                self._persist_track_to_db(tr, source, result)
            # dropped — vehicle gone
        self._tracks = kept

    def _flush_all_tracks(self, source: str, result: FrameResult) -> None:
        """End of video — persist anything still open."""
        for tr in self._tracks:
            if not tr.persisted:
                self._persist_track_to_db(tr, source, result)
        self._tracks = []

    def consume_video_tail_events(self) -> list[dict]:
        """Events persisted when the clip ends (caller: merge into UI after the loop)."""
        out = self._video_tail_events
        self._video_tail_events = []
        return out

    def _det_id_for_violation_bbox(
        self,
        detection: DetectionResult,
        vehicle_bbox: BBox,
        det_id_by_vehicle_idx: dict[int, int],
    ) -> int:
        try:
            vi = detection.vehicles.index(vehicle_bbox)
            return int(det_id_by_vehicle_idx.get(vi, 0))
        except ValueError:
            thr = float(config.TRACK_IOU_MATCH_THRESHOLD)
            for tr in self._tracks:
                if vehicle_bbox.iou(tr.bbox) >= thr and tr.persisted:
                    return int(tr.last_detection_id)
            return 0

    def _sync_tracks_for_frame(
        self,
        detection: DetectionResult,
        plate_map: dict[int, tuple[str, float]],
        helmet_statuses: dict[int, str],
        source: str,
        frame_number: int,
        snapshot_path: str,
        violations: list[ViolationEvent],
        result: FrameResult,
    ) -> dict[int, int]:
        """
        Match vehicles to tracks, update merged plate/helmet, flush stale tracks,
        persist immediately on violations. Returns vehicle_idx → detection_row_id.
        """
        det_id_by_vehicle_idx: dict[int, int] = {}
        vehicle_to_track: dict[int, _VehicleTrack] = {}

        matched_track_ids: set[int] = set()

        # Greedy match: each vehicle picks best unused track by IoU
        available = set(range(len(self._tracks)))

        for i, vehicle in enumerate(detection.vehicles):
            plate_text, pconf = plate_map.get(i, ("", 0.0))
            h = self._helmet_for_vehicle(detection, helmet_statuses, vehicle)

            best_j = -1
            best_iou = float(config.TRACK_IOU_MATCH_THRESHOLD)
            for j in available:
                iou = vehicle.iou(self._tracks[j].bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_j >= 0:
                available.discard(best_j)
                matched_track_ids.add(best_j)
                tr = self._tracks[best_j]
                tr.bbox = vehicle
                tr.label = vehicle.label
                tr.last_seen_frame = frame_number
                tr.missed = 0
                tr.max_vehicle_conf = max(tr.max_vehicle_conf, vehicle.confidence)
                if snapshot_path:
                    tr.snapshot_path = snapshot_path
                self._merge_plate_track(tr, plate_text, pconf)
                self._merge_helmet_track(tr, h)
                vehicle_to_track[i] = tr
            else:
                nt = _VehicleTrack(
                    bbox=vehicle,
                    label=vehicle.label,
                    last_seen_frame=frame_number,
                    missed=0,
                    max_vehicle_conf=vehicle.confidence,
                    snapshot_path=snapshot_path or "",
                )
                self._merge_plate_track(nt, plate_text, pconf)
                self._merge_helmet_track(nt, h)
                self._tracks.append(nt)
                vehicle_to_track[i] = nt

        for j, tr in enumerate(self._tracks):
            if j not in matched_track_ids:
                tr.missed += 1

        self._flush_stale_tracks(source, frame_number, result)

        # Re-build vehicle_to_track after removals — tracks may have been dropped;
        # re-match vehicles to tracks by IoU for violation linkage only
        def _find_track_for_vehicle(veh: BBox) -> _VehicleTrack | None:
            best: _VehicleTrack | None = None
            best_iou = float(config.TRACK_IOU_MATCH_THRESHOLD)
            for tr in self._tracks:
                iou = veh.iou(tr.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best = tr
            return best

        # Violations need a detection row the same frame (FK for challan workflow)
        for v_event in violations:
            if v_event.vehicle is None:
                continue
            vi: int | None = None
            try:
                vi = detection.vehicles.index(v_event.vehicle)
            except ValueError:
                vi = None

            tr = (
                vehicle_to_track.get(vi)
                if vi is not None
                else None
            ) or _find_track_for_vehicle(v_event.vehicle)
            if tr is None:
                continue
            if not tr.persisted:
                det_id = self._persist_track_to_db(tr, source, result)
                if vi is not None:
                    det_id_by_vehicle_idx[vi] = det_id
            elif vi is not None:
                det_id_by_vehicle_idx[vi] = tr.last_detection_id

        return det_id_by_vehicle_idx

    def _persist_every_vehicle_immediate(
        self,
        detection: DetectionResult,
        plate_map: dict[int, tuple[str, float]],
        helmet_statuses: dict[int, str],
        source: str,
        frame_number: int,
        snapshot_path: str,
        result: FrameResult,
    ) -> dict[int, int]:
        """Legacy path: one insert per vehicle per frame (live / non-video)."""
        det_id_by_vehicle_idx: dict[int, int] = {}
        for i, vehicle in enumerate(detection.vehicles):
            plate_text = plate_map.get(i, ("", 0.0))[0]
            h_status = self._helmet_for_vehicle(detection, helmet_statuses, vehicle)
            stored_helmet = config.finalize_track_helmet_status(h_status)
            caption_text = self.caption.caption_vehicle(vehicle, plate_text, stored_helmet)
            det_id = self.db.insert_detection(
                source=source,
                frame_number=frame_number,
                vehicle_type=vehicle.label,
                plate_number=plate_text,
                helmet_status=stored_helmet,
                confidence=vehicle.confidence,
                caption=caption_text,
                bbox_json=[vehicle.to_dict()],
                snapshot_path=snapshot_path,
            )
            det_id_by_vehicle_idx[i] = det_id
            result.new_db_events.append(
                {
                    "plate": plate_text or "—",
                    "vehicle": vehicle.label,
                    "helmet": stored_helmet,
                    "caption": caption_text,
                }
            )
        return det_id_by_vehicle_idx

    # ── Single frame ───────────────────────────────────────────────────────────

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        source: str = "live",
        save_snaps: bool = False,
    ) -> FrameResult:
        result = FrameResult(frame_number=frame_number)

        # 1. Detect vehicles + persons
        detection = self.detector.detect_frame(frame, frame_number)
        result.detection = detection

        # 2. Helmet check (HelmetChecker returns internal dict, scores, display)
        _h = self.helmet.check_frame(frame, detection, frame_idx=frame_number)
        helmet_statuses = _h[0] if isinstance(_h, tuple) else _h
        result.helmet_statuses = helmet_statuses

        # 3. ANPR
        plate_map = self.anpr.read_plates_in_frame(frame, detection.vehicles)
        result.plate_map = plate_map

        # 4. Captions (same helmet policy as DB — unknown → non-compliant for wording)
        for i, vehicle in enumerate(detection.vehicles):
            plate_text = plate_map.get(i, ("", 0.0))[0]
            h_status = self._helmet_for_vehicle(detection, helmet_statuses, vehicle)
            stored_h = config.finalize_track_helmet_status(h_status)
            cap = self.caption.caption_vehicle(
                vehicle=vehicle,
                plate=plate_text,
                helmet_status=stored_h,
            )
            result.captions.append(cap)

        # 5. Violation check
        violations = self.violation.evaluate(detection, helmet_statuses, plate_map)
        result.violations = violations

        # 6. Snapshot (optional)
        snapshot_path = ""
        if save_snaps and (violations or detection.vehicles):
            annotated = self.detector.draw_boxes(
                frame,
                detection,
                helmet_statuses,
                {i: plate_map.get(i, ("", 0.0))[0] for i in range(len(detection.vehicles))},
            )
            snapshot_path = save_snapshot(annotated, prefix="vgx")

        # 7. Persist detections (tracked consolidation for video sessions)
        if self._tracking_session_active:
            det_id_by_vehicle_idx = self._sync_tracks_for_frame(
                detection,
                plate_map,
                helmet_statuses,
                source,
                frame_number,
                snapshot_path,
                violations,
                result,
            )
        else:
            det_id_by_vehicle_idx = self._persist_every_vehicle_immediate(
                detection,
                plate_map,
                helmet_statuses,
                source,
                frame_number,
                snapshot_path,
                result,
            )

        # 8. Process violations → challans
        for v_event in violations:
            challan_id, pdf_path = self.challan.generate(
                violation=v_event,
                snapshot_path=snapshot_path,
                notify_email=config.NOTIFY_EMAIL_TO,
            )
            v_event.challan_id = challan_id
            result.challan_ids.append(challan_id)

            det_row_id = 0
            if v_event.vehicle is not None:
                det_row_id = self._det_id_for_violation_bbox(
                    detection, v_event.vehicle, det_id_by_vehicle_idx
                )

            self.db.insert_violation(
                detection_id=det_row_id,
                plate_number=v_event.plate_number,
                violation_type=v_event.violation_type,
                description=v_event.description,
                fine_inr=v_event.fine_inr,
                challan_id=challan_id,
                challan_path=pdf_path,
            )
            self.db.insert_challan(
                challan_id=challan_id,
                plate_number=v_event.plate_number,
                violation_type=v_event.violation_type,
                fine_inr=v_event.fine_inr,
                pdf_path=pdf_path,
            )

            cap = self.caption.caption_violation(
                plate=v_event.plate_number,
                violation_desc=v_event.description,
                fine_inr=v_event.fine_inr,
            )
            result.captions.append(cap)
            logger.info(f"Challan {challan_id} generated for {v_event.plate_number}")

        # 9. Build annotated frame
        plate_str_map = {
            i: plate_map.get(i, ("", 0.0))[0] for i in range(len(detection.vehicles))
        }
        annotated = self.detector.draw_boxes(frame, detection, helmet_statuses, plate_str_map)
        if violations:
            lines = [f"VIOLATION: {v.plate_number} — {v.description}" for v in violations]
            annotated = draw_overlay_text(annotated, lines, color=(0, 0, 230))
        result.annotated_frame = annotated

        return result

    # ── Video file ─────────────────────────────────────────────────────────────

    def process_video(
        self,
        video_path: str,
        progress_callback=None,
        max_frames: int | None = None,
        save_snaps: bool = True,
        source_label: str | None = None,
    ):
        """
        Generator — yields FrameResult for each processed frame.
        """
        import cv2 as _cv2

        cap = _cv2.VideoCapture(video_path)
        total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        source = source_label or os.path.basename(video_path)
        processed = 0
        self.violation.reset_cooldowns()
        self._reset_tracking()
        self._tracking_session_active = True
        self._video_tail_events = []
        last_result = FrameResult(frame_number=0)

        try:
            for frame, detection in self.detector.process_video(
                video_path, max_frames=max_frames
            ):
                result = self.process_frame(
                    frame=frame,
                    frame_number=detection.frame_number,
                    source=source,
                    save_snaps=save_snaps,
                )
                result.detection = detection
                last_result = result

                if progress_callback:
                    progress_callback(processed, total)
                processed += 1
                yield result
        finally:
            tail = FrameResult(frame_number=-1)
            self._flush_all_tracks(source, tail)
            self._video_tail_events = tail.new_db_events[:]
            self._tracking_session_active = False


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VisionGate X pipeline CLI")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--no-snaps", action="store_true", help="Skip snapshot saving")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    pipe = Pipeline()
    total_violations = 0

    for result in pipe.process_video(
        args.video,
        max_frames=args.max_frames,
        save_snaps=not args.no_snaps,
    ):
        if result.violations:
            for v in result.violations:
                total_violations += 1
                print(
                    f"  Frame {result.frame_number:>5} | {v.plate_number:<14} | "
                    f"{v.violation_type:<12} | ₹{v.fine_inr} | Challan: {v.challan_id}"
                )

    stats = pipe.db.get_summary_stats()
    print("\n── Summary ──────────────────────────────────")
    for k, v in stats.items():
        print(f"  {k:<22}: {v}")
