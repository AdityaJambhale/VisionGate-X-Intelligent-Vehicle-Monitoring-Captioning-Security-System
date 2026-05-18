"""
VisionGate X — Violation Engine
Evaluates detection results against traffic rules and
returns structured ViolationEvent objects.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.detector import BBox, DetectionResult


@dataclass
class ViolationEvent:
    violation_type: str          # key from config.VIOLATION_TYPES
    description: str
    fine_inr: float
    plate_number: str
    vehicle: BBox | None
    rider: BBox | None
    frame_number: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    challan_id: str = ""


class ViolationEngine:
    """
    Checks each frame's detections for rule violations.
    Currently implements: helmet violation.
    Extendable: add new check_* methods and call them from evaluate().
    """

    def __init__(self):
        self._seen_plates: dict[str, int] = {}   # plate → last_frame with violation
        self._cooldown_frames = 60               # suppress duplicate alerts

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        detection: DetectionResult,
        helmet_statuses: dict[int, str],
        plate_map: dict[int, tuple[str, float]],
    ) -> list[ViolationEvent]:
        """
        Run all violation checks on a single frame's results.
        Returns list of ViolationEvent (may be empty).
        """
        violations: list[ViolationEvent] = []
        violations.extend(
            self._check_helmet_violations(detection, helmet_statuses, plate_map)
        )
        return violations

    # ── Helmet rule ────────────────────────────────────────────────────────────

    def _check_helmet_violations(
        self,
        detection: DetectionResult,
        helmet_statuses: dict[int, str],
        plate_map: dict[int, tuple[str, float]],
    ) -> list[ViolationEvent]:
        events = []
        rule   = config.VIOLATION_TYPES["NO_HELMET"]
        pairs  = detection.rider_vehicle_pairs()

        for person, vehicle in pairs:
            try:
                person_idx = detection.persons.index(person)
            except ValueError:
                continue

            status = helmet_statuses.get(person_idx, "unknown")
            # Helmet once in any logic path is handled elsewhere; frame rule:
            # compliant only if helmet; unknown treats as non-compliant (not verified).
            if status == "helmet":
                continue

            # Get plate of the associated vehicle
            try:
                vehicle_idx = detection.vehicles.index(vehicle)
            except ValueError:
                vehicle_idx = -1

            plate_info  = plate_map.get(vehicle_idx, ("", 0.0))
            plate_number = plate_info[0] if plate_info else ""

            # Cooldown: don't spam violations for the same plate
            last_frame = self._seen_plates.get(plate_number, -999)
            if detection.frame_number - last_frame < self._cooldown_frames:
                continue

            self._seen_plates[plate_number] = detection.frame_number

            events.append(ViolationEvent(
                violation_type = "NO_HELMET",
                description    = rule["description"],
                fine_inr       = rule["fine_inr"],
                plate_number   = plate_number or "UNKNOWN",
                vehicle        = vehicle,
                rider          = person,
                frame_number   = detection.frame_number,
            ))

        return events

    # ── Future rule stubs ──────────────────────────────────────────────────────

    def _check_wrong_side(self, detection: DetectionResult) -> list[ViolationEvent]:
        # TODO: implement with vehicle trajectory tracking (DeepSORT)
        return []

    def _check_speed(self, detection: DetectionResult) -> list[ViolationEvent]:
        # TODO: implement with frame-timestamp based velocity estimation
        return []

    def reset_cooldowns(self):
        """Call between video files to clear per-plate cooldown state."""
        self._seen_plates.clear()
