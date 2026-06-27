"""Führt Outline-, Tag- und Bewegungs-Verifikation zusammen und zeichnet Overlays.

Kombination Farbe + Bewegung (gezielt, nicht global):
- FARBE+FORM findet rote Gegner-Outline-Kandidaten (kann auf statisches
  Rot/Deko anspringen).
- Pro Kandidat prüft der MotionVerifier, ob er sich anders bewegt als seine
  Umgebung (siehe movement.py). Nur dann gilt er als bewegter Gegner.

Darstellung:
    * nur Farbe (statisch)     -> orange
    * Farbe + Bewegung (Gegner)-> grün (dick, "GEGNER")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .movement import MotionVerifier, MovementRegion
from .outlines import OutlineDetection, RedOutlineDetector
from .tags import TagDetection, TagDetector


@dataclass
class FrameResult:
    """Gesammelte Erkennungen für ein einzelnes Frame."""

    outlines: list[OutlineDetection] = field(default_factory=list)
    tags: list[TagDetection] = field(default_factory=list)
    movements: list[MovementRegion] = field(default_factory=list)  # bewegte Outlines
    frame_id: int = 0

    def confirmed_boxes(self) -> list[tuple[int, int, int, int]]:
        """Boxen, die per Farbe UND Bewegung bestätigt sind (= Gegner)."""
        return [m.box for m in self.movements]

    def to_dict(self) -> dict:
        """Kompaktes Dict – genau das, was später über LAN an PC2 geht."""
        return {
            "frame": self.frame_id,
            "outlines": [o.to_dict() for o in self.outlines],
            "confirmed": [list(b) for b in self.confirmed_boxes()],
            "tags": [t.to_dict() for t in self.tags],
        }

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), separators=(",", ":"))


class DetectionPipeline:
    """Kombiniert Farb-Outlines, Bewegungs-Verifikation und (optional) OCR."""

    # BGR-Farben für die Overlays
    COLOR_ONLY = (0, 140, 255)     # orange = nur Farbe (statisch)
    CONFIRMED = (0, 255, 0)        # grün   = Farbe + Bewegung = Gegner
    COLOR_TAG = (255, 180, 0)      # cyan/blau

    def __init__(
        self,
        outline_detector: RedOutlineDetector | None = None,
        motion_verifier: MotionVerifier | None = None,
        tag_detector: TagDetector | None = None,
        use_ocr: bool = True,
        use_motion: bool = True,
        outlines_only: bool = True,
    ) -> None:
        self.outlines = outline_detector or RedOutlineDetector()
        self.motion = motion_verifier or MotionVerifier()
        self.tags = tag_detector or TagDetector()
        self.use_ocr = use_ocr
        self.use_motion = use_motion
        self.outlines_only = outlines_only

    def process(self, frame: np.ndarray) -> FrameResult:
        result = FrameResult()
        result.outlines = self.outlines.detect(frame, outlines_only=self.outlines_only)

        if self.use_motion:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if result.outlines:
                self.motion.update(gray)              # teurer Fluss nur bei Bedarf
                result.movements = [
                    MovementRegion(box=o.box, area=o.area)
                    for o in result.outlines
                    if self.motion.is_moving(o.box)
                ]
            else:
                self.motion.note_frame(gray)          # nur Frame merken

        if self.use_ocr:
            result.tags = self.tags.detect(frame)
        return result

    def draw(self, frame: np.ndarray, result: FrameResult, show_movement: bool = True) -> np.ndarray:
        """Zeichnet die erkannte rote Outline 1:1 nach (echte Silhouette statt
        Box). Mit Bewegungs-Verifikation: bewegter Gegner grün, statisches
        Rot orange."""
        out = frame.copy()
        confirmed = set(result.confirmed_boxes())

        def is_confirmed(o) -> bool:
            return show_movement and o.box in confirmed

        # Dezente, halbtransparente Füllung der bestätigten Gegner-Silhouetten,
        # damit sie hervorstechen (ein addWeighted-Pass für alle zusammen).
        conf_contours = [
            c for o in result.outlines if is_confirmed(o) for c in o.contours
        ]
        if conf_contours:
            overlay = out.copy()
            cv2.drawContours(overlay, conf_contours, -1, self.CONFIRMED, cv2.FILLED)
            cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)

        # Outline nachzeichnen.
        for o in result.outlines:
            is_conf = is_confirmed(o)
            color = self.CONFIRMED if is_conf else self.COLOR_ONLY
            thick = 2 if is_conf else 1
            if o.contours:
                cv2.polylines(out, o.contours, isClosed=True, color=color,
                              thickness=thick, lineType=cv2.LINE_AA)
            else:  # Fallback (sollte nicht vorkommen)
                x, y, w, h = o.box
                cv2.rectangle(out, (x, y), (x + w, y + h), color, thick)
            if is_conf:
                x, y, _, _ = o.box
                cv2.putText(out, "GEGNER", (x, max(0, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.CONFIRMED, 2,
                            cv2.LINE_AA)

        if show_movement:
            self._draw_legend(out)
        self._draw_tags(out, result)
        return out

    def _draw_legend(self, out: np.ndarray) -> None:
        items = [("nur Farbe (statisch)", self.COLOR_ONLY),
                 ("Gegner (bewegt)", self.CONFIRMED)]
        y = 24
        for text, color in items:
            cv2.rectangle(out, (12, y - 12), (28, y + 2), color, -1)
            cv2.putText(out, text, (34, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        color, 2, cv2.LINE_AA)
            y += 26

    def _draw_tags(self, out: np.ndarray, result: FrameResult) -> None:
        for t in result.tags:
            x, y, w, h = t.box
            cv2.rectangle(out, (x, y), (x + w, y + h), self.COLOR_TAG, 2)
            cv2.putText(out, t.text, (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_TAG, 2,
                        cv2.LINE_AA)
