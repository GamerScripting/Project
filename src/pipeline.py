"""Führt Outline-, Tag- und Movement-Erkennung zusammen und zeichnet Overlays."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .movement import MovementDetector, MovementRegion
from .outlines import OutlineDetection, RedOutlineDetector
from .tags import TagDetection, TagDetector


@dataclass
class FrameResult:
    """Gesammelte Erkennungen für ein einzelnes Frame."""

    outlines: list[OutlineDetection] = field(default_factory=list)
    tags: list[TagDetection] = field(default_factory=list)
    movements: list[MovementRegion] = field(default_factory=list)
    frame_id: int = 0

    def to_dict(self) -> dict:
        """Kompaktes Dict – genau das, was später über LAN an PC2 geht."""
        return {
            "frame": self.frame_id,
            "outlines": [o.to_dict() for o in self.outlines],
            "movements": [m.to_dict() for m in self.movements],
            "tags": [t.to_dict() for t in self.tags],
        }

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), separators=(",", ":"))


class DetectionPipeline:
    """Kombiniert alle Detektoren. Die OCR ist optional (langsam)."""

    # BGR-Farben für die Overlays
    COLOR_OUTLINE = (0, 255, 0)    # grün  = echte Outline
    COLOR_SOLID = (0, 165, 255)    # orange = massives rotes Objekt (verworfen)
    COLOR_TAG = (255, 180, 0)      # cyan/blau
    COLOR_MOVE = (0, 0, 255)       # rot

    def __init__(
        self,
        outline_detector: RedOutlineDetector | None = None,
        movement_detector: MovementDetector | None = None,
        tag_detector: TagDetector | None = None,
        use_ocr: bool = True,
        outlines_only: bool = True,
    ) -> None:
        self.outlines = outline_detector or RedOutlineDetector()
        self.movement = movement_detector or MovementDetector()
        self.tags = tag_detector or TagDetector()
        self.use_ocr = use_ocr
        # outlines_only=False behalten, um echte Outlines visuell von massiven
        # roten Objekten unterscheiden zu können (Debug). True = nur Outlines.
        self.outlines_only = outlines_only

    def process(self, frame: np.ndarray) -> FrameResult:
        result = FrameResult()
        result.outlines = self.outlines.detect(frame, outlines_only=self.outlines_only)
        result.movements = self.movement.detect(frame)
        if self.use_ocr:
            result.tags = self.tags.detect(frame)
        return result

    def draw(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """Zeichnet alle Erkennungen als Overlay auf eine Kopie des Frames."""
        out = frame.copy()

        for m in result.movements:
            x, y, w, h = m.box
            cv2.rectangle(out, (x, y), (x + w, y + h), self.COLOR_MOVE, 1)

        for o in result.outlines:
            x, y, w, h = o.box
            color = self.COLOR_OUTLINE if o.is_outline else self.COLOR_SOLID
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

        for t in result.tags:
            x, y, w, h = t.box
            cv2.rectangle(out, (x, y), (x + w, y + h), self.COLOR_TAG, 2)
            cv2.putText(
                out,
                t.text,
                (x, max(0, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.COLOR_TAG,
                2,
                cv2.LINE_AA,
            )
        return out
