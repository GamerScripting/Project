"""Führt Outline-, Tag- und Movement-Erkennung zusammen und zeichnet Overlays."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .movement import MovementDetector, MovementRegion
from .outlines import OutlineDetection, OutlineDetector
from .tags import TagDetection, TagDetector


@dataclass
class FrameResult:
    """Gesammelte Erkennungen für ein einzelnes Frame."""

    outlines: list[OutlineDetection] = field(default_factory=list)
    tags: list[TagDetection] = field(default_factory=list)
    movements: list[MovementRegion] = field(default_factory=list)


class DetectionPipeline:
    """Kombiniert alle Detektoren. Die OCR ist optional (langsam)."""

    # BGR-Farben für die Overlays
    COLOR_OUTLINE = (0, 255, 0)    # grün
    COLOR_TAG = (255, 180, 0)      # cyan/blau
    COLOR_MOVE = (0, 0, 255)       # rot

    def __init__(
        self,
        outline_detector: OutlineDetector | None = None,
        movement_detector: MovementDetector | None = None,
        tag_detector: TagDetector | None = None,
        use_ocr: bool = True,
    ) -> None:
        self.outlines = outline_detector or OutlineDetector()
        self.movement = movement_detector or MovementDetector()
        self.tags = tag_detector or TagDetector()
        self.use_ocr = use_ocr

    def process(self, frame: np.ndarray) -> FrameResult:
        result = FrameResult()
        result.outlines = self.outlines.detect(frame)
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
            cv2.rectangle(out, (x, y), (x + w, y + h), self.COLOR_OUTLINE, 2)

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
