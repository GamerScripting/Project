"""OCR-Texterkennung (Tags / Nametags) mit EasyOCR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TagDetection:
    """Ein erkannter Text mit Position und Konfidenz."""

    text: str
    box: tuple[int, int, int, int]  # x, y, w, h
    confidence: float

    @property
    def center(self) -> tuple[int, int]:
        x, y, w, h = self.box
        return (x + w // 2, y + h // 2)


class TagDetector:
    """Erkennt Text in Frames mittels EasyOCR.

    EasyOCR wird erst beim ersten Aufruf geladen (lazy), damit der Import des
    Moduls schnell bleibt und ohne installiertes EasyOCR nicht crasht.
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        gpu: bool = False,
        min_confidence: float = 0.4,
    ) -> None:
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.min_confidence = min_confidence
        self._reader = None  # wird lazy initialisiert

    def _ensure_reader(self):
        if self._reader is None:
            import easyocr  # lokaler Import: Abhängigkeit nur bei Bedarf

            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._reader

    def detect(self, frame: np.ndarray) -> list[TagDetection]:
        reader = self._ensure_reader()
        # readtext liefert: [ (box_4_punkte, text, confidence), ... ]
        results = reader.readtext(frame)

        detections: list[TagDetection] = []
        for points, text, conf in results:
            if conf < self.min_confidence:
                continue
            xs = [int(p[0]) for p in points]
            ys = [int(p[1]) for p in points]
            x, y = min(xs), min(ys)
            w, h = max(xs) - x, max(ys) - y
            detections.append(
                TagDetection(text=text.strip(), box=(x, y, w, h), confidence=float(conf))
            )
        return detections
