"""Erkennung farbiger Umrandungen / Highlights via HSV-Farbfilter."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class OutlineDetection:
    """Ein erkanntes Objekt mit farbiger Umrandung."""

    box: tuple[int, int, int, int]  # x, y, w, h
    area: float
    contour: np.ndarray = field(repr=False)

    @property
    def center(self) -> tuple[int, int]:
        x, y, w, h = self.box
        return (x + w // 2, y + h // 2)


class OutlineDetector:
    """Findet Objekte anhand einer bestimmten Outline-Farbe (in HSV).

    HSV ist robuster gegen Helligkeitsschwankungen als RGB. Die Default-Werte
    sind auf ein kräftiges Grün eingestellt – passt `lower`/`upper` an die
    Farbe eurer Outlines an.

    HSV-Bereiche in OpenCV:
        H: 0–179, S: 0–255, V: 0–255
    """

    def __init__(
        self,
        lower: tuple[int, int, int] = (40, 80, 80),
        upper: tuple[int, int, int] = (80, 255, 255),
        min_area: int = 150,
    ) -> None:
        self.lower = np.array(lower, dtype=np.uint8)
        self.upper = np.array(upper, dtype=np.uint8)
        # Konturen kleiner als min_area (Pixel²) werden als Rauschen verworfen.
        self.min_area = min_area
        # Kernel zum Schließen kleiner Lücken in der Maske.
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray) -> list[OutlineDetection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)

        # Morphologie: Lücken schließen + Rauschen entfernen.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: list[OutlineDetection] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append(OutlineDetection(box=(x, y, w, h), area=area, contour=cnt))
        return detections
