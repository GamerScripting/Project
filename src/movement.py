"""Bewegungserkennung per Frame-Differenz."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MovementRegion:
    """Eine Bildregion, in der Bewegung erkannt wurde."""

    box: tuple[int, int, int, int]  # x, y, w, h
    area: float

    @property
    def center(self) -> tuple[int, int]:
        x, y, w, h = self.box
        return (x + w // 2, y + h // 2)


class MovementDetector:
    """Erkennt Bewegung durch Vergleich aufeinanderfolgender Frames.

    Prinzip: Der absolute Differenzbetrag zwischen dem aktuellen und dem
    vorherigen (Graustufen-)Frame wird geschwellt. Wo sich etwas bewegt hat,
    entstehen helle Flecken, aus denen wir Bounding-Boxes bilden.
    """

    def __init__(self, threshold: int = 25, min_area: int = 500) -> None:
        self.threshold = threshold
        self.min_area = min_area
        self._prev_gray: np.ndarray | None = None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def detect(self, frame: np.ndarray) -> list[MovementRegion]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Leichtes Blur reduziert Kamera-/Encoding-Rauschen.
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return []

        diff = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray

        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, self._kernel, iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions: list[MovementRegion] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            regions.append(MovementRegion(box=(x, y, w, h), area=area))
        return regions

    def reset(self) -> None:
        """Vorheriges Frame vergessen (z. B. bei Szenenwechsel)."""
        self._prev_gray = None
