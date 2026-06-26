"""Erkennung farbiger (roter) Umrandungen via HSV-Farbfilter.

Besonderheiten:
- Rot liegt in HSV an beiden Enden des Hue-Kreises (~0 und ~180), deshalb
  werden ZWEI Farbbereiche kombiniert.
- Echte Outlines werden von massiven roten Objekten über die *Füllrate*
  unterschieden: Eine Outline ist ein dünner Rand um einen nicht-roten
  Innenbereich (wenig rote Pixel in der Box), ein rotes Objekt ist fast
  vollständig rot gefüllt.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class OutlineDetection:
    """Ein erkanntes rotes Element."""

    box: tuple[int, int, int, int]   # x, y, w, h (in Originalauflösung)
    area: float                      # Fläche der Kontur (Pixel²)
    fill_ratio: float                # Anteil roter Pixel in der Box (0..1)
    is_outline: bool                 # True = Outline, False = massives Objekt
    contour: np.ndarray = field(default=None, repr=False)

    @property
    def center(self) -> tuple[int, int]:
        x, y, w, h = self.box
        return (x + w // 2, y + h // 2)

    def to_dict(self) -> dict:
        """Kompakte, sendbare Darstellung (ohne die schwere Kontur)."""
        return {
            "box": list(self.box),
            "center": list(self.center),
            "fill": round(self.fill_ratio, 3),
        }


class RedOutlineDetector:
    """Findet rote Outlines und trennt sie von massiven roten Objekten.

    Args:
        lower1/upper1: erster Rot-Bereich in HSV (um Hue 0).
        lower2/upper2: zweiter Rot-Bereich in HSV (um Hue 180).
        min_area: Konturen kleiner als das (Pixel²) gelten als Rauschen.
        max_fill_ratio: Liegt der Rot-Anteil in der Box UNTER diesem Wert,
            wird das Element als Outline gewertet, sonst als massives Objekt.
        downscale: Faktor <1.0 verkleinert das Frame vor der Analyse
            (z. B. 0.5 = halbe Kantenlänge ≈ 4x schneller). Boxen werden
            wieder auf die Originalauflösung hochgerechnet.
    """

    def __init__(
        self,
        # Default deckt Orange BIS Rot ab (Hue 0–20), passend zu orangen
        # Glow-Outlines. Massive rote/orange Flächen (Health-Bars) fängt die
        # Füllrate ab, nicht die Farbe.
        lower1: tuple[int, int, int] = (0, 90, 90),
        upper1: tuple[int, int, int] = (20, 255, 255),
        lower2: tuple[int, int, int] = (170, 90, 90),
        upper2: tuple[int, int, int] = (180, 255, 255),
        min_area: int = 120,
        max_fill_ratio: float = 0.35,
        downscale: float = 1.0,
        roi: tuple[float, float, float, float] | None = None,
    ) -> None:
        self.lower1 = np.array(lower1, dtype=np.uint8)
        self.upper1 = np.array(upper1, dtype=np.uint8)
        self.lower2 = np.array(lower2, dtype=np.uint8)
        self.upper2 = np.array(upper2, dtype=np.uint8)
        self.min_area = min_area
        self.max_fill_ratio = max_fill_ratio
        self.downscale = max(0.05, min(1.0, downscale))
        # ROI als relative Anteile (x, y, w, h) in 0..1 – z. B. (0, 0.15, 1, 0.7)
        # blendet oberes und unteres HUD aus. None = ganzes Bild.
        self.roi = roi
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def _roi_px(self, frame: np.ndarray) -> tuple[int, int, int, int]:
        """Rechnet die relative ROI in Pixel um (oder volles Bild)."""
        h, w = frame.shape[:2]
        if self.roi is None:
            return (0, 0, w, h)
        rx, ry, rw, rh = self.roi
        x0, y0 = int(rx * w), int(ry * h)
        return (x0, y0, int(rw * w), int(rh * h))

    def red_mask(self, frame: np.ndarray) -> np.ndarray:
        """Binäre Maske aller roten Pixel (beide Hue-Bereiche kombiniert)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower1, self.upper1)
        mask |= cv2.inRange(hsv, self.lower2, self.upper2)
        # Nur ein kleiner Open-Schritt gegen Rauschen – KEIN Close, damit
        # der hohle Innenraum einer Outline hohl bleibt (wichtig für Füllrate).
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        return mask

    def detect(
        self, frame: np.ndarray, outlines_only: bool = True
    ) -> list[OutlineDetection]:
        """Erkennt rote Elemente.

        Args:
            outlines_only: Wenn True, werden massive rote Objekte herausgefiltert.
        """
        # Auf die ROI zuschneiden (HUD ausblenden + schneller).
        rx, ry, rw, rh = self._roi_px(frame)
        cropped = frame[ry:ry + rh, rx:rx + rw]

        scale = self.downscale
        if scale < 1.0:
            small = cv2.resize(cropped, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        else:
            small = cropped

        mask = self.red_mask(small)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        inv_scale = 1.0 / scale
        detections: list[OutlineDetection] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)

            # Füllrate: Anteil roter Pixel innerhalb der Bounding-Box.
            roi = mask[y:y + h, x:x + w]
            box_area = w * h
            fill_ratio = (cv2.countNonZero(roi) / box_area) if box_area else 0.0
            is_outline = fill_ratio <= self.max_fill_ratio

            if outlines_only and not is_outline:
                continue

            # Box zurück auf Originalauflösung skalieren + ROI-Offset addieren.
            box = (
                int(x * inv_scale) + rx, int(y * inv_scale) + ry,
                int(w * inv_scale), int(h * inv_scale),
            )
            detections.append(
                OutlineDetection(
                    box=box,
                    area=area * inv_scale * inv_scale,
                    fill_ratio=fill_ratio,
                    is_outline=is_outline,
                    contour=cnt,
                )
            )
        return detections


# Rückwärtskompatibler Alias
OutlineDetector = RedOutlineDetector
