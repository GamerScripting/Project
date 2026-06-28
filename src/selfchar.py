"""Eigenen 3rd-Person-Charakter (zentral, OHNE Glow-Outline) markieren.

Der eigene Held hat – anders als Gegner/Mitspieler – keine farbige
Glow-Outline. Frühere Versuche, ihn über lokale Kanten-/Detail-DICHTE zu
segmentieren, scheiterten auf dem rot-lastigen "Rot_max"-Environment: der
gleichfarbige Hintergrund (Kisten/Boden) und magenta VFX-Funken wurden
mitgerissen.

Robuster Ansatz (kopf-verankert):
  * Die 3rd-Person-Kamera folgt dem Helden -> sein KOPF liegt horizontal sehr
    konsistent in einer engen zentralen Spalte (~x 0.36-0.44).
  * Sein rot-oranges HAAR + die magenta GESICHTSMASKE bilden einen Farb-Anker.
    Wir suchen diesen Anker NUR in der zentralen Spalte (schließt seitliche
    VFX/Gegner/Umgebungs-Rot aus) und nehmen den obersten Treffer als Kopf.
  * Von dort wird eine proportionale Körper-Box nach unten aufgespannt.
    Springt der Held, wandert der Kopf-Anker hoch -> die Box wandert mit
    (die Beine sind dann ebenfalls höher, die Box bleibt bis zum Boden).

Bewusst eine saubere Box statt einer 1:1-Silhouette: ohne Glow lässt sich die
Silhouette auf gleichfarbigem Grund nicht zuverlässig tracen. Die Box markiert
dafür verlässlich NUR den eigenen Char und greift nie Hintergrund/Gegner/VFX.
Kosten ~1-2 ms/Frame -> per use_self abschaltbar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class SelfDetection:
    box: tuple[int, int, int, int]
    contour: object = field(default=None, repr=False)


class SelfCharacterDetector:
    def __init__(
        self,
        column: tuple[float, float] = (0.33, 0.48),   # enge zentrale x-Spalte
        head_y: tuple[float, float] = (0.24, 0.78),   # wo der Kopf liegen darf
        body_w_frac: float = 0.135,                   # Box-Breite (Bildanteil)
        body_bottom: float = 0.965,                   # Box-Unterkante (Bildanteil)
        min_cue_frac: float = 0.0005,                 # Mindest-Anker-Fläche
        fallback: bool = True,                        # feste Box, wenn kein Anker
    ) -> None:
        self.column = column
        self.head_y = head_y
        self.body_w_frac = body_w_frac
        self.body_bottom = body_bottom
        self.min_cue_frac = min_cue_frac
        self.fallback = fallback
        self._close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

    @staticmethod
    def _head_cue(hsv: np.ndarray) -> np.ndarray:
        """Maske aus rot-orangem Haar UND magenta Gesichtsmaske/Akzenten."""
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        hair = ((h <= 16) | (h >= 168)) & (s > 130) & (v > 110)
        mag = (h >= 145) & (h < 168) & (s > 110) & (v > 90)
        return (hair | mag).astype(np.uint8) * 255

    def _body_box(self, head_cx: int, head_top: int, w: int, h: int):
        bw = int(self.body_w_frac * w)
        x0 = max(0, head_cx - bw // 2)
        x1 = min(w, head_cx + bw // 2)
        y0 = max(0, head_top - int(0.01 * h))
        y1 = int(self.body_bottom * h)
        return x0, y0, x1 - x0, y1 - y0

    @staticmethod
    def _as_contour(box: tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = box
        return np.array([[[x, y]], [[x + w, y]],
                         [[x + w, y + h]], [[x, y + h]]], dtype=np.int32)

    def detect(self, frame: np.ndarray) -> SelfDetection | None:
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cue = self._head_cue(hsv)

        cx0, cx1 = int(self.column[0] * w), int(self.column[1] * w)
        cy0, cy1 = int(self.head_y[0] * h), int(self.head_y[1] * h)
        colmask = np.zeros((h, w), np.uint8)
        colmask[cy0:cy1, cx0:cx1] = 255
        cue = cue & colmask
        cue = cv2.morphologyEx(cue, cv2.MORPH_CLOSE, self._close)

        cnts, _ = cv2.findContours(cue, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.min_cue_frac * w * h
        cands = []
        for c in cnts:
            if cv2.contourArea(c) < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            cands.append((y, x + bw // 2))
        if cands:
            cands.sort(key=lambda t: t[0])     # obersten Treffer = Kopf
            head_top, head_cx = cands[0]
            box = self._body_box(head_cx, head_top, w, h)
            return SelfDetection(box=box, contour=self._as_contour(box))

        if not self.fallback:
            return None
        # Kein Anker (Haar verdeckt o. ä.): feste zentrale Box – der eigene
        # Char steht in 3rd-Person ohnehin immer hier.
        head_cx = int(0.40 * w)
        head_top = int(0.42 * h)
        box = self._body_box(head_cx, head_top, w, h)
        return SelfDetection(box=box, contour=self._as_contour(box))
