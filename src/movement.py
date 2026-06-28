"""Bewegungs-Verifikation per LOKALEM Optical-Flow-Kontrast.

Lehre aus den Tests: Globale Bewegungserkennung (Frame-Diff, MOG2, globale
Kamera-Kompensation) ist auf Gameplay-Footage wegen Parallaxe nicht sauber –
bei schnellen Schwenks bewegt sich alles unterschiedlich stark.

Schlauerer Ansatz – nicht global suchen, sondern gezielt VERIFIZIEREN:
Für jede bereits per Farbe gefundene Outline-Box wird geprüft, ob sie sich
ANDERS bewegt als ihre direkte Umgebung (ein "Ring" um die Box).

    * statische rote Deko / HUD  -> bewegt sich MIT dem lokalen Hintergrund
      (gleicher Fluss)             -> Kontrast ~ 0  -> NICHT bewegt
    * echter Gegner              -> bewegt sich unabhängig vom Hintergrund
                                   -> hoher Kontrast -> BEWEGT

Weil Box und Ring lokal benachbart sind (ähnliche Tiefe), hebt sich die
Kamerabewegung im Vergleich heraus – das ist robust gegen Parallaxe.
Schatten/Lichtwechsel erzeugen kaum kohärenten Fluss und fallen weg.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MovementRegion:
    """Eine Box, die als eigenständig bewegt verifiziert wurde."""

    box: tuple[int, int, int, int]  # x, y, w, h
    area: float
    score: float = 0.0              # Bewegungs-Score 0..1 (Kontrast+Kohärenz)

    @property
    def center(self) -> tuple[int, int]:
        x, y, w, h = self.box
        return (x + w // 2, y + h // 2)

    def to_dict(self) -> dict:
        return {"box": list(self.box), "center": list(self.center)}


class MotionVerifier:
    """Prüft pro Kandidaten-Box, ob sie sich anders bewegt als ihre Umgebung.

    Benutzung:
        mv.update(gray_frame)          # einmal pro Frame: Fluss berechnen
        moving = mv.is_moving(box)     # für jede Outline-Box

    Args:
        downscale: Faktor, auf den das Bild für den (teuren) dichten Optical
            Flow verkleinert wird (0.5 = halbe Kantenlänge).
        min_contrast: Mindest-Differenz (px im verkleinerten Bild) zwischen
            der mittleren Bewegung IN der Box und der ihres Rings.
        ring_frac: Wie weit der Vergleichs-Ring über die Box hinausgeht
            (Anteil der Box-Größe je Seite).
    """

    def __init__(
        self,
        downscale: float = 0.5,
        min_contrast: float = 1.2,
        ring_frac: float = 0.5,
        min_flow: float = 0.6,
        min_coherence: float = 0.35,
        align_deg: float = 35.0,
    ) -> None:
        self.scale = downscale
        self.min_contrast = min_contrast
        self.ring_frac = ring_frac
        # Bewegt sich ein CHARAKTER (zusammenhängend, gleichgerichtet) oder nur
        # ein Schatten/Lichtwechsel (inkohärenter Flacker-Fluss)?
        #   min_flow:      ab welcher Fluss-Stärke (px im verkleinerten Bild)
        #                  ein Pixel als "in Bewegung" zählt.
        #   min_coherence: Anteil der Box-Pixel, die sich kräftig UND in die
        #                  gleiche Richtung wie der Median bewegen müssen.
        #                  Ein Schatten erzeugt verstreuten, richtungslosen
        #                  Fluss -> niedrige Kohärenz -> verworfen.
        #   align_deg:     max. Winkelabweichung vom Median, die noch als
        #                  "gleiche Richtung" zählt.
        self.min_flow = min_flow
        self.min_coherence = min_coherence
        self._cos_align = float(np.cos(np.deg2rad(align_deg)))
        self._prev: np.ndarray | None = None
        self._flow: np.ndarray | None = None
        # DIS-Optical-Flow ist um ein Vielfaches schneller als Farneback.
        # Fallback auf Farneback, falls in dieser OpenCV-Build nicht vorhanden.
        try:
            self._dis = cv2.DISOpticalFlow_create(
                cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST
            )
        except AttributeError:
            self._dis = None

    def _to_small(self, gray: np.ndarray) -> np.ndarray:
        return cv2.resize(gray, None, fx=self.scale, fy=self.scale,
                          interpolation=cv2.INTER_AREA)

    def note_frame(self, gray: np.ndarray) -> None:
        """Frame nur merken (kein Fluss) – für Frames ohne Kandidaten."""
        self._prev = self._to_small(gray)
        self._flow = None

    def update(self, gray: np.ndarray) -> None:
        """Berechnet den dichten Fluss vom vorherigen zum aktuellen Frame."""
        small = self._to_small(gray)
        if self._prev is None:
            self._prev = small
            self._flow = None
            return
        if self._dis is not None:
            self._flow = self._dis.calc(self._prev, small, None)
        else:
            self._flow = cv2.calcOpticalFlowFarneback(
                self._prev, small, None,
                pyr_scale=0.5, levels=3, winsize=21,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
        self._prev = small

    def _median_flow(self, region: np.ndarray) -> tuple[float, float] | None:
        if region.size == 0:
            return None
        fx = float(np.median(region[..., 0]))
        fy = float(np.median(region[..., 1]))
        return fx, fy

    def is_moving(self, box: tuple[int, int, int, int]) -> bool:
        """True, wenn sich `box` deutlich anders bewegt als ihr Ring."""
        return self.evaluate(box)[0]

    def evaluate(self, box: tuple[int, int, int, int]) -> tuple[bool, float]:
        """(bewegt?, score 0..1). Der Score steigt mit Bewegungs-Kontrast UND
        -Kohärenz; 0.0, wenn nicht als bewegt bestätigt."""
        if self._flow is None:
            return (False, 0.0)
        fh, fw = self._flow.shape[:2]
        s = self.scale
        x, y, w, h = box
        # Box in Fluss-Koordinaten
        bx, by, bw, bh = int(x * s), int(y * s), int(w * s), int(h * s)
        bx2, by2 = bx + bw, by + bh
        if bw < 2 or bh < 2:
            return (False, 0.0)

        # Ring um die Box (geclippt)
        mx, my = int(bw * self.ring_frac), int(bh * self.ring_frac)
        rx, ry = max(0, bx - mx), max(0, by - my)
        rx2, ry2 = min(fw, bx2 + mx), min(fh, by2 + my)
        bx, by = max(0, bx), max(0, by)
        bx2, by2 = min(fw, bx2), min(fh, by2)
        if bx2 <= bx or by2 <= by:
            return (False, 0.0)

        inner = self._flow[by:by2, bx:bx2].reshape(-1, 2)
        # Ring = großer Block minus Box-Bereich
        ring_block = self._flow[ry:ry2, rx:rx2].reshape(-1, 2)
        if ring_block.shape[0] <= inner.shape[0]:
            return (False, 0.0)

        in_med = self._median_flow(inner)
        # Ring-Median: nutze den ganzen Block (Box-Anteil ist klein und zieht
        # den Median kaum) – einfache, robuste Näherung.
        ring_med = self._median_flow(ring_block)
        if in_med is None or ring_med is None:
            return (False, 0.0)

        dfx = in_med[0] - ring_med[0]
        dfy = in_med[1] - ring_med[1]
        contrast = (dfx * dfx + dfy * dfy) ** 0.5
        if contrast < self.min_contrast:
            return (False, 0.0)

        # --- Kohärenz: bewegt sich ein Charakter oder nur ein Schatten? ---
        # Ein Charakter = zusammenhängende Fläche, die sich gleichgerichtet
        # bewegt. Ein Schatten/Lichtwechsel erzeugt schwachen, richtungslosen
        # Fluss. Anteil der Box-Pixel verlangen, die kräftig UND ~parallel zum
        # Median ziehen.
        fx = inner[:, 0]
        fy = inner[:, 1]
        mag = np.sqrt(fx * fx + fy * fy)
        moving = mag >= self.min_flow
        n_moving = int(np.count_nonzero(moving))
        if n_moving == 0:
            return (False, 0.0)
        # Median-Richtung der bewegten Pixel
        mvx = float(np.median(fx[moving]))
        mvy = float(np.median(fy[moving]))
        med_mag = (mvx * mvx + mvy * mvy) ** 0.5
        if med_mag < 1e-6:
            return (False, 0.0)
        # Skalarprodukt der Einheitsvektoren = cos(Winkel zur Median-Richtung)
        cos_ang = (fx * mvx + fy * mvy) / (mag * med_mag + 1e-6)
        aligned = moving & (cos_ang >= self._cos_align)
        coherence = np.count_nonzero(aligned) / float(inner.shape[0])
        if coherence < self.min_coherence:
            return (False, 0.0)

        # --- Bewegungs-Score 0..1 (für Confidence) ---
        # Wie weit über den Schwellen liegen Kohärenz und Kontrast?
        def clamp(v: float) -> float:
            return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v

        coh_score = clamp((coherence - self.min_coherence)
                          / max(1e-6, 0.75 - self.min_coherence))
        con_score = clamp((contrast - self.min_contrast) / 2.0)
        score = 0.6 * coh_score + 0.4 * con_score
        return (True, score)

    def reset(self) -> None:
        self._prev = None
        self._flow = None
