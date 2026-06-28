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
from .selfchar import SelfCharacterDetector, SelfDetection
from .tags import TagDetection, TagDetector


@dataclass
class FrameResult:
    """Gesammelte Erkennungen für ein einzelnes Frame."""

    outlines: list[OutlineDetection] = field(default_factory=list)  # Gegner (rot)
    teammates: list[OutlineDetection] = field(default_factory=list)  # Mitspieler (blau)
    self_char: SelfDetection | None = None  # eigener Charakter (zentral, lila)
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
    EDGE_ENEMY = (40, 40, 255)     # ROT    = die erkannte Gegner-Outline-Kante
    FILL_ENEMY = (0, 165, 255)     # orange = Füllung der Silhouette (Verifikation)
    COLOR_TEAM = (220, 0, 255)     # lila-pink = Mitspieler (ausgeschlossen)
    COLOR_SELF = (255, 0, 140)     # violett   = eigener Charakter

    def __init__(
        self,
        outline_detector: RedOutlineDetector | None = None,
        motion_verifier: MotionVerifier | None = None,
        tag_detector: TagDetector | None = None,
        self_detector: SelfCharacterDetector | None = None,
        use_ocr: bool = True,
        use_motion: bool = True,
        use_self: bool = True,
        boost_red: bool = False,
        outlines_only: bool = True,
        ocr_every: int = 1,
    ) -> None:
        self.outlines = outline_detector or RedOutlineDetector()
        self.motion = motion_verifier or MotionVerifier()
        self.tags = tag_detector or TagDetector()
        self.selfdet = self_detector or SelfCharacterDetector()
        self.use_ocr = use_ocr
        self.use_motion = use_motion
        self.use_self = use_self
        self.boost_red = boost_red
        self.outlines_only = outlines_only
        # OCR-Throttle: EasyOCR ist mit Abstand der teuerste Schritt (~300 ms
        # GPU / ~1800 ms CPU pro Frame). HUD-Text ändert sich langsam, daher
        # nur jeden N-ten verarbeiteten Frame OCR rechnen und die Tags
        # dazwischen weiterverwenden. 1 = jedes Frame (alt).
        self.ocr_every = max(1, ocr_every)
        self._ocr_ctr = 0
        self._tag_cache: list[TagDetection] = []

    def set_ocr_every(self, n: int) -> None:
        """Setzt das OCR-Intervall in (verarbeiteten) Frames; >=1."""
        self.ocr_every = max(1, int(n))

    @staticmethod
    def _suppress_overlap(
        enemies: list[OutlineDetection], teammates: list[OutlineDetection]
    ) -> list[OutlineDetection]:
        """Verwirft rote Kandidaten, deren Zentrum in einer Teammate-Box liegt
        (z. B. blaue Mitspieler mit etwas rotem Schaden-Glimmer drauf)."""
        if not teammates:
            return enemies
        kept = []
        for e in enemies:
            cx, cy = e.center
            inside = any(
                tx <= cx <= tx + tw and ty <= cy <= ty + th
                for (tx, ty, tw, th) in (t.box for t in teammates)
            )
            if not inside:
                kept.append(e)
        return kept

    @staticmethod
    def _outside_self(
        dets: list[OutlineDetection], self_box: tuple[int, int, int, int] | None
    ) -> list[OutlineDetection]:
        """Verwirft Outlines, deren Zentrum in der eigenen "ICH"-Box liegt.

        Der eigene Char hat keinen Glow, aber sein Kostüm (z. B. blau) kann die
        Gegner-/Mitspieler-Maske auslösen -> er würde fälschlich als TEAM/GEGNER
        markiert. Da er separat als "ICH" geführt wird, blenden wir alles in
        seiner Box aus, damit NUR "ICH" auf ihm steht."""
        if self_box is None:
            return dets
        sx, sy, sw, sh = self_box
        kept = []
        for d in dets:
            cx, cy = d.center
            if sx <= cx <= sx + sw and sy <= cy <= sy + sh:
                continue
            kept.append(d)
        return kept

    def process(self, frame: np.ndarray) -> FrameResult:
        result = FrameResult()
        dets = self.outlines.detect(frame, outlines_only=self.outlines_only)
        teammates = [d for d in dets if d.team == "teammate"]
        enemies = [d for d in dets if d.team == "enemy"]

        # Eigenen Char zuerst bestimmen, damit wir Gegner-/Team-Treffer auf ihm
        # selbst (Kostümfarbe) ausblenden können -> auf ihm steht nur "ICH".
        if self.use_self:
            result.self_char = self.selfdet.detect(frame)
            self_box = result.self_char.box if result.self_char else None
            teammates = self._outside_self(teammates, self_box)
            enemies = self._outside_self(enemies, self_box)

        result.teammates = teammates
        # Gegner, die einen Mitspieler überdecken, NICHT doppelt als Gegner werten.
        result.outlines = self._suppress_overlap(enemies, teammates)

        if self.use_motion:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if result.outlines:
                self.motion.update(gray)              # teurer Fluss nur bei Bedarf
                for o in result.outlines:
                    moving, mscore = self.motion.evaluate(o.box)
                    if moving:
                        result.movements.append(
                            MovementRegion(box=o.box, area=o.area, score=mscore)
                        )
                    # Endgültige Confidence = Form-Anteil + Bewegungs-Anteil.
                    # Statische (nur-Farbe) Kandidaten erreichen so max. 0.6 –
                    # erst bestätigte Bewegung hebt Richtung 100 %.
                    o.confidence = 0.6 * o.shape_confidence + 0.4 * (mscore if moving else 0.0)
            else:
                self.motion.note_frame(gray)          # nur Frame merken

        if self.use_ocr:
            # Nur jeden ocr_every-ten Frame neu OCRen, sonst letzten Stand
            # weiterverwenden (HUD-Text ändert sich kaum).
            if self._ocr_ctr % self.ocr_every == 0:
                self._tag_cache = self.tags.detect(frame)
            self._ocr_ctr += 1
            result.tags = self._tag_cache
        return result

    # Confidence-Stufen für die Beschriftung (BGR)
    CONF_HIGH = (0, 255, 0)        # grün  >= 75 %
    CONF_MID = (0, 255, 255)       # gelb  50–75 %
    CONF_LOW = (0, 140, 255)       # orange < 50 %

    @staticmethod
    def _boost_red(out: np.ndarray) -> np.ndarray:
        """Hebt rot-dominante Pixel an, damit Rot vor der Erkennung besser
        heraussticht (rein visuell). Bewusst billig: nur Kanal-Arithmetik."""
        b, g, r = cv2.split(out)
        # Wie stark dominiert Rot über den helleren der anderen Kanäle?
        redness = cv2.subtract(r, cv2.max(g, b))
        boost = cv2.multiply(redness, 1.6)
        r = cv2.add(r, boost)
        # Grün/Blau leicht absenken, wo es rot ist -> Rot wirkt satter.
        g = cv2.subtract(g, cv2.multiply(redness, 0.4))
        b = cv2.subtract(b, cv2.multiply(redness, 0.4))
        return cv2.merge([b, g, r])

    @classmethod
    def _conf_color(cls, conf: float) -> tuple[int, int, int]:
        if conf >= 0.75:
            return cls.CONF_HIGH
        if conf >= 0.50:
            return cls.CONF_MID
        return cls.CONF_LOW

    def draw(self, frame: np.ndarray, result: FrameResult,
             show_movement: bool = True, show_confidence: bool = True) -> np.ndarray:
        """Zeichnet Gegner (rote Kante + orange Silhouetten-Füllung zur
        Verifikation) und Mitspieler (lila-pink, ausgeschlossen).

        Bewusst getrennte Farben: die KANTE ist rot = das tatsächlich erkannte
        Rot der Outline; die FÜLLUNG ist orange = der vom Loop umschlossene
        Körper. So sieht man sofort, ob die Hohl-Outline korrekt sitzt."""
        out = frame.copy()
        if self.boost_red:
            out = self._boost_red(out)
        confirmed = set(result.confirmed_boxes())

        def is_confirmed(o) -> bool:
            return show_movement and o.box in confirmed

        # 1) Orange Silhouetten-Füllung ALLER Gegner in einem addWeighted-Pass.
        #    Die gefüllten Konturen sind implizit geschlossen (drawContours
        #    FILLED verbindet Anfang/Ende) -> "Hilfslinie" bei unten offener
        #    Outline gratis.
        enemy_contours = [c for o in result.outlines for c in o.contours]
        if enemy_contours:
            overlay = out.copy()
            cv2.drawContours(overlay, enemy_contours, -1, self.FILL_ENEMY,
                             cv2.FILLED)
            cv2.addWeighted(overlay, 0.30, out, 0.70, 0, out)

        # 2) Gegner-Outline-KANTE in Rot nachzeichnen (dicker, wenn bewegt).
        for o in result.outlines:
            is_conf = is_confirmed(o)
            thick = 3 if is_conf else 2
            if o.contours:
                cv2.polylines(out, o.contours, isClosed=True,
                              color=self.EDGE_ENEMY, thickness=thick,
                              lineType=cv2.LINE_AA)
            else:  # Fallback (sollte nicht vorkommen)
                x, y, w, h = o.box
                cv2.rectangle(out, (x, y), (x + w, y + h), self.EDGE_ENEMY, thick)
            x, y, _, _ = o.box
            if show_confidence:
                pct = f"{o.confidence * 100:.0f}%"
                label = f"GEGNER {pct}" if is_conf else pct
                cv2.putText(out, label, (x, max(12, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            self._conf_color(o.confidence), 2, cv2.LINE_AA)
            elif is_conf:
                cv2.putText(out, "GEGNER", (x, max(0, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.CONFIRMED, 2,
                            cv2.LINE_AA)

        # 3) Mitspieler (blau erkannt) -> lila-pink, ausgeschlossen.
        for t in result.teammates:
            if t.contours:
                cv2.polylines(out, t.contours, isClosed=True,
                              color=self.COLOR_TEAM, thickness=2,
                              lineType=cv2.LINE_AA)
            tx, ty, _, _ = t.box
            cv2.putText(out, "TEAM", (tx, max(12, ty - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_TEAM, 2,
                        cv2.LINE_AA)

        # 4) Eigener Charakter (zentral, kein Glow) -> violett.
        if result.self_char is not None and result.self_char.contour is not None:
            cv2.polylines(out, [result.self_char.contour], isClosed=True,
                          color=self.COLOR_SELF, thickness=3, lineType=cv2.LINE_AA)
            sx, sy, _, _ = result.self_char.box
            cv2.putText(out, "ICH", (sx, max(12, sy - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_SELF, 2,
                        cv2.LINE_AA)

        self._draw_legend(out, show_movement, show_confidence)
        self._draw_tags(out, result)
        return out

    def _draw_legend(self, out: np.ndarray, show_movement: bool = True,
                     show_confidence: bool = False) -> None:
        items: list[tuple[str, tuple[int, int, int]]] = [
            ("Gegner-Kante (rot)", self.EDGE_ENEMY),
            ("Gegner-Fuellung (orange)", self.FILL_ENEMY),
            ("Mitspieler (ausgeschl.)", self.COLOR_TEAM),
            ("Eigener Char (ICH)", self.COLOR_SELF),
        ]
        if show_confidence:
            items += [("Confidence hoch >=75%", self.CONF_HIGH),
                      ("Confidence mittel 50-75%", self.CONF_MID),
                      ("Confidence niedrig <50%", self.CONF_LOW)]
        if not items:
            return
        # Unter das Missions-Banner des Spiels (oben links) setzen und mit
        # halbtransparentem Panel hinterlegen, damit es immer lesbar bleibt.
        x0, y0 = 12, 120
        line_h, pad = 24, 8
        panel_h = line_h * len(items) + pad
        panel = out[y0 - line_h:y0 - line_h + panel_h, x0 - 6:x0 + 270]
        if panel.size:
            panel[:] = (panel * 0.35).astype(panel.dtype)
        y = y0
        for text, color in items:
            cv2.rectangle(out, (x0, y - 12), (x0 + 16, y + 2), color, -1)
            cv2.putText(out, text, (x0 + 22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1, cv2.LINE_AA)
            y += line_h

    def _draw_tags(self, out: np.ndarray, result: FrameResult) -> None:
        for t in result.tags:
            x, y, w, h = t.box
            cv2.rectangle(out, (x, y), (x + w, y + h), self.COLOR_TAG, 2)
            cv2.putText(out, t.text, (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_TAG, 2,
                        cv2.LINE_AA)
