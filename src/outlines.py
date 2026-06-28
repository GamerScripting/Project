"""Erkennung dünner roter/pinker Charakter-Outlines (Gegner-Glow).

Ansatz = FARBE + FORM, optimiert auf DÜNNE Linien:

1. Farb-Maske (HSV, zwei Rot-Bereiche wegen Hue-Wrap) klassifiziert jeden
   Pixel als "Outline-Farbe" oder nicht.  ->  `core`-Maske (dünn, roh)
2. KEINE Erosion/Opening!  Ein Opening mit 3x3-Kernel löscht eine 2-px-Linie
   komplett aus – genau das, was wir behalten wollen.  Stattdessen ein
   CLOSE, das die (durch Crosshair/Text/Effekte) zerrissene Outline wieder
   zu EINEM zusammenhängenden Silhouetten-Loop verbindet.  ->  `connected`
3. Konturen auf `connected`; pro Kandidat entscheiden FORM-Merkmale, ob es
   ein Charakter ist:
     - Seitenverhältnis (ein Mensch ist hoch, kein breiter HUD-Balken)
     - Größe (kein Mini-Rauschen, nicht der halbe Bildschirm)
     - Thinness / Füllrate (eine OUTLINE ist eine dünne Linie, keine
       "fette Ummantellung" = massiv rot gefülltes Objekt).  Gemessen auf
       der `core`-Maske, damit das CLOSE die Füllrate nicht verfälscht.

GPU: Nur die (teure) Farb-Klassifikation läuft via PyTorch/cv2.cuda auf der
GPU; Morphologie + findContours sind auf der binären Maske billig und laufen
auf der CPU (cv2.findContours hat ohnehin kein CUDA).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as _F
    _TORCH = True
except ImportError:
    _TORCH = False


# ---------------------------------------------------------------------------
# GPU-Hilfsfunktion: BGR -> HSV (OpenCV-Maßstab) auf der GPU
# ---------------------------------------------------------------------------

def _bgr_to_hsv_torch(bgr: "torch.Tensor") -> "torch.Tensor":
    """BGR-uint8-Tensor (H,W,3) -> HSV (H:0-180, S/V:0-255) als float-Tensor."""
    b = bgr[:, :, 0].float() / 255.0
    g = bgr[:, :, 1].float() / 255.0
    r = bgr[:, :, 2].float() / 255.0
    maxc = torch.max(torch.max(r, g), b)
    minc = torch.min(torch.min(r, g), b)
    diff = maxc - minc
    eps = 1e-6
    s = torch.where(maxc > eps, diff / (maxc + eps), torch.zeros_like(maxc))
    h = torch.zeros_like(maxc)
    mr = (maxc == r) & (diff > eps)
    mg = (maxc == g) & (diff > eps) & ~mr
    mb = (diff > eps) & ~mr & ~mg
    h[mr] = (((g - b) / (diff + eps)) % 6)[mr]
    h[mg] = (((b - r) / (diff + eps)) + 2)[mg]
    h[mb] = (((r - g) / (diff + eps)) + 4)[mb]
    h = (h * 30.0) % 180.0          # OpenCV-Hue: 0..180
    return torch.stack([h, s * 255.0, maxc * 255.0], dim=2)


# ---------------------------------------------------------------------------
# Datenklasse
# ---------------------------------------------------------------------------

@dataclass
class OutlineDetection:
    """Ein erkannter Gegner-Outline / Charakter-Kandidat."""

    box: tuple[int, int, int, int]   # x, y, w, h (Originalauflösung)
    area: float                      # Bounding-Box-Fläche (Pixel², Original)
    fill_ratio: float                # roter Pixel-Anteil in der Box (Thinness)
    aspect: float                    # Höhe / Breite
    is_outline: bool                 # immer True (Kandidat hat Filter bestanden)
    # "enemy" (rote Outline) oder "teammate" (blaue Outline). Teammates werden
    # NUR markiert/ausgeschlossen, nicht als Gegner gewertet.
    team: str = "enemy"
    circularity: float = 0.0         # 4*pi*A/U^2 (1=Kreis -> rundes Icon)
    # Echte Silhouetten-Konturen (Originalkoordinaten) zum 1:1-Nachzeichnen
    # der roten Outline statt einer Box.
    contours: list = field(default_factory=list, repr=False)
    solidity: float = 1.0            # Konturfläche / Konvexhülle (Form-Maß)
    enclosed_frac: float = 0.0       # gefüllte Silhouette / Box (Loop umschließt Körper)
    solid_ratio: float = 0.0         # rote Pixel / Silhouette (0=hohl, 1=voller Klecks)
    # Confidence 0..1 wie gut die FORM zu einer Charakter-Outline passt
    # (nur Farbe+Form). Die Pipeline überschreibt `confidence` mit dem um die
    # Bewegung angereicherten Endwert; `shape_confidence` bleibt der Formanteil.
    shape_confidence: float = 0.0
    confidence: float = 0.0

    @property
    def center(self) -> tuple[int, int]:
        x, y, w, h = self.box
        return (x + w // 2, y + h // 2)

    def to_dict(self) -> dict:
        return {
            "box": list(self.box),
            "center": list(self.center),
            "team": self.team,
            "fill": round(self.fill_ratio, 3),
            "aspect": round(self.aspect, 2),
            "conf": round(self.confidence, 3),
        }


# ---------------------------------------------------------------------------
# Detektor
# ---------------------------------------------------------------------------

class RedOutlineDetector:
    """Findet dünne rote/pinke Charakter-Outlines (Farbe + Form).

    HSV-Farbe (Default auf den Gegner-Glow abgestimmt):
        Bereich 1 (Rot/Coral/Orange-Seite):  H 0-14
        Bereich 2 (Pink/Magenta-Seite):      H 150-180
        S >= 60, V >= 110  (heller, gesättigter Leucht-Rand;
        S-Untergrenze niedrig genug für den weichen Glow)

    Form-Filter (das "Form"-Teil von Farbe+Form):
        aspect_range:  erlaubtes Höhe/Breite-Verhältnis (Mensch ist hoch);
                       blockt breite HUD-Balken & horizontale Linien.
        min_area / max_area_frac:  Größenfenster der Bounding-Box.
        max_fill_ratio:  obere Schranke für den roten Pixel-Anteil – eine
                         dünne Outline füllt die Box kaum; eine "fette
                         Ummantellung" (massiv rotes Objekt) wird verworfen.
        min_fill_ratio:  es muss überhaupt eine Linie da sein.

    Verbindung:
        connect_ksize / connect_iter steuern das CLOSE, das die zerrissene
        Outline zu einem zusammenhängenden Loop verbindet.

    GPU:
        use_cuda=True -> PyTorch-CUDA, sonst cv2.cuda, sonst CPU.
        pip install torch --index-url https://download.pytorch.org/whl/cu124
    """

    def __init__(
        self,
        # --- Farbe (HSV) ---
        # Rot-Bereich (0-14): NUR ergänzend. Gegner-Glow ist überwiegend PINK
        # (H 150-180); reines Rot (H~8) ist fast immer Damage-Vignette, rote
        # Laternen/Kerzen/Deko. Deshalb hier hohe S/V-Schwelle, damit matte
        # Rottöne (z. B. Kerze S~60) NICHT mehr durchrutschen.
        lower1: tuple[int, int, int] = (0,  120, 120),
        upper1: tuple[int, int, int] = (10, 255, 255),
        # Pink/Magenta = das eigentliche Gegner-Signal, S-Schwelle moderat.
        # Untergrenze leicht gesenkt (S 70->55, V 110->100), damit der weiche,
        # dünne Glow weiter außen noch mitgenommen wird (mehr Recall).
        lower2: tuple[int, int, int] = (150,  55, 100),
        upper2: tuple[int, int, int] = (180, 255, 255),
        # --- Teammate-Outline (BLAU) ---
        # Eigene Mitspieler werden durch Wände BLAU umrandet (gemessen am
        # Team-Balken: H~99, S~195, V~190). Wir erkennen sie mit demselben
        # Form-Filter, markieren sie aber NUR (pink) und schließen sie aus.
        lower_blue: tuple[int, int, int] = (90,  80, 120),
        upper_blue: tuple[int, int, int] = (128, 255, 255),
        detect_teammates: bool = True,
        # --- Form (auf "generische Spielerform/-größe" getrimmt) ---
        min_area: int = 450,                 # min. Bounding-Box-Fläche (Orig-px²)
        # Ein einzelner Spieler füllt nie den halben Screen. Klein halten,
        # damit Vignette/zusammengemergte Effekt-Flecken rausfallen.
        max_area_frac: float = 0.12,         # max. Anteil am Gesamtbild
        max_w_frac: float = 0.30,            # max. Box-Breite als Bildanteil
        max_h_frac: float = 0.75,            # max. Box-Höhe als Bildanteil
        # h/w. Ein Spieler steht ~aufrecht (aspect >= ~1). Breite Wolken
        # (Kirschblüten ~0.4-0.9, Lotus ~0.5) fallen raus. Untergrenze 0.9
        # ist der Preis dafür, dass liegende/geworfene Gegner verfehlt werden
        # können – bewusst zugunsten der Präzision (kein Blüten-Spam).
        aspect_range: tuple[float, float] = (0.3, 5.0),
        # Bildschirm-relative Mindesthöhe: nur GROBES Mini-Rauschen raus.
        # ACHTUNG: ferne Gegner sind klein (~40 px @1080p) und gleich groß wie
        # Kerzen-Specks -> Größe trennt die beiden NICHT. Die eigentliche
        # Kerze-vs-Charakter-Trennung macht enclose_min + max_solid_ratio.
        # Deshalb niedrig: 0.03 = 3 % der Bildhöhe (1080p -> ~32 px). 0.0 = aus.
        min_h_frac: float = 0.03,
        # dichte rote Flächen (VFX-Bursts, gefüllte Deko) raus; eine echte
        # Outline ist eine dünne Linie -> niedrige Füllrate. Untergrenze klein
        # halten: eine OUTLINE ist NICHT gefüllt, dünne Ränder sollen NICHT
        # an einer hohen min_fill scheitern (genau dafür gibt es enclose_min).
        min_fill_ratio: float = 0.04,        # nur Rausch-Untergrenze
        max_fill_ratio: float = 0.35,        # darüber = "fette Ummantellung"/VFX
        # --- Charakter-Form: HOHLER Loop, kein gefüllter Klecks ---
        # Schlüssel-Unterscheidung Outline vs. Kerze: Eine Charakter-Outline
        # umschließt (gefüllt gedacht) einen KÖRPER -> die Silhouette deckt
        # einen guten Teil der Box ab. Ein Kringel/Bogen/Streifen umschließt
        # fast nichts.
        # 0.16 (gemessen): echte, dünn umrandete Gegner liefern eine
        # Silhouetten-Füllung von ~0.14-0.23. 0.22 war zu streng und hat sie
        # knapp verworfen; 0.16 holt sie rein, ohne leere Kringel zu öffnen.
        enclose_min: float = 0.16,           # Silhouette / Box-Fläche, min.
        # ...und innerhalb dieser Silhouette ist nur der dünne RAND rot, die
        # Mitte (Charakterkörper) NICHT. Ein gefüllter roter Klecks (Kerze,
        # Laterne, VFX-Burst) ist innen voll rot -> solid_ratio ~1 -> raus.
        max_solid_ratio: float = 0.70,       # rote Pixel / Silhouette, max.
        # Rundheit (4*pi*Fläche/Umfang^2): ein Kreis ~1.0. Runde Icons
        # (Iron-Man-Portrait, KO-Badge, Killfeed, Lotus, rundes Fenster) sind
        # nahezu kreisrund -> raus. Eine Charakterform (Kopf schmal, Körper
        # breiter, Beine) ist deutlich unrunder. >0 schaltet den Filter ein.
        max_circularity: float = 0.82,
        # Solidity = Konturfläche / Konvexhülle. Ein Charakter-Loop ist
        # zusammenhängend-kompakt; zerfaserte/sternförmige VFX-Bursts und
        # verstreute Deko haben eine löchrige Hülle -> niedrige Solidity.
        # 0.0 = aus. ~0.3 verwirft nur extrem zerfranste Formen.
        min_solidity: float = 0.30,
        # --- Verbindung der zerrissenen Outline ---
        connect_ksize: int = 5,              # Dilations-Kernel (verbindet Lücken)
        connect_iter: int = 2,               # mehr = größere Lücken überbrückt
        # 8 (gemessen): bei 22 verschmolzen rote Beams/Bars + benachbarte
        # Gegner zu EINER bildschirmbreiten Mega-Box, deren Silhouette die Box
        # kaum füllte -> alle darin steckenden Gegner fielen durch enclose_min.
        # 8 lässt die Outlines zu sauberen Einzel-Gegnern zerfallen (mehr Recall).
        merge_gap: int = 8,                  # Boxen näher als das verschmelzen
        # --- Performance / Region ---
        downscale: float = 1.0,
        # ROI blendet das HUD am Rand aus (oben Objective/Timer, unten
        # Health/Ability-Bar, seitlich Portraits/Killfeed). Default deckt die
        # zentrale Spielfläche ab; None = ganzes Bild.
        roi: tuple[float, float, float, float] | None = (0.05, 0.08, 0.90, 0.75),
        use_cuda: bool = False,
    ) -> None:
        self.lower1 = np.array(lower1, dtype=np.uint8)
        self.upper1 = np.array(upper1, dtype=np.uint8)
        self.lower2 = np.array(lower2, dtype=np.uint8)
        self.upper2 = np.array(upper2, dtype=np.uint8)
        self.lower_blue = np.array(lower_blue, dtype=np.uint8)
        self.upper_blue = np.array(upper_blue, dtype=np.uint8)
        self.detect_teammates = detect_teammates

        self.min_area = min_area
        self.max_area_frac = max_area_frac
        self.max_w_frac = max_w_frac
        self.max_h_frac = max_h_frac
        self.min_h_frac = min_h_frac
        self.aspect_range = aspect_range
        self.min_fill_ratio = min_fill_ratio
        self.max_fill_ratio = max_fill_ratio
        self.enclose_min = enclose_min
        self.max_solid_ratio = max_solid_ratio
        self.max_circularity = max_circularity
        self.min_solidity = min_solidity

        self.connect_iter = connect_iter
        self.merge_gap = merge_gap
        self._connect_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (connect_ksize, connect_ksize)
        )

        self.downscale = max(0.05, min(1.0, downscale))
        self.roi = roi

        # GPU-Backend ermitteln
        self._cuda_backend: str | None = None
        if use_cuda:
            if _TORCH and torch.cuda.is_available():
                self._cuda_backend = "torch"
                print(f"GPU aktiv (PyTorch): {torch.cuda.get_device_name(0)}")
            else:
                try:
                    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                        self._cuda_backend = "cv2"
                        print("GPU aktiv (cv2.cuda)")
                except (cv2.error, AttributeError):
                    pass
            if self._cuda_backend is None:
                print(
                    "GPU nicht verfügbar – CPU-Fallback.\n"
                    "  Tipp: pip install torch --index-url "
                    "https://download.pytorch.org/whl/cu124"
                )

    # ------------------------------------------------------------------
    # ROI
    # ------------------------------------------------------------------

    def _roi_px(self, frame: np.ndarray) -> tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        if self.roi is None:
            return (0, 0, w, h)
        rx, ry, rw, rh = self.roi
        return (int(rx * w), int(ry * h), int(rw * w), int(rh * h))

    # ------------------------------------------------------------------
    # Farb-Maske (core) – CPU + GPU, OHNE Morphologie (Linie bleibt dünn)
    # ------------------------------------------------------------------

    def _color_mask_cpu(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower1, self.upper1)
        mask |= cv2.inRange(hsv, self.lower2, self.upper2)
        return mask

    def _color_mask_torch(self, bgr: np.ndarray) -> np.ndarray:
        gpu = torch.from_numpy(bgr).cuda()
        hsv = _bgr_to_hsv_torch(gpu)
        lo1 = torch.tensor(self.lower1, device="cuda", dtype=torch.float32)
        hi1 = torch.tensor(self.upper1, device="cuda", dtype=torch.float32)
        lo2 = torch.tensor(self.lower2, device="cuda", dtype=torch.float32)
        hi2 = torch.tensor(self.upper2, device="cuda", dtype=torch.float32)
        m1 = ((hsv >= lo1) & (hsv <= hi1)).all(dim=2)
        m2 = ((hsv >= lo2) & (hsv <= hi2)).all(dim=2)
        mask = (m1 | m2)
        return (mask.to(torch.uint8) * 255).cpu().numpy()

    def _color_mask_cv2cuda(self, bgr: np.ndarray) -> np.ndarray:
        gpu = cv2.cuda_GpuMat()
        gpu.upload(bgr)
        gpu_hsv = cv2.cuda.cvtColor(gpu, cv2.COLOR_BGR2HSV)
        m1 = cv2.cuda.inRange(gpu_hsv, self.lower1, self.upper1)
        m2 = cv2.cuda.inRange(gpu_hsv, self.lower2, self.upper2)
        return cv2.cuda.bitwise_or(m1, m2).download()

    def _core_mask(self, bgr: np.ndarray) -> np.ndarray:
        if self._cuda_backend == "torch":
            return self._color_mask_torch(bgr)
        if self._cuda_backend == "cv2":
            return self._color_mask_cv2cuda(bgr)
        return self._color_mask_cpu(bgr)

    def _blue_mask_cpu(self, bgr: np.ndarray) -> np.ndarray:
        """Blaue Teammate-Outline-Maske (CPU, billig genug)."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, self.lower_blue, self.upper_blue)

    def red_mask(self, frame: np.ndarray) -> np.ndarray:
        """Öffentliche Farb-Maske (für Tuning-Tool) – volle Auflösung, CPU."""
        return self._color_mask_cpu(frame)

    # ------------------------------------------------------------------
    # Bounding-Boxen verschmelzen (Fragmente einer Outline -> 1 Charakter)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Form-Confidence: wie klar sieht der Kandidat nach Charakter-Outline aus?
    # ------------------------------------------------------------------

    def _shape_confidence(
        self, enclosed_frac: float, solid_ratio: float,
        aspect: float, solidity: float,
    ) -> float:
        """0..1 aus den Form-Maßen, jeweils relativ zu den eingestellten
        Schwellen. Höher = klarer hohler, körper-umschließender Loop."""
        def clamp(v: float) -> float:
            return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v

        # Umschließung: weiter über dem Minimum = klarer ein Körper im Loop.
        s_enc = (enclosed_frac - self.enclose_min) / max(1e-6, 0.90 - self.enclose_min)
        # Hohlheit: niedrige solid_ratio = klarer dünner Rand (Outline),
        # hohe = voller Klecks (Kerze/VFX).
        s_hollow = (self.max_solid_ratio - solid_ratio) / max(1e-6, self.max_solid_ratio - 0.12)
        # Seitenverhältnis: aufrechter Mensch ~1.3..2.8 = ideal, Ränder abfallend.
        if aspect < 1.3:
            s_asp = (aspect - self.aspect_range[0]) / max(1e-6, 1.3 - self.aspect_range[0])
        elif aspect > 2.8:
            s_asp = (self.aspect_range[1] - aspect) / max(1e-6, self.aspect_range[1] - 2.8)
        else:
            s_asp = 1.0
        # Kompaktheit des Loops.
        s_sol = (solidity - self.min_solidity) / max(1e-6, 0.95 - self.min_solidity)

        s_enc, s_hollow, s_asp, s_sol = (
            clamp(s_enc), clamp(s_hollow), clamp(s_asp), clamp(s_sol)
        )
        # Umschließung + Hohlheit sind die Kern-Unterscheider (Charakter vs.
        # Kerze), Aspect/Solidity nur feinjustierend.
        return 0.35 * s_enc + 0.35 * s_hollow + 0.15 * s_asp + 0.15 * s_sol

    @staticmethod
    def _merge_boxes(
        boxes: list[tuple[int, int, int, int]], gap: int,
        cap_w: int | None = None, cap_h: int | None = None,
    ) -> list[tuple[int, int, int, int]]:
        """Vereint Boxen, die sich überlappen oder näher als `gap` liegen.

        `cap_w`/`cap_h`: eine Verschmelzung wird VERWEIGERT, wenn die
        Ergebnis-Box breiter/höher als der Cap würde. So können rote
        Attack-Beams / Health-Bars NICHT mehr eine bildschirmbreite Kette
        bilden, die echte Gegner verschluckt – sie bleiben getrennt."""
        boxes = list(boxes)
        changed = True
        while changed:
            changed = False
            out: list[tuple[int, int, int, int]] = []
            while boxes:
                x, y, w, h = boxes.pop()
                x2, y2 = x + w, y + h
                i = 0
                while i < len(out):
                    ox, oy, ow, oh = out[i]
                    ox2, oy2 = ox + ow, oy + oh
                    if (x < ox2 + gap and ox < x2 + gap
                            and y < oy2 + gap and oy < y2 + gap):
                        nx, ny = min(x, ox), min(y, oy)
                        nx2, ny2 = max(x2, ox2), max(y2, oy2)
                        if ((cap_w is None or nx2 - nx <= cap_w)
                                and (cap_h is None or ny2 - ny <= cap_h)):
                            x, y, x2, y2 = nx, ny, nx2, ny2
                            out.pop(i)
                            changed = True
                            i = 0
                            continue
                    i += 1
                out.append((x, y, x2 - x, y2 - y))
            boxes = out
        return boxes

    # ------------------------------------------------------------------
    # Haupt-Detektion
    # ------------------------------------------------------------------

    def detect(
        self, frame: np.ndarray, outlines_only: bool = True
    ) -> list[OutlineDetection]:
        """Erkennt rote Gegner- UND blaue Teammate-Outlines (Farbe + Form).

        Beide Teams durchlaufen denselben Form-Filter; getrennt werden sie nur
        über die Farb-Maske. `outlines_only` bleibt aus Kompatibilität."""
        rx, ry, rw, rh = self._roi_px(frame)
        cropped = frame[ry:ry + rh, rx:rx + rw]

        scale = self.downscale
        if scale < 1.0:
            small = cv2.resize(cropped, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        else:
            small = cropped
        inv_scale = 1.0 / scale

        frame_h, frame_w = frame.shape[:2]

        # ROTE Outline = Gegner.
        detections = self._candidates_from_mask(
            self._core_mask(small), "enemy",
            frame_h, frame_w, inv_scale, rx, ry,
        )
        # BLAUE Outline = Teammate (nur markieren/ausschließen).
        if self.detect_teammates:
            detections += self._candidates_from_mask(
                self._blue_mask_cpu(small), "teammate",
                frame_h, frame_w, inv_scale, rx, ry,
            )
        return detections

    def _candidates_from_mask(
        self, core: np.ndarray, team: str,
        frame_h: int, frame_w: int, inv_scale: float, rx: int, ry: int,
    ) -> list[OutlineDetection]:
        """Form-Filter-Pipeline für EINE Farb-Maske (rot oder blau)."""
        frame_area = frame_h * frame_w

        # Lücken überbrücken: reine Dilation verbindet die zerrissene Outline
        # (CLOSE würde die Brücken sofort wieder wegerodieren).
        connected = cv2.dilate(core, self._connect_kernel,
                               iterations=self.connect_iter)

        contours, _ = cv2.findContours(
            connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        frags = []
        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw > 1 and bh > 1:
                frags.append(((bx, by, bw, bh), c))
        raw = [r for r, _ in frags]
        # Verschmelzungs-Caps in (ggf. verkleinerten) Maskenkoordinaten: ein
        # einzelner Charakter ist nie bildschirmbreit. Verhindert, dass Beams/
        # Bars eine durchgehende Kette bilden, die Gegner verschluckt.
        cap_w = int(frame_w * self.max_w_frac / inv_scale)
        cap_h = int(frame_h * self.max_h_frac / inv_scale)
        boxes = self._merge_boxes(raw, self.merge_gap, cap_w, cap_h)

        detections: list[OutlineDetection] = []

        for (x, y, w, h) in boxes:
            if w == 0 or h == 0:
                continue

            box_area_orig = (w * h) * inv_scale * inv_scale
            w_orig = w * inv_scale
            h_orig = h * inv_scale

            # --- Größe ---
            if box_area_orig < self.min_area:
                continue
            # Bildschirm-relative Mindesthöhe: kein winziger Kerzen-Glow.
            if h_orig < frame_h * self.min_h_frac:
                continue
            if box_area_orig > frame_area * self.max_area_frac:
                continue
            # Dimensions-Caps: eine bildschirm-spannende Box (Damage-Vignette,
            # zusammengemergte Effekt-Flecken) ist kein einzelner Gegner.
            if w_orig > frame_w * self.max_w_frac:
                continue
            if h_orig > frame_h * self.max_h_frac:
                continue

            # --- Form: Seitenverhältnis (Mensch ist hoch) ---
            aspect = h / w
            if not (self.aspect_range[0] <= aspect <= self.aspect_range[1]):
                continue

            # --- Thinness: Outline = dünne Linie, keine "fette Ummantellung" ---
            # Auf der CORE-Maske gemessen, damit das CLOSE nicht verfälscht.
            core_box = core[y:y + h, x:x + w]
            red_px = cv2.countNonZero(core_box)
            fill_ratio = red_px / float(w * h)
            if fill_ratio < self.min_fill_ratio:
                continue
            if fill_ratio > self.max_fill_ratio:
                continue

            # --- Member-Konturen einsammeln (die in diese Box fallen) ---
            bx2, by2 = x + w, y + h
            members = [
                c for (fx, fy, fw, fh), c in frags
                if fx < bx2 and x < fx + fw and fy < by2 and y < fy + fh
            ]
            if not members:
                continue

            # --- Charakter-Form: HOHLER Loop, kein gefüllter Klecks ---
            # Member-Konturen gefüllt in eine box-lokale Maske zeichnen =
            # die Silhouette, die der Outline-Loop umschließt.
            local = np.zeros((h, w), dtype=np.uint8)
            shifted = [c - np.array([[x, y]], dtype=c.dtype) for c in members]
            cv2.drawContours(local, shifted, -1, 255, cv2.FILLED)
            sil_area = cv2.countNonZero(local)
            if sil_area == 0:
                continue
            # 1) Umschließt der Loop überhaupt einen Körper? (kein Kringel/Bogen)
            enclosed_frac = sil_area / float(w * h)
            if enclosed_frac < self.enclose_min:
                continue
            # 2) Ist das Innere HOHL? Nur der dünne Rand ist rot; ein gefüllter
            #    roter Klecks (Kerze/Laterne/VFX) ist innen voll -> raus.
            red_in_sil = cv2.countNonZero(cv2.bitwise_and(core_box, local))
            solid_ratio = red_in_sil / float(sil_area)
            if solid_ratio > self.max_solid_ratio:
                continue

            # --- Form: Solidity der größten Member-Kontur ---
            # (Charakter-Loop = kompakt; sternförmige VFX = zerfasert.)
            biggest = max(members, key=cv2.contourArea)
            c_area = cv2.contourArea(biggest)
            hull_area = cv2.contourArea(cv2.convexHull(biggest))
            solidity = c_area / hull_area if hull_area > 0 else 0.0
            if self.min_solidity > 0.0 and solidity < self.min_solidity:
                continue

            # --- Form: Rundheit (rundes Icon raus) ---
            # 4*pi*A/U^2: Kreis ~1.0. Ein rundes Portrait/KO-Badge/Lotus ist
            # nahezu kreisrund; eine Charakterform (Kopf->Schultern->Beine) ist
            # deutlich unrunder. Auf die SILHOUETTE (gefüllte Member) bezogen,
            # damit ein hohler Ring nicht künstlich "unrund" wirkt.
            sil_perim = 0.0
            sil_contours, _ = cv2.findContours(
                local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if sil_contours:
                sil_perim = cv2.arcLength(
                    max(sil_contours, key=cv2.contourArea), True
                )
            circularity = (
                4.0 * np.pi * sil_area / (sil_perim * sil_perim)
                if sil_perim > 1e-6 else 0.0
            )
            if self.max_circularity > 0.0 and circularity > self.max_circularity:
                continue

            # Konturen in Originalkoordinaten umrechnen (Downscale + ROI).
            scaled = []
            for c in members:
                cc = c.astype(np.float32)
                cc[:, :, 0] = cc[:, :, 0] * inv_scale + rx
                cc[:, :, 1] = cc[:, :, 1] * inv_scale + ry
                scaled.append(cc.astype(np.int32))

            box = (
                int(x * inv_scale) + rx, int(y * inv_scale) + ry,
                int(w * inv_scale), int(h * inv_scale),
            )
            conf = self._shape_confidence(
                enclosed_frac, solid_ratio, aspect, solidity
            )
            detections.append(
                OutlineDetection(
                    box=box,
                    area=box_area_orig,
                    fill_ratio=fill_ratio,
                    aspect=aspect,
                    is_outline=True,
                    team=team,
                    circularity=circularity,
                    contours=scaled,
                    solidity=solidity,
                    enclosed_frac=enclosed_frac,
                    solid_ratio=solid_ratio,
                    shape_confidence=conf,
                    confidence=conf,
                )
            )
        return detections


# Rückwärtskompatibler Alias
OutlineDetector = RedOutlineDetector
