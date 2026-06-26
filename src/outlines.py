"""Erkennung farbiger (roter/pink) Umrandungen via HSV-Farbfilter.

Besonderheiten:
- Rot liegt in HSV an beiden Enden des Hue-Kreises (~0 und ~180), deshalb
  werden ZWEI Farbbereiche kombiniert.
- Echte Outlines werden von massiven roten Objekten über die *Füllrate*
  unterschieden: Eine Outline ist ein dünner Rand um einen nicht-roten
  Innenbereich (wenig rote Pixel in der Box), ein rotes Objekt ist fast
  vollständig rot gefüllt.
- GPU-Beschleunigung via PyTorch (torch.cuda) wenn verfügbar, sonst
  cv2.cuda, sonst CPU-Fallback.
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
# GPU-Hilfsfunktionen (PyTorch)
# ---------------------------------------------------------------------------

def _bgr_to_hsv_torch(bgr: "torch.Tensor") -> "torch.Tensor":
    """BGR-uint8-Tensor (H,W,3) auf GPU → HSV im OpenCV-Maßstab (H:0-180, S/V:0-255)."""
    b = bgr[:, :, 0].float() / 255.0
    g = bgr[:, :, 1].float() / 255.0
    r = bgr[:, :, 2].float() / 255.0
    rgb = torch.stack([r, g, b], 2)
    maxc = rgb.max(2).values
    minc = rgb.min(2).values
    diff = maxc - minc
    v = maxc
    s = torch.where(maxc > 1e-6, diff / maxc, torch.zeros_like(maxc))
    h = torch.zeros_like(maxc)
    eps = 1e-6
    mr = (maxc == r) & (diff > eps)
    mg = (maxc == g) & (diff > eps) & ~mr
    mb = ~mr & ~mg & (diff > eps)
    h[mr] = ((g - b) / (diff + eps))[mr] % 6
    h[mg] = ((b - r) / (diff + eps) + 2)[mg]
    h[mb] = ((r - g) / (diff + eps) + 4)[mb]
    h = (h * 30) % 180
    return torch.stack([h, s * 255, v * 255], 2)


def _morph_open_torch(mask: "torch.Tensor", ksize: int = 3) -> "torch.Tensor":
    """Binäres Opening (Erosion→Dilation) auf einem 2-D GPU-Float-Tensor."""
    m = mask.float().unsqueeze(0).unsqueeze(0)
    p = ksize // 2
    m = -_F.max_pool2d(-m, ksize, stride=1, padding=p)   # erosion
    m = _F.max_pool2d(m, ksize, stride=1, padding=p)     # dilation
    return m.squeeze(0).squeeze(0)


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------

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
        return {
            "box": list(self.box),
            "center": list(self.center),
            "fill": round(self.fill_ratio, 3),
        }


# ---------------------------------------------------------------------------
# Detektor
# ---------------------------------------------------------------------------

class RedOutlineDetector:
    """Findet rote/pinke Charakter-Outlines (z. B. Marvel Rivals Gegner-Glow).

    GPU-Beschleunigung:
        use_cuda=True aktiviert automatisch PyTorch-CUDA wenn verfügbar,
        danach cv2.cuda als Fallback, dann CPU.
        Tipp: pip install torch --index-url https://download.pytorch.org/whl/cu124

    HSV-Standardwerte (angepasst auf den Charakter-Outline-Glow):
        Hue  0-12  (Rot/Coral-Seite) und 155-180 (Pink/Magenta-Seite)
        Sat  70+   (niedriger Wert fängt den weichen Glow-Effekt)
        Val  140+  (hell/leuchtend)
    """

    def __init__(
        self,
        # Salmon/Coral-Rot (untere Hue-Seite) – breiter Bereich für Glow
        lower1: tuple[int, int, int] = (0,   70, 140),
        upper1: tuple[int, int, int] = (12, 255, 255),
        # Pink/Magenta (obere Hue-Seite)
        lower2: tuple[int, int, int] = (155,  70, 140),
        upper2: tuple[int, int, int] = (180, 255, 255),
        min_area: int = 150,
        max_fill_ratio: float = 0.35,
        downscale: float = 1.0,
        roi: tuple[float, float, float, float] | None = None,
        use_cuda: bool = False,
    ) -> None:
        self.lower1 = np.array(lower1, dtype=np.uint8)
        self.upper1 = np.array(upper1, dtype=np.uint8)
        self.lower2 = np.array(lower2, dtype=np.uint8)
        self.upper2 = np.array(upper2, dtype=np.uint8)
        self.min_area = min_area
        self.max_fill_ratio = max_fill_ratio
        self.downscale = max(0.05, min(1.0, downscale))
        self.roi = roi
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # GPU-Backend ermitteln
        self._cuda_backend: str | None = None
        if use_cuda:
            if _TORCH:
                import torch
                if torch.cuda.is_available():
                    self._cuda_backend = "torch"
                    print(f"GPU aktiv (PyTorch): {torch.cuda.get_device_name(0)}")
            if self._cuda_backend is None:
                try:
                    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                        self._gpu_morph = cv2.cuda.createMorphologyFilter(
                            cv2.MORPH_OPEN, cv2.CV_8UC1, self._kernel
                        )
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
    # Hilfsmethoden
    # ------------------------------------------------------------------

    def _roi_px(self, frame: np.ndarray) -> tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        if self.roi is None:
            return (0, 0, w, h)
        rx, ry, rw, rh = self.roi
        return (int(rx * w), int(ry * h), int(rw * w), int(rh * h))

    # ------------------------------------------------------------------
    # CPU-Pfad
    # ------------------------------------------------------------------

    def red_mask(self, frame: np.ndarray) -> np.ndarray:
        """Binäre Maske aller roten/pinken Pixel (CPU)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower1, self.upper1)
        mask |= cv2.inRange(hsv, self.lower2, self.upper2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        return mask

    # ------------------------------------------------------------------
    # GPU-Pfade
    # ------------------------------------------------------------------

    def _mask_torch(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """GPU-Pfad via PyTorch: Resize + HSV + Masken auf CUDA."""
        import torch
        scale = self.downscale
        gpu = torch.from_numpy(frame).cuda()           # H, W, 3 uint8
        if scale < 1.0:
            t = gpu.float().permute(2, 0, 1).unsqueeze(0)
            nh, nw = int(frame.shape[0] * scale), int(frame.shape[1] * scale)
            t = _F.interpolate(t, (nh, nw), mode="area")
            gpu = t.squeeze(0).permute(1, 2, 0).byte()
        hsv = _bgr_to_hsv_torch(gpu)
        lo1 = torch.tensor(self.lower1, device="cuda").float()
        hi1 = torch.tensor(self.upper1, device="cuda").float()
        lo2 = torch.tensor(self.lower2, device="cuda").float()
        hi2 = torch.tensor(self.upper2, device="cuda").float()
        m1 = ((hsv >= lo1) & (hsv <= hi1)).all(2)
        m2 = ((hsv >= lo2) & (hsv <= hi2)).all(2)
        mask_gpu = (m1 | m2).float()
        mask_gpu = _morph_open_torch(mask_gpu)
        return (mask_gpu.cpu().numpy() * 255).astype(np.uint8), 1.0 / scale

    def _mask_cv2cuda(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """GPU-Pfad via cv2.cuda."""
        scale = self.downscale
        gpu = cv2.cuda_GpuMat()
        gpu.upload(frame)
        if scale < 1.0:
            nw, nh = int(frame.shape[1] * scale), int(frame.shape[0] * scale)
            gpu = cv2.cuda.resize(gpu, (nw, nh), interpolation=cv2.INTER_AREA)
        gpu_hsv = cv2.cuda.cvtColor(gpu, cv2.COLOR_BGR2HSV)
        m1 = cv2.cuda.inRange(gpu_hsv, self.lower1, self.upper1)
        m2 = cv2.cuda.inRange(gpu_hsv, self.lower2, self.upper2)
        gpu_mask = cv2.cuda.bitwise_or(m1, m2)
        gpu_mask = self._gpu_morph.apply(gpu_mask)
        return gpu_mask.download(), 1.0 / scale

    # ------------------------------------------------------------------
    # Haupt-Detektion
    # ------------------------------------------------------------------

    def detect(
        self, frame: np.ndarray, outlines_only: bool = True
    ) -> list[OutlineDetection]:
        rx, ry, rw, rh = self._roi_px(frame)
        cropped = frame[ry:ry + rh, rx:rx + rw]

        if self._cuda_backend == "torch":
            mask, inv_scale = self._mask_torch(cropped)
        elif self._cuda_backend == "cv2":
            mask, inv_scale = self._mask_cv2cuda(cropped)
        else:
            scale = self.downscale
            small = (cv2.resize(cropped, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA)
                     if scale < 1.0 else cropped)
            mask = self.red_mask(small)
            inv_scale = 1.0 / scale

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: list[OutlineDetection] = []
        frame_area = frame.shape[0] * frame.shape[1]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Bounding-Box darf nicht den ganzen Frame füllen
            if (w * h * inv_scale * inv_scale) > frame_area * 0.5:
                continue

            roi_mask = mask[y:y + h, x:x + w]
            box_area = w * h
            fill_ratio = (cv2.countNonZero(roi_mask) / box_area) if box_area else 0.0
            is_outline = fill_ratio <= self.max_fill_ratio

            if outlines_only and not is_outline:
                continue

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
