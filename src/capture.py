"""Liest Frames aus einer Videodatei."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class VideoSource:
    """Kapselt das Einlesen einer Videodatei mit OpenCV.

    Benutzung:
        with VideoSource("clip.mp4", skip=1) as src:
            for frame in src:
                ...
    """

    def __init__(self, path: str | Path, skip: int = 1) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video nicht gefunden: {self.path}")
        # Nur jeden `skip`-ten Frame liefern (>=1). Spart Rechenzeit.
        self.skip = max(1, skip)
        self.cap: cv2.VideoCapture | None = None

    def __enter__(self) -> "VideoSource":
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Video konnte nicht geöffnet werden: {self.path}")
        return self

    def __exit__(self, *exc) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def fps(self) -> float:
        if self.cap is None:
            return 0.0
        return self.cap.get(cv2.CAP_PROP_FPS) or 0.0

    @property
    def size(self) -> tuple[int, int]:
        """(Breite, Höhe) des Videos."""
        if self.cap is None:
            return (0, 0)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def __iter__(self) -> Iterator[np.ndarray]:
        if self.cap is None:
            raise RuntimeError("VideoSource muss als Context-Manager genutzt werden.")
        index = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            if index % self.skip == 0:
                yield frame
            index += 1
