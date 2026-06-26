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

    def __init__(
        self,
        path: str | Path,
        skip: int = 1,
        start_sec: float = 0.0,
        max_frames: int | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video nicht gefunden: {self.path}")
        # Nur jeden `skip`-ten Frame liefern (>=1). Spart Rechenzeit.
        self.skip = max(1, skip)
        # An welcher Sekunde einsteigen (Seek) und wie viele Frames max. liefern.
        # Praktisch beim Tunen großer Dateien – nicht jedes Mal alles durchlaufen.
        self.start_sec = max(0.0, start_sec)
        self.max_frames = max_frames
        self.cap: cv2.VideoCapture | None = None

    def __enter__(self) -> "VideoSource":
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Video konnte nicht geöffnet werden: {self.path}")
        if self.start_sec > 0:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, self.start_sec * 1000.0)
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
        yielded = 0
        while True:
            if self.max_frames is not None and yielded >= self.max_frames:
                break
            ok, frame = self.cap.read()
            if not ok:
                break
            if index % self.skip == 0:
                yield frame
                yielded += 1
            index += 1
