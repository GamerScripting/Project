"""Interaktives Tuning-Tool für die Outline-Farbe.

Lädt ein Bild oder ein einzelnes Video-Frame und zeigt:
  - links:  Originalbild mit erkannten Outlines (grün) / Objekten (orange)
  - rechts: die aktuelle Farbmaske

Bedienung:
  - Trackbars (oben): HSV-Grenzen, Mindestfläche, Füllrate, Downscale
  - LINKSKLICK ins Bild: gibt den HSV-Wert des Pixels aus (zum Farbe finden)
  - 'p' druckt die aktuellen Werte als fertige main.py-Argumente
  - 'q' beendet

Benutzung:
    python tools/tune.py --image screenshot.png
    python tools/tune.py --video clip.mp4 --frame 500
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from src.outlines import RedOutlineDetector  # noqa: E402

WIN = "Tuning (q=quit, p=print, Klick=HSV)"


def load_frame(args) -> np.ndarray:
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"Bild nicht lesbar: {args.image}")
        return img
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Video nicht lesbar: {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Frame {args.frame} nicht lesbar.")
    return img


def _nop(_):  # Trackbar-Callback (ungenutzt)
    pass


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param["frame"]
        if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
            hsv = cv2.cvtColor(frame[y:y + 1, x:x + 1], cv2.COLOR_BGR2HSV)[0, 0]
            print(f"Klick @({x},{y})  HSV = {tuple(int(v) for v in hsv)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--video")
    ap.add_argument("--frame", type=int, default=0)
    args = ap.parse_args()
    if not args.image and not args.video:
        raise SystemExit("Bitte --image ODER --video angeben.")

    frame = load_frame(args)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_mouse, {"frame": frame})

    # Trackbars (Startwerte = Orange-Default)
    defs = [
        ("H low", 0, 180), ("H high", 20, 180),
        ("S low", 90, 255), ("S high", 255, 255),
        ("V low", 90, 255), ("V high", 255, 255),
        ("min area", 120, 5000),
        ("fill x100", 35, 100),
        ("downscale x100", 100, 100),
    ]
    for name, val, mx in defs:
        cv2.createTrackbar(name, WIN, val, mx, _nop)

    def g(n):
        return cv2.getTrackbarPos(n, WIN)

    while True:
        hl, hh = g("H low"), g("H high")
        sl, sh = g("S low"), g("S high")
        vl, vh = g("V low"), g("V high")
        downscale = max(0.1, g("downscale x100") / 100.0)
        det = RedOutlineDetector(
            lower1=(hl, sl, vl), upper1=(hh, sh, vh),
            lower2=(170, sl, vl), upper2=(180, sh, vh),
            min_area=max(1, g("min area")),
            max_fill_ratio=g("fill x100") / 100.0,
            downscale=downscale,
        )
        dets = det.detect(frame, outlines_only=False)
        mask = det.red_mask(frame)

        vis = frame.copy()
        for d in dets:
            x, y, w, h = d.box
            color = (0, 255, 0) if d.is_outline else (0, 165, 255)
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(vis, f"{d.fill_ratio:.2f}", (x, max(0, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combo = np.hstack([vis, mask_bgr])
        cv2.imshow(WIN, combo)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            print("\n--- main.py-Werte (Outline = grün) ---")
            print(f"H {hl}-{hh}  S {sl}-{sh}  V {vl}-{vh}  "
                  f"| --fill-ratio {g('fill x100')/100:.2f} "
                  f"--min-area {g('min area')} --downscale {downscale:.2f}\n")

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
