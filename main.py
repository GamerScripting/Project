"""CLI-Einstiegspunkt: analysiert ein Video auf Outlines, Tags und Movement."""

from __future__ import annotations

import argparse
import sys

import cv2

from src.capture import VideoSource
from src.movement import MovementDetector
from src.outlines import OutlineDetector
from src.pipeline import DetectionPipeline
from src.tags import TagDetector


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Outlines / Tags / Movement im Video erkennen.")
    p.add_argument("--video", required=True, help="Pfad zur Videodatei")
    p.add_argument("--skip", type=int, default=1, help="Nur jeden N-ten Frame analysieren")
    p.add_argument("--no-ocr", action="store_true", help="OCR (Tags) abschalten – schneller")
    p.add_argument("--gpu", action="store_true", help="EasyOCR auf der GPU laufen lassen")
    p.add_argument("--save", help="Annotiertes Ergebnis als Videodatei speichern")
    p.add_argument("--no-window", action="store_true", help="Kein Vorschaufenster anzeigen")

    # Outline-Farbe (HSV). Default = kräftiges Grün.
    p.add_argument("--hsv-lower", type=int, nargs=3, metavar=("H", "S", "V"),
                   default=[40, 80, 80], help="Untere HSV-Grenze der Outline-Farbe")
    p.add_argument("--hsv-upper", type=int, nargs=3, metavar=("H", "S", "V"),
                   default=[80, 255, 255], help="Obere HSV-Grenze der Outline-Farbe")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    pipeline = DetectionPipeline(
        outline_detector=OutlineDetector(
            lower=tuple(args.hsv_lower), upper=tuple(args.hsv_upper)
        ),
        movement_detector=MovementDetector(),
        tag_detector=TagDetector(gpu=args.gpu),
        use_ocr=not args.no_ocr,
    )

    writer = None
    try:
        with VideoSource(args.video, skip=args.skip) as src:
            if args.save:
                w, h = src.size
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps = src.fps or 30.0
                writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

            for frame in src:
                result = pipeline.process(frame)
                annotated = pipeline.draw(frame, result)

                if writer is not None:
                    writer.write(annotated)

                if not args.no_window:
                    cv2.imshow("Detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
