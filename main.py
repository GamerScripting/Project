"""CLI-Einstiegspunkt: analysiert ein Video auf rote Outlines, Tags und Movement."""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from src.capture import VideoSource
from src.movement import MovementDetector
from src.outlines import RedOutlineDetector
from src.pipeline import DetectionPipeline
from src.tags import TagDetector


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rote Outlines / Tags / Movement im Video erkennen.")
    p.add_argument("--video", required=True, help="Pfad zur Videodatei")
    p.add_argument("--skip", type=int, default=1, help="Nur jeden N-ten Frame analysieren")
    p.add_argument("--start", type=float, default=0.0, help="Bei Sekunde X ins Video einsteigen")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Höchstens N Frames verarbeiten (gut zum Tunen großer Dateien)")
    p.add_argument("--no-ocr", action="store_true", help="OCR (Tags) abschalten – schneller")
    p.add_argument("--gpu", action="store_true", help="EasyOCR auf der GPU laufen lassen")
    p.add_argument("--save", help="Annotiertes Ergebnis als Videodatei speichern")
    p.add_argument("--no-window", action="store_true", help="Kein Vorschaufenster anzeigen")
    p.add_argument("--show-solid", action="store_true",
                   help="Massive rote Objekte mit anzeigen (orange) statt verwerfen")
    p.add_argument("--bench", action="store_true",
                   help="Pro Frame die reine Detection-Zeit (ms) ausgeben")

    # Outline-Trennung & Geschwindigkeit
    p.add_argument("--fill-ratio", type=float, default=0.35,
                   help="Max. Rot-Anteil in der Box, damit es als Outline gilt (0..1)")
    p.add_argument("--downscale", type=float, default=1.0,
                   help="Frame vor Outline-Analyse verkleinern (z. B. 0.5 = schneller)")
    p.add_argument("--min-area", type=int, default=120,
                   help="Mindestfläche einer Kontur in Pixel²")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    pipeline = DetectionPipeline(
        outline_detector=RedOutlineDetector(
            min_area=args.min_area,
            max_fill_ratio=args.fill_ratio,
            downscale=args.downscale,
        ),
        movement_detector=MovementDetector(),
        tag_detector=TagDetector(gpu=args.gpu),
        use_ocr=not args.no_ocr,
        outlines_only=not args.show_solid,
    )

    writer = None
    n_frames = 0
    total_ms = 0.0
    try:
        with VideoSource(args.video, skip=args.skip,
                         start_sec=args.start, max_frames=args.max_frames) as src:
            if args.save:
                w, h = src.size
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps = src.fps or 30.0
                writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

            for frame in src:
                t0 = time.perf_counter()
                result = pipeline.process(frame)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                n_frames += 1
                result.frame_id = n_frames
                total_ms += dt_ms
                if args.bench:
                    print(f"Frame {n_frames}: {dt_ms:6.2f} ms  "
                          f"(outlines={len(result.outlines)}, "
                          f"moves={len(result.movements)}, tags={len(result.tags)})")

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

    if n_frames:
        avg = total_ms / n_frames
        print(f"\nØ {avg:.2f} ms/Frame über {n_frames} Frames "
              f"(~{1000.0 / avg:.0f} FPS möglich).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
