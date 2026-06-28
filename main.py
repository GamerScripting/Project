"""CLI-Einstiegspunkt: analysiert ein Video auf rote Outlines, Tags und Movement."""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from src.capture import VideoSource
from src.movement import MotionVerifier
from src.outlines import RedOutlineDetector
from src.pipeline import DetectionPipeline
from src.tags import TagDetector


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rote Outlines / Tags / Movement im Video erkennen.")
    p.add_argument("--video", required=True, help="Pfad zur Videodatei")
    p.add_argument("--skip", type=int, default=3,
                   help="Nur jeden N-ten Frame analysieren (Default 3 = schneller)")
    p.add_argument("--start", type=float, default=0.0, help="Bei Sekunde X ins Video einsteigen")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Höchstens N Frames verarbeiten (gut zum Tunen großer Dateien)")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="Nach N Sekunden Verarbeitungszeit beenden und das bis "
                        "dahin gerenderte Video als fertig speichern")
    p.add_argument("--no-ocr", action="store_true", help="OCR (Tags) abschalten – schneller")
    p.add_argument("--ocr-interval-ms", type=float, default=100.0,
                   help="OCR (EasyOCR, teuerster Schritt) nur alle N "
                        "Millisekunden Videozeit rechnen; Tags dazwischen "
                        "weiterverwenden. Höher = schneller. 0 = jedes Frame.")
    p.add_argument("--gpu", action="store_true", help="EasyOCR auf der GPU laufen lassen")
    p.add_argument("--cuda", action="store_true",
                   help="Outline-Detection auf der NVIDIA-GPU (CUDA) beschleunigen")
    p.add_argument("--save", help="Annotiertes Ergebnis als Videodatei speichern")
    p.add_argument("--no-window", action="store_true", help="Kein Vorschaufenster anzeigen")
    p.add_argument("--no-motion", action="store_true",
                   help="Bewegungs-Verifikation aus (alle Outlines orange)")
    p.add_argument("--no-confidence", action="store_true",
                   help="Confidence-Prozentwerte im Overlay ausblenden")
    p.add_argument("--no-self", action="store_true",
                   help="Eigenen Charakter NICHT segmentieren/markieren (spart ~20ms)")
    p.add_argument("--boost-red", action="store_true",
                   help="Rot im Bild visuell anheben (besser sichtbar)")
    p.add_argument("--boost-only", action="store_true",
                   help="NUR die Rot-Bildbearbeitung zeigen: jeden Frame durch "
                        "_boost_red schicken, KEINE Erkennung/Overlays zeichnen. "
                        "Sehr schnell (keine Detection/OCR).")
    p.add_argument("--show-solid", action="store_true",
                   help="Massive rote Objekte mit anzeigen (orange) statt verwerfen")
    p.add_argument("--bench", action="store_true",
                   help="Pro Frame die reine Detection-Zeit (ms) ausgeben")

    # Outline-Trennung & Geschwindigkeit
    p.add_argument("--fill-ratio", type=float, default=0.35,
                   help="Max. Rot-Anteil in der Box, damit es als Outline gilt (0..1)")
    p.add_argument("--min-fill", type=float, default=0.04,
                   help="Min. Rot-Anteil in der Box (nur Rausch-Untergrenze; "
                        "eine Outline ist NICHT gefüllt -> niedrig lassen)")
    p.add_argument("--min-height", type=float, default=0.03,
                   help="Min. Box-Höhe als Bildanteil (0..1). Nur Mini-Rausch "
                        "raus; ferne Gegner sind klein. 0 = aus.")
    p.add_argument("--enclose-min", type=float, default=0.16,
                   help="Min. gefüllte Silhouette / Box. Der Loop muss einen "
                        "Körper umschließen (kein Kringel/Streifen). 0.16 = "
                        "gemessener Recall-Sweet-Spot für dünne Outlines.")
    p.add_argument("--merge-gap", type=int, default=8,
                   help="Boxen näher als N px verschmelzen. Niedrig (8) hält "
                        "benachbarte Gegner/Beams getrennt -> mehr Recall; "
                        "hoch fasst Outline-Fragmente stärker zusammen.")
    p.add_argument("--max-solid", type=float, default=0.70,
                   help="Max. Rot-Anteil INNERHALB der Silhouette. Hoch = "
                        "voller roter Klecks (Kerze/VFX), keine hohle Outline.")
    p.add_argument("--max-circularity", type=float, default=0.82,
                   help="Max. Rundheit (1=Kreis). Runde Icons (Portrait/KO/"
                        "Lotus) raus; Charakterform ist unrunder. 0 = aus.")
    p.add_argument("--no-teammates", action="store_true",
                   help="Blaue Mitspieler-Outlines NICHT erkennen/ausschließen")
    p.add_argument("--aspect-min", type=float, default=0.3,
                   help="Min. Höhe/Breite einer Box. Niedrig (0.3) = Max-Recall, "
                        "fängt Sprung-/Ausfall-Posen (holt aber Lotus/VFX rein).")
    p.add_argument("--aspect-max", type=float, default=5.0,
                   help="Max. Höhe/Breite einer Box")
    p.add_argument("--min-solidity", type=float, default=0.30,
                   help="Min. Solidity (Konturfläche/Konvexhülle). Höher = nur "
                        "kompakte Charakterformen, zerfaserte VFX raus. 0 = aus.")
    p.add_argument("--motion-coherence", type=float, default=0.35,
                   help="Anteil der Box-Pixel, die sich gleichgerichtet bewegen "
                        "müssen (Charakter vs. Schatten). 0 = aus.")
    p.add_argument("--motion-min-flow", type=float, default=0.6,
                   help="Min. Fluss-Stärke, ab der ein Pixel als bewegt zählt.")
    p.add_argument("--downscale", type=float, default=1.0,
                   help="Frame vor Outline-Analyse verkleinern (z. B. 0.5 = schneller)")
    p.add_argument("--min-area", type=int, default=450,
                   help="Mindest-Bounding-Box-Fläche in Pixel²")
    p.add_argument("--roi", type=float, nargs=4, metavar=("X", "Y", "W", "H"),
                   default=None,
                   help="Detection-Region als Anteile 0..1, z. B. 0 0.15 1 0.7. "
                        "Ohne Angabe gilt der HUD-Default 0.05 0.08 0.90 0.75.")
    p.add_argument("--no-roi", action="store_true",
                   help="ROI abschalten – ganzes Bild analysieren (inkl. HUD)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # ROI-Auflösung: --no-roi = ganzes Bild, --roi X Y W H überschreibt,
    # sonst greift der HUD-Default (zentrale Spielfläche).
    if args.no_roi:
        roi = None
    elif args.roi:
        roi = tuple(args.roi)
    else:
        roi = (0.05, 0.08, 0.90, 0.75)

    pipeline = DetectionPipeline(
        outline_detector=RedOutlineDetector(
            min_area=args.min_area,
            min_h_frac=args.min_height,
            max_fill_ratio=args.fill_ratio,
            min_fill_ratio=args.min_fill,
            enclose_min=args.enclose_min,
            merge_gap=args.merge_gap,
            max_solid_ratio=args.max_solid,
            max_circularity=args.max_circularity,
            detect_teammates=not args.no_teammates,
            aspect_range=(args.aspect_min, args.aspect_max),
            min_solidity=args.min_solidity,
            downscale=args.downscale,
            roi=roi,
            use_cuda=args.cuda,
        ),
        motion_verifier=MotionVerifier(
            min_coherence=args.motion_coherence,
            min_flow=args.motion_min_flow,
        ),
        tag_detector=TagDetector(gpu=args.gpu),
        use_ocr=not args.no_ocr,
        use_motion=not args.no_motion,
        use_self=not args.no_self,
        boost_red=args.boost_red,
        outlines_only=not args.show_solid,
    )

    writer = None
    n_frames = 0
    total_ms = 0.0
    try:
        with VideoSource(args.video, skip=args.skip,
                         start_sec=args.start, max_frames=args.max_frames) as src:
            # OCR-Intervall (ms Videozeit) -> Anzahl VERARBEITETER Frames.
            # Verarbeitete Frames liegen skip/fps Sekunden auseinander.
            if args.ocr_interval_ms and args.ocr_interval_ms > 0:
                src_fps = src.fps or 30.0
                frame_dt_ms = 1000.0 * args.skip / src_fps
                ocr_every = max(1, round(args.ocr_interval_ms / frame_dt_ms))
                pipeline.set_ocr_every(ocr_every)
                print(f"OCR-Throttle: alle {args.ocr_interval_ms:.0f} ms "
                      f"= jeder {ocr_every}. verarbeitete Frame")
            if args.save:
                w, h = src.size
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                # WICHTIG: Wir schreiben nur jeden skip-ten Frame. Die
                # Writer-fps muss daher src.fps/skip sein, sonst läuft das
                # Ergebnis skip-fach zu schnell (Default skip=3 -> 3x).
                fps = (src.fps or 30.0) / args.skip
                writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

            loop_start = time.perf_counter()
            for frame in src:
                if (args.max_seconds is not None
                        and time.perf_counter() - loop_start >= args.max_seconds):
                    print(f"\nZeitlimit ({args.max_seconds:.0f}s) erreicht – beende.")
                    break
                t0 = time.perf_counter()
                if args.boost_only:
                    # Reiner Bildbearbeitungs-Modus: nur Rot anheben, sonst
                    # nichts erkennen/zeichnen.
                    annotated = DetectionPipeline._boost_red(frame)
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    n_frames += 1
                    total_ms += dt_ms
                else:
                    result = pipeline.process(frame)
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    n_frames += 1
                    result.frame_id = n_frames
                    total_ms += dt_ms
                    if args.bench:
                        print(f"Frame {n_frames}: {dt_ms:6.2f} ms  "
                              f"(outlines={len(result.outlines)}, "
                              f"moves={len(result.movements)}, tags={len(result.tags)})")

                    annotated = pipeline.draw(frame, result,
                                              show_movement=not args.no_motion,
                                              show_confidence=not args.no_confidence)

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
