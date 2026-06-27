"""OCR-Lern-Demo mit EasyOCR.

Ziel: verstehen, wie OCR (Optical Character Recognition = Texterkennung im Bild)
funktioniert. Dieses Skript liest Text aus EINEM Bild und erklärt jeden Schritt
auf der Konsole.

Benutzung:
    python tools/ocr_demo.py --image bild.png
    python tools/ocr_demo.py --image bild.png --lang en de   # mehrere Sprachen
    python tools/ocr_demo.py --image bild.png --save out.png  # Boxen einzeichnen

Was OCR im Kern macht (zwei Stufen):
  1) DETECTION  – "Wo im Bild ist überhaupt Text?"  -> findet Text-Regionen (Boxen)
  2) RECOGNITION– "Welche Buchstaben stehen da?"     -> liest die Zeichen in jeder Box
EasyOCR macht beides mit neuronalen Netzen und gibt pro Fund zurück:
  (Box-Koordinaten, erkannter Text, Konfidenz 0..1)
"""

from __future__ import annotations

import argparse
import time


def parse_args():
    p = argparse.ArgumentParser(description="OCR-Lern-Demo (EasyOCR)")
    p.add_argument("--image", required=True, help="Pfad zum Bild")
    p.add_argument("--lang", nargs="+", default=["en"],
                   help="Sprachen, z. B. 'en' oder 'en de'")
    p.add_argument("--gpu", action="store_true", help="GPU nutzen (schneller)")
    p.add_argument("--save", help="Bild mit eingezeichneten Boxen speichern")
    p.add_argument("--min-conf", type=float, default=0.0,
                   help="Nur Funde ab dieser Konfidenz zeigen (0..1)")
    return p.parse_args()


def main():
    args = parse_args()

    # Importe hier drin, damit die --help-Ausgabe auch ohne installierte
    # Pakete funktioniert.
    import cv2
    import easyocr

    # ------------------------------------------------------------------
    # 1) Bild laden
    # ------------------------------------------------------------------
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Bild nicht lesbar: {args.image}")
    print(f"Bild geladen: {args.image}  ({img.shape[1]}x{img.shape[0]} px)")

    # ------------------------------------------------------------------
    # 2) OCR-"Reader" erstellen.
    #    Beim ERSTEN Mal lädt EasyOCR die neuronalen Modelle herunter
    #    (Detection- + Recognition-Netz). Das dauert einmalig, danach
    #    liegen sie lokal im Cache.
    # ------------------------------------------------------------------
    print(f"Lade OCR-Modelle für Sprache(n): {args.lang} ...")
    t0 = time.perf_counter()
    reader = easyocr.Reader(args.lang, gpu=args.gpu)
    print(f"Modelle bereit in {time.perf_counter() - t0:.1f}s "
          f"(GPU={args.gpu})")

    # ------------------------------------------------------------------
    # 3) OCR ausführen.
    #    readtext() läuft intern beide Stufen (Detection + Recognition)
    #    und liefert eine Liste aus (box, text, confidence).
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    results = reader.readtext(img)
    dt = time.perf_counter() - t0
    print(f"\nOCR fertig in {dt * 1000:.0f} ms – {len(results)} Textregionen gefunden:\n")

    # ------------------------------------------------------------------
    # 4) Ergebnisse anzeigen + verstehen.
    #    Jede 'box' sind 4 Eckpunkte (Polygon), weil Text auch schräg
    #    stehen kann. Daraus machen wir ein einfaches Rechteck zum Zeichnen.
    # ------------------------------------------------------------------
    for i, (box, text, conf) in enumerate(results, 1):
        if conf < args.min_conf:
            continue
        xs = [int(p[0]) for p in box]
        ys = [int(p[1]) for p in box]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        # Konfidenz als simple Balkenanzeige zum Gefühl-Entwickeln:
        bar = "#" * int(conf * 20)
        print(f"  [{i:2}] '{text}'")
        print(f"       Konfidenz: {conf:.2f}  {bar}")
        print(f"       Position : x={x1}-{x2}, y={y1}-{y2}")

        if args.save:
            color = (0, 200, 0) if conf >= 0.5 else (0, 165, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, text, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    if args.save:
        cv2.imwrite(args.save, img)
        print(f"\nBild mit Boxen gespeichert: {args.save}")

    # Kleines Fazit zum Lernen
    print("\nMerke dir:")
    print("  - Hohe Konfidenz (~>0.8) = sicher erkannt.")
    print("  - Niedrige Konfidenz = kleiner/unscharfer/schräger Text.")
    print("  - Tipp: Text wird besser erkannt, wenn er groß & kontrastreich ist.")


if __name__ == "__main__":
    main()
