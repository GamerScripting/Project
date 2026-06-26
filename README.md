# Detection Pipeline (Outlines · Tags · Movement)

Eine modulare Computer-Vision-Pipeline, die ein **Video** Frame für Frame
analysiert und drei Dinge erkennt:

| Modul | Was es erkennt | Methode |
|-------|----------------|---------|
| `outlines.py` | Farbige Umrandungen / Highlights | HSV-Farbfilter + Konturen |
| `tags.py`     | Text / Nametags                  | OCR (EasyOCR)            |
| `movement.py` | Bewegung im Bild                 | Frame-Differenz         |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Hinweis:** EasyOCR lädt beim ersten Start automatisch ein Sprachmodell
> herunter. Mit einer GPU (CUDA) läuft die OCR deutlich schneller.

## Benutzung

```bash
python main.py --video pfad/zum/video.mp4
```

Wichtige Optionen:

```bash
python main.py --video clip.mp4 \
    --skip 2 \              # nur jeden 2. Frame analysieren (schneller)
    --no-ocr \              # OCR abschalten (OCR ist langsam)
    --save out.mp4          # annotiertes Ergebnis als Video speichern
```

Drücke `q` im Vorschaufenster zum Beenden.

## Projektstruktur

```
src/
├── capture.py    # Liest Frames aus einer Videodatei
├── outlines.py   # Erkennung farbiger Umrandungen
├── tags.py       # OCR-Texterkennung (EasyOCR)
├── movement.py   # Bewegungserkennung per Frame-Differenz
└── pipeline.py   # Führt alle Detektoren zusammen + zeichnet Overlays
main.py           # CLI-Einstiegspunkt
```

## Nächste Schritte / Ideen

- Outline-Farbe an euer Spiel anpassen (`--help` zeigt die HSV-Parameter).
- OCR nur auf Regionen anwenden, in denen Outlines gefunden wurden (schneller).
- Bewegungserkennung mit Objekt-Tracking koppeln (z. B. ID pro Objekt).
