# Stage-Buzzi

This repository collects machine learning utilities for image classification and segmentation. The code is organized into separate folders for classification models, segmentation pipelines and setup scripts.

## Directory overview

- `Classificazione/` – CNN and ViT models plus utilities for building datasets and generating the `esperimenti.csv` file.
- `Segmentazione/` – segmentation models including a small CNN and an embedded copy of Meta's Segment Anything.
- `Setup/` – environment notes and formatting tools (`Formattazione commit/` contains an optional `pre-commit` configuration).
- `Commenti/` – assorted project notes.
- `requirements.txt` – core Python dependencies.

## Quick start

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the classification models. The dataset utilities will
   automatically create `esperimenti.csv` the first time you run them.
   For example:
   ```bash
   python -m Classificazione.CNN.model_temperatura
   ```
   Replace `model_temperatura` with the desired module (e.g. `model_tempo`, `model_raffreddamento`, ...).

The repository also provides an optional pre‑commit configuration in `Setup/Formattazione commit/` which formats Python files with **black** and **isort**.
