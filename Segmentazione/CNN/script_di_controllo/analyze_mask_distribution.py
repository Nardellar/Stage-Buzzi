"""
Script per analizzare la distribuzione dei pixel per classe in tutte le maschere
del dataset di segmentazione. Mostra una tabella riassuntiva a terminale.
"""

from __future__ import annotations

import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Percorso delle maschere (relativo allo script)
BASE_DIR = Path(__file__).resolve().parent.parent
MASKS_DIR = (BASE_DIR / "../images/Maschere").resolve()

# Nomi delle classi usate nel progetto di segmentazione
CLASS_NAMES = {
    0: "Background",
    1: "Resina",
    2: "Pori/Imperfezioni",
    3: "Fase Fusa",
    4: "Belite",
    5: "Alite",
    6: "Calce libera",
}


def load_mask(mask_path: str) -> np.ndarray | None:
    """Carica una maschera TIFF e la converte in un array 2D di interi.
    Restituisce None se il file è vuoto o illeggibile."""
    if os.path.getsize(mask_path) == 0:
        return None
    try:
        mask = np.array(Image.open(mask_path))
    except Exception:
        return None
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    return mask.astype(np.int32)


def analyze_masks(masks_dir: str) -> None:
    mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.tif")))
    if not mask_paths:
        print(f"Nessuna maschera trovata in {masks_dir}")
        sys.exit(1)

    print(f"Trovate {len(mask_paths)} maschere in {masks_dir}\n")

    # Contatori globali per classe
    global_counts: defaultdict[int, int] = defaultdict(int)
    # Per ogni maschera salviamo i conteggi (per la tabella dettagliata)
    per_mask_counts: list[tuple[str, dict[int, int]]] = []
    # Raccogliamo tutte le classi presenti
    all_classes: set[int] = set()

    skipped = 0
    for mask_path in mask_paths:
        mask = load_mask(mask_path)
        if mask is None:
            skipped += 1
            continue
        unique, counts = np.unique(mask, return_counts=True)
        counts_dict = dict(zip(unique.tolist(), counts.tolist()))

        for cls_id, count in counts_dict.items():
            global_counts[cls_id] += count
            all_classes.add(cls_id)

        per_mask_counts.append((Path(mask_path).stem, counts_dict))

    if skipped:
        print(f"⚠️  {skipped}/{len(mask_paths)} maschere saltate (file vuoti o illeggibili)\n")
    if not per_mask_counts:
        print("Nessuna maschera leggibile trovata. I file .tif potrebbero essere vuoti (es. placeholder Git LFS).")
        sys.exit(1)

    # Ordina le classi
    sorted_classes = sorted(all_classes)

    # ── Tabella riassuntiva globale ──
    print("=" * 72)
    print("  DISTRIBUZIONE GLOBALE DEI PIXEL PER CLASSE")
    print("=" * 72)

    total_pixels = sum(global_counts.values())
    header = f"{'Classe':<5} {'Nome':<22} {'Pixel':>14} {'%':>8}"
    print(header)
    print("-" * len(header))
    for cls_id in sorted_classes:
        name = CLASS_NAMES.get(cls_id, f"Classe {cls_id}")
        count = global_counts[cls_id]
        pct = 100.0 * count / total_pixels if total_pixels > 0 else 0
        print(f"{cls_id:<5} {name:<22} {count:>14,} {pct:>7.2f}%")
    print("-" * len(header))
    print(f"{'':>28}{'TOTALE':} {total_pixels:>14,}")
    print()

    # ── Tabella: pixel non-background (solo classi utili) ──
    print("=" * 72)
    print("  DISTRIBUZIONE PIXEL NON-BACKGROUND (classi 1-5)")
    print("=" * 72)
    fg_classes = [c for c in sorted_classes if c != 0]
    fg_total = sum(global_counts[c] for c in fg_classes)
    header2 = f"{'Classe':<5} {'Nome':<22} {'Pixel':>14} {'% (fg)':>8}"
    print(header2)
    print("-" * len(header2))
    for cls_id in fg_classes:
        name = CLASS_NAMES.get(cls_id, f"Classe {cls_id}")
        count = global_counts[cls_id]
        pct = 100.0 * count / fg_total if fg_total > 0 else 0
        print(f"{cls_id:<5} {name:<22} {count:>14,} {pct:>7.2f}%")
    print("-" * len(header2))
    print(f"{'':>28}{'TOTALE fg':} {fg_total:>14,}")
    print()

    # ── Tabella per maschera ──
    print("=" * 90)
    print("  DETTAGLIO PER MASCHERA")
    print("=" * 90)

    # Intestazione dinamica
    cls_headers = [CLASS_NAMES.get(c, f"Cls{c}") for c in sorted_classes]
    col_w = 12
    row_header = f"{'Maschera':<35}"
    for h in cls_headers:
        row_header += f"{h:>{col_w}}"
    print(row_header)
    print("-" * len(row_header))

    for stem, counts_dict in per_mask_counts:
        # Tronca nomi lunghi
        display_name = stem if len(stem) <= 33 else stem[:30] + "..."
        row = f"{display_name:<35}"
        for cls_id in sorted_classes:
            count = counts_dict.get(cls_id, 0)
            row += f"{count:>{col_w},}"
        print(row)

    print("-" * len(row_header))
    print(f"\nAnalisi completata su {len(per_mask_counts)} maschere leggibili (su {len(mask_paths)} totali).")

    # ── Riepilogo finale richiesto: totale pixel per classe + percentuali ──
    print("\n" + "=" * 72)
    print("  RIEPILOGO FINALE DATASET (TOTALI PER CLASSE)")
    print("=" * 72)
    final_header = f"{'Classe':<5} {'Nome':<22} {'Pixel Totali':>14} {'Percentuale':>12}"
    print(final_header)
    print("-" * len(final_header))
    for cls_id in sorted_classes:
        name = CLASS_NAMES.get(cls_id, f"Classe {cls_id}")
        count = global_counts[cls_id]
        pct = 100.0 * count / total_pixels if total_pixels > 0 else 0.0
        print(f"{cls_id:<5} {name:<22} {count:>14,} {pct:>11.2f}%")
    print("-" * len(final_header))
    print(f"{'':>28}{'TOTALE':} {total_pixels:>14,} {100.00:>11.2f}%")


if __name__ == "__main__":
    analyze_masks(str(MASKS_DIR))
