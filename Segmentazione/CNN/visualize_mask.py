import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # <-- ECCO LA CORREZIONE

# --- CONFIGURAZIONE ---
# Modifica questi percorsi per farli corrispondere alla tua struttura
IMAGE_DIR = "../images/Immagini/"
MASK_DIR = "../images/Maschere/"

# Seleziona quale coppia immagine/maschera vuoi visualizzare tramite nome file.
# Puoi usare sia il nome completo (es. "1253--27.png") sia solo lo stem (es. "1253--27").
FILE_NAME = "siniscola_OPC_10x--101"

# Se FILE_NAME è vuoto, viene usato il fallback per indice.
FILE_INDEX = 3
#NON ADDESTRARE LA CLASSE Modifica questo indice per visualizzare file diversi


def _resolve_image_mask_paths(image_files, mask_files):
    """Restituisce i path di immagine e maschera a partire da FILE_NAME o FILE_INDEX."""
    if FILE_NAME and FILE_NAME.strip():
        requested_name = FILE_NAME.strip()
        requested_stem = Path(requested_name).stem

        image_match = next((f for f in image_files if Path(f).stem == requested_stem or f == requested_name), None)
        if image_match is None:
            raise FileNotFoundError(f"Immagine '{requested_name}' non trovata in {IMAGE_DIR}")

        mask_matches = [f for f in mask_files if Path(f).stem == requested_stem]
        if not mask_matches:
            raise FileNotFoundError(f"Maschera associata a '{requested_name}' non trovata in {MASK_DIR}")
        if len(mask_matches) > 1:
            raise ValueError(
                f"Trovate più maschere per '{requested_stem}': {mask_matches}. Risolvi l'ambiguità mantenendo un solo file."
            )

        return (
            os.path.join(IMAGE_DIR, image_match),
            os.path.join(MASK_DIR, mask_matches[0]),
            image_match,
            mask_matches[0],
        )

    if FILE_INDEX >= len(image_files):
        raise IndexError(f"L'indice {FILE_INDEX} è troppo grande. Ci sono solo {len(image_files)} file immagine.")

    image_match = image_files[FILE_INDEX]
    requested_stem = Path(image_match).stem
    mask_matches = [f for f in mask_files if Path(f).stem == requested_stem]
    if not mask_matches:
        raise FileNotFoundError(f"Maschera associata a '{image_match}' non trovata in {MASK_DIR}")
    if len(mask_matches) > 1:
        raise ValueError(
            f"Trovate più maschere per '{requested_stem}': {mask_matches}. Risolvi l'ambiguità mantenendo un solo file."
        )

    return (
        os.path.join(IMAGE_DIR, image_match),
        os.path.join(MASK_DIR, mask_matches[0]),
        image_match,
        mask_matches[0],
    )


def visualize_single_mask():
    """Carica e visualizza una coppia immagine/maschera con una legenda a colori."""
    print("Avvio visualizzazione...")

    try:
        image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        mask_files = sorted(
            [f for f in os.listdir(MASK_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
    except FileNotFoundError:
        print(f"ERRORE: Una delle cartelle non è stata trovata. Controlla i percorsi IMAGE_DIR e MASK_DIR.")
        return

    if not image_files or not mask_files:
        print("Nessun file trovato nelle cartelle.")
        return

    try:
        img_path, mask_path, image_name, mask_name = _resolve_image_mask_paths(image_files, mask_files)
    except (FileNotFoundError, ValueError, IndexError) as exc:
        print(f"ERRORE: {exc}")
        return

    # Carica l'immagine e la maschera
    image = cv2.imread(img_path)
    if image is None:
        print(f"ERRORE: impossibile leggere l'immagine {img_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converti per matplotlib
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        print(f"ERRORE: impossibile leggere la maschera {mask_path}")
        return

    print(f"\nVisualizzazione di:")
    print(f"  - Immagine: {image_name}")
    print(f"  - Maschera: {mask_name}")

    unique_values = np.unique(mask)
    print(f"Valori trovati in questa maschera: {unique_values}")

    # Crea una maschera colorata per la visualizzazione
    colors = [
        [0, 0, 0],  # Classe 0: Nero
        [255, 0, 0],  # Classe 1: Rosso
        [0, 255, 0],  # Classe 2: Verde
        [0, 0, 255],  # Classe 3: Blu
        [255, 255, 0],  # Classe 4: Giallo
        [255, 0, 255],  # Classe 5: Magenta
        [0, 255, 255],  # Classe 6: Ciano
        [255, 128, 0],  # Classe 7: Arancione
    ]

    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    legend_patches = []

    for value in unique_values:
        if value < len(colors):
            colored_mask[mask == value] = colors[value]
            # La riga seguente ora funzionerà perché mpatches è stato importato correttamente
            legend_patches.append(mpatches.Patch(color=np.array(colors[value]) / 255., label=f'Classe {value}'))
        else:
            print(f"Attenzione: valore {value} senza un colore definito nella lista.")

    # Crea la visualizzazione
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(image)
    axes[0].set_title("Immagine Originale")
    axes[0].axis('off')

    axes[1].imshow(colored_mask)
    axes[1].set_title("Maschera a Colori")
    axes[1].legend(handles=legend_patches, loc='lower right')
    axes[1].axis('off')

    plt.suptitle(f"Analisi di '{mask_name}'", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    visualize_single_mask()