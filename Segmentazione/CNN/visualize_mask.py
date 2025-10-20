import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # <-- ECCO LA CORREZIONE

# --- CONFIGURAZIONE ---
# Modifica questi percorsi per farli corrispondere alla tua struttura
IMAGE_DIR = "../images/Immagini/"
MASK_DIR = "../images/Maschere/"

# Seleziona quale coppia immagine/maschera vuoi visualizzare (es. il primo file, il decimo, ecc.)
FILE_INDEX = 3
#NON ADDESTRARE LA CLASSE Modifica questo indice per visualizzare file diversi


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

    if FILE_INDEX >= len(image_files):
        print(f"ERRORE: L'indice {FILE_INDEX} è troppo grande. Ci sono solo {len(image_files)} file.")
        return

    # Seleziona i file in base all'indice
    img_path = os.path.join(IMAGE_DIR, image_files[FILE_INDEX])
    mask_path = os.path.join(MASK_DIR, mask_files[FILE_INDEX])

    # Carica l'immagine e la maschera
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converti per matplotlib
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    print(f"\nVisualizzazione di:")
    print(f"  - Immagine: {image_files[FILE_INDEX]}")
    print(f"  - Maschera: {mask_files[FILE_INDEX]}")

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

    plt.suptitle(f"Analisi di '{mask_files[FILE_INDEX]}'", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    visualize_single_mask()