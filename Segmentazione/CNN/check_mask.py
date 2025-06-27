# Script per analizzare i file delle maschere e stampare i valori unici dei pixel. nel senso a che valori numerici sono mappati i pixel delle maschere.

import os
import cv2
import numpy as np

# Imposta il percorso della cartella delle maschere
MASK_DIR = "../images/Maschere/"

# Quanti file analizzare (mettine un numero piccolo, 5-10 sono sufficienti)
NUM_FILES_TO_CHECK = 10

def check_mask_values():
    """Analizza i file delle maschere e stampa i valori unici dei pixel."""
    print(f"Controllo dei valori dei pixel nella cartella: {MASK_DIR}")

    try:
        mask_files = [f for f in os.listdir(MASK_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        if not mask_files:
            print("Nessun file di maschera trovato. Controlla il percorso MASK_DIR.")
            return

        # Limita il numero di file da controllare
        files_to_process = mask_files[:NUM_FILES_TO_CHECK]
        print(f"Analisi dei primi {len(files_to_process)} file...")

        all_unique_values = set()

        for filename in files_to_process:
            path = os.path.join(MASK_DIR, filename)
            # Leggi la maschera con OpenCV
            mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)

            if mask is None:
                print(f"ATTENZIONE: Impossibile leggere il file {filename}")
                continue

            # Trova i valori unici e aggiungili al set
            unique_values = np.unique(mask)
            all_unique_values.update(unique_values)

        print("\n--- RISULTATO DIAGNOSI ---")
        if all_unique_values:
            print(f"Valori unici trovati in tutti i file analizzati: {sorted(list(all_unique_values))}")
            print("Questi sono i valori che devi mappare a 0, 1, 2 nel tuo script di training.")
        else:
            print("Nessun valore trovato. Strano, controlla i file.")

    except FileNotFoundError:
        print(f"ERRORE: La cartella non esiste: {os.path.abspath(MASK_DIR)}")
        print("Assicurati che il percorso MASK_DIR sia corretto.")

if __name__ == '__main__':
    check_mask_values()