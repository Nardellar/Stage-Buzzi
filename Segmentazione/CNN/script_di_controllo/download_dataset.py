import os
import zipfile
import gdown


FILE_ID = "1a-FT0Idh236Tbhmkqj3oGcL7q0yIgK-l"

# Costruisce l'URL diretto per il download
URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Trova la cartella esatta in cui si trova questo script (script_di_controllo)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Sale di due cartelle (fuori da script_di_controllo -> fuori da CNN) e punta a "images"
EXTRACT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../images"))

# Salva lo zip temporaneo direttamente dentro la cartella images
ZIP_PATH = os.path.join(EXTRACT_DIR, "dataset_temp.zip")


def main():
    print("Inizio il download del dataset da Google Drive...")
    print("Potrebbe volerci qualche minuto in base alla tua connessione.\n")

    # Scarica il file usando gdown (quiet=False mostra la barra di progresso)
    gdown.download(URL, ZIP_PATH, quiet=False)

    if not os.path.exists(ZIP_PATH):
        print("\n[ERRORE] Il download è fallito. Controlla l'ID del file e i permessi di condivisione.")
        return

    print(f"\nEstrazione dei file nella cartella '{EXTRACT_DIR}'...")
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    try:
        # Estrae tutto il contenuto dello zip
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print("Estrazione completata con successo!")
    except zipfile.BadZipFile:
        print("\n[ERRORE] Il file scaricato non è uno .zip valido.")
        return
    finally:
        # Pulizia: elimina il file .zip temporaneo anche se c'è stato un errore
        if os.path.exists(ZIP_PATH):
            print("Eliminazione del file .zip temporaneo...")
            os.remove(ZIP_PATH)

    print("\n✅ Fatto! Dataset pronto all'uso.")


if __name__ == "__main__":
    main()