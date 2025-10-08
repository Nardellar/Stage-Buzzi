# CNN/predict.py
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model_vit_unet import ViT_Unet  # Importa il modello per caricarlo
"""
Scopo: Dopo aver addestrato il modello, questo script ti permette di usarlo per vedere come si comporta su nuove 
immagini.

Funzioni Principali:
- Carica il modello addestrato che hai salvato (vit_unet_pytorch.pth).
- Prende un'immagine di input, la prepara nello stesso modo in cui sono state preparate le immagini di addestramento.
- Passa l'immagine al modello per ottenere una maschera di segmentazione.
- Visualizza l'immagine originale, la maschera reale (se disponibile) e la maschera predetta dal modello, 
  per permetterti di valutare visivamente il risultato.

In breve: È lo strumento per vedere se il modello ha imparato bene e cosa è in grado """

def create_color_map():
    """Crea una mappa di colori per la visualizzazione delle maschere."""
    return np.array([
        [0, 0, 0],  # Classe 0: Sfondo (Nero)
        [255, 0, 0],  # Classe 1: Alite (Rosso)
        [0, 255, 0],  # Classe 2: Belite (Verde)
    ], dtype=np.uint8)


def mask_to_rgb(mask, color_map):
    """Converte una maschera di etichette in un'immagine RGB."""
    if len(mask.shape) == 3 and mask.shape[-1] == 1:
        mask = np.squeeze(mask, axis=-1)

    rgb_mask = color_map[mask]
    return rgb_mask


def predict_single_image(model, image_path, device, size=(224, 224)):
    """Carica un'immagine, la pre-processa ed esegue la predizione."""
    model.eval()  # Imposta il modello in modalità valutazione

    # Carica l'immagine originale
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Trasformazioni (devono essere le stesse della validazione)
    transform = A.Compose([
        A.Resize(size[0], size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    augmented = transform(image=original_image.copy())  # Usa una copia per non modificare l'originale
    image_tensor = augmented['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)

    # Ottieni la classe predetta per ogni pixel
    predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    return original_image, predicted_mask


def main():
    # --- Configurazione ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "vit_unet_pytorch.pth"  # Il modello che hai salvato
    IMAGE_DIR = "../images/Immagini/"
    MASK_DIR = "../images/Maschere/"
    NUM_CLASSES = 3
    VALIDATION_SPLIT = 0.2
    NUM_PREDICTIONS_TO_SHOW = 5  # Quante immagini di validazione vuoi visualizzare

    # --- Carica il modello ---
    print(f"Caricamento del modello da: {MODEL_PATH}")
    model = ViT_Unet(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
    print("Modello caricato.")

    # --- Prepara i dati di validazione per i test ---
    image_files = sorted([os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)])
    mask_files = sorted([os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR)])

    _, val_img_paths, _, val_mask_paths = train_test_split(
        image_files, mask_files, test_size=VALIDATION_SPLIT, random_state=42
    )

    color_map = create_color_map()

    # --- Esegui e visualizza le predizioni su alcune immagini di validazione ---
    for i in range(min(NUM_PREDICTIONS_TO_SHOW, len(val_img_paths))):
        img_path = val_img_paths[i]
        mask_path = val_mask_paths[i]

        original_image, predicted_mask = predict_single_image(model, img_path, DEVICE)

        # Carica la maschera reale per confronto
        true_mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        true_mask_resized = cv2.resize(true_mask_raw, (224, 224), interpolation=cv2.INTER_NEAREST)

        # --- Visualizza i risultati ---
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title("Immagine Originale")
        plt.imshow(cv2.resize(original_image, (224, 224)))
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Maschera Reale (Ground Truth)")
        plt.imshow(mask_to_rgb(true_mask_resized, color_map))
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Maschera Predetta dal Modello")
        plt.imshow(mask_to_rgb(predicted_mask, color_map))
        plt.axis('off')

        plt.suptitle(f"Confronto per: {os.path.basename(img_path)}")
        plt.show()


if __name__ == '__main__':
    main()