# CNN/train.py
"""
Scopo: Questo è lo script principale che esegui per avviare il processo di addestramento. Coordina tutti gli altri file.
Funzioni Principali:
- Chiama data_loader.py per preparare i dati di addestramento e di validazione.
- Chiama model_vit_unet.py per costruire l'architettura del modello.
- Definisce i parametri dell'addestramento, come il numero di epoche (quante volte vedere l'intero dataset),
  il learning rate (quanto velocemente il modello impara) e la funzione di perdita (loss function, che misura
  l'errore del modello).

- Esegue il ciclo di addestramento: passa i dati al modello, calcola l'errore, aggiorna i pesi del modello per
  migliorare, e ripete il processo.

Salva il modello addestrato (il file .pth) quando ottiene i risultati migliori sul set di validazione.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Importa le nostre classi e funzioni
from data_loader import SegmentationDataset
from model_vit_unet import ViT_Unet


def main():
    # --- Configurazione ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Parametri di addestramento
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    NUM_CLASSES = 3
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4  # Potresti dover ridurre questo valore a seconda della VRAM della GPU
    NUM_EPOCHS = 50

    # Percorsi
    IMAGE_DIR = "../images/Immagini/"
    MASK_DIR = "../images/Maschere/"
    MODEL_SAVE_PATH = "vit_unet_pytorch.pth"

    # --- 1. Preparazione dei dati ---
    image_files = sorted(
        [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted(
        [os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR) if f.lower().endswith(('.tif', '.tiff'))])

    train_img, val_img, train_mask, val_mask = train_test_split(
        image_files, mask_files, test_size=0.2, random_state=42
    )

    # --- 2. Trasformazioni e Augmentation ---
    train_transform = A.Compose([
        A.Resize(IMG_HEIGHT, IMG_WIDTH),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(IMG_HEIGHT, IMG_WIDTH),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # --- 3. Creazione dei Dataset e DataLoader ---
    train_dataset = SegmentationDataset(train_img, train_mask, transform=train_transform, num_classes=NUM_CLASSES)
    val_dataset = SegmentationDataset(val_img, val_mask, transform=val_transform, num_classes=NUM_CLASSES)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 4. Modello, Loss e Optimizer ---
    model = ViT_Unet(num_classes=NUM_CLASSES).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 5. Ciclo di Addestramento ---
    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        # Training loop
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]"):
            images = images.to(DEVICE)
            masks = masks.squeeze(1).to(DEVICE)  # Rimuove la dimensione del canale

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, masks)

            # Backward e ottimizzazione
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Val]"):
                images = images.to(DEVICE)
                masks = masks.squeeze(1).to(DEVICE)

                outputs = model(images)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Salva il miglior modello
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    main()