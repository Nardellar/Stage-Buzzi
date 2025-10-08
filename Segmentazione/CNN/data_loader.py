# CNN/data_loader.py
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
"""Questo file è il responsabile della preparazione dei tuoi dati. Il suo compito è prendere i file di immagine (.png) 
e di maschera (.tif) dalle loro cartelle, ridimensionarli, e convertirli in un formato che PyTorch può utilizzare 
(i tensori).
Trova e abbina le immagini con le maschere corrispondenti.
Applica data augmentation
Riorganizza i dati in "batch": invece di elaborare un'immagine alla volta, li raggruppa in piccoli lotti (es. 8 immagini per volta) per rendere l'addestramento più efficiente."""

class SegmentationDataset(Dataset):
    """
    Dataset per la segmentazione di immagini di cemento.
    Carica un'immagine e la sua maschera corrispondente.
    """

    def __init__(self, image_paths, mask_paths, transform=None, mask_transform=None, num_classes=3):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.mask_transform = mask_transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Carica l'immagine
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Carica la maschera
        mask_path = self.mask_paths[idx]
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        # Rimappa i valori della maschera se necessario
        # Esempio: se le classi sono 0, 100, 200, le mappiamo a 0, 1, 2.
        # Questo è solo un esempio, adattalo ai valori reali delle tue maschere.
        # Se i valori sono già 0, 1, 2..., puoi commentare questa parte.
        # new_mask = np.zeros_like(mask, dtype=np.uint8)
        # new_mask[mask == 100] = 1 # Esempio per la classe Alite
        # new_mask[mask == 200] = 2 # Esempio per la classe Belite
        # mask = new_mask

        # Assicura che non ci siano valori fuori range
        mask[mask >= self.num_classes] = 0

        # Applica le trasformazioni (es. resize, to_tensor, data augmentation)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.long()