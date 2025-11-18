"""
Funzioni di supporto per caricare e preparare il dataset di segmentazione.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set

import albumentations as A
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def load_dataset_stateless(
    images_dir: str,
    masks_dir: str,
    image_size: Tuple[int, int],
    use_grayscale: bool = False,
    image_names: Optional[Iterable[str]] = None,
    return_paths: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Carica immagini PNG e maschere TIFF, applicando il resize richiesto e convertendole in scala di grigi se necessario.
    Restituisce tensori normalizzati in [0, 1], maschere int32 e percorsi delle immagini se richiesto.
    """ 
    #cerca tutte le immagini e maschere e le ordina
    img_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.tif")))

    if not img_paths or not mask_paths:
        raise FileNotFoundError(
            f"Nessuna immagine trovata in {images_dir} o nessuna maschera in {masks_dir}"
        )
    #crea un dizionario che mappa il nome dell'immagine senza estensione (stem) al percorso file completo della maschera corrispondente
    mask_map = {Path(mask_path).stem: mask_path for mask_path in mask_paths}
    #Trasformiamo "image_names" da una lista ad un set, che rende i controlli di appartenenza più veloci (O(1) rispetto a O(n) per una lista)
    image_names_set = set(image_names) if image_names is not None else None

    #liste di accumulo di immagini/maschere già pre-processate
    images = []
    masks = []
    selected_paths: List[str] = []
    
    #per ogni immagine nella cartella delle immagini:
    for img_path in img_paths:
        stem = Path(img_path).stem
        #se l'immagine non e' tra quelle richieste, saltala
        if image_names_set is not None and stem not in image_names_set:
            continue
        #cerco la maschera corrispondente all'immagine
        mask_path = mask_map.get(stem)
        #se non trovo la maschera, lancio un errore
        if mask_path is None:
            raise FileNotFoundError(f"Nessuna maschera trovata per l'immagine {img_path}")

        #PRE-PROCESSING
        #carico l'immagine (in formato BGR)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Impossibile leggere l'immagine {img_path}")
        #converto l'l'ordine dei canali da BGR a RGB (perche' il modello CNN si aspetta immagini RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #se voglio l'immagine in scala di grigi
        if use_grayscale:
            #converto l'immagine in grigio ma su un solo canale (produce una matrice 2D (Height, Width) con i valori di grigio (luminanza) per ogni pixel)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            #Aggiugniamo l'asse dei canali per ottenere un tensore 3D (Height, Width, Canali) e inseriamo i valori di grigio replicati su 3 canali (RGB)
            img = np.repeat(gray[:, :, None], 3, axis=2)
        #ridimensiono l'immagine alla dimensione richiesta e applico interpolazione lineare (per mantenere valori dei pixel interi)
        img = cv2.resize(img, image_size, interpolation=cv2.INTER_LINEAR)
        #normalizzo l'immagine in [0,1] e la salvo nella lista delle immagini pre-processate
        images.append(img.astype(np.float32) / 255.0)

        #apriamo la maschera con PIL (gestisce meglio i .tiff) e la convertiamo in un array NumPy
        mask = Image.open(mask_path)
        #convertiamo la maschera in un array NumPy
        mask_np = np.array(mask)
        #controlliamo se la maschera ha 3 canali (RGB) e nel caso la convertiamo ad un solo canale (grigio)
        #serve perche' e' il formato che tensorflow/XGBoost si aspettano
        #le maschere ora sono una matrice 2D (Height, Width) con i valori di classe per ogni pixel
        if mask_np.ndim == 3:
            mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
        #ridimensioniamo la maschera alla stessa dimensione delle immagini (sempre con inteprolazione Inter_nearest) (puo' capitare che alcuni tiff siano salvati su 3 canali, mettiamo il controllo per sicurezza)
        mask_np = cv2.resize(
            mask_np,
            image_size,
            interpolation=cv2.INTER_NEAREST,
        )
        #convertiamo i valori dei pixel in interi e aggiugiamo la maschera all'elenco delle pre-processate
        masks.append(mask_np.astype(np.int32))
        #aggiungiamo il percorso dell'immagine alla lista dei percorsi selezionati
        selected_paths.append(str(img_path))

    #convertiamo le liste in array NumPy in forma:
    #images_np = (Num_immagini, Height, Width, Canali(3))
    #masks_np = (Num_immagini, Height, Width)
    images_np = np.asarray(images)
    masks_np = np.asarray(masks)
    #se non ci sono immagini o maschere, lancio un errore
    if images_np.size == 0 or masks_np.size == 0:
        raise ValueError("Dataset vuoto dopo il caricamento.")

    if return_paths:
        return images_np, masks_np, selected_paths
    return images_np, masks_np


def augment_dataset(images: np.ndarray,masks: np.ndarray,) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applica una semplice augmentazione duplicando ogni esempio con trasformazioni casuali.
    """
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5), #flip orizzontale con probabilita' 50%
            A.VerticalFlip(p=0.5), #flip verticale con probabilita' 50%
            A.RandomRotate90(p=0.5), #ruota di 90 gradi con probabilita' 50%
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5), #trasla, scala e ruota con probabilita' 50% (shift_limit: limite di traslazione, scale_limit: limite di scala, rotate_limit: limite di rotazione)
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.5
            ), #cambia la luminosita' e il contrasto con probabilita' 50%
        ]
    )
    #liste di accumulo di immagini/maschere originali + augmentate
    augmented_images = []
    augmented_masks = []
    #per ogni immagine e maschera, salviamo le originali e salviamo anche la loro versione augmentata randomicamente
    for img, mask in zip(images, masks):
        augmented_images.append(img)
        augmented_masks.append(mask)

        aug = transform(image=img, mask=mask)
        augmented_images.append(aug["image"])
        augmented_masks.append(aug["mask"])

    return np.asarray(augmented_images), np.asarray(augmented_masks)


def compute_class_weights_dict(masks: np.ndarray, num_classes: int,) -> Optional[Dict[int, float]]:
    """
    Calcola pesi bilanciati ignorando la classe 0 (background).
    """
    #appiattiamo le machere (Height, Width) in un array monodimensionale (Height * Width)
    labels = masks.flatten()
    #eliminiamo i pixel di background (classe 0) e sottraimo 1 alle classi restanti per farle cominciare da 0
    labels = labels[labels != 0] - 1

    #trovo le classi uniche presenti nelle maschere
    unique_labels = np.unique(labels)
    #calcolo i pesi bilanciati per ogni classe
    weights = compute_class_weight("balanced", classes=unique_labels, y=labels)
    #verifichiamo che tutte le classi presenti siano nel range previsto
    invalid = [int(k) for k in unique_labels if not (0 <= k < num_classes)]
    if invalid:
        raise ValueError(
            f"Classi fuori range rilevate nelle maschere: {invalid}. "
            "Controlla che i label siano compresi tra 1 e num_classes."
        )

    #creiamo il dizionario "label": "peso label"
    weights_dict = dict(zip(unique_labels.tolist(), weights.tolist()))
    #converto il dizionario in un dizionario di tipo int: float (per sicurezza do' dati di tipi nativi python al booster)
    return {int(label): float(weight) for label, weight in weights_dict.items()}


def _match_image_mask_pairs(images_dir: str, masks_dir: str) -> List[Tuple[Path, Path]]:
    """
    Calcola tutte le coppie immagine/maschera valide nelle cartelle specificate.
    """
    #trovo tutte le immagini nella cartella delle immagini
    image_paths = sorted(Path(images_dir).glob("*.png"))
    #creo un dizionario che mappa "nome maschera" : "path" per ogni maschera
    mask_map = {path.stem: path for path in Path(masks_dir).glob("*.tif")}
    pairs = []
    #per ogni immagine, cerco la maschera corrispondente e la aggiungo alla lista delle coppie
    for img_path in image_paths:
        mask_path = mask_map.get(img_path.stem)
        if mask_path is not None:
            pairs.append((img_path, mask_path))
    return pairs

#Serve piu avanti per stratificare il dataset in train e eval
def _dominant_class(mask_path: Path) -> int:
    """
    Data una maschera, restiuisce (senza considerare il background" ka label piu' presente)
    """
    #apriamo la maschera con PIL e la convertiamo in un array NumPy
    mask = np.array(Image.open(mask_path))
    #controlliamo se la maschera ha 3 canali (RGB) e nel caso la convertiamo ad un solo canale (grigio) (lo facciamo per avere un array 2D)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    #di ogni maschera consideriamo solo i pixel non della classe 0
    valid = mask[mask > 0]
    if valid.size == 0:
        return 0
    unique, counts = np.unique(valid, return_counts=True)
    #restituisco la classe con frequenza piu alta
    return int(unique[np.argmax(counts)])


def prepare_dataset_splits(images_dir: str, masks_dir: str, split_dir: str, train_ratio: float = 0.8, seed: int = 42,) -> Tuple[List[str], List[str]]:
    """
    Genera (o ricarica) la lista di file per train ed eval (80/20).
    L'eval set viene usato sia come validation che come test.
    """
    #definisco i percorsi file in cui salvare informazioni sugli split
    split_path = Path(split_dir)
    train_file = split_path / "train.txt"
    eval_file = split_path / "eval.txt"

    #se gli split sono stati gia' generati in passato, li leggiamo e li riusiamo
    if train_file.exists() and eval_file.exists():
        #leggo i file e rimuovo spazi bianchi e righe vuote
        train_ids = [line.strip() for line in train_file.read_text().splitlines() if line.strip()]
        eval_ids = [line.strip() for line in eval_file.read_text().splitlines() if line.strip()]
        if train_ids and eval_ids:
            return train_ids, eval_ids

    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio deve essere compreso tra 0 e 1 (esclusi).")

    pairs = _match_image_mask_pairs(images_dir, masks_dir)
    if not pairs:
        raise FileNotFoundError("Nessuna coppia immagine/maschera valida per creare gli split.")

    #creo liste per gli stem delle immagini e le loro label dominanti
    images = []
    dominant_labels = []
    #per ogni coppia immagine/maschera, aggiungo lo stem dell'immagine e la label dominante della maschera
    for img_path, mask_path in pairs:
        images.append(img_path.stem)
        dominant_labels.append(_dominant_class(mask_path))

    #Calcolo lo split test/val stratificando per attributo
    try:
        train_ids, eval_ids = train_test_split(
            images,
            test_size=1.0 - train_ratio,
            random_state=seed,
            stratify=dominant_labels,
        )
    #se lo stratify fallisce (a causa per esempio di pochi esempi di una certa label), eseguo lo split senza stratificazione
    except ValueError:
        print(
            "Impossibile stratificare lo split (probabilmente una classe ha troppi pochi campioni). "
            "Eseguo split senza stratificazione."
        )
        train_ids, eval_ids = train_test_split(
            images,
            test_size=1.0 - train_ratio,
            random_state=seed,
        )

    #creo la cartella per gli split, e salvo i due file su disco
    split_path.mkdir(parents=True, exist_ok=True)
    train_file.write_text("\n".join(train_ids))
    eval_file.write_text("\n".join(eval_ids))
    #restituisco gli split calcolati
    return train_ids, eval_ids


__all__ = [
    "load_dataset_stateless",
    "augment_dataset",
    "compute_class_weights_dict",
    "prepare_dataset_splits",
]
