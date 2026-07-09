"""
Funzioni di utilità condivise tra training ed evaluation per la gestione del dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pandas as pd
from datasets import Dataset
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[2]))
from Classificazione.ViT.gestione_dataset import csv_config


def load_attribute_metadata(attribute: str) -> Tuple[List[str], Dict[str, int], Dict[int, str], Dict[str, str]]:
    """
    Carica il CSV degli esperimenti e costruisce tutte le mappature necessarie per l'attributo scelto.

    - attribute: nome della colonna da classificare (es: "temperatura")
    - classes: valori possibili dell'attributo (es: ["1300", "1400", "1500"])
    - class_to_id: mappa classe -> ID numerico (es: "1300" -> 0)
    - id_to_class: mappa ID numerico -> classe (es: 0 -> "1300")
    - experiment_to_class: mappa ID esperimento -> classe (es: "EXP01" -> "1300")
    """
    # Controlla che il CSV con le informazioni sugli esperimenti esista, altrimenti lo crea.
    root_dir = Path(__file__).resolve().parents[2]
    csv_path = root_dir / "esperimenti.csv"
    if not csv_path.exists():
        csv_config.create_csv(csv_path)
    df = pd.read_csv(csv_path)

    if attribute not in df.columns:
        raise ValueError(
            f"L'attributo '{attribute}' non esiste nel CSV. Colonne disponibili: {list(df.columns)}"
        )

    # Prende la colonna dell'attributo scelto, rimuove i valori null e li trasforma in stringa.
    df_attribute = df[attribute].dropna().astype(str)
    # Toglie i valori duplicati e li ordina alfabeticamente (es: ["1300", "1400", "1500"] per temperatura).
    classes = sorted(df_attribute.unique())
    # Crea dizionario che mappa: classe → ID numerico (es: "1300" → 0, "1400" → 1, "1500" → 2).
    class_to_id = {class_name: idx for idx, class_name in enumerate(classes)}
    # Crea dizionario opposto: ID numerico → classe (es: 0 → "1300", 1 → "1400", 2 → "1500").
    id_to_class = {idx: class_name for class_name, idx in class_to_id.items()}
    # Crea un dizionario che mappa: ID esperimento → classe (es: "EXP01" → "1300", "EXP02" → "1400", ...).
    experiment_to_class = df.set_index("ID")[attribute].dropna().astype(str).to_dict()

    return classes, class_to_id, id_to_class, experiment_to_class


def add_class_id_column(dataset: Dataset, class_to_id: Dict[str, int], experiment_to_class: Dict[str, str]) -> Dataset:
    """
    Aggiunge al dataset HuggingFace una colonna "class_id" con l'ID numerico della classe.
    """
    label_feature = dataset.features["label"]

    def add_class_id(example: Dict) -> Dict:
        # Prende l'ID esperimento a cui è associata l'immagine (es: "EXP01").
        experiment_id = label_feature.int2str(example["label"])
        # Cerca nel CSV la classe per questo esperimento (es: "EXP01" → "1300").
        class_name = experiment_to_class.get(experiment_id)
        # Converte la classe nell'ID numerico del modello (es: "1300" → 0).
        example["class_id"] = class_to_id.get(class_name, -1)
        return example

    return dataset.map(add_class_id)


def filter_missing_class(dataset: Dataset, class_field: str = "class_id") -> Dataset:
    """
    Rimuove dal dataset gli esempi a cui non è stata assegnata una classe valida.
    """
    return dataset.filter(lambda ex: ex[class_field] != -1)


def build_vit_batch_preprocessor(processor, use_grayscale: bool, label_field: str = "class_id") -> Callable:
    """
    Genera la funzione di preprocessing che prepara un batch completo per il ViT.

    La funzione risultante:
    1. Preprocessa le immagini: crop (950px) e opzionale conversione in scala di grigi.
    2. Converte le immagini in tensori PyTorch tramite AutoImageProcessor (resize, normalize).
    3. Aggiunge le labels al batch per permettere al Trainer/evaluation di calcolare loss/metriche.
    
    Returns:
        Funzione transform(batch) che prende un batch del dataset e ritorna un dict con:
        - pixel_values: tensori delle immagini preprocessate
        - labels: ID numerici delle classi
    """

    def transform(batch):
        # Ritaglia l'immagine in basso a 950 pixel per togliere l'etichetta di misura sull'immagine.
        def crop(image: Image.Image) -> Image.Image:
            width, height = image.size
            # Taglia solo se l'immagine è più alta di 950 pixel.
            crop_height = min(950, height)
            if crop_height == height:
                return image
            # Parametri: (sinistra, alto, destra, basso) - manteniamo la parte superiore dell'immagine.
            return image.crop((0, 0, width, crop_height))

        # Converte l'immagine in scala di grigi se il parametro --grayscale è specificato.
        def maybe_grayscale(image: Image.Image) -> Image.Image:
            # Se non è richiesto, manteniamo l'immagine originale in RGB.
            if not use_grayscale:
                return image.convert("RGB")
            # Converte RGB in scala di grigi (L = Luminance, un solo canale).
            gray = image.convert("L")
            # Replica su tre canali lo stesso canale grigio poiché il ViT si aspetta immagini RGB (3 canali).
            return Image.merge("RGB", (gray, gray, gray))

        # Applica crop + (eventuale) scala di grigi a tutte le immagini del batch.
        images = [maybe_grayscale(crop(img)) for img in batch["image"]]
        # Usa il processor HuggingFace per convertire le immagini in tensori PyTorch.
        inputs_vit = processor(images, return_tensors="pt")
        # Aggiungiamo le label per permettere al Trainer/eval di calcolare la loss o le metriche.
        inputs_vit["labels"] = batch[label_field]
        return inputs_vit

    return transform


# Alias per retrocompatibilità
build_vit_transform = build_vit_batch_preprocessor
