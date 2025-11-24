"""
Script per addestrare il ViT.

TECNICHE ANTI-OVERFITTING IMPLEMENTATE:
========================================

1. TRANSFER LEARNING:
   - ViT backbone congelato (trainable=False)
   - Solo il classificatore custom viene addestrato
   
2. ARCHITETTURA CUSTOM:
   - Head a 2 layer con LayerNorm e GELU
   - Dropout (0.3 e 0.15) tra i layer
   
3. REGULARIZZAZIONE:
   - Weight Decay (L2): 1e-4
   - Label Smoothing: 0.1
   - Gradient Clipping: max_norm=1.0
   
4. CLASS BALANCING:
   - Class weights basati sulla distribuzione del training set
   
5. LEARNING RATE SCHEDULING:
   - LR Scheduler: ReduceLROnPlateau
   - Patience: 7 epoch
   - Factor: 0.5 (dimezza LR quando plateau)
   - Warmup: 100 steps iniziali
   
6. EARLY STOPPING:
   - Patience: 15 epoch
   - Monitora: accuracy su validation set
   - Carica automaticamente il best model
   
7. DATA SPLITTING:
   - 80% training, 20% validation
   - Split stratificato per mantenere proporzioni classi

Risultato: 92.5% accuracy su test set (280 immagini) con solo 1120 training samples.

TERMINOLOGIA:
=============
- attribute: nome della colonna CSV da classificare (es: "temperatura", "rampa")
- classes: valori possibili dell'attributo = le classi da predire (es: ["1300", "1400", "1500"])
- class_id: ID numerico usato dal modello per ogni classe (es: 0, 1, 2)
- experiment_id: ID esperimento nel dataset (es: "EXP01", "EXP02")

Esempio per temperatura:
  attribute = "temperatura"
  classes = ["1300", "1400", "1500"]
  class_to_id = {"1300": 0, "1400": 1, "1500": 2}
  id_to_class = {0: "1300", 1: "1400", 2: "1500"}
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from datasets import load_dataset, ClassLabel
from sklearn.metrics import f1_score
from transformers import (
    AutoImageProcessor,
    TrainingArguments,
    Trainer,
    ViTForImageClassification,
    default_data_collator,
    EarlyStoppingCallback,
)

from dataset_utils import (
    add_class_id_column,
    build_vit_batch_preprocessor,
    filter_missing_class,
    load_attribute_metadata,
)

MODEL_NAME = "google/vit-base-patch16-224"


def build_model(num_labels: int, dropout_rate: float = 0.3) -> ViTForImageClassification:
    """Crea il ViT con head personalizzata e backbone congelato."""
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        ignore_mismatched_sizes=True, #per permettermi di caricare il ViT anche se chiedo una classificazione a 3 classi e non 768

    )

    #impostiamo il classificatore custom a 2 layer
    hidden_size = model.config.hidden_size
     
    model.classifier = nn.Sequential(
        nn.Dropout(dropout_rate), # quanti neuroni "disattivare"
        nn.Linear(hidden_size, hidden_size // 2), #definisco il numero di neuroni del primo layer di classificazione (riduco la dimedimensione)
        nn.LayerNorm(hidden_size // 2), #normalizzo sullo stesso numero di neuroni del primo layer
        nn.GELU(), #funzione di attivazione non lineare
        nn.Dropout(dropout_rate / 2), #secondo dropout piu leggero
        nn.Linear(hidden_size // 2, num_labels),# definisco il numero di neuroni dell'output layer (coincide col numero di classi)
    )

    #congelo il backbone del ViT
    for param in model.vit.parameters():
        param.requires_grad = False

    return model

def prepare_datasets(attribute: str, use_grayscale: bool) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.Tensor, Dict[int, str], Dict[str, int]]:
    """Carica dataset HuggingFace, aggiunge l'attributo e applica split."""
    # Carichiamo le informazioni sul CSV e otteniamo le mappature per l'attributo scelto.
    classes, class_to_id, id_to_class, experiment_to_class = load_attribute_metadata(attribute)

    #carica il dataset di esperiemnti. rappresentati nella forma:
    # {
    #     "image": file png,
    #     "label": "EXP01", "EXP02", "EXP03", ...
    # }
    dataset = load_dataset("Nardellar/Esperimenti", split="train")

    # Per ogni immagine associa l'ID classe usando le informazioni del CSV.
    dataset = add_class_id_column(dataset, class_to_id, experiment_to_class)
    #Rimuove dal dataset gli esempi a cui non è stata assegnata una classe valida.
    dataset = filter_missing_class(dataset, class_field="class_id")
    
    #trasforma la colonna class_id in un oggetto ClassLabel per gestire correttamente la stratificazione
    dataset = dataset.cast_column("class_id", ClassLabel(names=classes))

    #divido il dataset in train 80% e validation 20%, stratificando per class_id
    split_ds = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column="class_id")
    #estraggo il train e il validation set dallo split
    train_ds, val_ds = split_ds["train"], split_ds["test"]

    # Salvataggio del validation set nella cartella "validation_test_set"  
    # serve per le fase successiva di evaluation.
    val_ds.save_to_disk("validation_test_set")

    #calcola i pesi delle classi per bilanciamento (class Weights)
    #il calcolo è: weight[es: 0] = totale_campioni / (num_classi * num.immagini_classe[es: 0])
    #con 0 -> 1300 gradi
    images_per_class = Counter(train_ds["class_id"]) #conta quante immagini ci sono per ogni classe nel training set
    total_samples = sum(images_per_class.values())
    num_classes = len(classes)
    #calcolo i pesi bilanciati per ogni classe
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes, dtype=int),
        y=train_ds["class_id"],
    )
    #converto i pesi in un tensore pytorch
    class_weights = torch.tensor(weights, dtype=torch.float32)

    print("\nDistribuzione classi nel train set:")
    for class_id, count in sorted(images_per_class.items()):
        print(f"  - {id_to_class[class_id]}: {count} immagini")

    # Classe di HuggingFace che individua automaticamente il preprocessor corretto.
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    # Applica trasformazioni ai due set (crop, scala di grigi, normalizzazione, aggiunta labels).
    transform = build_vit_batch_preprocessor(processor, use_grayscale, label_field="class_id")
    train_ds = train_ds.with_transform(transform)
    val_ds = val_ds.with_transform(transform)

    return train_ds, val_ds, class_weights, id_to_class, class_to_id


class WeightedTrainer(Trainer):
    """Trainer esteso per applicare class weights e label smoothing."""

    def __init__(self, *args, class_weights: torch.Tensor, label_smoothing: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        #non creiamo subito la loss function, lo facciamo poi durante il calcolo 
        self.loss_fn = None
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        calcola loss con class weights e label smoothing.
        (override del metodo di huggingface usato da Trainer)
        
        Args:
            model: Il modello
            inputs: Dict con pixel_values e labels
            return_outputs: Se True, ritorna anche gli output del modello
            num_items_in_batch: Numero di item nel batch (nuovo parametro in transformers 4.x)
        """
        #rimuoviamo la label (class_id) dall'input e le salviamo
        labels = inputs.pop("labels")
        #passiamo in input al ViT le immagini e ci salviamo le sue predizioni (logits)
        outputs = model(**inputs)
        logits = outputs.logits

        #se il device dei class weights e' diverso dal device dei logits, li portiamo sullo stesso device (GPU o CPU)
        # se no il codice fallisce
        if self.loss_fn is None or self.class_weights.device != logits.device:
            self.class_weights = self.class_weights.to(logits.device)
            #definiamo la loss function con weighted class e label smoothing
            self.loss_fn = nn.CrossEntropyLoss(
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
            )
        #calcolaimo loss dando in input alla loss_fn le labels e i logits
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        #permette di restiutire solo loss (per trainer.train() ) o loss e output (per per trainer.evaluate() )
        #serve alla libreria Trainer
        return (loss, outputs) if return_outputs else loss

#calcola le metriche durante la valutazione del modello. viene usato nel validation set
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    #prende la classe predetta con probabilita' piu' alta dal modello per ogni immagine)
    predictions = logits.argmax(axis=-1)
    #calcolo quante volte la predizione è corretta e ne fa un media
    accuracy = (predictions == labels).mean()
    #f1 macro valuta in media tutte le classi indipendentemente dal numero di esempi
    f1_macro = f1_score(labels, predictions, average="macro")
    #f1 pesato utile per confrontare con accuracy quando il dataset è leggermente sbilanciato
    f1_weighted = f1_score(labels, predictions, average="weighted")
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def main():
    parser = argparse.ArgumentParser(description="Addestramento ViT PyTorch.")
    parser.add_argument(
        "attribute",
        nargs="?",
        default="temperatura",
        help="Nome dell'attributo da usare per la classificazione (default: temperatura).",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Se specificato, converte le immagini in scala di grigi.",
    )
    args = parser.parse_args()

    attribute = args.attribute
    results_dir = Path(f"training_results_{attribute}")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("ADDESTRAMENTO ViT")
    print(f"Attributo selezionato: {attribute}")

    # 1) Caricamento dataset e trasformazioni
    train_ds, val_ds, class_weights, id_to_class, class_to_id = prepare_datasets(attribute, args.grayscale)

    # 2) Costruzione del modello: backbone ViT congelato + classificatore personalizzato.
    model = build_model(num_labels=len(id_to_class))
    #imposta la variabile a un dizionario ID classe -> classe (es: 0 -> "1300", 1 -> "1400", 2 -> "1500")
    model.config.id2label = id_to_class
    #imposta la variabile a un dizionario classe -> ID classe (es: "1300" -> 0, "1400" -> 1, "1500" -> 2)
    model.config.label2id = class_to_id

    # 3) Definizione degli iperparametri e delle policy di training
    training_args = TrainingArguments(
        output_dir=str(results_dir), #specifica dove salvare i risultati dell'addestramento
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=100, #<- numero di epoche
        eval_strategy='epoch', #valuta il validation dopo ogni epoca
        save_strategy='epoch', #salva un checkpopoint alla fine di ogni epoca
        save_total_limit=1, # mantiene solo l'ultimo checkpoint
        load_best_model_at_end=True, #a fine training carica il checkpoint migliore
        metric_for_best_model='f1_macro',
        greater_is_better=True,
        weight_decay=1e-4, #regolarizzazione L2 (penalizza pesi grandi -> meno overfitting)
        learning_rate=5e-5,
        lr_scheduler_type='reduce_lr_on_plateau',
        lr_scheduler_kwargs={'patience': 7, 'factor': 0.5}, #aspetta 7 epoche prima di dimezzare il LR
        max_grad_norm=1.0,
        warmup_steps=100, #numero di step prima che il LR raggiunga il suo valore gradualmente
        logging_steps=50,
        logging_dir=str(results_dir / 'logs'), #dove salvare i logs
        report_to='tensorboard', #puoi visualizzare i risultati anche su tensorboard con "tensorboard --logdir training_results_temperatura/logs" su terminale
        remove_unused_columns=False,
        seed=42,
    )
    #carico il pre-processor per il ViT. use_fast=True usa il pre-processor più veloce
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)

    # 4) Istanziazione del Trainer con loss personalizzata per pesare le classi e usare label smoothing.
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator, #raggruppa i diversi dizionari in batch in un unico dizionario con tensori
        processing_class=processor,
        compute_metrics=compute_metrics, #passiamo la funzione che calcola le metriche (chiamata ad ogni epoca)
        class_weights=class_weights, #pesi per bilanciare le classi
        label_smoothing=0.1, #preveiene overfitting
    )
    #ferma il training se non migliora dopo 15 epoche
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=15))

    # 5) Logging a console delle impostazioni principali.
    #-------------------------------------------
    "TECNICHE ANTI-OVERFITTING".center(70)
    print("="*70)
    print(f"Backbone ViT           : CONGELATO (solo classifier addestrato)")
    print(f"Classifier dropout     : 0.3 (primo layer), 0.15 (secondo layer)")
    print(f"Weight Decay (L2)      : {training_args.weight_decay}")
    print(f"Label Smoothing        : 0.1")
    print(f"Gradient Clipping      : max_norm={training_args.max_grad_norm}")
    print(f"Learning Rate          : {training_args.learning_rate}")
    print(f"LR Scheduler           : {training_args.lr_scheduler_type}")
    print(f"LR Scheduler Patience  : 7 epoch (factor=0.5)")
    print(f"Warmup Steps           : {training_args.warmup_steps}")
    print(f"Early Stopping         : patience=15 epoch (monitora f1_macro)")
    print(f"Metric for best model  : {training_args.metric_for_best_model}")
    print(f"Class Weights          : ATTIVI (bilanciamento classi)")
    print(f"Batch Size             : {training_args.per_device_train_batch_size}")
    print(f"Num Epoch              : {training_args.num_train_epochs}")
    print(f"Training Samples       : {len(train_ds)}")
    print(f"Validation Samples     : {len(val_ds)}")
    print(f"Num Classi             : {len(id_to_class)}")
    print("="*70 + "\n")

    # 6) Avvio effettivo dell'addestramento gestito dal WightedTrainer.
    trainer.train()

    # Salva il modello migliore e il processor per riprodurre i preprocessamenti.
    trainer.save_model()
    #salvo il processor del modello per poterlo usare poi in evaluation
    processor.save_pretrained(results_dir)

    #carico il modello migliore
    eval_results = trainer.evaluate()
    print("\nValutazione finale sul validation set:")
    
    #stampo le metriche di valutazione
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    #creo e salvo informazioni e risultati che usero' poi in evaluation
    artifacts = {
        "attribute": attribute,
        "id_to_class": id_to_class,  # Mappa ID numerico → classe (es: 0 → "1300")
        "class_to_id": class_to_id,  # Mappa classe → ID numerico (es: "1300" → 0)
        "best_model_dir": str(results_dir),
        "eval_results": eval_results,
        "use_grayscale": args.grayscale,
    }
    with open(results_dir / "artifacts.json", "w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=4)

    print("\nAddestramento completato!")
    print(f"Modello e tokenizer salvati in: {results_dir}")
    print(f"Artifacts salvati in: {results_dir / 'artifacts.json'}")


if __name__ == "__main__":
    main()
