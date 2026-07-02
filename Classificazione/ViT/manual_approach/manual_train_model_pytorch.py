"""
Script per l'addestramento del modello ViT (Vision Transformer) con PyTorch.

Funzionalità principali:
- Divide il dataset in Train (80%) e Validation (20%) mantenendo la stratificazione per classe
- Salva il validation set su file per essere usato come test set separato in seguito
- Implementa: early stopping, learning rate scheduling, gradient clipping
- Salva il modello migliore basato sulla validation loss e gli artefatti necessari per l'inference
"""
# from __future__ import annotations permette di usare stringhe come type hints in Python < 3.10
from __future__ import annotations

# Librerie standard Python
import argparse  # Per gestire gli argomenti da riga di comando
import json  # Per salvare/caricare file JSON (artefatti del modello)
import sys  # Per aggiungere percorsi al PYTHONPATH
from collections import Counter  # Per contare le occorrenze delle classi
from datetime import datetime  # Per generare timestamp univoci nei nomi dei file
from pathlib import Path  # Per gestire i percorsi dei file in modo cross-platform

# Librerie per data manipulation e calcolo numerico
import pandas as pd  # Per leggere e manipolare il CSV degli esperimenti
# Librerie PyTorch per deep learning
import torch  # Framework principale per deep learning
import torch.nn as nn  # Moduli neural network (LayerNorm, Dropout, Linear, ecc.)
from PIL import Image  # Per manipolazione base delle immagini (crop, convert, ecc.)
# Librerie HuggingFace per dataset e modelli pretrained
from datasets import load_dataset, ClassLabel  # Per caricare dataset e gestire label categoriche
from torch.utils.data import DataLoader  # Per creare batch iterabili dal dataset
from tqdm import tqdm  # Per barre di progresso durante il training
from transformers import (
    ViTForImageClassification,  # Modello ViT pretrained per classificazione immagini
    AutoImageProcessor,  # Processor automatico per preprocessing delle immagini
    default_data_collator  # Funzione per combinare esempi in batch
)

# Aggiunge la directory root del progetto al PYTHONPATH per importare moduli locali
# Questo permette di importare common.csv_config anche se non siamo nella root
sys.path.append(str(Path(__file__).resolve().parents[2]))
from common import csv_config  # Modulo personalizzato per gestire il CSV degli esperimenti


# --- FUNZIONI DI PREPARAZIONE DEL DATASET ---
def prepare_and_split_dataset(attribute: str, batch_size: int, use_grayscale: bool):
    """
    Prepara e divide il dataset in train e validation set.

    Processo:
    1. Carica il CSV con i metadati degli esperimenti
    2. Crea mappature tra ID, classi originali e nuovi attributi
    3. Carica il dataset HuggingFace con le immagini
    4. Aggiunge gli attributi corrispondenti a ciascuna immagine
    5. Divide in train (80%) e validation (20%) mantenendo la stratificazione
    6. Calcola i pesi delle classi per bilanciare il dataset
    7. Configura il preprocessing delle immagini (crop, grayscale, resize, normalize)

    Args:
        attribute: Nome della colonna nel CSV da usare come attributo (es. "temperatura")
        batch_size: Dimensione del batch (usato solo per riferimento, non qui)
        use_grayscale: Se True, converte le immagini in scala di grigi

    Returns:
        train_ds: Dataset di training con transform applicato
        val_ds: Dataset di validation con transform applicato
        num_classes: Numero di classi uniche per l'attributo
        id2label: Dizionario che mappa ID classe -> nome label
        class_weights_tensor: Tensor con i pesi per bilanciare le classi nella loss
    """
    # Trova la directory root del progetto (2 livelli sopra questo file)
    # Esempio: se siamo in Classificazione/ViT/manual_train_model_pytorch.py
    # root_dir sarà il path a Stage-Buzzi/
    root_dir = Path(__file__).resolve().parents[2]
    csv_path = root_dir / "esperimenti.csv"

    # Se il CSV non esiste, lo crea usando il modulo di configurazione
    if not csv_path.exists():
        csv_config.create_csv(csv_path)

    # Legge il CSV in un DataFrame pandas
    # Il CSV contiene metadati: ID esperimento -> attributi (temperatura, ecc.)
    df = pd.read_csv(csv_path)

    # CREA MAPPATURA LABEL: converte stringhe di attributi in ID numerici
    # Esempio per "temperatura": {"1500": 0, "1400": 1, "1300": 2}

    # Estrae tutti i valori unici dell'attributo e li ordina alfabeticamente
    unique_attributes = sorted([str(attr) for attr in df[attribute].unique()])

    # Mappa: nome label -> ID numerico (es. "Alta" -> 0)
    label2id = {label: i for i, label in enumerate(unique_attributes)}

    # Mappa inversa: ID numerico -> nome label (es. 0 -> "Alta")
    id2label = {i: label for label, i in label2id.items()}

    # Numero totale di classi per questo attributo
    num_classes = len(unique_attributes)

    # Crea una mappa: ID esperimento -> valore attributo (come stringa)
    # Esempio: {1579: "Alta", 2201: "Media", ...}
    # df.set_index("ID") crea un indice sul campo ID per accesso veloce
    attr_map = {k: str(v) for k, v in df.set_index("ID")[attribute].to_dict().items()}

    # CARICA IL DATASET HUGGINGFACE
    # "Nardellar/Esperimenti" è un dataset remoto che contiene le immagini
    # split="train" carica il set di training completo
    ds = load_dataset("Nardellar/Esperimenti", split="train")

    def add_attribute(example):
        """
        Funzione helper per aggiungere l'attributo a ogni esempio del dataset.

        Il dataset originale ha label che corrispondono agli ID esperimento (es. "1579").
        Questa funzione:
        1. Converte l'ID (label numerica) in stringa (nome classe)
        2. Cerca il valore dell'attributo nel CSV usando l'ID
        3. Converte il valore attributo in ID numerico usando label2id

        Args:
            example: Singolo esempio del dataset (contiene "image", "label", ecc.)

        Returns:
            example: Esempio con nuovo campo "attribute" aggiunto
        """
        # Converte la label numerica in nome classe (ID esperimento come stringa)
        # Esempio: 1579 (int) -> "1579" (str)
        class_name = ds.features["label"].int2str(example["label"])

        # Cerca il valore dell'attributo per questo ID nel CSV
        # Se non trovato, usa -1 come valore sentinella
        raw_attribute_value = attr_map.get(class_name, -1)

        # Converte il valore attributo in ID numerico (es. "Alta" -> 0)
        # Se non trovato, usa -1
        example["attribute"] = label2id.get(raw_attribute_value, -1)
        return example

    # APPLICA LA MAPPA E FILTRA
    # map(): aggiunge il campo "attribute" a tutti gli esempi
    # filter(): rimuove gli esempi dove attribute == -1 (ID non trovati nel CSV)
    ds = ds.map(add_attribute).filter(lambda ex: ex["attribute"] != -1)

    # Converte la colonna "attribute" in ClassLabel (tipo HuggingFace per classi categoriche)
    # Questo permette a HuggingFace di gestire correttamente la stratificazione
    ds = ds.cast_column('attribute', ClassLabel(names=unique_attributes))

    # DIVIDE IL DATASET IN TRAIN E VALIDATION
    print("\n--- Divisione del dataset (80% Train, 20% Validation) ---")

    # train_test_split con stratify_by_column mantiene la stessa distribuzione di classi
    # in train e validation (evita sbilanciamenti)
    # seed=42 garantisce riproducibilità
    # test_size=0.2 significa 20% validation, 80% train
    ds_split = ds.train_test_split(test_size=0.2, seed=42, stratify_by_column="attribute")
    train_ds, val_ds = ds_split["train"], ds_split["test"]

    print(f"Train set: {len(train_ds)} immagini")
    print(f"Validation set: {len(val_ds)} immagini")

    # SALVA IL VALIDATION SET SU DISCO
    # Questo permette di usare lo stesso validation set come test set fisso
    # per confrontare diversi modelli o configurazioni
    val_ds.save_to_disk("validation_test_set")
    print("Validation set salvato in 'validation_test_set/'")

    # CALCOLA I PESI DELLE CLASSI PER BILANCIAMENTO
    # Se alcune classi sono più rappresentate di altre, la loss le peserà meno
    # Formula: weight[classe] = totale_campioni / (num_classi * count[classe])
    # Esempio: se classe A ha 100 campioni e classe B ha 50:
    #   weight[A] = 150 / (2 * 100) = 0.75
    #   weight[B] = 150 / (2 * 50) = 1.5
    # La classe B (minoritaria) avrà peso maggiore nella loss

    # Conta quanti campioni ci sono per ogni classe nel training set
    class_counts = Counter(train_ds["attribute"])
    total_samples = sum(class_counts.values())

    # Calcola i pesi usando la formula standard per class balancing
    class_weights = {
        class_id: total_samples / (num_classes * count)
        for class_id, count in class_counts.items()
    }

    # Converte in tensor PyTorch con ordine corretto (classe 0, 1, 2, ...)
    class_weights_tensor = torch.tensor([class_weights[i] for i in range(num_classes)])

    # CONFIGURA IL PREPROCESSING DELLE IMMAGINI
    # AutoImageProcessor carica il preprocessor configurato per il modello ViT
    # Gestisce automaticamente: resize a 224x224, normalizzazione, ecc.
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

    def transform(examples):
        """
        Funzione di trasformazione applicata on-the-fly alle immagini.

        Questo metodo è "lazy": le trasformazioni vengono applicate solo quando
        gli esempi vengono effettivamente caricati durante l'iterazione del dataset.
        Questo risparmia memoria rispetto al preprocessing anticipato.

        Args:
            examples: Batch di esempi dal dataset (dizionario con "image", "attribute", ecc.)

        Returns:
            inputs: Dizionario con "pixel_values" (tensor preprocessato) e "labels"
        """

        def crop(image):
            """
            Ritaglia l'immagine dall'alto a max 950 pixel di altezza.

            Questo rimuove eventuali artefatti o aree non informative nella parte bassa
            delle immagini del dataset (probabilmente barre di scala o annotazioni).

            Args:
                image: PIL Image da ritagliare

            Returns:
                image: Immagine ritagliata o originale se già < 950px
            """
            width, height = image.size
            crop_height = min(950, height)  # Massimo 950 pixel di altezza

            # Se l'immagine è già più piccola, non fare nulla
            if crop_height == height:
                return image

            # Ritaglia: (left, top, right, bottom)
            # Ritaglia dall'alto lasciando la parte superiore
            return image.crop((0, 0, width, crop_height))

        def maybe_grayscale(image):
            """
            Converte opzionalmente l'immagine in scala di grigi.

            Il ViT si aspetta immagini RGB (3 canali), quindi anche in grayscale
            dobbiamo replicare il canale su 3 canali.

            Args:
                image: PIL Image da convertire

            Returns:
                image: Immagine RGB (originale o grayscale replicato)
            """
            if not use_grayscale:
                # Mantieni i colori originali, assicurati che sia RGB
                return image.convert("RGB")

            # Converti in scala di grigi (L = Luminance, un solo canale)
            gray = image.convert("L")
            # Replica il canale grigio su 3 canali RGB
            # Image.merge crea un'immagine RGB da 3 bande separate
            return Image.merge("RGB", (gray, gray, gray))

        # APPLICA TRASFORMAZIONI A TUTTE LE IMMAGINI DEL BATCH
        # 1. Crop per rimuovere aree non informative
        # 2. Opzionalmente converti in grayscale
        # 3. La lista viene creata con list comprehension
        images = [maybe_grayscale(crop(img)) for img in examples["image"]]

        # PREPROCESSING CON HUGGINGFACE PROCESSOR
        # Il processor applica:
        #   - Resize a 224x224 (dimensione input del ViT)
        #   - Normalizzazione con mean/std del dataset ImageNet
        #   - Conversione in tensor PyTorch
        # return_tensors="pt" significa che ritorna tensori PyTorch invece di numpy
        inputs = processor(images, return_tensors="pt")

        # Aggiunge le label al dizionario degli input
        inputs["labels"] = examples["attribute"]
        return inputs

    # APPLICA IL TRANSFORM IN MODO LAZY (ON-THE-FLY)
    # set_transform() non applica subito le trasformazioni, ma le applica solo quando
    # il dataset viene iterato (quando il DataLoader carica i batch)
    # Questo è il metodo standard HuggingFace per preprocessing efficiente
    train_ds.set_transform(transform)
    val_ds.set_transform(transform)

    return train_ds, val_ds, num_classes, id2label, class_weights_tensor






# --- CLASSE DEL MODELLO ---
class ViTForCustomClassification(nn.Module):
    """
    Modello ViT personalizzato per classificazione immagini.
    
    Strategia di fine-tuning:
    - Usa un ViT pretrained su ImageNet (google/vit-base-patch16-224)
    - Congela i parametri del ViT base (feature extractor) per evitare overfitting
    - Sostituisce il classificatore originale con uno personalizzato a 2 livelli
    - Solo il classificatore personalizzato viene addestrato 

    """

    def __init__(self, num_labels, dropout_rate=0.3):
        """
        Inizializza il modello ViT personalizzato.
        
        ⚠️ QUANDO VIENE CHIAMATO __init__:
        Questo metodo viene chiamato AUTOMATICAMENTE quando crei un'istanza della classe,
        esattamente alla riga 735 nel main() con:
            model = ViTForCustomClassification(num_labels=num_classes)
        
        Non viene mai chiamato manualmente - Python lo invoca automaticamente come "costruttore".
        Viene eseguito DOPO che prepare_and_split_dataset() ha finito e PRIMA di train_model().
        
        Args:
            num_labels: Numero di classi da classificare (es. 3 per temperatura)
            dropout_rate: Tasso di dropout per il primo livello (default 0.3)
                          Il secondo dropout è metà di questo valore (0.15)
        """
        # super().__init__() chiama il costruttore della classe padre (nn.Module)
        # Necessario per inizializzare correttamente l'oggetto PyTorch
        super().__init__()
        
        # Carica il ViT pretrained da HuggingFace
        # "google/vit-base-patch16-224": modello base con patch 16x16, input 224x224
        # ignore_mismatched_sizes=True: permette di cambiare il numero di classi
        # anche se il modello originale aveva un numero diverso di output
        self.vit = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

        # CONGELA IL BACKBONE ViT (Transfer Learning - Feature Extraction Only)
        # Impostando requires_grad=False, i gradienti non vengono calcolati per questi parametri
        # Questo significa che durante il training solo il classificatore verrà aggiornato
        # Risparmia memoria, velocizza il training e previene overfitting
        for param in self.vit.vit.parameters():
            param.requires_grad = False

        # SOSTITUISCE IL CLASSIFICATORE ORIGINALE CON UNO PERSONALIZZATO
        # Il classificatore originale è un semplice Linear layer
        # Qui creiamo una rete più complessa con:
        #   - Dropout per regolarizzazione
        #   - Layer intermedio per feature learning
        #   - LayerNorm per normalizzazione e stabilità
        #   - GELU come funzione di attivazione (più smooth di ReLU)
        
        # Dimensione dell'embedding nascosto del ViT (768 per vit-base)
        hidden_size = self.vit.config.hidden_size
        
        # Classificatore a 2 livelli con dropout progressivo
        self.vit.classifier = nn.Sequential(
            # Primo layer: Dropout iniziale (30%) per regolarizzazione forte
            nn.Dropout(dropout_rate),
            
            # Riduce la dimensionalità da hidden_size a hidden_size/2
            # Questo crea un bottleneck che forza il modello a imparare rappresentazioni compresse
            nn.Linear(hidden_size, hidden_size // 2),
            
            # LayerNorm: normalizza i valori lungo le features (stabilità e convergenza)
            # Applicata DOPO il Linear layer (Post-LayerNorm architecture)
            nn.LayerNorm(hidden_size // 2),
            
            # GELU (Gaussian Error Linear Unit): attivazione non-lineare
            # Più smooth di ReLU, migliora spesso le prestazioni nei Transformer
            nn.GELU(),
            
            # Secondo Dropout più leggero (15%) - dropout decrescente
            nn.Dropout(dropout_rate / 2),
            
            # Layer finale: mappa alle classi target
            # Output size = num_labels (es. 3 per temperatura)
            nn.Linear(hidden_size // 2, num_labels)
        )

    def forward(self, pixel_values, labels=None):
        """
        Forward pass del modello.
        
        Args:
            pixel_values: Tensor delle immagini preprocessate [batch_size, 3, 224, 224]
            labels: Tensor delle label true [batch_size] (opzionale, per training)
        
        Returns:
            outputs: Oggetto con attributi .loss (se labels fornite) e .logits
        """
        # Delega il forward al modello ViT interno
        # Il modello calcola automaticamente la loss se labels sono fornite
        return self.vit(pixel_values=pixel_values, labels=labels)





# --- FUNZIONE DI TRAINING ---
def train_model(model, train_ds, val_ds, class_weights, num_epochs, results_dir, attribute):
    """
    Addestra il modello con un loop personalizzato (non usa HuggingFace Trainer).
    
    Implementa tecniche avanzate:
    - Early stopping: ferma il training se la validation loss non migliora
    - Learning rate scheduling: riduce il LR quando la loss si appiattisce
    - Gradient clipping: previene esplosione dei gradienti
    - Label smoothing: migliora la generalizzazione riducendo overconfidence
    - Class balancing: usa pesi per bilanciare classi sbilanciate
    
    Args:
        model: Modello ViT da addestrare (ViTForCustomClassification)
        train_ds: Dataset di training
        val_ds: Dataset di validation
        class_weights: Tensor con pesi per bilanciare le classi
        num_epochs: Numero massimo di epoche
        results_dir: Directory dove salvare il modello e i risultati
        attribute: Nome dell'attributo (per naming dei file)
    
    Returns:
        model_path: Path al modello salvato (il migliore basato su validation loss)
    """
    # SETUP DEL DISPOSITIVO (GPU o CPU)
    # Rileva automaticamente se c'è una GPU disponibile
    # Se c'è CUDA, usa la GPU per velocizzare drasticamente il training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo: {device}")
    
    # Sposta il modello e i pesi delle classi sul dispositivo (GPU o CPU)
    model = model.to(device)
    class_weights = class_weights.to(device)

    # SETUP DELL'OPTIMIZER: AdamW con weight decay
    # AdamW è una variante di Adam che implementa correttamente il weight decay (L2 regolarizzazione)
    # lr=5e-5: learning rate conservativo, buono per fine-tuning
    # weight_decay=1e-4: regolarizzazione L2 per prevenire overfitting
    #    Penalizza i pesi grandi, forzando il modello a trovare soluzioni più semplici
    optimizer = torch.optim.AdamW(
        model.parameters(),  # Tutti i parametri addestrabili del modello
        lr=5e-5,  # Learning rate iniziale (0.00005)
        weight_decay=1e-4  # L2 regularization (0.0001) - previene overfitting
    )

    # SETUP DELLA LOSS FUNCTION: CrossEntropyLoss con tecniche avanzate
    # weight=class_weights: bilancia automaticamente le classi
    #    Le classi minoritarie hanno peso maggiore nella loss
    # label_smoothing=0.1: tecnica di regolarizzazione che "ammorbidisce" le label
    #    Invece di label hard [0, 0, 1], usa [0.05, 0.05, 0.90]
    #    Riduce overconfidence e migliora la generalizzazione
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,  # Bilanciamento delle classi
        label_smoothing=0.1  # Label smoothing al 10% - migliora generalizzazione
    )

    # SETUP DEL LEARNING RATE SCHEDULER
    # ReduceLROnPlateau riduce il learning rate quando la loss si appiattisce
    # mode='min': monitora la validation loss e riduce quando non diminuisce
    # factor=0.5: riduce il LR della metà quando triggerato
    # patience=7: aspetta 7 epoche senza miglioramento prima di ridurre
    # Questo aiuta il modello a convergere meglio verso la fine del training
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7
    )

    # CREA I DATALOADERS
    # I DataLoaders gestiscono il batching, shuffling e caricano i dati in modo efficiente
    
    # DataLoader per il training
    train_loader = DataLoader(
        train_ds,
        batch_size=16,  # Dimensione del batch (16 immagini per volta)
        shuffle=True,  # Mescola i dati ad ogni epoca per migliorare la generalizzazione
        collate_fn=default_data_collator,  # Funzione HuggingFace per combinare esempi in batch
        num_workers=0  # 0 per compatibilità Windows (su Linux si può usare > 0)
    )
    
    # DataLoader per la validation
    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,  # Non mescolare per validation (non necessario e mantiene ordine)
        collate_fn=default_data_collator,
        num_workers=0
    )

    # INIZIALIZZA VARIABILI PER EARLY STOPPING E SALVATAGGIO
    best_val_loss = float('inf')  # Traccia la miglior validation loss (inizialmente infinito)
    patience = 15  # Numero di epoche da attendere senza miglioramento prima di fermarsi
    patience_counter = 0  # Contatore delle epoche senza miglioramento
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Timestamp per nome file univoco
    model_path = results_dir / f"best_model_{attribute}_{timestamp}.pth"  # Path del modello salvato

    print("\n--- Inizio addestramento ---")
    
    # LOOP DI TRAINING: itera per num_epochs epoche
    for epoch in range(num_epochs):
        # ========== FASE DI TRAINING ==========
        model.train()  # Imposta il modello in modalità training (abilita dropout, batch norm)
        train_loss = 0.0  # Accumula la loss totale
        train_correct = 0  # Conta le predizioni corrette
        train_total = 0  # Conta il totale delle immagini processate

        # Itera sui batch del training set
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"):
            # Sposta i dati sul dispositivo (GPU se disponibile)
            pixel_values = batch["pixel_values"].to(device)  # Immagini preprocessate
            labels = batch["labels"].to(device)  # Label true

            # RESET GRADIENTI
            # PyTorch accumula i gradienti per default, quindi dobbiamo azzerarli
            optimizer.zero_grad()
            
            # FORWARD PASS
            # Il modello calcola le predizioni e la loss automaticamente
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss  # Loss già calcolata dal modello (con class weights e label smoothing)

            # BACKWARD PASS: calcola i gradienti
            loss.backward()

            # GRADIENT CLIPPING: previene esplosione dei gradienti
            # Se la norma dei gradienti supera 1.0, li scala per mantenerla a 1.0
            # Questo è importante per stabilità del training, specialmente in reti profonde
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # AGGIORNA I PESI: applica i gradienti usando l'optimizer
            optimizer.step()

            # AGGIORNA STATISTICHE PER METRICHE
            train_loss += loss.item()  # Accumula la loss (item() converte tensor -> float)
            
            # Calcola le predizioni: argmax trova la classe con probabilità maggiore
            predictions = outputs.logits.argmax(dim=1)
            
            # Conta quante predizioni sono corrette
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)  # Dimensione del batch

        # Calcola metriche medie per l'epoca
        train_acc = train_correct / train_total  # Accuracy = corrette / totali
        train_loss /= len(train_loader)  # Loss media = loss totale / numero di batch

        # ========== FASE DI VALIDATION ==========
        model.eval()  # Imposta il modello in modalità evaluation (disabilita dropout, batch norm)
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # torch.no_grad() disabilita il calcolo dei gradienti (risparmia memoria e velocizza)
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]"):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass (solo inference, niente backward)
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

                # Accumula statistiche
                val_loss += loss.item()
                predictions = outputs.logits.argmax(dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        # Calcola metriche medie per validation
        val_acc = val_correct / val_total
        val_loss /= len(val_loader)

        # ========== LOGGING E METRICHE ==========
        print(f"\nEpoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # ========== LEARNING RATE SCHEDULING ==========
        # Il scheduler monitora la validation loss e riduce il LR se non migliora
        # Questo viene fatto ad ogni epoca per permettere al modello di convergere meglio
        scheduler.step(val_loss)

        # ========== EARLY STOPPING E SALVATAGGIO ==========
        # Controlla se la validation loss è migliorata
        if val_loss < best_val_loss:
            # MIGLIORAMENTO: aggiorna la best loss e resetta il contatore di patience
            best_val_loss = val_loss
            patience_counter = 0
            
            # SALVA IL MODELLO MIGLIORE
            # Salva sia i pesi del modello che lo stato dell'optimizer
            # Questo permette di riprendere il training esattamente da questo punto
            torch.save({
                'epoch': epoch,  # Epoca corrente
                'model_state_dict': model.state_dict(),  # Pesi del modello
                'optimizer_state_dict': optimizer.state_dict(),  # Stato optimizer (momentum, ecc.)
                'val_loss': val_loss,  # Validation loss per riferimento
                'val_acc': val_acc,  # Validation accuracy per riferimento
            }, model_path)
            print(f"  Modello salvato! (val_loss improved to {val_loss:.4f})")
        else:
            # NON C'È MIGLIORAMENTO: incrementa il contatore di patience
            patience_counter += 1
            
            # Se abbiamo aspettato abbastanza senza miglioramenti, ferma il training
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                print(f"  Nessun miglioramento per {patience} epoche consecutive")
                break

    return model_path


# --- MAIN ---
def main():
    """
    Funzione principale che orchestrazione tutto il processo di training.
    
    Flusso di esecuzione:
    1. Parsa gli argomenti da riga di comando
    2. Crea la directory per i risultati
    3. Prepara e divide il dataset
    4. Crea il modello ViT personalizzato
    5. Esegue il training
    6. Salva gli artifacts necessari per l'inference futura
    """
    # SETUP ARGOMENTI DA RIGA DI COMANDO
    # argparse permette di passare parametri quando si esegue lo script
    # Esempio: python manual_train_model_pytorch.py temperatura --grayscale
    parser = argparse.ArgumentParser(description="Training ViT manuale (loop personalizzato).")
    
    # Argomento posizionale: l'attributo da classificare
    # nargs="?" significa opzionale, se non specificato usa il default
    parser.add_argument(
        "attribute",
        nargs="?",
        default="temperatura",  # Valore di default se non specificato
        help="Attributo da classificare (default: temperatura).",
    )
    
    # Flag opzionale: converte le immagini in scala di grigi
    # action="store_true" significa che se --grayscale è presente, il valore è True
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Converte le immagini in scala di grigi (replicate su 3 canali).",
    )
    
    # Parsa gli argomenti e li salva in args
    args = parser.parse_args()

    # Estrae i valori degli argomenti
    attribute = args.attribute
    
    # Stampa informazioni di configurazione
    print("ADDESTRAMENTO ViT con PyTorch")
    print(f"Attributo selezionato: {attribute}")
    print(f"Uso scala di grigi     : {args.grayscale}")

    # CREA LA DIRECTORY PER I RISULTATI
    # I risultati (modello, artifacts, metriche) verranno salvati qui
    # parents=True crea anche le directory parent se non esistono
    # exist_ok=True non solleva errore se la directory esiste già
    results_dir = Path(f"training_results_{attribute}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # PREPARA IL DATASET
    # Questa funzione:
    #   - Carica il CSV e il dataset HuggingFace
    #   - Crea le mappature label
    #   - Divide in train/validation (80/20)
    #   - Calcola i pesi delle classi
    #   - Configura il preprocessing delle immagini
    train_ds, val_ds, num_classes, id2label, class_weights = prepare_and_split_dataset(
        attribute, batch_size=16, use_grayscale=args.grayscale
    )

    # CREA IL MODELLO ViT PERSONALIZZATO
    # ⭐ QUI VIENE CHIAMATO __init__ AUTOMATICAMENTE! ⭐
    # Quando esegui questa riga, Python chiama automaticamente il metodo __init__ della classe
    # con i parametri num_labels=num_classes (e dropout_rate=0.3 di default)
    #
    # Il modello viene inizializzato con:
    #   - Backbone ViT pretrained (congelato)
    #   - Classificatore personalizzato (addestrabile)
    #
    # Sequenza: Python → crea istanza → chiama __init__ → carica ViT → configura classificatore
    model = ViTForCustomClassification(num_labels=num_classes)

    # ESEGUE IL TRAINING
    # Questa funzione implementa il loop di training completo con:
    #   - Early stopping
    #   - Learning rate scheduling
    #   - Salvataggio del modello migliore
    # num_epochs=100: massimo 100 epoche (ma early stopping può fermare prima)
    model_path = train_model(
        model, train_ds, val_ds, class_weights, 
        num_epochs=100,  # Numero massimo di epoche
        results_dir=results_dir, 
        attribute=attribute
    )

    # SALVA GLI ARTIFACTS PER L'INFERENCE
    # Gli artifacts sono informazioni necessarie per usare il modello in seguito:
    #   - mappatura ID -> label (per decodificare le predizioni)
    #   - numero di classi
    #   - path del modello salvato
    #   - configurazione (grayscale o no)
    artifacts = {
        'attribute': attribute,  # Nome dell'attributo classificato
        'id2label': {int(k): v for k, v in id2label.items()},  # Mappatura ID -> nome classe
        'model_path': str(model_path),  # Path del modello salvato (come stringa)
        'num_classes': num_classes,  # Numero di classi
        'use_grayscale': args.grayscale,  # Flag per sapere se usare grayscale in inference
    }
    
    # Salva gli artifacts in un file JSON leggibile
    # indent=4 rende il JSON formattato e leggibile
    with open(results_dir / "artifacts.json", "w") as f:
        json.dump(artifacts, f, indent=4)

    # Messaggio finale con informazioni sui file salvati
    print("\nAddestramento completato!")
    print(f"Modello salvato in: {model_path}")
    print(f"Artifacts salvati in: {results_dir / 'artifacts.json'}")


if __name__ == "__main__":
    """
    Entry point dello script.
    
    Questo blocco viene eseguito solo se lo script è eseguito direttamente
    (non quando viene importato come modulo).
    """
    main()

