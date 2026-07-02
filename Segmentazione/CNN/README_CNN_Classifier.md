# CNN + Classificatore per Segmentazione

Questo sistema implementa un approccio moderno per la segmentazione delle componenti del cemento, basato sull'architettura del notebook `Segmentazione_best_(optimization).ipynb` ma con miglioramenti significativi.

## Architettura

Il sistema utilizza un approccio a due fasi:

1. **Encoder CNN**: Estrae features multi-scala da un modello pre-addestrato (ResNet50, EfficientNet, o VGG16)
2. **Classificatore XGBoost**: Classifica ogni pixel nelle 6 classi utili

### Classi di Segmentazione

- **Classe 0**: Pixel non etichettati (esclusi dall'addestramento)
- **Classe 1**: Resina
- **Classe 2**: Pori e imperfezioni
- **Classe 3**: Fase fusa
- **Classe 4**: Belite
- **Classe 5**: Alite
- **Classe 6**: Classe 6 (label sconosciuta)

## File Principali

- `cnn_classifier_model.py`: Classe principale del modello
- `train_cnn_classifier.py`: Script di training
- `test_cnn_classifier.py`: Script di test e visualizzazione
- `README_CNN_Classifier.md`: Questa documentazione

## Installazione Dipendenze

```bash
pip install tensorflow
pip install tf-keras
pip install xgboost
pip install optuna
pip install opencv-python
pip install pillow
pip install scikit-learn
pip install matplotlib
pip install tqdm
```

## Utilizzo

### 1. Training del Modello

```bash
# Training con ResNet50 (raccomandato)
python train_cnn_classifier.py --cnn_model resnet50 --image_size 512 512 --batch_size 2

# Training con EfficientNet #CONSIGLIATO FOR REAL
python train_cnn_classifier.py --cnn_model efficientnet --image_size 512 512 --batch_size 2

# Training senza ottimizzazione parametri (più veloce)
python train_cnn_classifier.py --skip_optimization

# Training con parametri personalizzati
python train_cnn_classifier.py --cnn_model resnet50 --optimize_trials 50 --model_save_path my_model.pkl
```

### 2. Test del Modello

```bash
# Test su una singola immagine
python test_cnn_classifier.py --image_path ../images/Immagini/1579--03.png

# Test su più immagini
python test_cnn_classifier.py --image_dir ../images/Immagini/ --num_images 5

# Confronto con ground truth
python test_cnn_classifier.py --image_path ../images/Immagini/1579--03.png --mask_path ../images/Maschere/1579--03.tif

# Salva i risultati
python test_cnn_classifier.py --image_dir ../images/Immagini/ --num_images 3 --save_results
```

## Caratteristiche del Sistema

### Vantaggi rispetto al notebook originale:

1. **Architetture moderne**: Supporta ResNet50, EfficientNet, VGG16
2. **Gestione memoria ottimizzata**: Elaborazione in batch per gestire grandi immagini
3. **Ottimizzazione automatica**: Usa Optuna per ottimizzare i parametri XGBoost
4. **Visualizzazione avanzata**: Confronto con ground truth e statistiche dettagliate
5. **Codice modulare**: Facilmente estendibile e modificabile

### Gestione delle Classi:

- **Filtraggio automatico**: Esclude automaticamente i pixel di classe 0 (non etichettati)
- **Rimappatura**: Converte le etichette da 1-6 a 0-5 per il classificatore
- **Validazione**: Verifica la distribuzione delle classi durante il caricamento

### Ottimizzazioni:

- **Features multi-scala**: Combina features da diversi livelli della CNN
- **Upsampling intelligente**: Riconcilia le dimensioni delle features
- **Augmentation**: Supporto per data augmentation (opzionale)
- **CRF**: Possibilità di applicare Conditional Random Fields (da implementare)

## Parametri di Training

### Modelli CNN Supportati:

- `resnet50`: ResNet50 pre-addestrato su ImageNet (raccomandato)
- `efficientnet`: EfficientNet-B0 pre-addestrato su ImageNet
- `vgg16`: VGG16 pre-addestrato su ImageNet

### Parametri XGBoost:

Il sistema ottimizza automaticamente:
- `max_depth`: Profondità dell'albero (4-10)
- `learning_rate`: Tasso di apprendimento (0.01-0.3)
- `n_estimators`: Numero di alberi (100-500)
- `subsample`: Percentuale di dati utilizzati (0.6-1.0)
- `colsample_bytree`: Percentuale di features utilizzate (0.6-1.0)
- E molti altri parametri...

## Struttura dei Dati

### Input:
- **Immagini**: PNG/JPG in `../images/Immagini/`
- **Maschere**: TIFF in `../images/Maschere/`
- **Dimensioni**: 1024x1024 (ridimensionate automaticamente)

### Output:
- **Modello**: File .pkl con classificatore addestrato
- **Visualizzazioni**: Immagini con risultati della segmentazione
- **Statistiche**: Report dettagliati delle performance

## Troubleshooting

### Errori Comuni:

1. **Memoria insufficiente**: Riduci `batch_size` o `image_size`
2. **GPU non disponibile**: Il sistema usa automaticamente la CPU
3. **File non trovati**: Verifica i percorsi delle immagini e maschere
4. **Errori di import**: Installa tutte le dipendenze richieste

### Performance:

- **Training**: 20-60 minuti (dipende da GPU e numero di trial)
- **Inference**: 1-5 secondi per immagine
- **Memoria**: 4-8 GB RAM raccomandati

## Estensioni Future

Possibili miglioramenti:
1. **CRF post-processing**: Applicazione di Conditional Random Fields
2. **Ensemble models**: Combinazione di più modelli
3. **Transfer learning**: Fine-tuning dell'encoder CNN
4. **Data augmentation**: Augmentation automatica durante il training
5. **Validation cross-fold**: Validazione incrociata per stime più robuste

## Contatto

Per domande o problemi, consulta la documentazione del progetto o contatta il team di sviluppo.


