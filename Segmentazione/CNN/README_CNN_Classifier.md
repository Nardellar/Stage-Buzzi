# GPU-Optimized CNN + Boosting per Segmentazione Clinker

Questo progetto implementa una pipeline ibrida avanzata per la segmentazione semantica delle componenti del clinker tramite immagini al microscopio. 

Il sistema supera gli approcci tradizionali combinando la capacità di estrazione spaziale delle Reti Neurali Convoluzionali (CNN) con la potenza e l'interpretabilità dei classificatori gradient boosting, supportati da ottimizzazione automatica degli iperparametri e raffinamento spaziale.

## 🏗️ Architettura della Pipeline

L'approccio si divide in tre fasi sequenziali:

1. **Feature Extraction (CNN):** Un backbone pre-addestrato (ResNet50 o ConvNeXt-Tiny) analizza l'immagine ed estrae feature map ricche di contesto spaziale, eliminando il rumore dei decoder non addestrati.
2. **Pixel-wise Classification (Boosting):** I vettori di feature di ogni pixel vengono classificati da un modello XGBoost o LightGBM (ottimizzato su GPU). Optuna gestisce la ricerca degli iperparametri ideali (massimizzando la metrica mIoU).
3. **Raffinamento Spaziale (DenseCRF):** Un Conditional Random Field (CRF) applica vincoli di prossimità e colore (Bilateral e Gaussian energy) alle predizioni probabilistiche del booster, eliminando il rumore e definendo nettamente i contorni.

---
## 🚀 Quick Start

### 1. Installazione Dipendenze
Assicurati di avere un ambiente virtuale attivo, quindi installa i requisiti:

```bash
pip install -r requirements.txt
```

### 2. Addestramento del Modello
Per avviare l'estrazione delle feature e l'ottimizzazione del classificatore. Puoi scegliere il backbone CNN tramite riga di comando.

```bash
# Training standard (usa ConvNeXt-Tiny per default)
python train_gpu_optimized_classifier.py

# Training usando ResNet50 come estrattore
python train_gpu_optimized_classifier.py --cnn_model resnet50

# Training con conversione delle immagini in scala di grigi
python train_gpu_optimized_classifier.py --grayscale
```
*I file del modello addestrato (estrattore `.keras` e classificatore `.pkl`/`.json`) verranno salvati nella root del progetto.*

### 3. Valutazione e Inferenza
Per testare il modello sulle immagini di validazione/holdout. Lo script genererà in automatico un report delle metriche e salverà le anteprime.

```bash
# Valutazione completa con raffinamento DenseCRF attivato (Raccomandato)
python evaluate_gpu_model.py

# Valutazione disabilitando il CRF (restituisce le predizioni grezze del booster)
python evaluate_gpu_model.py --no_crf
```
*Gli output visivi (`gpu_model_preview.png` e `gpu_confusion_matrix.png`) verranno generati nella cartella corrente.*

---
### Classi di Segmentazione
Il sistema mappa le seguenti classi (ignorando i pixel di background non etichettati durante il training):
* **0**: Resina
* **1**: Pori/Imperfezioni
* **2**: Fase Fusa
* **3**: Belite
* **4**: Alite
* **5**: Calce libera

---

## 📂 Struttura dei File Core

* `gpu_optimized_cnn_classifier.py`: Contiene la classe principale che gestisce l'integrazione tra la CNN in Keras e l'estimatore XGBoost/LightGBM.
* `train_gpu_optimized_classifier.py`: Script di avvio per il training della pipeline e il tuning di Optuna.
* `evaluate_gpu_model.py`: Script di test finale. Applica l'inferenza, il filtraggio DenseCRF e genera le metriche (mIoU, F1) e le matrici di confusione.
* `data_module.py`: Gestisce il caricamento immagini, il partizionamento stratificato e la Data Augmentation tramite Albumentations.
* `tuning.py`: Gestisce l'ottimizzazione degli iperparametri (LightGBM/XGBoost) tramite framework Optuna.
* `script_di_controllo/*`: Utilities standalone per sovrapporre e ispezionare visivamente singole maschere e immagini originali e per clacolare la distribuzione delle classi nel dataset.

---



## ⚙️ Configurazione (Costanti)

Per modificare il comportamento interno, puoi agire direttamente sulle costanti dichiarate in cima al file `train_gpu_optimized_classifier.py`:

* `BATCH_SIZE`: default: 2.
* `CLASSIFIER_TYPE`: Scegli tra `"xgboost"` o `"lightgbm"`.
* `USE_GPU`: Imposta a `True` per sfruttare l'accelerazione hardware nel training dei booster.
* `MAX_PIXELS_PER_IMAGE`: Controlla il campionamento (stratificato) dei pixel per prevenire l'esaurimento della RAM (default: 20000).
* `TRIALS`: Numero di iterazioni concesse a Optuna per trovare i parametri ideali (default: 75).
* `AUGMENTATION_PROFILE`: Scegli tra `"standard"` o `"aggressive"`.