# 🔬 Analisi di Immagini del Clinker tramite Deep Learning

Questo repository contiene una coppia di modelli avanzati di Computer Vision e Machine Learning sviluppati per l'analisi automatica delle componenti del clinker (cemento) tramite immagini acquisite al microscopio.

Il progetto è strutturato in due moduli principali e indipendenti, ognuno dedicato a un task specifico di analisi: **Segmentazione Semantica** e **Classificazione**.

---

## 📂 Struttura del Progetto

Il repository è diviso in due grandi sezioni. Ogni sezione contiene i propri script di addestramento, inferenza e una documentazione dettagliata dedicata.

### 1. [Segmentazione (CNN + Gradient Boosting)](./Segmentazione/CNN/README.md)
Pipeline ibrida che combina l'estrazione spaziale di feature tramite Reti Neurali Convoluzionali (ResNet50 / ConvNeXt) con l'efficienza dei classificatori Gradient Boosting (XGBoost / LightGBM) e il raffinamento spaziale tramite DenseCRF.
* **Obiettivo:** Segmentazione pixel-perfect delle componenti (Alite, Belite, Fase Fusa, Pori, Calce libera).

### 2. [Classificazione (Vision Transformers - ViT)](Classificazione/ViT/README.md)
Modelli basati su architettura Transformer (ViT) per analizzare e classificare intere immagini di clinker in base a specifiche condizioni di produzione (temperatura, tempi di rampa, tempo totale, raffreddamento).
* **Obiettivo:** Classificazione globale e analisi dell'attenzione del modello (Attention Maps) sulle caratteristiche macroscopiche.

---

## 🚀 Setup Rapido Globale

Tutti i moduli condividono un unico ambiente virtuale per evitare conflitti di dipendenze tra i vari framework (TensorFlow, PyTorch, XGBoost, ecc.).

### Installazione dell'ambiente

**1.** Clona il repository:
   ```bash
   git clone https://github.com/Nardellar/Stage-Buzzi.git
   cd Stage-Buzzi
  ```
    
**2.** Crea un ambiente virtuale (consigliato Python 3.11):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   #.venv\Scripts\activate     # Windows
   ```
**3.** Installa tutte le dipendenze unificate:
    ```bash
    pip install -r requirements.txt
    ```

### Gestione Dati e Modelli (Note Importanti)
Per mantenere il repository leggero e rispettare i limiti di GitHub, il dataset per il task di segmentazione è incompleto, ma è possibile scaricarlo leggendo le [istruzioni specifiche](./Segmentazione/CNN/README.md) all'interno delle cartella CNN; stesso discordo per  i pesi dei modelli addestrati (.keras, .pth, .pkl).


## Tesi e Progetto
Per approfondire le scelte tecniche, l'analisi dei dati e i risultati completi, è possibile consultare la tesi integrale allegata al repository: \
**[Leggi la Tesi Completa (Tesi_Stage_Buzzi.pdf)](./Tesi_Stage_Buzzi.pdf)**