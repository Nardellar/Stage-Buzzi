# ViT Classification - PyTorch Implementation

## 📋 Overview
Questa cartella contiene la pipeline di **classificazione di attributi** basata su **Vision Transformer (ViT)** in PyTorch/HuggingFace.
## 🚀 Quick Start

### Installazione dipendenze
```bash
pip install -r requirements.txt
```

### Training Classificazione (ViT)
```bash
python -m Classificazione.ViT.create_and_train_model "(attributo)"
```

### Evaluation
```bash
python -m Classificazione.ViT.evaluate_model "(attributo)"
```

## 🌐 Risorse Hugging Face

Lo script usa risorse online Hugging Face:
- dataset: `Nardellar/Esperimenti`
- modello base: `google/vit-base-patch16-224`

Al primo avvio serve connessione internet per scaricare dataset/modello (poi vengono cache-ati localmente).

## 📌 Documentazione tecnica

- Scelte progettuali (head + regolarizzazione + configurazione): `DESIGN_CLASSIFICAZIONE.md`

## 🔧 Architettura

### Modello
- **Base**: `google/vit-base-patch16-224` pretrained
- **Backbone**: congelato (no fine-tuning end-to-end)
- **Classification Head custom** (MLP a 2 layer):
  - Dropout (0.3)
  - Linear layer (768 → 384)
  - LayerNorm
  - GELU activation
  - Dropout (0.15)
  - Linear layer (384 → num_classes)

### Parametri di Training
- **Optimizer**: AdamW (lr=5e-5, weight_decay=1e-4)
- **Loss**: CrossEntropy con class weights + label smoothing (0.1)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=7)
- **Early Stopping**: patience=15 epochs
- **Metriche**: accuracy, F1 macro, F1 weighted (best model su F1 macro)

Per motivazioni e dettagli di design: vedi `DESIGN_CLASSIFICAZIONE.md`.


## 📁 File Structure

```
Classificazione/ViT/
├── create_and_train_model.py    # Training (Trainer HF + loss pesata)
├── evaluate_model.py            # Evaluation + report/plot
├── manual_train_model_pytorch.py# Variante con training loop manuale
├── training_results_*/          # Training outputs
│   └── artifacts.json           # Metadata
├── evaluation_results_*/        # Evaluation outputs
│   ├── classification_report_*.json
│   ├── confusion_matrix_*.png
│   └── attention_maps_*.png
```
## 📊 Output del Training

Il training salva:
- directory con checkpoint/best model (gestita da HuggingFace Trainer)
- `artifacts.json` con metadati (attributo, mapping classi, flag grayscale, metriche)

## 🎓 Evaluation Metrics

L'evaluation produce:
1. **Classification Report** (JSON) - precision, recall, F1-score per classe
2. **Confusion Matrix** (PNG) - visualizzazione matrice di confusione
3. **Attention Maps** (PNG) - visualizzazione delle attention maps

## 📚 References

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [HuggingFace Preprocessing Guide](https://huggingface.co/docs/transformers/preprocessing)


