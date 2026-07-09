# ViT Classification - PyTorch Implementation

## 📋 Overview
.........................
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

Puoi personalizzarle da CLI:

```bash
python -m Classificazione.ViT.create_and_train_model "(attributo)" \
--dataset-name "(Nome_Dataset)" \
  --model-name "(Nome_Modello)"
```
Al primo avvio serve connessione internet per scaricare dataset/modello (poi vengono cache-ati localmente).

## 🔧 Architettura

### Modello: `ViTForCustomClassification`
- **Base**: `google/vit-base-patch16-224` pretrained
- **Classification Head custom**: 
  - Dropout (0.3)
  - Linear layer (768 → 384)
  - LayerNorm
  - GELU activation
  - Dropout (0.15)
  - Linear layer (384 → num_classes)

### Parametri di Training
- **Optimizer**: AdamW (lr=5e-5, weight_decay=1e-4)
- **Loss**: CrossEntropyLoss con class weights
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=7)
- **Early Stopping**: patience=15 epochs
- **Batch Size**: 16


## 📁 File Structure

```
Classificazione/ViT/
├── train_model_pytorch.py       # Training script
├── evaluate_model_pytorch.py    # Evaluation script
├── training_results_*/          # Training outputs
│   ├── best_model_*.pth         # Saved model
│   └── artifacts.json           # Metadata
├── evaluation_results_*/        # Evaluation outputs
│   ├── classification_report_*.json
│   ├── confusion_matrix_*.png
│   └── attention_maps_*.png
```
## 📊 Output del Training

Il modello salvato contiene:
```python
{
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_loss': val_loss,
    'val_acc': val_acc,
}
```

## 🎓 Evaluation Metrics

L'evaluation produce:
1. **Classification Report** (JSON) - precision, recall, F1-score per classe
2. **Confusion Matrix** (PNG) - visualizzazione matrice di confusione
3. **Attention Maps** (PNG) - visualizzazione delle attention maps

## 📚 References

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [HuggingFace Preprocessing Guide](https://huggingface.co/docs/transformers/preprocessing)


