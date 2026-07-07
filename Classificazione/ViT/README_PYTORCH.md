# ViT Classification - PyTorch Implementation

## 📋 Overview

Questa implementazione usa **PyTorch nativo** con le **best practices di HuggingFace** per la classificazione di immagini con Vision Transformer (ViT).

## 🎯 Vantaggi rispetto alla versione TensorFlow

1. ✅ **Nessuna conversione PyTorch→TensorFlow** - modello nativo
2. ✅ **Serializzazione affidabile** - `torch.save()` sempre funziona
3. ✅ **Consistenza garantita** - stesso preprocessing per train/eval
4. ✅ **Coerente con segmentazione** - tutto il progetto usa PyTorch
5. ✅ **Best practices HuggingFace** - approccio standard della libreria

## 🚀 Quick Start

### Installazione dipendenze
```bash
pip install -r requirements.txt
```

### Training
```bash
python -m Classificazione.ViT.create_and_train_model "(attributo)"
```

### Evaluation
```bash
python -m Classificazione.ViT.evaluate_model temperatura
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
- **Classificatore custom**: 
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

## 📊 Dataset Preprocessing

### HuggingFace Best Practices

```python
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

def transform(examples):
    # 1. Converti in RGB (gestisce anche grayscale)
    images = [img.convert("RGB") for img in examples["image"]]
    
    # 2. Preprocessa con AutoImageProcessor
    #    - Resize automatico a 224x224
    #    - Normalizzazione con mean/std del modello
    inputs = processor(images, return_tensors="pt")
    
    # 3. Aggiungi labels
    inputs["labels"] = examples["attribute"]
    return inputs

# Applica on-the-fly (lazy loading)
dataset.set_transform(transform)
```

### DataLoader con Collator HuggingFace

```python
from transformers import default_data_collator

loader = DataLoader(
    dataset,
    batch_size=16,
    collate_fn=default_data_collator,  # Gestisce batching automatico
    num_workers=0  # Windows compatibility
)
```

## 🔑 Best Practices Implementate

### 1. **`AutoImageProcessor`**
- Preprocessing automatico specifico per il modello
- Gestisce resize, normalizzazione, formato corretto

### 2. **`set_transform()`**
- Preprocessing on-the-fly (non pre-processa tutto)
- Più efficiente in memoria
- Metodo standard HuggingFace

### 3. **`.convert("RGB")`**
- Converte esplicitamente in RGB
- Gestisce immagini in scala di grigi o altri formati

### 4. **`default_data_collator`**
- Collator HuggingFace per batching
- Gestisce automaticamente tensori di dimensioni diverse
- Più robusto del collate_fn di default di PyTorch

### 5. **Consistenza Training/Evaluation**
- IDENTICA trasformazione in train e eval
- Garantisce risultati consistenti
- Risolve il problema principale della versione TensorFlow

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


