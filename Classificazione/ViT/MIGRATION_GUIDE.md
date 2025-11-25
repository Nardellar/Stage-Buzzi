# 🔄 Guida alla Migrazione: TensorFlow → PyTorch

## 📋 Sommario dei Cambiamenti

### ✅ **Problema Risolto**
Il modello TensorFlow aveva **risultati inconsistenti** tra training e evaluation a causa di:
- Problemi di serializzazione del `TFViTModel` di HuggingFace
- Pesi del ViT che venivano reinizializzati durante il caricamento
- Mix di librerie (HuggingFace PyTorch → TensorFlow)

### 🎯 **Soluzione**
Migrazione completa a **PyTorch nativo** con **best practices HuggingFace**.

---

## 🆕 Nuovi File

| File | Descrizione |
|------|-------------|
| `train_model_pytorch.py` | Training script PyTorch (sostituisce `train_model.py`) |
| `evaluate_model_pytorch.py` | Evaluation script PyTorch (sostituisce `evaluate_model.py`) |
| `README_PYTORCH.md` | Documentazione completa |
| `MIGRATION_GUIDE.md` | Questa guida |

---

## 🚀 Come Usare i Nuovi Script
### 2️⃣ **Training**
Avvia il training con:

```bash
python create_and_train_model.py temperatura
```

oppure usa l'attributo che preferisci come argomento.

### 3️⃣ **Evaluation**
Dopo il training, valuta il modello:

```bash
python evaluate_model_pytorch.py
```

---

## 🔑 Best Practices HuggingFace Implementate

### 1. **`AutoImageProcessor`**
```python
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
```
- ✅ Preprocessing automatico specifico per il modello
- ✅ Gestisce resize, normalizzazione, formato

### 2. **`set_transform()` con conversione RGB**
```python
def transform(examples):
    # Converti in RGB (anche da grayscale)
    images = [img.convert("RGB") for img in examples["image"]]
    
    # Preprocessa
    inputs = processor(images, return_tensors="pt")
    inputs["labels"] = examples["attribute"]
    return inputs

dataset.set_transform(transform)
```
- ✅ Preprocessing on-the-fly (lazy loading)
- ✅ Efficiente in memoria
- ✅ Standard HuggingFace

### 3. **`default_data_collator`**
```python
from transformers import default_data_collator

loader = DataLoader(
    dataset,
    batch_size=16,
    collate_fn=default_data_collator  # ← Collator HuggingFace
)
```
- ✅ Batching automatico
- ✅ Gestisce tensori di dimensioni diverse
- ✅ Più robusto del default PyTorch

### 4. **Consistenza Training/Evaluation**
- ✅ IDENTICO preprocessing in entrambi gli script
- ✅ Garantisce risultati consistenti
- ✅ Risolve il problema principale

---

## 🗂️ Formato dei File Salvati



### PyTorch
```
training_results_temperatura/
├── best_model_temperatura_*.pth     # Modello completo
└── artifacts.json
```

## 📞 Riferimenti

- [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers)
- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)




