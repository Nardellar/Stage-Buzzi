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
| `test_pytorch_setup.py` | Script di test dell'ambiente |
| `MIGRATION_GUIDE.md` | Questa guida |

---

## 🚀 Come Usare i Nuovi Script

### 1️⃣ **Test dell'Ambiente**
Prima di tutto, verifica che tutto sia installato correttamente:

```bash
cd Classificazione/ViT
python test_pytorch_setup.py
```

Se tutti i test passano, procedi al passo 2.

### 2️⃣ **Training**
Avvia il training con:

```bash
python train_model_pytorch.py temperatura
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

## 📊 Confronto: TensorFlow vs PyTorch

| Aspetto | TensorFlow (vecchio) | PyTorch (nuovo) |
|---------|---------------------|----------------|
| **Serializzazione** | ❌ Problemi con TFViTModel | ✅ torch.save() affidabile |
| **Consistenza** | ❌ Risultati diversi train/eval | ✅ Risultati consistenti |
| **Conversione** | ❌ PyTorch → TensorFlow | ✅ Nativo PyTorch |
| **Coerenza progetto** | ❌ Segmentazione usa PyTorch | ✅ Tutto PyTorch |
| **Maintenance** | ❌ Codice complesso | ✅ Codice pulito |

---

## 🗂️ Formato dei File Salvati

### TensorFlow (vecchio)
```
training_results_temperatura/
├── best_model_temperatura_*.npz     # Solo pesi addestrabili
└── artifacts.json
```

### PyTorch (nuovo)
```
training_results_temperatura/
├── best_model_temperatura_*.pth     # Modello completo
└── artifacts.json
```

---

## 🔄 Migrazione di un Modello Esistente

Se hai già addestrato un modello con TensorFlow e vuoi migrare:

### ❌ **Non compatibile direttamente**
I pesi `.npz` di TensorFlow **non sono compatibili** con PyTorch.

### ✅ **Soluzione**
Ri-addestra il modello con la nuova versione PyTorch:

```bash
# 1. Backup del vecchio (opzionale)
mv training_results_temperatura training_results_temperatura_OLD

# 2. Training nuovo
python train_model_pytorch.py temperatura

# 3. Evaluation nuovo
python evaluate_model_pytorch.py
```

Il **dataset di validation** (`validation_test_set/`) è **compatibile** tra le due versioni, quindi i risultati saranno comparabili.

---

## 🧪 Verifica della Consistenza

Per verificare che training e evaluation diano risultati consistenti:

1. **Controlla l'ultima epoch del training**
   ```
   Epoch 100/100:
     Train Loss: 0.1234, Train Acc: 0.9500
     Val Loss: 0.2500, Val Acc: 0.8500  ← Questo valore
   ```

2. **Confronta con l'evaluation**
   ```
   Accuracy sul Test Set: 0.8500  ← Dovrebbe essere simile
   ```

Se i valori sono **simili** (±0.01), la consistenza è garantita! ✅

---

## 📚 File da Tenere

### ✅ **Mantieni questi file**
- `train_model_pytorch.py` - nuovo training
- `evaluate_model_pytorch.py` - nuovo evaluation
- `README_PYTORCH.md` - documentazione
- `test_pytorch_setup.py` - test ambiente

### 🗑️ **Opzionale: Backup vecchi file**
Puoi tenere i vecchi file TensorFlow come backup:
- `train_model.py` → `train_model_OLD.py`
- `evaluate_model.py` → `evaluate_model_OLD.py`

---

## 🐛 Troubleshooting

### Problema: "Can't pickle local object"
**Causa**: Multiprocessing su Windows con `num_workers > 0`

**Soluzione**: ✅ Già risolto con `num_workers=0`

### Problema: "CUDA out of memory"
**Causa**: GPU con poca memoria

**Soluzione**: Riduci `batch_size` da 16 a 8 o 4

### Problema: "Module not found"
**Causa**: Pacchetto non installato

**Soluzione**: 
```bash
pip install torch transformers datasets pillow
```

---

## ✅ Checklist Post-Migrazione

- [ ] `test_pytorch_setup.py` passa tutti i test
- [ ] Training completa senza errori
- [ ] Evaluation completa senza errori
- [ ] Accuracy di validation ≈ accuracy di evaluation
- [ ] File `.pth` e `artifacts.json` creati correttamente
- [ ] Confusion matrix e attention maps generate

---

## 🎓 Prossimi Passi

1. ✅ Testa il training su un piccolo numero di epochs
2. ✅ Verifica la consistenza dei risultati
3. ✅ Addestra il modello completo (100 epochs)
4. ✅ Confronta le performance con la versione TensorFlow
5. 🚀 Deploy del modello migliore!

---

## 📞 Riferimenti

- [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers)
- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)

---

**Data migrazione**: Ottobre 2024  
**Versione PyTorch**: 2.0+  
**Versione Transformers**: 4.30+

