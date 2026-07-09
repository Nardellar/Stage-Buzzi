# Design classificazione (ViT) — scelte architetturali e training

Questo documento descrive le scelte progettuali implementate per la classificazione

---

## Architettura: backbone ViT + head personalizzata

### Backbone
- **Modello base**: `google/vit-base-patch16-224`
- **Backbone/encoder congelato**: durante il training non vengono aggiornati i pesi del ViT; si addestra solo la testa di classificazione.

**Motivazione:**
- dataset relativamente piccolo (1400 immagini per attributo)
- poche classi per attributo (tipicamente 2–5)
- obiettivo: massimizzare generalizzazione e stabilità evitando overfitting e costi di fine-tuning end-to-end

### Classification head (MLP a 2 layer)

Schema effettivo:

```
features (hidden_size=768)
  → Dropout(0.3)
  → Linear(768 → 384)
  → LayerNorm(384)
  → GELU
  → Dropout(0.15)
  → Linear(384 → num_labels)
  → logits
```

#### Perché sostituire la head lineare “di default”?
La head standard di `ViTForImageClassification` è essenzialmente una proiezione lineare \(768 → C\). Nel vostro contesto:
- **task diverso dal pretraining** (il nummero di classi di ImageNet vs numero di classi di attributo analizzato)
- una sola linear può essere troppo rigida (confini lineari) o instabile nella generalizzazione

#### Perché 2 layer + non-linearità?
Due layer con attivazione in mezzo consentono **decisioni non lineari** mantenendo una head semplcie con basso rischio di overfitting.

#### Perché bottleneck 768 → 384?
È una **compressione regolarizzante**:
- riduce la dimensionalità del latente e il rischio di adattarsi al rumore
- mantiene un rapporto capacità/overfitting più adatto a poche classi e pochi esempi

#### Perché LayerNorm?
Stabilizza le attivazioni della head e il training:
- indipendente dalla batch size (preferibile a BatchNorm con batch piccoli)
- coerente con lo stile di normalizzazione dei Transformer

#### Perché GELU?
È la non-linearità tipica dei Transformer (ViT/BERT):
- comportamento smooth, gradienti più informativi rispetto a ReLU in questo contesto
- coerenza con il backbone

#### Perché Dropout 0.3 e 0.15?
Regolarizzazione forte sul primo mapping e più leggera vicino ai logits:
- 0.3 limita co-adattamento e overfitting quando si addestra solo la head
- 0.15 riduce rumore sui logits mantenendo capacità discriminativa

---

## Output del modello: logits vs probabilità

L’output di `ViTForImageClassification` è:
- **logits**: valori reali non normalizzati (shape: `batch x num_labels`)
- **probabilità**: si ottengono applicando `softmax(logits)`

In training si usa `CrossEntropyLoss` per contrastare bias verso le classi maggioritarie.

---

## Strategia di training e regolarizzazione (scelte “stabili”)

### Bilanciamento classi
- **Class weights** nella Cross Entropy: aumentano l’importanza delle classi minoritarie durante l’ottimizzazione.

### Label smoothing
Riduce overconfidence e migliora generalizzazione/calibrazione:
- tipicamente `label_smoothing = 0.1`

### Weight decay (AdamW)
Regolarizzazione L2 “decoupled” sugli aggiornamenti dei pesi della head:
- tipicamente `weight_decay = 1e-4`

### Gradient clipping
Stabilizza l’ottimizzazione limitando la norma dei gradienti:
- tipicamente `max_grad_norm = 1.0`

### Scheduler + warmup + early stopping
Per stabilità e convergenza:
- warmup iniziale (100 step)
- riduzione LR su plateau (patience e factor)
- early stopping (patience 15) per evitare overfitting

---

## Riassunto onfigurazione 

Questi valori sono quelli usati nella pipeline PyTorch/Trainer attuale:

| Componente | Scelta |
|---|---|
| backbone | ViT congelato (solo head addestrata) |
| head | Dropout 0.3 → Linear 768→384 → LayerNorm → GELU → Dropout 0.15 → Linear 384→C |
| optimizer | AdamW |
| learning rate | 5e-5 |
| weight decay | 1e-4 |
| loss | CrossEntropy con class weights + label smoothing 0.1 |
| gradient clipping | 1.0 |
| scheduler | ReduceLROnPlateau (factor=0.5, patience=7) + warmup 100 step |
| early stopping | patience 15 |
| metrica selezione best model | F1 macro |

