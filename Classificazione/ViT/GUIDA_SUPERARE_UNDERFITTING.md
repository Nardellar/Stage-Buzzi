# Guida per Superare l'Underfitting nel Modello "Migliorato"

## 🎯 Obiettivo
Portare il modello "migliorato" da **53% a >80%** di accuracy

---

## 📊 Parametri Attuali vs Raccomandati

### 1. **EPOCHE** (CRITICO!)

| Configurazione | Epoche Attuali | Epoche Raccomandate | Motivo |
|----------------|----------------|---------------------|--------|
| **Training Normale** | 25 | **50-100** | Convergenza completa |
| **Cross-Validation** | 10 | **25** | Valutazione accurata |

**NUMERO IDEALE: 75-100 EPOCHE**

#### Perché più epoche?
- Il modello con regularizzazione impara più lentamente
- 25 epoche non sono sufficienti per convergere
- Il modello originale raggiunge 81% dopo 10 epoche, ma senza regularizzazione
- Con regularizzazione serve 3-4x più tempo

---

### 2. **LEARNING RATE** (MOLTO IMPORTANTE!)

```python
# ❌ ATTUALE (troppo basso)
learning_rate = 1e-5

# ✅ RACCOMANDATO
learning_rate = 5e-5  # Come il modello originale
# O anche
learning_rate = 1e-4  # Se vuoi convergenza più veloce
```

**Impatto**: +20-30% di accuracy

---

### 3. **DROPOUT RATE** (IMPORTANTE!)

```python
# ❌ ATTUALE (troppo aggressivo)
dropout_rate = 0.3

# ✅ RACCOMANDATO
dropout_rate = 0.1  # Molto più leggero
# O anche
dropout_rate = 0.15  # Moderato
```

**Impatto**: +10-15% di accuracy

---

### 4. **L2 REGULARIZATION** (IMPORTANTE!)

```python
# ❌ ATTUALE (troppo forte)
kernel_regularizer = tf.keras.regularizers.l2(0.01)

# ✅ RACCOMANDATO
kernel_regularizer = tf.keras.regularizers.l2(0.001)  # 10x più leggero
# O anche
kernel_regularizer = None  # Rimuovi completamente
```

**Impatto**: +5-10% di accuracy

---

### 5. **WEIGHT DECAY** (MODERATO)

```python
# ❌ ATTUALE
weight_decay = 1e-4

# ✅ RACCOMANDATO
weight_decay = 1e-5  # 10x più leggero
# O anche
weight_decay = 0  # Rimuovi se hai già L2
```

**Impatto**: +5% di accuracy

---

### 6. **OVERSAMPLING** (IMPORTANTE!)

```python
# ❌ ATTUALE
# Solo class weights, dataset sbilanciato

# ✅ RACCOMANDATO
# Riabilita oversampling come nel modello originale
# Oppure usa SMOTE o altre tecniche di bilanciamento
```

**Impatto**: +10-15% di accuracy

---

### 7. **AUGMENTATION** (UTILE)

```python
# ❌ ATTUALE
# Disabilitata per problemi tecnici

# ✅ RACCOMANDATO
# Riabilita augmentation moderata:
# - Random flip (orizzontale/verticale)
# - Random brightness (±10%)
# - Random contrast (±10%)
```

**Impatto**: +5-10% di accuracy

---

## 🔧 Configurazione Ottimale Raccomandata

```python
# PARAMETRI OTTIMALI PER SUPERARE UNDERFITTING

# 1. EPOCHE
EPOCHS = 75  # Minimo 50, ideale 75-100

# 2. LEARNING RATE
learning_rate = 5e-5  # Come originale

# 3. OPTIMIZER
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=1e-5,  # Molto leggero
    beta_1=0.9,
    beta_2=0.999
)

# 4. REGULARIZZAZIONE LEGGERA
dropout_rate = 0.1  # Molto più leggero
l2_regularization = 0.001  # 10x più leggero

# 5. CALLBACKS
callbacks = [
    EarlyStopping(patience=15),  # Più paziente
    ReduceLROnPlateau(patience=7, factor=0.5)
]

# 6. OVERSAMPLING
# Riabilita oversampling del modello originale

# 7. AUGMENTATION
# Riabilita augmentation moderata
```

---

## 📊 Performance Attese con Parametri Ottimali

### Scenario Conservativo
- **Accuracy**: 75-80%
- **Stabilità**: ±3-5%
- **Epoche necessarie**: 50-75

### Scenario Realistico
- **Accuracy**: 80-85%
- **Stabilità**: ±2-3%
- **Epoche necessarie**: 75-100

### Scenario Ottimistico
- **Accuracy**: 85-90%
- **Stabilità**: ±2%
- **Epoche necessarie**: 100+

---

## 🚀 Piano di Implementazione

### Step 1: Fix Immediati (Impatto Alto)
1. ✅ Learning rate: 1e-5 → 5e-5
2. ✅ Epoche: 25 → 75
3. ✅ Dropout: 0.3 → 0.1

**Impatto Atteso**: +25-30% accuracy

### Step 2: Ottimizzazioni (Impatto Medio)
1. ✅ L2 regularization: 0.01 → 0.001
2. ✅ Weight decay: 1e-4 → 1e-5
3. ✅ Riabilita oversampling

**Impatto Atteso**: +10-15% accuracy

### Step 3: Raffinamenti (Impatto Basso)
1. ✅ Riabilita augmentation
2. ✅ Ottimizza callbacks
3. ✅ Fine-tuning iperparametri

**Impatto Atteso**: +5-10% accuracy

---

## ⏱️ Tempo di Training Stimato

Con 75 epoche:
- **Tempo per epoca**: ~6 minuti
- **Tempo totale**: ~450 minuti (7.5 ore)
- **Con early stopping**: Probabilmente si ferma a ~50-60 epoche (5-6 ore)

---

## 🎯 Conclusione

Per superare l'underfitting del modello "migliorato":

**NUMERO IDEALE DI EPOCHE: 75-100**

Ma prima di aumentare le epoche, **FIX CRITICI**:
1. Learning rate: 5e-5
2. Dropout: 0.1
3. L2 reg: 0.001

Con questi fix + 75 epoche, dovresti raggiungere **~80-85% di accuracy**.
