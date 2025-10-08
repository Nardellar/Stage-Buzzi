# Risultati Cross-Validation: Confronto Modelli

## 📊 Riepilogo Esecutivo

**Data**: 8 Ottobre 2025  
**Attributo Testato**: Temperatura (3 classi: 1300°C, 1400°C, 1500°C)  
**Dataset**: 1400 immagini totali

---

## 🏆 Modello Originale - VINCITORE

### Risultati Cross-Validation (5 fold, 10 epoche)
```
Accuracy Media:  81.07% ± 2.18%
Loss Media:      0.6201 ± 0.0233

Fold 1: 79.64%
Fold 2: 82.14%
Fold 3: 77.50%
Fold 4: 82.86%
Fold 5: 83.21%
```

### Valutazione
- ✅ Performance ECCELLENTI (>80%)
- ✅ Stabilità ECCELLENTE (std < 3%)
- ✅ NO Overfitting
- ✅ Pronto per Produzione

---

## ❌ Modello "Migliorato" - FALLITO

### Risultati Cross-Validation (5 fold, 10 epoche)
```
Accuracy Media:  53.43% ± 11.44%
Loss Media:      1.0084 ± 0.1315

Fold 1: 48.57%
Fold 2: 33.21% (DISASTROSO!)
Fold 3: 62.86%
Fold 4: 63.57%
Fold 5: 58.93%
```

### Valutazione
- ❌ Performance SCARSE (<60%)
- ❌ Stabilità PESSIMA (std > 11%)
- ❌ UNDERFITTING Grave
- ❌ NON Utilizzabile

---

## 📈 Confronto

| Metrica | Originale | "Migliorato" | Differenza |
|---------|-----------|--------------|------------|
| CV Accuracy | 81.07% | 53.43% | **-27.64%** |
| CV Std Dev | 2.18% | 11.44% | **+9.26%** |
| Stabilità | ECCELLENTE | PESSIMA | ❌ |

---

## 🔍 Cause del Fallimento

1. **Learning Rate Troppo Basso**: 1e-5 vs 5e-5 (5x più lento)
2. **Regularizzazione Eccessiva**: 4 tecniche simultanee
3. **Mancanza di Oversampling**: Solo class weights
4. **Augmentation Disabilitata**: Problemi tecnici

---

## 💡 Raccomandazione

**USARE IL MODELLO ORIGINALE**

È stabile, affidabile, e ha performance eccellenti (81% ± 2%).
