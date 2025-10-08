"""
Configurazione OTTIMALE per il modello ViT
Basata sui risultati della cross-validation e analisi dell'underfitting
"""

# ============================================================================
# CONFIGURAZIONE OTTIMALE PER SUPERARE UNDERFITTING
# ============================================================================

# 🎯 EPOCHE (CRITICO!)
EPOCHS_TRAINING = 75  # Minimo 50, ideale 75-100
EPOCHS_CV = 25        # Per cross-validation

# 📊 LEARNING RATE (MOLTO IMPORTANTE!)
LEARNING_RATE = 5e-5  # Come il modello originale (NON 1e-5!)

# 🔧 OPTIMIZER
OPTIMIZER_CONFIG = {
    'learning_rate': 5e-5,
    'weight_decay': 1e-5,  # Molto leggero (NON 1e-4!)
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-8
}

# 🛡️ REGULARIZZAZIONE (LEGGERA!)
DROPOUT_RATE = 0.1  # Molto più leggero (NON 0.3!)
L2_REGULARIZATION = 0.001  # 10x più leggero (NON 0.01!)

# 📦 BATCH SIZE
BATCH_SIZE = 16  # OK, mantieni

# ⚖️ BILANCIAMENTO
USE_OVERSAMPLING = True  # Riabilita oversampling
USE_CLASS_WEIGHTS = True  # Usa entrambi per sicurezza

# 🎨 AUGMENTATION
USE_AUGMENTATION = True  # Riabilita se possibile
AUGMENTATION_CONFIG = {
    'horizontal_flip': True,
    'vertical_flip': True,
    'brightness': 0.1,  # Moderato
    'contrast': 0.1,    # Moderato
    'rotation': 0,      # Disabilita per evitare problemi
}

# 🔔 CALLBACKS
EARLY_STOPPING_PATIENCE = 15  # Più paziente (era 10)
REDUCE_LR_PATIENCE = 7        # Più paziente (era 5)
REDUCE_LR_FACTOR = 0.5        # OK

# ============================================================================
# CONFRONTO: PARAMETRI ORIGINALE vs MIGLIORATO vs OTTIMALE
# ============================================================================

COMPARISON = """
┌─────────────────────┬──────────────┬──────────────┬──────────────┐
│ Parametro           │ Originale    │ "Migliorato" │ OTTIMALE     │
├─────────────────────┼──────────────┼──────────────┼──────────────┤
│ Epoche              │ 25           │ 25           │ 75-100       │
│ Learning Rate       │ 5e-5         │ 1e-5 ❌      │ 5e-5 ✅      │
│ Dropout             │ 0 (nessuno)  │ 0.3 ❌       │ 0.1 ✅       │
│ L2 Regularization   │ 0 (nessuno)  │ 0.01 ❌      │ 0.001 ✅     │
│ Weight Decay        │ 0 (nessuno)  │ 1e-4 ❌      │ 1e-5 ✅      │
│ Oversampling        │ Sì ✅        │ No ❌        │ Sì ✅        │
│ Augmentation        │ No           │ No ❌        │ Sì ✅        │
│                     │              │              │              │
│ CV Accuracy         │ 81.07% ✅    │ 53.43% ❌    │ ~80-85% 🎯   │
│ CV Std Dev          │ 2.18% ✅     │ 11.44% ❌    │ ~3-5% 🎯     │
└─────────────────────┴──────────────┴──────────────┴──────────────┘
"""

# ============================================================================
# IMPATTO STIMATO DELLE MODIFICHE
# ============================================================================

IMPACT_ANALYSIS = """
Modifica                        Impatto Atteso    Priorità
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Learning Rate: 1e-5 → 5e-5      +20-30%          🔴 CRITICO
Epoche: 25 → 75                 +15-25%          🔴 CRITICO
Dropout: 0.3 → 0.1              +10-15%          🟠 ALTO
Oversampling: No → Sì           +10-15%          🟠 ALTO
L2 Reg: 0.01 → 0.001            +5-10%           🟡 MEDIO
Weight Decay: 1e-4 → 1e-5       +5%              🟡 MEDIO
Augmentation: No → Sì           +5-10%           🟢 BASSO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTALE STIMATO                  +70-110%         
Performance Attesa              80-85%           
"""

# ============================================================================
# PIANO DI IMPLEMENTAZIONE GRADUALE
# ============================================================================

IMPLEMENTATION_PLAN = """
FASE 1: FIX CRITICI (Implementa SUBITO)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Learning rate: 1e-5 → 5e-5
2. Epoche: 25 → 75
3. Dropout: 0.3 → 0.1

Performance Attesa: 70-75%
Tempo: ~7-8 ore

FASE 2: OTTIMIZZAZIONI (Se Fase 1 funziona)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. L2 reg: 0.01 → 0.001
5. Weight decay: 1e-4 → 1e-5
6. Riabilita oversampling

Performance Attesa: 75-80%
Tempo: +2 ore

FASE 3: RAFFINAMENTI (Opzionale)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. Riabilita augmentation
8. Fine-tuning iperparametri
9. Test con più epoche (100+)

Performance Attesa: 80-85%
Tempo: +3-4 ore
"""

# ============================================================================
# CODICE PER APPLICARE I FIX
# ============================================================================

def get_optimal_config():
    """Restituisce la configurazione ottimale"""
    return {
        'epochs': 75,
        'learning_rate': 5e-5,
        'dropout_rate': 0.1,
        'l2_regularization': 0.001,
        'weight_decay': 1e-5,
        'batch_size': 16,
        'early_stopping_patience': 15,
        'reduce_lr_patience': 7,
        'use_oversampling': True,
        'use_augmentation': True
    }

# ============================================================================
# STIMA TEMPO DI TRAINING
# ============================================================================

TIME_ESTIMATES = """
Con i parametri ottimali:

Training Normale (75 epoche):
  - Tempo per epoca: ~6 minuti
  - Tempo totale: ~450 minuti (7.5 ore)
  - Con early stopping: ~300-360 minuti (5-6 ore)

Cross-Validation (5 fold × 25 epoche):
  - Tempo per fold: ~150 minuti (2.5 ore)
  - Tempo totale: ~750 minuti (12.5 ore)
  - Con early stopping: ~500-600 minuti (8-10 ore)

RACCOMANDAZIONE:
  1. Testa prima con training normale (75 epoche)
  2. Se funziona, esegui CV per validazione
  3. Esegui durante la notte per risparmiare tempo
"""

if __name__ == "__main__":
    print("=" * 70)
    print("CONFIGURAZIONE OTTIMALE PER MODELLO ViT")
    print("=" * 70)
    
    config = get_optimal_config()
    
    print("\n📋 PARAMETRI OTTIMALI:")
    for key, value in config.items():
        print(f"  {key:.<30} {value}")
    
    print("\n" + COMPARISON)
    print(IMPACT_ANALYSIS)
    print(IMPLEMENTATION_PLAN)
    print(TIME_ESTIMATES)
    
    print("\n" + "=" * 70)
    print("🎯 NUMERO IDEALE DI EPOCHE: 75-100")
    print("=" * 70)
    print("\n💡 Per applicare questi parametri, modifica:")
    print("   Classificazione/ViT/vit_from_hf_attribute_improved.py")
    print("\n   Cerca e modifica le seguenti righe:")
    print("   - Riga ~360: learning_rate = 5e-5")
    print("   - Riga ~30: dropout_rate = 0.1")
    print("   - Riga ~42: l2(0.001)")
    print("   - Riga ~551: epochs = 75")
