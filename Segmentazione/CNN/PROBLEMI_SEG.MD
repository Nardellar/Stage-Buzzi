
# 🔍 Analisi Criticità — Segmentazione CNN + Boosting

Dataset: **79 immagini**, 5 classi, sbilanciamento marcato.

---

## 🔴 CRITICITÀ GRAVI

### 1. Evaluation = Validation (data leakage nella selezione del modello)
**File:** `train_gpu_optimized_classifier.py` riga 127–130, `evaluate_gpu_model.py`

Lo **stesso identico split `eval_ids`** viene usato sia come:
- **validation set** durante Optuna (per guidare la selezione del trial migliore)
- **test set** per la valutazione finale del modello

Questo è un **data leakage**: la metrica finale è ottimisticamente biased perché Optuna ha già visto e ottimizzato su quei dati. Con 75 trial, il rischio di overfitting sugli eval è concreto.

> **Fix:** Dividere in 3 split (train / validation / test) oppure usare **cross-validation nested** dentro Optuna e tenere eval come test puro. Con 79 immagini, un 60/20/20 darebbe ~47 train, ~16 val, ~16 test.

---

### 2. Il campionamento pixel è random uniforme, non stratificato per classe
**File:** `gpu_optimized_cnn_classifier.py` righe 216–225

Il downsampling dei pixel (`max_pixels_per_image=20000`) usa `rng.choice` **uniforme**: i pixel campionati rispecchiano la distribuzione originale. Se un'immagine ha il 95% di Alite e il 5% di Pori, il campione da 20k pixel avrà ~1000 pixel di Pori e ~19000 di Alite.

Con dataset piccolo + sbilanciamento marcato, le classi rare vengono ulteriormente sottorappresentate nel training set del booster.

> **Fix:** Sostituire il campionamento uniforme con un **campionamento stratificato per classe**, assicurando che ogni classe abbia un numero minimo di rappresentanti nel campione. Esempio:
> ```python
> # per ogni classe, campiona almeno min_per_class pixel
> indices_per_class = []
> for c in np.unique(label_filtered):
>     idx_c = np.where(label_filtered == c)[0]
>     n_sample = max(min_per_class, len(idx_c) * budget_fraction)
>     indices_per_class.append(rng.choice(idx_c, size=min(n_sample, len(idx_c)), replace=False))
> sample_indices = np.concatenate(indices_per_class)
> ```

---

### 3. L'evaluation finale usa solo `accuracy_score` (incoerente col tuning)
**File:** `evaluate_gpu_model.py` riga 237

Dopo aver corretto il tuning per usare F1 macro, la **valutazione finale** usa ancora solo `accuracy_score`:
```python
accuracy = accuracy_score(ground_truth, predictions)
```
Questo crea un'incoerenza: selezioni il modello con F1 macro ma lo valuti con accuracy. Dovresti riportare **almeno** F1 macro, recall per classe, e IoU per classe nella valutazione finale.

> **Fix:** Aggiungere nel report di valutazione: F1 macro, F1 per classe, IoU (Jaccard) per classe, e balanced accuracy.

---

## 🟠 CRITICITÀ MEDIE

### 4. Augmentation troppo timida per un dataset così piccolo
**File:** `data_module.py` righe 116–137

L'augmentation **raddoppia** il dataset (da 79 a ~158 immagini) con trasformazioni conservative:
- `shift_limit=0.05`, `scale_limit=0.05`, `rotate_limit=15°`, `brightness_limit=0.15`

Con solo 79 immagini e 5 classi sbilanciate, servirebbero trasformazioni più aggressive e un fattore di moltiplicazione maggiore (3x–5x).

> **Fix suggerito:** 
> - Generare **3-5 copie augmentate** per immagine invece di 1
> - Aggiungere trasformazioni più aggressive: `ElasticTransform`, `GridDistortion`, `GaussNoise`, `CLAHE`
> - Eventualmente fare **oversampling guidato per classe**: più augmentation per le immagini che contengono più pixel delle classi rare

---

### 5. I file di split sono vuoti → ogni run rigenera (potenziale inconsistenza)
**File:** `splits/train.txt` e `splits/eval.txt`

Entrambi i file sono **vuoti**. Questo significa che `prepare_dataset_splits` rigenerava gli split ad ogni esecuzione. Dato il seed fisso (42) l'output è deterministico, ma se qualcuno cambiasse il seed o aggiungesse immagini, train e eval non sarebbero più coerenti tra training ed evaluation.

> **Fix:** Eseguire una volta lo script e verificare che i file vengano popolati correttamente, oppure fare un commit dei file di split nel repo.

---

### 6. `compute_class_weights_dict` non gestisce classi assenti
**File:** `data_module.py` righe 153–170

Se nel training set manca completamente una classe (plausibile con 79 immagini e sbilanciamento), `compute_class_weight` non la includerà nel dizionario. Successivamente, `calculate_pixels_weights` in `tuning.py` farà un `KeyError` perché cercherà il peso di quella classe.

> **Fix:** Dopo il calcolo, assicurarsi che tutte le classi da 0 a `num_classes-1` abbiano un peso, assegnando un peso di default (es. il massimo dei pesi trovati) alle classi mancanti.

---

### 7. Il re-training finale post-Optuna perde l'early stopping
**File:** `tuning.py` righe 230–260 (LightGBM) e 270–310 (XGBoost)

Dopo aver trovato il miglior trial, il modello finale viene **ri-addestrato da zero** su tutto il train set con `n_estimators = best_iteration + 1` **senza early stopping e senza validation set**. Questo significa:
- Il numero ottimale di iterazioni è stato trovato su un certo split, ma il re-training cambia la distribuzione dei dati
- Senza early stopping, il modello potrebbe overfittare se `best_iteration` non è più ottimale per il dataset completo

> **Fix:** O si usa direttamente il modello del trial migliore (già addestrato), oppure si ri-addestra con early stopping su un validation set.

---

## 🟡 CRITICITÀ MINORI

### 8. Stampe incoerenti dopo il cambio di metrica
**File:** `gpu_optimized_cnn_classifier.py` riga 350
```python
print(f"Migliore accuracy (val): {best_metric:.4f}")
```
E in `train_gpu_optimized_classifier.py` riga 172:
```python
print(f"Accuracy best trial     : {best_accuracy:.4f}")
```
Ora che la metrica è F1 macro, queste stampe sono fuorvianti.

### 9. DenseCRF applicato solo in evaluation, non durante il tuning
Il CRF migliora le predizioni ma **non viene usato durante la selezione del modello**. Il trial migliore secondo F1 macro *senza CRF* potrebbe non essere il migliore *con CRF*. Questo non è necessariamente un errore, ma è una discrepanza da tenere presente.

### 10. Nessuna Cross-Validation
Con solo ~63 immagini di training (80% di 79), una singola partizione train/val è fragile. Una **k-fold stratificata** (anche k=3 o k=5) darebbe stime molto più robuste, specialmente con classi rare.

---

## 📋 PRIORITÀ DEGLI INTERVENTI

| # | Criticità | Impatto | Sforzo |
|---|-----------|---------|--------|
| 1 | Eval = Validation (leakage) | 🔴 Alto | Medio |
| 2 | Campionamento pixel non stratificato | 🔴 Alto | Basso |
| 3 | Eval finale solo accuracy | 🔴 Alto | Basso |
| 6 | Classi assenti → crash | 🟠 Medio | Basso |
| 4 | Augmentation troppo debole | 🟠 Medio | Basso |
| 7 | Re-training senza early stopping | 🟠 Medio | Medio |
| 8 | Stampe incoerenti | 🟡 Basso | Banale |
| 5 | Split vuoti | 🟡 Basso | Banale |

Vuoi che implementi una o più di queste correzioni?


Nel tuo codice attuale, pero': c'e' una sfumatura importante.
Da come hai implementato la pipeline (_build_feature_extractor + solo predict, senza fit end-to-end), il decoder viene creato con pesi iniziali random e non viene realmente addestrato insieme alla CNN. In questo scenario, quella prima conv puo' risultare poco utile o addirittura rumorosa, perche' trasforma feature pretrained con filtri non ottimizzati.

Il decoder non addestrato è un problema?
Dipende da cosa gli chiedi di fare.
Nel tuo caso attuale: non è un problema grave, ma è uno spreco
La tua pipeline usa il decoder solo per fare upsampling delle feature map del backbone, poi passa i vettori pixel al booster (LightGBM/XGBoost). Il booster è quello che realmente "impara" a classificare.
Quindi il decoder con pesi random:
non rovina le feature del backbone (le mescola un po', ma il segnale passa comunque)
non aggiunge informazione utile (i filtri random non estraggono pattern significativi)
aggiunge rumore alle feature, rendendo il lavoro del booster più difficil
In pratica stai facendo un upsampling bilineare mascherato da convoluzione, ma con rumore in più.
Le tue tre opzioni
Opzione 1: Rimuovere le conv e tenere solo upsampling deterministico
Consigliata per la tua pipeline attuale.
Sostituisci le conv non addestrate con semplice UpSampling2D + Resizing. Nessun peso da addestrare, nessun rumore aggiunto, risultato più pulito e riproducibile.
Vantaggi:
zero parametri learnable nel decoder
feature del backbone passano "pure" al booster
più semplice da spiegare in tesi
