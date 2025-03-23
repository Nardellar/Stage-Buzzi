# 📊 Esperimenti Immagini - Analisi con TensorFlow

Questo progetto analizza un dataset di immagini associato a un file CSV, esegue il mapping di attributi e mostra un batch di immagini con TensorFlow.

---

## 🧰 Requisiti

- Python 3.7+
- pip

---

## ⚙️ Setup ambiente

### 1. Clona il repository

```bash
git clone https://github.com/tuo-username/tuo-repo.git
cd tuo-repo
```

### 2. Crea e attiva l'ambiente virtuale

```bash
python3 -m venv .venv
source .venv/bin/activate         # Su Windows: .venv\Scripts\activate
```

### 3. Installa le dipendenze

```bash
pip install -r requirements.txt
```

---

## ▶️ Esegui lo script

Assicurati di avere il file `esperimenti.csv` nella stessa cartella dello script, poi:

```bash
python main.py
```

Il codice scaricherà automaticamente il dataset da Google Drive (se non già presente), chiederà quale attributo analizzare e mostrerà un batch di immagini.

---

## ✅ (Opzionale) Setup pre-commit hook

Questo progetto include un file `.pre-commit-config.yaml` per garantire formattazione del codice con [black](https://github.com/psf/black) e [isort](https://github.com/pycqa/isort).

### Per attivare i hook pre-commit:

1. Installa `pre-commit`:

```bash
pip install pre-commit
```

2. Installa i hook nel repository:

```bash
pre-commit install
```

3. Ora ogni volta che farai un `git commit`, i file Python verranno automaticamente formattati.

⚠️ **Nota:** i pre-commit hook sono opzionali. Il codice funziona anche se non vengono installati.

---

## 📦 Contenuti del progetto

| File / Cartella     | Descrizione                                  |
|---------------------|----------------------------------------------|
| `main.py`           | Script principale per analizzare le immagini |
| `esperimenti.csv`   | File CSV con i dati associati alle immagini  |
| `.pre-commit-config.yaml` | Configurazione per i pre-commit hook        |
| `requirements.txt`  | Dipendenze Python                            |

---

## 🧹 Suggerimento extra

Per semplificare la configurazione su nuovi PC, puoi usare lo script:

```bash
./setup_dev.sh
```

Esso crea l’ambiente virtuale, installa le dipendenze e configura i pre-commit hook.

---

## 👤 Autore

- [@tuo-username](https://github.com/tuo-username)
