# Stage-Buzzi
## Documentazione del codice

### Introduzione
Il seguente script in Python utilizza TensorFlow e Pandas per caricare un dataset di immagini da una directory e associarlo a un file CSV contenente attributi aggiuntivi. Il codice permette di visualizzare immagini in base a un attributo scelto dall'utente.
per avere tutte le librerie necessarie, eseguire il seguente comando:
```bash pip install -r requirements.txt
e successivamente eseguire il comando:
```bash python main.py

### Librerie Utilizzate

- sys: Per terminare il programma in caso di errore.

- numpy: Per operazioni numeriche.

- tensorflow e keras: Per gestire dataset di immagini.

- pandas: Per leggere e manipolare dati in formato CSV.

- matplotlib.pyplot: Per visualizzare immagini.

### Inizializzazione dei Percorsi

- base_path: Percorso della cartella contenente le immagini.

- df: DataFrame contenente i metadati delle immagini.

### Caricamento del Dataset di Immagini

- Il dataset viene creato partendo dalla directory base.

- Le etichette vengono inferite dai nomi delle cartelle.

- Le immagini vengono ridimensionate a (108, 192).

- Il batch size è impostato a 32.

- Il dataset viene stampato per mostrare le classi trovate.

### Funzione map_labels_to_attribute

- Converte l'attributo in minuscolo e rimuove spazi.

- Controlla se l'attributo esiste nel DataFrame, altrimenti mostra un errore.

- Crea una mappa tra ID esperimenti e valori dell'attributo.

- Scorre il dataset di immagini e associa ogni immagine al valore dell'attributo.

- Restituisce un dataset con immagini e valori dell'attributo.

### Funzione show_images

- Mostra un massimo di 32 immagini dal dataset.

- Ridimensiona la figura e organizza le immagini in una griglia.

### Interazione con l'utente

- L'utente inserisce un attributo.

- Se l'input è vuoto o contiene solo spazi, il programma termina.

- Se l'attributo non esiste nel DataFrame, il programma termina.

### Creazione del dataset filtrato

- Filtra il dataset in base all'attributo scelto.

- Stampa il numero di immagini corrispondenti.

### Visualizzazione delle immagini

- Se il dataset filtrato contiene immagini, vengono mostrate.

- Se nessuna immagine soddisfa i criteri, viene mostrato un messaggio di errore.

