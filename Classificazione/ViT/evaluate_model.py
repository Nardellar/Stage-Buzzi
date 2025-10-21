"""
Script per la valutazione di un modello ViT già addestrato.
- Carica i pesi del modello e gli artefatti salvati dallo script di training.
- Carica il validation/test set salvato.
- Esegue la valutazione finale, calcolando metriche, matrice di confusione
  e mappe di attenzione.
"""
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from transformers import AutoImageProcessor
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


# È necessario importare la classe del modello per poter ricreare l'architettura
# prima di caricare i pesi.
from train_model import ViTForCustomClassificationImproved




# --- FUNZIONI DI VALUTAZIONE E VISUALIZZAZIONE ---


# (in evaluate_model.py)


def load_test_data(batch_size: int):
    try:
        test_ds = load_dataset('json', data_files='validation_test_set.json', split='train')
    except FileNotFoundError:
        print("❌ ERRORE: File 'validation_test_set.json' non trovato.")
        return None


    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")


    def transform(batch):

        # ✅ CORREZIONE CRITICA: Il dataset salvato contiene percorsi, non immagini caricate
        # Dobbiamo caricare le immagini dai percorsi
        images_loaded = [Image.open(img_data['path']) for img_data in batch["image"]]
        processed = processor(images=images_loaded, return_tensors="tf")
        batch["pixel_values"] = processed["pixel_values"]
        batch["labels"] = tf.convert_to_tensor(batch["attribute"], dtype=tf.int32)

        # ✅ CORREZIONE: Gestisci il caso in cui original_label_id non esiste
        if "original_label_id" in batch:
            batch["original_labels"] = tf.convert_to_tensor(batch["original_label_id"], dtype=tf.int32)
        else:
            # Se non esiste, usa le stesse etichette di attribute
            batch["original_labels"] = tf.convert_to_tensor(batch["attribute"], dtype=tf.int32)

        return batch


    test_ds = test_ds.map(transform, batched=True, batch_size=batch_size)


    # --- INIZIO CORREZIONE ---


    # 1. Crea il dataset TF (questo crea la tupla (features, labels))
    test_tf = test_ds.to_tf_dataset(
        columns=["pixel_values"],  # ✅ CORREZIONE: Solo pixel_values per semplicità
        label_cols=["labels"],
        batch_size=batch_size,
        shuffle=False
    )


    # 2. Mappa la tupla al formato dizionario che il modello si aspetta
    #    (pixel_values, labels) -> (dict, tensor)
    test_tf_mapped = test_tf.map(
        lambda pixel_values, labels: ({'pixel_values': pixel_values}, labels),  # ✅ CORREZIONE: Gestisci correttamente le features
        num_parallel_calls=tf.data.AUTOTUNE
    )


    # 3. Ritorna SOLO il dataset mappato
    return test_tf_mapped
    # --- FINE CORREZIONE ---




def evaluate_and_report(model, test_dataset, id2label, results_dir, attribute):
    print("\n🔍 Valutazione del modello sul Test Set...")
    y_true, y_pred = [], []
    for images, labels in test_dataset:
        predictions = model(images, training=False)
        y_true.extend(labels.numpy())
        y_pred.extend(tf.argmax(predictions['logits'], axis=1).numpy())


    class_names = [id2label[str(i)] for i in range(len(id2label))]
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_path = results_dir / f"classification_report_{attribute}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"📊 Accuracy sul Test Set: {report['accuracy']:.4f}")
    print(f"📄 Report di classificazione salvato in: {report_path}")


    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Matrice di Confusione (Test Set) - {attribute}')
    plt.xlabel('Predizioni');
    plt.ylabel('Valori Reali')
    cm_path = results_dir / f"confusion_matrix_{attribute}.png"
    plt.savefig(cm_path);
    plt.close()
    print(f"🖼️ Matrice di confusione salvata in: {cm_path}")




def save_attention_maps_on_test(model, test_dataset, id2label, results_dir, attribute, num_images=8):
    print("\n🎨 Generazione Mappe di Attenzione sul Test Set...")
    # ... (Il codice qui è funzionalmente identico alla versione precedente)
    for inputs, labels in test_dataset.take(1):
        outputs = model(inputs, training=False, output_attentions=True)
        attentions = outputs["attentions"][-1]
        avg_attentions = tf.reduce_mean(attentions, axis=1)
        cls_token_attention = avg_attentions[:, 0, 1:]
        num_patches_side = int(np.sqrt(cls_token_attention.shape[-1]))
        attention_maps = tf.reshape(cls_token_attention, (-1, num_patches_side, num_patches_side))
        predictions = tf.argmax(outputs["logits"], axis=-1)


        plt.figure(figsize=(20, 10))
        for i in range(min(num_images, len(inputs['pixel_values']))):
            img = inputs['pixel_values'][i].numpy().transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())
            heatmap = tf.image.resize(tf.expand_dims(attention_maps[i], -1), [img.shape[0], img.shape[1]])


            plt.subplot(2, num_images // 2, i + 1)
            plt.imshow(img);
            plt.imshow(heatmap, cmap='jet', alpha=0.5)
            true_label = id2label[str(labels[i].numpy())]
            pred_label = id2label[str(predictions[i].numpy())]
            plt.title(f"Vero: {true_label}\nPredetto: {pred_label}",
                      color=("green" if true_label == pred_label else "red"))
            plt.axis("off")
    map_path = results_dir / f"attention_maps_{attribute}.png"
    plt.savefig(map_path);
    plt.close()
    print(f"🖼️ Mappe di attenzione salvate in: {map_path}")




# --- FUNZIONE PRINCIPALE DI VALUTAZIONE ---
def main_evaluate():
    print("🚀 SCRIPT DI VALUTAZIONE MODELLO 🚀")


    training_dirs = [d for d in Path(".").iterdir() if d.is_dir() and d.name.startswith("training_results_")]
    if not training_dirs:
        print("❌ ERRORE: Nessuna cartella 'training_results_*' trovata.")
        return


    print("Seleziona la sessione di training da valutare:")
    for i, d in enumerate(training_dirs):
        print(f"  [{i}] - {d.name}")


    choice = int(input("Inserisci il numero: ").strip())
    training_dir = training_dirs[choice]
    artifacts_path = training_dir / "artifacts.json"


    with open(artifacts_path) as f:
        artifacts = json.load(f)


    attribute = artifacts['attribute']
    id2label = artifacts['id2label']
    model_weights_path = artifacts['model_weights_path']
    num_classes = artifacts['num_classes']


    eval_results_dir = Path(f"evaluation_results_{attribute}")
    eval_results_dir.mkdir(parents=True, exist_ok=True)


    print(f"\n🔄 Ricostruzione dell'architettura del modello...")
    model = ViTForCustomClassificationImproved(num_labels=num_classes)

    # ✅ CORREZIONE CRITICA: Forza la costruzione completa dell'architettura
    print("🔄 Costruzione dell'architettura con input dummy...")
    dummy_input = {'pixel_values': tf.zeros([1, 3, 224, 224])}

    # Forza la costruzione di TUTTI i layer
    _ = model(dummy_input, training=False)

    # Verifica che l'architettura sia costruita
    print(f"✅ Architettura costruita. Classifier units: {model.classifier.units}")

    # ✅ VERIFICA: Controlla che il file dei pesi esista
    if not Path(model_weights_path).exists():
        print(f"❌ ERRORE: File dei pesi non trovato: {model_weights_path}")
        return

    print(f"🔄 Caricamento dei pesi da: {model_weights_path}")
    print(f"📁 Dimensione file: {Path(model_weights_path).stat().st_size} bytes")

    try:
        model.load_weights(model_weights_path)
        print("✅ Pesi caricati con successo")
    except Exception as e:
        print(f"❌ ERRORE nel caricamento dei pesi: {e}")
        return

    # ✅ VERIFICA CRITICA: Testa il modello con predizioni diverse
    test_output = model(dummy_input, training=False)
    logits = test_output['logits']
    predictions = tf.nn.softmax(logits)

    print(f"✅ Modello testato. Output shape: {logits.shape}")
    print(f"🔍 Predizioni di test: {predictions.numpy()[0]}")

    # Verifica che le predizioni non siano tutte uguali (segno di modello non addestrato)
    if tf.math.reduce_std(predictions) < 0.01:
        print("⚠️ ATTENZIONE: Le predizioni sono troppo uniformi, il modello potrebbe non essere addestrato!")
    else:
        print("✅ Le predizioni mostrano varianza, il modello sembra addestrato")


    test_tf = load_test_data(batch_size=16)
    if test_tf is None: return


    evaluate_and_report(model, test_tf, id2label, eval_results_dir, attribute)
    save_attention_maps_on_test(model, test_tf, id2label, eval_results_dir, attribute)


    print("\n✅ Valutazione completata!")
    print(f"📊 I risultati sono stati salvati nella cartella: '{eval_results_dir}'")




if __name__ == "__main__":
    main_evaluate()


