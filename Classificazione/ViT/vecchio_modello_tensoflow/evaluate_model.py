"""
Script per la valutazione di un modello ViT già addestrato.
- Carica i pesi del modello e gli artefatti salvati dallo script di training.
- Carica il validation/test set salvato.
- Esegue la valutazione finale, calcolando metriche, matrice di confusione
  e mappe di attenzione.
"""
import json
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoImageProcessor

from train_model import ViTForCustomClassificationImproved  # Correzione: Import della classe


# --- FUNZIONI DI VALUTAZIONE E VISUALIZZAZIONE ---


# (in evaluate_model.py)


def load_test_data(batch_size: int):
    try:
        # ✅ CORREZIONE: Carica dataset HuggingFace invece di JSON
        from datasets import Dataset
        test_ds = Dataset.load_from_disk("validation_test_set")
        print(f"✅ Dataset HuggingFace caricato: {len(test_ds)} campioni")
    except (FileNotFoundError, ValueError):
        try:
            # ✅ CORREZIONE: Gestisci anche il formato JSON
            import json
            with open("validation_test_set.json", "r") as f:
                test_data = [json.loads(line) for line in f]
            print(f"✅ Dataset JSON caricato: {len(test_data)} campioni")
            test_ds = Dataset.from_list(test_data)
        except FileNotFoundError:
            print("❌ ERRORE: Nessun dataset di validazione trovato.")
            return None

    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

    def preprocess_images(batch, processor):
        """
        ✅ CORREZIONE: STESSO preprocessing del training per consistenza totale
        """
        # batch["image"] contiene già oggetti PIL dal dataset HuggingFace
        processed = processor(images=batch["image"], return_tensors="tf")
        batch["pixel_values"] = processed["pixel_values"]
        batch["labels"] = tf.convert_to_tensor(batch["attribute"], dtype=tf.int32)
        batch["original_labels"] = tf.convert_to_tensor(batch["original_label_id"], dtype=tf.int32)
        return batch

    test_ds = test_ds.map(lambda batch: preprocess_images(batch, processor), batched=True, batch_size=batch_size)

    # ✅ CORREZIONE CRITICA: Usa lo STESSO formato del training
    test_tf = test_ds.to_tf_dataset(
        columns=["pixel_values", "original_labels"],  # ✅ STESSO del training
        label_cols=["labels"],
        batch_size=batch_size,
        shuffle=False
    )

    # ✅ CORREZIONE CRITICA: Usa la STESSA funzione format_for_model del training
    def format_for_model(features, labels):
        """Formatta i dati per il modello - IDENTICO al training"""
        if isinstance(features, dict):
            return features, labels
        elif isinstance(features, tuple):
            pixel_values, original_labels = features
            return {'pixel_values': pixel_values}, labels
        else:
            return {'pixel_values': features}, labels

    test_tf_mapped = test_tf.map(
        format_for_model,  # ✅ STESSA funzione del training
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return test_tf_mapped


def evaluate_and_report(model, test_dataset, id2label, results_dir, attribute):
    print("\n🔍 Valutazione del modello sul Test Set...")
    y_true, y_pred = [], []
    for batch in test_dataset:
        # ✅ CORREZIONE: Gestisci correttamente il formato del batch
        if isinstance(batch, tuple) and len(batch) == 2:
            images, labels = batch
        else:
            # Se è già un dizionario con features e labels
            images = batch
            labels = batch.get('labels', None)
            if labels is None:
                continue

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

    for batch in test_dataset.take(1):
        # ✅ CORREZIONE: Gestisci correttamente il formato del batch
        if isinstance(batch, tuple) and len(batch) == 2:
            inputs, labels = batch
        else:
            inputs = batch
            labels = batch.get('labels', None)
            if labels is None:
                continue

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
        print("ERRORE: Nessuna cartella 'training_results_*' trovata.")
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

    # ✅ CORREZIONE: Gestisci entrambi i formati (vecchio e nuovo)
    if 'model_path' in artifacts:
        model_path = artifacts['model_path']  # Nuovo formato: modello completo
        print("📁 Caricamento modello completo (.keras)")
    elif 'model_weights_path' in artifacts:
        model_path = artifacts['model_weights_path']  # Vecchio formato: solo pesi
        print("⚠️  ATTENZIONE: Caricamento solo pesi (.weights.h5) - risultati potrebbero essere inconsistenti")
    else:
        print("❌ ERRORE: Nessun percorso del modello trovato negli artifacts")
        return

    num_classes = artifacts['num_classes']

    eval_results_dir = Path(f"evaluation_results_{attribute}")
    eval_results_dir.mkdir(parents=True, exist_ok=True)

    # ✅ CORREZIONE: Carica il modello COMPLETO (architettura + pesi + statistiche BN)
    print(f"\n🔄 Caricamento del modello completo...")

    # ✅ VERIFICA: Controlla che il file del modello esista
    if not Path(model_path).exists():
        print(f"❌ ERRORE: File del modello non trovato: {model_path}")
        return

    print(f"📁 Dimensione file: {Path(model_path).stat().st_size} bytes")

    try:
        if model_path.endswith('.keras'):
            # Correzione: Carica il modello completo con keras.models.load_model
            model = keras.models.load_model(model_path)
            print("Modello completo caricato con successo")
        else:
            # Ricostruisci l'architettura e carica i pesi
            print("Ricostruzione dell'architettura del modello...")
            model = ViTForCustomClassificationImproved(num_labels=num_classes)

            # IMPORTANTE: Congela il ViT come nel training
            model.vit.trainable = False
            print("ViT congelato (trainable=False) - stesso stato del training")

            # Inizializza il modello con input dummy PER COSTRUIRE TUTTI I LAYER
            print("Inizializzazione del modello...")
            dummy_input = {'pixel_values': tf.zeros([1, 3, 224, 224])}
            _ = model(dummy_input, training=False)

            # Carica i pesi addestrabili (solo dropout, batch_norm, classifier)
            print(f"Caricamento pesi addestrabili da: {model_path}")
            if model_path.endswith('.npz'):
                import numpy as np
                weights = np.load(model_path)

                # Ricostruisci le liste di pesi per ogni layer
                dropout_weights = [weights['dropout_0']] if 'dropout_0' in weights else []
                batch_norm_weights = [
                    weights['batch_norm_0'],
                    weights['batch_norm_1'],
                    weights['batch_norm_2'],
                    weights['batch_norm_3']
                ]
                classifier_weights = [
                    weights['classifier_0'],
                    weights['classifier_1']
                ]

                # Carica i pesi
                if len(dropout_weights) > 0:
                    model.dropout.set_weights(dropout_weights)
                model.batch_norm.set_weights(batch_norm_weights)
                model.classifier.set_weights(classifier_weights)

                print("Pesi addestrabili caricati con successo!")
                print("Il ViT e' stato ricaricato fresco da HuggingFace (sempre uguale perche congelato)")
            elif model_path.endswith('.pkl'):
                # Formato pickle (vecchio)
                import pickle
                with open(model_path, 'rb') as f:
                    weights_dict = pickle.load(f)
                model.dropout.set_weights(weights_dict['dropout'])
                model.batch_norm.set_weights(weights_dict['batch_norm'])
                model.classifier.set_weights(weights_dict['classifier'])
                print("Pesi addestrabili caricati (formato pickle)")
            else:
                # Formato vecchio (.weights.h5)
                model.load_weights(model_path)
                print("Pesi caricati (formato .h5 vecchio)")
    except Exception as e:
        print(f"❌ ERRORE nel caricamento del modello: {e}")
        return

    # ✅ VERIFICA CRITICA: Testa il modello con predizioni diverse
    dummy_input = {'pixel_values': tf.zeros([1, 3, 224, 224])}
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

    # ✅ DEBUG: Testa con un batch reale per verificare la consistenza
    print("\n🔍 Test con batch reale per verificare consistenza...")
    test_tf = load_test_data(batch_size=1)
    if test_tf is not None:
        for batch in test_tf.take(1):
            if isinstance(batch, tuple) and len(batch) == 2:
                images, labels = batch
            else:
                images = batch
                labels = batch.get('labels', None)

            real_output = model(images, training=False)
            real_predictions = tf.nn.softmax(real_output['logits'])
            print(f"🔍 Predizioni batch reale: {real_predictions.numpy()[0]}")
            print(f"🔍 Label reale: {labels.numpy()[0] if labels is not None else 'N/A'}")
            break

    test_tf = load_test_data(batch_size=16)
    if test_tf is None: return

    evaluate_and_report(model, test_tf, id2label, eval_results_dir, attribute)
    save_attention_maps_on_test(model, test_tf, id2label, eval_results_dir, attribute)

    print("\n✅ Valutazione completata!")
    print(f"📊 I risultati sono stati salvati nella cartella: '{eval_results_dir}'")


if __name__ == "__main__":
    main_evaluate()
