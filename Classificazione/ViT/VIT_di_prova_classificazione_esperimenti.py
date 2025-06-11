from datasets import load_dataset
from transformers import AutoImageProcessor, TFViTForImageClassification
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ViTTemperaturaTrainer:
    def __init__(self, model_name="google/vit-base-patch16-224", num_labels=14):
        self.model_name = model_name
        self.num_labels = num_labels
        self.processor = None
        self.model = None
        self.history = None

    def load_and_prepare_datasets(self, dataset_name="Nardellar/Esperimenti",
                                  batch_size=16, test_size=0.2, val_size=0.1, seed=42):
        """
        Carica e prepara i dataset con split train/val/test
        """
        logger.info(f"Caricamento dataset {dataset_name}")

        # Carica il dataset
        ds = load_dataset(dataset_name, split="train")
        logger.info(f"Dataset caricato: {len(ds)} campioni")

        # Prima divisione: train + val/test
        ds_split = ds.train_test_split(test_size=test_size + val_size, seed=seed)
        train_ds = ds_split["train"]

        # Seconda divisione: val e test
        temp_ds = ds_split["test"]
        val_test_split = temp_ds.train_test_split(
            test_size=test_size / (test_size + val_size), seed=seed
        )
        val_ds = val_test_split["train"]
        test_ds = val_test_split["test"]

        logger.info(f"Split completato - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

        # Inizializza il processor
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)

        # Applica le trasformazioni
        def transform_fn(batch):
            # Data augmentation per il training set
            images = batch["image"]
            outputs = self.processor(images=images, return_tensors="tf")
            return {
                "pixel_values": outputs["pixel_values"],
                "labels": batch["label"],
            }

        # Applica trasformazioni
        train_ds = train_ds.with_transform(transform_fn)
        val_ds = val_ds.with_transform(transform_fn)
        test_ds = test_ds.with_transform(transform_fn)

        # Converti in TensorFlow datasets
        train_tf = train_ds.to_tf_dataset(
            columns=["pixel_values"],
            label_cols=["labels"],
            batch_size=batch_size,
            shuffle=True,
            drop_remainder=True  # Per evitare problemi con batch incompleti
        )

        val_tf = val_ds.to_tf_dataset(
            columns=["pixel_values"],
            label_cols=["labels"],
            batch_size=batch_size,
            shuffle=False
        )

        test_tf = test_ds.to_tf_dataset(
            columns=["pixel_values"],
            label_cols=["labels"],
            batch_size=batch_size,
            shuffle=False
        )

        # Cache e prefetch per ottimizzare le performance
        train_tf = train_tf.cache().prefetch(tf.data.AUTOTUNE)
        val_tf = val_tf.cache().prefetch(tf.data.AUTOTUNE)
        test_tf = test_tf.cache().prefetch(tf.data.AUTOTUNE)

        return train_tf, val_tf, test_tf

    def create_model(self, learning_rate=5e-5):
        """
        Crea e compila il modello ViT
        """
        # Mappa delle etichette (da personalizzare in base al dataset)
        label_names = {
            0: "molto_freddo", 1: "freddo", 2: "fresco", 3: "mite",
            4: "temperato", 5: "caldo", 6: "molto_caldo", 7: "torrido",
            8: "classe_8", 9: "classe_9", 10: "classe_10",
            11: "classe_11", 12: "classe_12", 13: "classe_13"
        }

        self.model = TFViTForImageClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=label_names,
            label2id={v: k for k, v in label_names.items()},
            ignore_mismatched_sizes=True,
        )

        # Optimizer con learning rate scheduling
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # Compila il modello
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy", "sparse_top_k_categorical_accuracy"]
        )

        logger.info("Modello creato e compilato")
        return self.model

    def train(self, train_ds, val_ds, epochs=10, save_dir="./models/vit_temperatura"):
        """
        Addestra il modello con callbacks avanzati
        """
        # Crea directory per i modelli
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Callbacks
        callbacks = [
            # Early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),

            # Riduzione learning rate
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7,
                verbose=1
            ),

            # Salvataggio del miglior modello
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{save_dir}/best_model.h5",
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),

            # TensorBoard logging
            tf.keras.callbacks.TensorBoard(
                log_dir=f"{save_dir}/logs",
                histogram_freq=1,
                write_graph=True
            )
        ]

        logger.info(f"Inizio training per {epochs} epoche")

        # Training
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        # Salva il modello finale
        self.model.save_pretrained(save_dir)
        logger.info(f"Modello salvato in {save_dir}")

        return self.history

    def evaluate(self, test_ds, save_dir="./models/vit_temperatura"):
        """
        Valuta il modello sul test set con metriche dettagliate
        """
        logger.info("Valutazione sul test set")

        # Valutazione standard
        test_loss, test_accuracy, test_top_k = self.model.evaluate(test_ds, verbose=1)

        # Predizioni per metriche dettagliate
        predictions = []
        true_labels = []

        for batch in test_ds:
            logits = self.model(batch[0], training=False).logits
            pred_labels = tf.argmax(logits, axis=-1)
            predictions.extend(pred_labels.numpy())
            true_labels.extend(batch[1].numpy())

        # Classification report
        report = classification_report(
            true_labels, predictions,
            target_names=[self.model.config.id2label[i] for i in range(self.num_labels)],
            output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)

        # Salva risultati
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_top_k_accuracy': test_top_k,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }

        # Salva in JSON
        import json
        with open(f"{save_dir}/evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        # Plot confusion matrix
        self.plot_confusion_matrix(cm, save_dir)

        logger.info(f"Accuracy finale: {test_accuracy:.4f}")
        return results

    def plot_confusion_matrix(self, cm, save_dir):
        """
        Crea e salva la matrice di confusione
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[self.model.config.id2label[i] for i in range(self.num_labels)],
                    yticklabels=[self.model.config.id2label[i] for i in range(self.num_labels)])
        plt.title('Matrice di Confusione')
        plt.ylabel('Etichette Vere')
        plt.xlabel('Predizioni')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_training_history(self, save_dir="./models/vit_temperatura"):
        """
        Visualizza l'andamento del training
        """
        if self.history is None:
            logger.warning("Nessuna history disponibile")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/training_history.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """
    Funzione principale per il training
    """
    # Configurazione
    CONFIG = {
        'dataset_name': "Nardellar/Esperimenti",
        'model_name': "google/vit-base-patch16-224",
        'num_labels': 14,
        'batch_size': 16,
        'epochs': 10,
        'learning_rate': 5e-5,
        'test_size': 0.15,
        'val_size': 0.15,
        'save_dir': "./models/vit_temperatura"
    }

    # Inizializza trainer
    trainer = ViTTemperaturaTrainer(
        model_name=CONFIG['model_name'],
        num_labels=CONFIG['num_labels']
    )

    # Carica dataset
    train_ds, val_ds, test_ds = trainer.load_and_prepare_datasets(
        dataset_name=CONFIG['dataset_name'],
        batch_size=CONFIG['batch_size'],
        test_size=CONFIG['test_size'],
        val_size=CONFIG['val_size']
    )

    # Crea modello
    trainer.create_model(learning_rate=CONFIG['learning_rate'])

    # Training
    trainer.train(
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=CONFIG['epochs'],
        save_dir=CONFIG['save_dir']
    )

    # Valutazione
    trainer.evaluate(test_ds=test_ds, save_dir=CONFIG['save_dir'])

    # Plot risultati
    trainer.plot_training_history(save_dir=CONFIG['save_dir'])

    logger.info("Training completato!")


if __name__ == "__main__":
    main()