from datasets import load_dataset
from transformers import AutoImageProcessor, TFViTForImageClassification
import tensorflow as tf

def load_temperatura_datasets(batch_size=16, test_size=0.2, seed=42):
    # 1) Carica il dataset “Esperimenti”
    ds = load_dataset("Nardellar/Esperimenti", split="train")

    # 2) Splitta in train/val
    ds = ds.train_test_split(test_size=test_size, seed=seed)
    train_ds, val_ds = ds["train"], ds["test"]

    # 3) Processor ViT
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

    # 4) Transform batch-wise
    def transform_fn(batch):
        outputs = processor(images=batch["image"], return_tensors="tf")
        return {
            "pixel_values": outputs["pixel_values"],  # [B,3,224,224]
            "labels":        batch["label"],          # interi 0–13
        }

    train_ds = train_ds.with_transform(transform_fn)
    val_ds   = val_ds.with_transform(transform_fn)

    # 5) to_tf_dataset: specifica columns + label_cols
    train_tf = train_ds.to_tf_dataset(
        columns=["pixel_values"],
        label_cols=["labels"],
        batch_size=batch_size,
        shuffle=True
    )
    val_tf = val_ds.to_tf_dataset(
        columns=["pixel_values"],
        label_cols=["labels"],
        batch_size=batch_size,
        shuffle=False
    )

    return train_tf, val_tf


if __name__ == "__main__":
    train_ds, val_ds = load_temperatura_datasets(batch_size=16)

    model = TFViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=14,
        id2label={i: str(i) for i in range(14)},
        label2id={str(i): i for i in range(14)},
        ignore_mismatched_sizes=True,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(train_ds, validation_data=val_ds, epochs=3)
    model.save_pretrained("vit_temperatura")
