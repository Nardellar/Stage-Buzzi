import dataset_organization
import model_script


def main():

    train_ds, val_ds = dataset_organization.get_dataset("tempo totale")

    mapping_dict = {
        15: 0,
        30: 1,
        45: 2
    }

    train_ds = train_ds.map(dataset_organization.remap_labels(mapping_dict))
    val_ds = val_ds.map(dataset_organization.remap_labels(mapping_dict))

    model = model_script.create_classification_model(
        input_shape=(112, 112, 3),
        num_classes=3,  # per esempio
        base_trainable=False
    )

    history = model_script.train_model(
        model=model,
        train_dataset=train_ds,
        validation_dataset=val_ds,
        epochs=30
    )

    # Per esempio, salva il modello finale
    model.save("Models/Tempo/modello_tempo.keras")


if __name__ == "__main__":
    main()