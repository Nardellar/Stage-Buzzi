from common import dataset_organization
import model_script

def main():

    train_ds, val_ds = dataset_organization.get_dataset("rampa")

    mapping_dict = {
        10: 0,
        25: 1,
        40: 2
    }

    train_ds = train_ds.map(dataset_organization.remap_labels(mapping_dict))
    val_ds = val_ds.map(dataset_organization.remap_labels(mapping_dict))


    model = model_script.create_classification_model(
        input_shape=(112, 112, 3),
        num_classes=5,  # per esempio
        base_trainable=False
    )

    history = model_script.train_model(
        model=model,
        train_dataset=train_ds,
        validation_dataset=val_ds,
        epochs=30
    )

    # Per esempio, salva il modello finale
    model.save("Models/Rampa/modello_rampa.keras")


if __name__ == "__main__":
    main()