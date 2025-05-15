import dataset_organization
import model_script


def main():

    train_ds, val_ds = dataset_organization.get_dataset("raffreddamento")


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
    model.save("Models/Raffreddamento/modello_raffreddamento.keras")


if __name__ == "__main__":
    main()