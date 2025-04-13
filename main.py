import dataset_organization
from model_temperatura import create_classification_model, train_model


def main():

    train_ds, val_ds = dataset_organization.get_dataset()



    model = create_classification_model(
        input_shape=(112, 112, 3),
        num_classes=3,  # per esempio
        base_trainable=False
    )

    history = train_model(
        model=model,
        train_dataset=train_ds,
        validation_dataset=val_ds,
        epochs=30,
        checkpoint_filepath='Models/best_modello_temperatura.keras'
    )

    # Per esempio, salva il modello finale
    model.save("Models/modello_temperatura.keras")


if __name__ == "__main__":
    main()
