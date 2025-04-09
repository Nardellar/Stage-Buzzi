import dataset_organization
from model import create_classification_model, train_model


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
        epochs=10,
        checkpoint_filepath='best_model.h5'
    )

    # Per esempio, salva il modello finale
    model.save("modello_finale.h5")


if __name__ == "__main__":
    main()
