import dataset_organization
from model import create_classification_model, train_model


def main():

    #riceve il training set (train_ds) e il validation set(validation_ds)
    train_ds, validation_ds = dataset_organization.get_dataset()


    #crea un modello di classificazione
    model = create_classification_model(
        input_shape=(224, 224, 3),   #le immmagini in input sono 224x224 con 3 canali (RGB)
        num_classes=3,  #in che senso????????????????????? definisce il problema come una classificazione a 3 classi
        base_trainable=False #il modello base (VGG16) non viene addestrato
    )
    #addestra il modello e passa come input...
    history = train_model(
        model=model, #il modello creato
        train_dataset=train_ds, #il training set
        validation_dataset=validation_ds, #il validation set
        epochs=10, #il numero di epoche d'addestramento
        checkpoint_filepath='Modelli_addestrati/best_model.h5' #salva il miglior modello in questo percorso
    )

    #salviamo il modello finito l'addestramento (non e' detto sia il migliore)
    #???????????'ci serve veramente?
    model.save("modello_finale.h5")


if __name__ == "__main__":
    main()
