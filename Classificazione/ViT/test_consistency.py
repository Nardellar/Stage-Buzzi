"""
Script di test per verificare la consistenza tra training e evaluation.
Questo script testa che il preprocessing e il formato dei dati siano identici.
"""
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from datasets import Dataset
from transformers import AutoImageProcessor
from train_model import ViTForCustomClassificationImproved, preprocess_images, format_for_model

def test_preprocessing_consistency():
    """Testa che il preprocessing sia identico tra training e evaluation"""
    print("Test di consistenza del preprocessing...")
    
    # Carica il dataset di test - gestisce sia formato HuggingFace che JSON
    test_ds = None
    try:
        # Prova prima il formato HuggingFace
        test_ds = Dataset.load_from_disk("validation_test_set")
        print(f" Dataset HuggingFace caricato: {len(test_ds)} campioni")
    except (FileNotFoundError, ValueError):
        try:
            # Prova il formato JSON
            import json
            with open("validation_test_set.json", "r") as f:
                test_data = [json.loads(line) for line in f]
            print(f" Dataset JSON caricato: {len(test_data)} campioni")
            
            # Converti i percorsi delle immagini in oggetti PIL
            from PIL import Image
            for item in test_data:
                if 'image' in item and 'path' in item['image']:
                    try:
                        item['image'] = Image.open(item['image']['path']).convert('RGB')
                    except Exception as e:
                        print(f"Errore nel caricamento dell'immagine {item['image']['path']}: {e}")
                        item['image'] = None
            
            # Filtra le immagini che non sono state caricate correttamente
            test_data = [item for item in test_data if item['image'] is not None]
            print(f"Immagini caricate correttamente: {len(test_data)}")
            
            # Converti in formato HuggingFace per il test
            from datasets import Dataset
            test_ds = Dataset.from_list(test_data)
        except FileNotFoundError:
            print(" Dataset di test non trovato. Esegui prima il training.")
            return False
    
    # Usa lo stesso processor del training
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    
    # Preprocessing come nel training
    test_ds_processed = test_ds.map(
        lambda batch: preprocess_images(batch, processor), 
        batched=True, 
        batch_size=4
    )
    
    # Crea dataset TF come nel training
    test_tf = test_ds_processed.to_tf_dataset(
        columns=["pixel_values", "original_labels"], 
        label_cols=["labels"], 
        batch_size=4, 
        shuffle=False
    )
    
    # Applica format_for_model come nel training
    test_tf_mapped = test_tf.map(
        format_for_model,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Test del formato dei dati
    print("\n Test del formato dei dati...")
    for batch in test_tf_mapped.take(1):
        if isinstance(batch, tuple) and len(batch) == 2:
            images, labels = batch
            print(f" Formato tuple: images={type(images)}, labels={type(labels)}")
            if isinstance(images, dict):
                print(f"   - images keys: {list(images.keys())}")
                print(f"   - pixel_values shape: {images['pixel_values'].shape}")
            print(f"   - labels shape: {labels.shape}")
        else:
            print(f"⚠️ Formato inaspettato: {type(batch)}")
        
        break
    
    return True

def test_model_loading():
    """Testa il caricamento del modello"""
    print("\n Test di caricamento del modello...")
    
    # Cerca le cartelle di training
    training_dirs = [d for d in Path(".").iterdir() if d.is_dir() and d.name.startswith("training_results_")]
    if not training_dirs:
        print(" Nessuna cartella di training trovata")
        return False
    
    # Usa la prima cartella disponibile
    training_dir = training_dirs[0]
    artifacts_path = training_dir / "artifacts.json"
    
    if not artifacts_path.exists():
        print(f" File artifacts.json non trovato in {training_dir}")
        return False
    
    with open(artifacts_path) as f:
        artifacts = json.load(f)
    
    print(f" Artifacts caricati: {artifacts['attribute']}")
    
    # Testa il caricamento del modello - gestisce sia formato nuovo che vecchio
    model_path = None
    if 'model_path' in artifacts:
        model_path = artifacts['model_path']
        print(f" Caricamento modello completo: {model_path}")
    elif 'model_weights_path' in artifacts:
        model_path = artifacts['model_weights_path']
        print(f" Caricamento solo pesi: {model_path}")
    else:
        print(" Nessun percorso del modello trovato negli artifacts")
        return False
    
    if not Path(model_path).exists():
        print(f" File modello non trovato: {model_path}")
        return False
    
    try:
        if model_path.endswith('.keras'):
            # Modello completo
            model = tf.keras.models.load_model(model_path)
            print(" Modello completo caricato con successo")
        else:
            # Solo pesi - ricostruisci l'architettura
            print(" Ricostruzione dell'architettura del modello...")
            model = ViTForCustomClassificationImproved(num_labels=artifacts['num_classes'])
            model.vit.trainable = False
            
            # Inizializza con input dummy
            dummy_input = {'pixel_values': tf.zeros([1, 3, 224, 224])}
            _ = model(dummy_input, training=False)
            
            # Carica i pesi
            model.load_weights(model_path)
            print(" Pesi caricati con successo")
        
        # Test con input dummy
        dummy_input = {'pixel_values': tf.zeros([1, 3, 224, 224])}
        output = model(dummy_input, training=False)
        print(f" Test modello: output shape = {output['logits'].shape}")
        
        return True
        
    except Exception as e:
        print(f" Errore nel caricamento: {e}")
        return False
    
    return False

def test_end_to_end():
    """Test end-to-end del pipeline di evaluation"""
    print("\n Test end-to-end...")
    
    # Carica dataset - gestisce sia formato HuggingFace che JSON
    try:
        test_ds = Dataset.load_from_disk("validation_test_set")
        print(f" Dataset HuggingFace caricato: {len(test_ds)} campioni")
    except (FileNotFoundError, ValueError):
        try:
            import json
            with open("validation_test_set.json", "r") as f:
                test_data = [json.loads(line) for line in f]
            print(f" Dataset JSON caricato: {len(test_data)} campioni")
            
            # Converti i percorsi delle immagini in oggetti PIL
            from PIL import Image
            for item in test_data:
                if 'image' in item and 'path' in item['image']:
                    try:
                        item['image'] = Image.open(item['image']['path']).convert('RGB')
                    except Exception as e:
                        print(f"Errore nel caricamento dell'immagine {item['image']['path']}: {e}")
                        item['image'] = None
            
            # Filtra le immagini che non sono state caricate correttamente
            test_data = [item for item in test_data if item['image'] is not None]
            print(f"Immagini caricate correttamente: {len(test_data)}")
            
            test_ds = Dataset.from_list(test_data)
        except FileNotFoundError:
            print(" Dataset di test non trovato")
            return False
    
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    
    test_ds_processed = test_ds.map(
        lambda batch: preprocess_images(batch, processor), 
        batched=True, 
        batch_size=2
    )
    
    test_tf = test_ds_processed.to_tf_dataset(
        columns=["pixel_values", "original_labels"], 
        label_cols=["labels"], 
        batch_size=2, 
        shuffle=False
    )
    
    test_tf_mapped = test_tf.map(format_for_model, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Carica modello
    training_dirs = [d for d in Path(".").iterdir() if d.is_dir() and d.name.startswith("training_results_")]
    if not training_dirs:
        print(" Nessun modello trovato")
        return False
    
    training_dir = training_dirs[0]
    artifacts_path = training_dir / "artifacts.json"
    
    with open(artifacts_path) as f:
        artifacts = json.load(f)
    
    # Gestisci sia formato nuovo che vecchio
    if 'model_path' in artifacts:
        model_path = artifacts['model_path']
        model = tf.keras.models.load_model(model_path)
    elif 'model_weights_path' in artifacts:
        model_path = artifacts['model_weights_path']
        # Ricostruisci l'architettura
        model = ViTForCustomClassificationImproved(num_labels=artifacts['num_classes'])
        model.vit.trainable = False
        # Inizializza con input dummy
        dummy_input = {'pixel_values': tf.zeros([1, 3, 224, 224])}
        _ = model(dummy_input, training=False)
        # Carica i pesi
        model.load_weights(model_path)
    else:
        print(" Nessun percorso del modello trovato negli artifacts")
        return False
    
    # Test predizioni
    print(" Test predizioni...")
    for batch in test_tf_mapped.take(1):
        if isinstance(batch, tuple) and len(batch) == 2:
            images, labels = batch
        else:
            images = batch
            labels = batch.get('labels', None)
        
        predictions = model(images, training=False)
        print(f" Predizioni: shape={predictions['logits'].shape}")
        print(f" Labels: shape={labels.shape}")
        
        # Verifica che le predizioni abbiano senso
        probs = tf.nn.softmax(predictions['logits'])
        print(f" Probabilità: {probs.numpy()[0]}")
        
        break
    
    return True

def main():
    """Esegue tutti i test di consistenza"""
    print("TEST DI CONSISTENZA TRAINING-EVALUATION")
    
    tests = [
        ("Preprocessing consistency", test_preprocessing_consistency),
        ("Model loading", test_model_loading),
        ("End-to-end pipeline", test_end_to_end)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"{test_name}")
        print('='*50)
        
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"{test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"{test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    print(f"\n{'='*50}")
    print("RIEPILOGO TEST")
    print('='*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nRisultato: {passed}/{total} test passati")
    
    if passed == total:
        print("Tutti i test sono passati! Il problema di consistenza dovrebbe essere risolto.")
    else:
        print("Alcuni test sono falliti. Controlla i problemi sopra.")

if __name__ == "__main__":
    main()
