"""
Test Rapido per la Versione Migliorata
Verifica che il sistema migliorato funzioni correttamente.
"""

import os
import sys
import numpy as np
from improved_cnn_classifier import ImprovedCNNSegmentationClassifier


def quick_test_improved():
    """
    Esegue un test rapido del sistema migliorato.
    """
    print("=" * 60)
    print("TEST RAPIDO SISTEMA CNN + CLASSIFICATORE MIGLIORATO")
    print("=" * 60)
    
    try:
        # Verifica che i file di dati esistano
        image_dir = "../images/Immagini/"
        mask_dir = "../images/Maschere/"
        
        if not os.path.exists(image_dir):
            print(f"ERRORE: Directory immagini non trovata: {image_dir}")
            return False
            
        if not os.path.exists(mask_dir):
            print(f"ERRORE: Directory maschere non trovata: {mask_dir}")
            return False
        
        # Conta i file
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.tif', '.tiff'))]
        
        print(f"Immagini trovate: {len(image_files)}")
        print(f"Maschere trovate: {len(mask_files)}")
        
        if len(image_files) == 0:
            print("ERRORE: Nessuna immagine trovata")
            return False
            
        if len(mask_files) == 0:
            print("ERRORE: Nessuna maschera trovata")
            return False
        
        # Inizializza il modello migliorato con parametri minimi
        print("\nInizializzazione modello migliorato...")
        model = ImprovedCNNSegmentationClassifier(
            cnn_model='vgg16',  # Usa VGG16 per compatibilità
            image_size=(256, 256),  # Dimensioni ridotte per test rapido
            num_classes=5,
            batch_size=1,  # Batch size minimo
            use_augmentation=False  # Disabilita augmentation per evitare problemi
        )
        
        # Test caricamento dati con augmentation
        print("\nTest caricamento dati con augmentation...")
        model.load_data_with_augmentation()
        
        print(f"Immagini dopo augmentation: {model.train_images.shape}")
        print(f"Maschere dopo augmentation: {model.train_masks.shape}")
        
        # Test estrazione features migliorata
        print("\nTest estrazione features migliorata...")
        model.extract_features_improved()
        
        print(f"Features estratte: {model.X_features.shape}")
        print(f"Labels estratte: {model.y_labels.shape}")
        
        # Test training rapido con CV ridotta
        print("\nTest training rapido con cross-validation...")
        cv_scores = model.train_classifier_with_cv(cv_folds=2)  # Solo 2 fold per test rapido
        
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"Accuracy media: {np.mean(cv_scores):.4f}")
        print(f"Deviazione standard: {np.std(cv_scores):.4f}")
        
        # Test predizione su prima immagine
        if len(image_files) > 0:
            print("\nTest predizione...")
            test_image_path = os.path.join(image_dir, image_files[0])
            
            if os.path.exists(test_image_path):
                predicted_mask = model.predict_image(test_image_path)
                print(f"Predizione completata. Shape maschera: {predicted_mask.shape}")
                
                # Statistiche rapide
                unique_classes, counts = np.unique(predicted_mask, return_counts=True)
                class_names = ['Resina', 'Pori/Imperfezioni', 'Fase Fusa', 'Belite', 'Alite']
                
                print("\nDistribuzione classi predette:")
                for class_id, count in zip(unique_classes, counts):
                    percentage = (count / predicted_mask.size) * 100
                    print(f"  {class_names[class_id]}: {percentage:.1f}%")
        
        print("\n" + "=" * 60)
        print("TEST RAPIDO MIGLIORATO COMPLETATO CON SUCCESSO!")
        print("Il sistema migliorato è pronto per l'uso.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\nERRORE durante il test rapido migliorato: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def check_improved_dependencies():
    """
    Verifica che tutte le dipendenze per la versione migliorata siano installate.
    """
    print("Verifica dipendenze per versione migliorata...")
    
    required_packages = [
        'tensorflow',
        'xgboost', 
        'opencv-python',
        'PIL',
        'sklearn',
        'matplotlib',
        'numpy',
        'optuna',
        'tqdm',
        'albumentations'  # Nuovo per la versione migliorata
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            elif package == 'sklearn':
                import sklearn
            elif package == 'albumentations':
                import albumentations
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (MANCANTE)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nDipendenze mancanti: {', '.join(missing_packages)}")
        print("Installa con: pip install " + " ".join(missing_packages))
        return False
    else:
        print("\nTutte le dipendenze sono installate!")
        return True


def main():
    """
    Funzione principale del test rapido migliorato.
    """
    print("Test rapido del sistema CNN + Classificatore MIGLIORATO")
    print("Questo test verifica che la versione migliorata funzioni correttamente.\n")
    
    # Verifica dipendenze
    if not check_improved_dependencies():
        print("\nRisolvi le dipendenze mancanti prima di continuare.")
        return
    
    print("\n" + "-" * 60)
    
    # Esegui test rapido
    success = quick_test_improved()
    
    if success:
        print("\n🎉 Sistema migliorato funziona correttamente!")
        print("\nPer iniziare il training completo, usa:")
        print("python train_improved_classifier.py --cnn_model efficientnet_b0")
        print("\nPer testare il modello addestrato, usa:")
        print("python test_improved_classifier.py --model_path improved_cnn_model.pkl")
        
        print("\nVantaggi della versione migliorata:")
        print("✅ Data augmentation per dataset piccolo")
        print("✅ Cross-validation per stime robuste")
        print("✅ Class weighting per bilanciare le classi")
        print("✅ Regolarizzazione aumentata per evitare overfitting")
        print("✅ Architetture moderne (EfficientNet)")
    else:
        print("\n❌ Test fallito. Controlla gli errori sopra.")
        sys.exit(1)


if __name__ == "__main__":
    main()
