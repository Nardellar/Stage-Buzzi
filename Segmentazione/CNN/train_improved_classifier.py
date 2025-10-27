"""
Script di Training per la Versione Migliorata
Usa improved_cnn_classifier.py con tecniche per dataset piccolo.
"""

import os
import sys
import argparse
from improved_cnn_classifier import ImprovedCNNSegmentationClassifier


def main():
    parser = argparse.ArgumentParser(description='Training CNN + Classificatore Migliorato')
    parser.add_argument('--cnn_model', type=str, default='efficientnet_b0', 
                       choices=['efficientnet_b0', 'resnet50', 'vgg16'],
                       help='Modello CNN da utilizzare (efficientnet_b0 consigliato)')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size per il training (1 per GPU piccola, 4-8 per GPU grande)')
    parser.add_argument('--classifier', type=str, default='xgboost', 
                       choices=['lightgbm', 'xgboost'],
                       help='Classificatore da utilizzare (xgboost consigliato)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TRAINING CNN + CLASSIFICATORE MIGLIORATO")
    print("=" * 60)
    print(f"Modello CNN: {args.cnn_model}")
    print(f"Classificatore: {args.classifier}")
    print(f"Dimensioni immagine: 512x512 (fisso)")
    print(f"Batch size: {args.batch_size}")
    print(f"Ottimizzazione Optuna: 50 trials (fisso)")
    print(f"Data augmentation: Sì (fisso)")
    print("=" * 60)
    
    try:
        # Inizializza il modello migliorato
        model = ImprovedCNNSegmentationClassifier(
            cnn_model=args.cnn_model,
            image_size=(512, 512),  # Fisso
            num_classes=5,
            batch_size=args.batch_size,
            use_augmentation=True,  # Sempre abilitata
            classifier=args.classifier
        )
        
        print("\n1. Caricamento dati con augmentation...")
        model.load_data_with_augmentation()
        
        print("\n2. Estrazione features migliorata...")
        model.extract_features_improved()
        
        print("\n3. Training con ottimizzazione Optuna...")
        # Riduci trials per EfficientNetB0 (più lento)
        n_trials = 20 if args.cnn_model == 'efficientnet_b0' else 50
        print(f"   Trials ottimizzati: {n_trials} (ridotto per EfficientNetB0)")
        test_accuracy = model.train_classifier_with_optuna(n_trials=n_trials)
        
        print("\n4. Salvataggio modello...")
        model.save_model_complete('improved_cnn_model.pkl')  # Fisso
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETATO CON SUCCESSO!")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Modello salvato in: improved_cnn_model.pkl")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERRORE durante il training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()