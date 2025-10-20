"""
Script di Training per U-Net Moderna
Addestra un modello U-Net end-to-end più efficace del CNN + XGBoost.
"""

import os
import sys
import argparse
from modern_unet_segmentation import ModernUNet


def main():
    parser = argparse.ArgumentParser(description='Training U-Net Moderna per Segmentazione')
    parser.add_argument('--encoder', type=str, default='efficientnet_b3', 
                       choices=['efficientnet_b3', 'efficientnet_b4', 'resnet50'],
                       help='Encoder da utilizzare')
    parser.add_argument('--input_size', type=int, nargs=2, default=[512, 512],
                       help='Dimensioni input (width height)')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size per il training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Numero di epoche')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--no_augmentation', action='store_true',
                       help='Disabilita data augmentation')
    parser.add_argument('--model_save_path', type=str, default='best_unet_model.h5',
                       help='Percorso per salvare il modello')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TRAINING U-NET MODERNA PER SEGMENTAZIONE")
    print("=" * 60)
    print(f"Encoder: {args.encoder}")
    print(f"Dimensioni input: {args.input_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epoche: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Data augmentation: {'No' if args.no_augmentation else 'Sì'}")
    print("=" * 60)
    
    try:
        # Inizializza il modello
        unet = ModernUNet(
            encoder_name=args.encoder,
            input_shape=(*args.input_size, 3),
            num_classes=5,
            dropout_rate=0.3
        )
        
        print("\n1. Caricamento dati...")
        unet.load_data()
        
        print("\n2. Costruzione modello...")
        model = unet.build_model()
        
        print("\n3. Training...")
        history = unet.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_augmentation=not args.no_augmentation
        )
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETATO CON SUCCESSO!")
        print(f"Modello salvato in: {args.model_save_path}")
        print("=" * 60)
        
        # Test rapido su un'immagine
        test_image = "../images/Immagini/1579--03.png"
        if os.path.exists(test_image):
            print(f"\nTest su immagine: {test_image}")
            predicted_mask = unet.predict_image(test_image)
            print(f"Predizione completata. Shape: {predicted_mask.shape}")
            
            # Salva visualizzazione
            unet.visualize_prediction(test_image, 'test_prediction.png')
        
    except Exception as e:
        print(f"\nERRORE durante il training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


