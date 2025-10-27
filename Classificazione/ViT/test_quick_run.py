"""
Quick test per verificare che il codice funzioni senza errori.
Testa solo con 2 epochs per velocità.
"""
import sys
from pathlib import Path

def test_quick_training():
    """Test rapido del training (solo 2 epochs)."""
    print("\n" + "=" * 60)
    print("TEST RAPIDO TRAINING (2 EPOCHS)")
    print("=" * 60)
    
    try:
        from train_model_pytorch import prepare_and_split_dataset, ViTForCustomClassification, train_model
        import torch
        from datetime import datetime
        
        print("\n1. Preparazione dataset...")
        attribute = "temperatura"
        train_ds, val_ds, num_classes, id2label, class_weights = prepare_and_split_dataset(attribute, batch_size=16)
        print(f"   Train: {len(train_ds)} campioni")
        print(f"   Val: {len(val_ds)} campioni")
        print(f"   Classi: {num_classes}")
        
        print("\n2. Creazione modello...")
        model = ViTForCustomClassification(num_labels=num_classes)
        print(f"   Modello creato con {num_classes} classi")
        
        print("\n3. Test training (2 epochs)...")
        results_dir = Path("test_results_quick")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Modifica temporanea per testare solo 2 epochs
        print("   [INIZIO] Training con 2 epochs...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Dispositivo: {device}")
        
        from torch.utils.data import DataLoader
        from transformers import default_data_collator
        import torch.nn as nn
        
        model = model.to(device)
        class_weights_device = class_weights.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(weight=class_weights_device)
        
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=default_data_collator, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=default_data_collator, num_workers=0)
        
        # Training loop semplificato
        for epoch in range(2):
            model.train()
            train_loss = 0.0
            batches = 0
            
            for batch in train_loader:
                if batches >= 3:  # Solo 3 batch per epoch per velocità
                    break
                    
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                
                optimizer.zero_grad()
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                batches += 1
            
            train_loss /= batches
            print(f"   Epoch {epoch+1}/2: Train Loss = {train_loss:.4f}")
        
        print("\n[OK] Training test completato senza errori!")
        
        # Cleanup
        import shutil
        if results_dir.exists():
            shutil.rmtree(results_dir)
        
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Errore durante il training test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("QUICK TEST - VERIFICA CODICE PYTORCH")
    print("=" * 60)
    print("\nQuesto test esegue un training rapido di 2 epochs con")
    print("solo 3 batch per epoca per verificare che non ci siano errori.")
    
    result = test_quick_training()
    
    print("\n" + "=" * 60)
    print("RISULTATO")
    print("=" * 60)
    
    if result:
        print("\n[SUCCESS] Tutti i test sono passati!")
        print("Il codice funziona correttamente.")
        print("\nPuoi ora eseguire il training completo con:")
        print("  python train_model_pytorch.py temperatura")
    else:
        print("\n[FAILED] Il test ha generato errori.")
        print("Controlla l'output sopra per i dettagli.")
    
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())

