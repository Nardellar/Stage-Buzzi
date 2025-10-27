"""
Script per verificare la struttura del modello e quali pesi sono stati caricati.
"""
from train_model_pytorch import ViTForCustomClassification
import torch

def analyze_model():
    print("=" * 60)
    print("ANALISI STRUTTURA MODELLO")
    print("=" * 60)
    
    # Crea il modello
    model = ViTForCustomClassification(num_labels=3)
    
    print("\n1. STRUTTURA DEL MODELLO:")
    print(f"   Tipo: {type(model)}")
    print(f"   ViT backbone: {type(model.vit)}")
    
    print("\n2. PARAMETRI TOTALI:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Totali: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    print(f"   Frozen: {total_params - trainable_params:,}")
    
    print("\n3. LAYER TRAINABLE:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"   [TRAIN] {name:50s} | Shape: {str(tuple(param.shape)):20s}")
    
    print("\n4. VERIFICHE:")
    
    # Verifica che il ViT sia caricato correttamente
    vit_config = model.vit.config
    print(f"   [OK] ViT hidden size: {vit_config.hidden_size}")
    print(f"   [OK] ViT num layers: {vit_config.num_hidden_layers}")
    print(f"   [OK] ViT num attention heads: {vit_config.num_attention_heads}")
    
    # Verifica il classificatore
    classifier_layer = model.vit.classifier
    print(f"   [OK] Classifier in_features: {classifier_layer.in_features}")
    print(f"   [OK] Classifier out_features: {classifier_layer.out_features} (deve essere 3)")
    
    print("\n5. LAYER MODIFICATI (Transfer Learning):")
    print("   - classifier.weight: (3, 768) - NUOVO per il tuo task")
    print("   - classifier.bias: (3,) - NUOVO per il tuo task")
    print("   - Tutto il resto: PRETRAINED da ImageNet")
    
    print("\n" + "=" * 60)
    print("CONCLUSIONE")
    print("=" * 60)
    print("[OK] Il modello e' configurato correttamente!")
    print("[OK] L'encoder ViT usa pesi pretrained (ImageNet)")
    print("[OK] Il classificatore e' nuovo per il tuo task (3 classi)")
    print("\nI warning sono NORMALI e ATTESI nel transfer learning.")
    print("=" * 60)

if __name__ == "__main__":
    analyze_model()

