"""
Script di test per verificare che l'ambiente PyTorch sia configurato correttamente.
"""
import sys
from pathlib import Path

def test_imports():
    """Testa che tutti i moduli necessari siano installati."""
    print("Test degli import...")
    
    try:
        import torch
        print(f"[OK] PyTorch: {torch.__version__}")
        print(f"   CUDA disponibile: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"[FAIL] PyTorch non installato: {e}")
        return False
    
    try:
        import transformers
        print(f"[OK] Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"[FAIL] Transformers non installato: {e}")
        return False
    
    try:
        import datasets
        print(f"[OK] Datasets: {datasets.__version__}")
    except ImportError as e:
        print(f"[FAIL] Datasets non installato: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"[OK] PIL/Pillow installato")
    except ImportError as e:
        print(f"[FAIL] PIL/Pillow non installato: {e}")
        return False
    
    try:
        import numpy as np
        print(f"[OK] NumPy: {np.__version__}")
    except ImportError as e:
        print(f"[FAIL] NumPy non installato: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"[OK] Pandas: {pd.__version__}")
    except ImportError as e:
        print(f"[FAIL] Pandas non installato: {e}")
        return False
    
    try:
        import sklearn
        print(f"[OK] Scikit-learn: {sklearn.__version__}")
    except ImportError as e:
        print(f"[FAIL] Scikit-learn non installato: {e}")
        return False
    
    try:
        import matplotlib
        print(f"[OK] Matplotlib: {matplotlib.__version__}")
    except ImportError as e:
        print(f"[FAIL] Matplotlib non installato: {e}")
        return False
    
    try:
        import seaborn
        print(f"[OK] Seaborn: {seaborn.__version__}")
    except ImportError as e:
        print(f"[FAIL] Seaborn non installato: {e}")
        return False
    
    try:
        from tqdm import tqdm
        print(f"[OK] TQDM installato")
    except ImportError as e:
        print(f"[FAIL] TQDM non installato: {e}")
        return False
    
    return True


def test_model_loading():
    """Testa il caricamento del modello ViT."""
    print("\nTest caricamento modello...")
    
    try:
        from transformers import ViTForImageClassification, AutoImageProcessor
        
        print("   Caricamento processor...")
        processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        print(f"[OK] Processor caricato: {processor.size}")
        
        print("   Caricamento modello...")
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        print(f"[OK] Modello caricato: {model.config.num_labels} classi")
        
        return True
    except Exception as e:
        print(f"[FAIL] Errore nel caricamento del modello: {e}")
        return False


def test_dataset_access():
    """Testa l'accesso al dataset HuggingFace."""
    print("\nTest accesso dataset...")
    
    try:
        from datasets import load_dataset
        
        print("   Caricamento dataset (prime 10 immagini)...")
        ds = load_dataset("Nardellar/Esperimenti", split="train[:10]")
        print(f"[OK] Dataset caricato: {len(ds)} campioni")
        print(f"   Colonne: {ds.column_names}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Errore nel caricamento del dataset: {e}")
        return False


def test_preprocessing():
    """Testa il preprocessing delle immagini."""
    print("\nTest preprocessing...")
    
    try:
        from transformers import AutoImageProcessor
        from datasets import load_dataset
        import torch
        
        processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        ds = load_dataset("Nardellar/Esperimenti", split="train[:2]")
        
        # Test transform
        def transform(examples):
            images = [img.convert("RGB") for img in examples["image"]]
            inputs = processor(images, return_tensors="pt")
            inputs["labels"] = [0, 1]  # Fake labels
            return inputs
        
        result = transform({"image": [ds[0]["image"], ds[1]["image"]]})
        print(f"[OK] Preprocessing funziona")
        print(f"   Shape pixel_values: {result['pixel_values'].shape}")
        print(f"   Labels: {result['labels']}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Errore nel preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("TEST SETUP PYTORCH PER VIT CLASSIFICATION")
    print("=" * 60)
    
    results = []
    
    # Test imports
    results.append(("Import", test_imports()))
    
    # Test model loading (richiede internet)
    print("\nATTENZIONE: I test seguenti richiedono connessione internet...")
    results.append(("Model Loading", test_model_loading()))
    results.append(("Dataset Access", test_dataset_access()))
    results.append(("Preprocessing", test_preprocessing()))
    
    # Riepilogo
    print("\n" + "=" * 60)
    print("RIEPILOGO TEST")
    print("=" * 60)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{name:20s}: {status}")
    
    all_passed = all(r for _, r in results)
    
    if all_passed:
        print("\nTutti i test sono passati!")
        print("L'ambiente e' configurato correttamente.")
        print("\nPuoi procedere con:")
        print("   python train_model_pytorch.py temperatura")
    else:
        print("\nAlcuni test sono falliti.")
        print("Controlla gli errori sopra e installa i pacchetti mancanti.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

