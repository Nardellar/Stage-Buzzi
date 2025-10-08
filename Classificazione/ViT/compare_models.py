"""
Script per confrontare il modello originale con quello migliorato
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def compare_results():
    """Confronta i risultati tra modello originale e migliorato"""
    
    print("📊 CONFRONTO MODELLI: Originale vs Migliorato")
    print("=" * 60)
    
    # Cerca i file di risultati
    results_dir = Path(".")
    original_files = list(results_dir.glob("results_*/performance_report_*.txt"))
    improved_files = list(results_dir.glob("results_improved_*/performance_report_*.txt"))
    
    print(f"📁 File risultati trovati:")
    print(f"  - Originali: {len(original_files)}")
    print(f"  - Migliorati: {len(improved_files)}")
    
    # Analizza risultati originali
    original_results = {}
    for file_path in original_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Estrai metriche dal report
            lines = content.split('\n')
            for line in lines:
                if 'Accuratezza (Validazione):' in line:
                    acc = float(line.split(':')[1].strip())
                elif 'Loss (Validazione):' in line:
                    loss = float(line.split(':')[1].strip())
            
            attr = file_path.parent.name.replace('results_', '')
            original_results[attr] = {'accuracy': acc, 'loss': loss}
            print(f"  ✅ {attr}: Acc={acc:.4f}, Loss={loss:.4f}")
            
        except Exception as e:
            print(f"  ❌ Errore leggendo {file_path}: {e}")
    
    # Analizza risultati migliorati (se disponibili)
    improved_results = {}
    for file_path in improved_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Estrai metriche dal report
            lines = content.split('\n')
            for line in lines:
                if 'Accuratezza (Validazione):' in line:
                    acc = float(line.split(':')[1].strip())
                elif 'Loss (Validazione):' in line:
                    loss = float(line.split(':')[1].strip())
            
            attr = file_path.parent.name.replace('results_improved_', '')
            improved_results[attr] = {'accuracy': acc, 'loss': loss}
            print(f"  ✅ {attr} (migliorato): Acc={acc:.4f}, Loss={loss:.4f}")
            
        except Exception as e:
            print(f"  ❌ Errore leggendo {file_path}: {e}")
    
    # Confronto diretto
    print(f"\n🔄 CONFRONTO DIRETTO:")
    print("-" * 40)
    
    for attr in original_results:
        if attr in improved_results:
            orig_acc = original_results[attr]['accuracy']
            impr_acc = improved_results[attr]['accuracy']
            orig_loss = original_results[attr]['loss']
            impr_loss = improved_results[attr]['loss']
            
            acc_diff = impr_acc - orig_acc
            loss_diff = impr_loss - orig_loss
            
            print(f"\n📊 {attr.upper()}:")
            print(f"  Accuracy:")
            print(f"    Originale:  {orig_acc:.4f}")
            print(f"    Migliorato: {impr_acc:.4f}")
            print(f"    Differenza: {acc_diff:+.4f} ({'✅' if acc_diff > 0 else '❌'})")
            
            print(f"  Loss:")
            print(f"    Originale:  {orig_loss:.4f}")
            print(f"    Migliorato: {impr_loss:.4f}")
            print(f"    Differenza: {loss_diff:+.4f} ({'✅' if loss_diff < 0 else '❌'})")
            
            # Interpretazione
            if acc_diff > 0.01 and loss_diff < -0.01:
                print(f"  🎉 MIGLIORAMENTO SIGNIFICATIVO!")
            elif acc_diff > 0 and loss_diff < 0:
                print(f"  ✅ Miglioramento leggero")
            elif acc_diff < 0 and loss_diff > 0:
                print(f"  ⚠️ Peggioramento")
            else:
                print(f"  ➡️ Risultati simili")
    
    # Grafico di confronto (se matplotlib disponibile)
    try:
        create_comparison_plot(original_results, improved_results)
    except Exception as e:
        print(f"\n⚠️ Impossibile creare grafico: {e}")
    
    return original_results, improved_results

def create_comparison_plot(original_results, improved_results):
    """Crea un grafico di confronto"""
    
    attributes = list(original_results.keys())
    orig_accs = [original_results[attr]['accuracy'] for attr in attributes]
    impr_accs = [improved_results.get(attr, {}).get('accuracy', 0) for attr in attributes]
    
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(attributes))
    width = 0.35
    
    plt.bar(x - width/2, orig_accs, width, label='Originale', alpha=0.8)
    plt.bar(x + width/2, impr_accs, width, label='Migliorato', alpha=0.8)
    
    plt.xlabel('Attributi')
    plt.ylabel('Accuracy')
    plt.title('Confronto Accuracy: Modello Originale vs Migliorato')
    plt.xticks(x, attributes, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_plot.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Grafico salvato come 'comparison_plot.png'")
    plt.show()

def run_quick_test():
    """Esegue un test rapido per confrontare i modelli"""
    
    print("🧪 TEST RAPIDO - Confronto Modelli")
    print("=" * 50)
    
    # Simula risultati per test
    print("📊 Simulazione risultati:")
    print("  - Modello originale: Accuracy ~0.91, Loss ~0.36")
    print("  - Modello migliorato: Accuracy ~0.94, Loss ~0.28")
    print("  - Miglioramento: +0.03 accuracy, -0.08 loss")
    print("  - Interpretazione: 🎉 MIGLIORAMENTO SIGNIFICATIVO!")
    
    print(f"\n💡 Per testare realmente:")
    print(f"  1. Esegui: python vit_from_hf_attribute.py (modello originale)")
    print(f"  2. Esegui: python vit_from_hf_attribute_improved.py (modello migliorato)")
    print(f"  3. Esegui: python compare_models.py (confronto)")

if __name__ == "__main__":
    print("🔍 ANALISI RISULTATI")
    print("Scegli opzione:")
    print("1. Analizza risultati esistenti")
    print("2. Test rapido (simulazione)")
    
    choice = input("➡️ Scegli (1/2): ").strip()
    
    if choice == "1":
        original, improved = compare_results()
    elif choice == "2":
        run_quick_test()
    else:
        print("❌ Scelta non valida")

