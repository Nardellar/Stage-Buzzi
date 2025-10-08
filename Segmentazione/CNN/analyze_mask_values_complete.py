"""
Script COMPLETO per analizzare i valori nelle maschere di segmentazione
Usa tutte le librerie disponibili nell'ambiente virtuale
"""
import os
import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
import tifffile

def analyze_all_masks():
    """Analizza tutte le maschere per trovare i valori unici"""
    
    MASK_DIR = "../images/Maschere/"
    
    print("🔍 ANALISI COMPLETA VALORI NELLE MASCHERE")
    print("=" * 60)
    
    # Lista tutti i file maschera
    mask_files = [f for f in os.listdir(MASK_DIR) if f.lower().endswith(('.tif', '.tiff'))]
    
    print(f"📁 Trovate {len(mask_files)} maschere")
    
    # Analizza ogni maschera
    all_unique_values = set()
    mask_analysis = {}
    
    for i, mask_file in enumerate(mask_files):
        mask_path = os.path.join(MASK_DIR, mask_file)
        
        print(f"📄 Analizzando {i+1}/{len(mask_files)}: {mask_file}")
        
        try:
            # Prova diversi metodi di caricamento
            mask = None
            
            # Metodo 1: tifffile (migliore per TIFF)
            try:
                mask = tifffile.imread(mask_path)
                print(f"   ✅ Caricato con tifffile")
            except:
                pass
            
            # Metodo 2: OpenCV
            if mask is None:
                try:
                    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                    if mask is not None:
                        print(f"   ✅ Caricato con OpenCV")
                except:
                    pass
            
            # Metodo 3: PIL
            if mask is None:
                try:
                    mask = np.array(Image.open(mask_path))
                    print(f"   ✅ Caricato con PIL")
                except:
                    pass
            
            if mask is None:
                print(f"   ❌ Errore nel caricare {mask_file}")
                continue
            
            # Analizza la maschera
            unique_values = np.unique(mask)
            all_unique_values.update(unique_values)
            
            # Conta occorrenze
            value_counts = Counter(mask.flatten())
            
            mask_analysis[mask_file] = {
                'unique_values': unique_values,
                'value_counts': value_counts,
                'shape': mask.shape,
                'dtype': mask.dtype,
                'min_value': np.min(mask),
                'max_value': np.max(mask)
            }
            
            print(f"   📊 Valori: {sorted(unique_values)}")
            print(f"   📐 Shape: {mask.shape}, Dtype: {mask.dtype}")
            print(f"   📈 Range: {np.min(mask)} - {np.max(mask)}")
            
        except Exception as e:
            print(f"   ❌ Errore nell'analisi di {mask_file}: {e}")
    
    # Analisi globale
    print(f"\n📊 ANALISI GLOBALE:")
    print(f"   Valori unici trovati in TUTTE le maschere: {sorted(all_unique_values)}")
    print(f"   Numero totale di classi diverse: {len(all_unique_values)}")
    
    # Verifica se ci sono più classi del previsto
    if len(all_unique_values) == 3:
        print("✅ OK: Esattamente 3 classi (come previsto)")
    elif len(all_unique_values) > 3:
        print(f"⚠️  PROBLEMA: Ci sono {len(all_unique_values)} classi invece di 3!")
        print("   Questo spiega i problemi di addestramento!")
    else:
        print("❌ PROBLEMA: Meno di 3 classi trovate")
    
    # Analizza distribuzione delle classi
    print(f"\n📈 DISTRIBUZIONE CLASSI:")
    for value in sorted(all_unique_values):
        count = 0
        for mask_file, data in mask_analysis.items():
            if value in data['unique_values']:
                count += 1
        percentage = (count / len(mask_files)) * 100
        print(f"   Classe {value}: presente in {count}/{len(mask_files)} maschere ({percentage:.1f}%)")
    
    # Salva risultati dettagliati
    save_detailed_results(mask_analysis, sorted(all_unique_values))
    
    # Crea visualizzazione
    create_visualization(mask_analysis, sorted(all_unique_values))
    
    return mask_analysis, sorted(all_unique_values)

def save_detailed_results(mask_analysis, unique_values):
    """Salva i risultati dettagliati su file"""
    
    with open("mask_analysis_detailed.txt", "w", encoding="utf-8") as f:
        f.write("ANALISI DETTAGLIATA MASCHERE DI SEGMENTAZIONE\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Valori unici trovati in TUTTE le maschere: {unique_values}\n")
        f.write(f"Numero totale di classi diverse: {len(unique_values)}\n\n")
        
        f.write("DETTAGLI PER MASCHERA:\n")
        f.write("-" * 40 + "\n")
        
        for mask_file, data in mask_analysis.items():
            f.write(f"\n{mask_file}:\n")
            f.write(f"  Valori: {sorted(data['unique_values'])}\n")
            f.write(f"  Shape: {data['shape']}\n")
            f.write(f"  Dtype: {data['dtype']}\n")
            f.write(f"  Range: {data['min_value']} - {data['max_value']}\n")
            
            # Mostra distribuzione dei valori
            f.write("  Distribuzione:\n")
            total_pixels = np.prod(data['shape'])
            for value, count in data['value_counts'].most_common():
                percentage = (count / total_pixels) * 100
                f.write(f"    Classe {value}: {count:,} pixel ({percentage:.2f}%)\n")
    
    print(f"📄 Risultati dettagliati salvati in: mask_analysis_detailed.txt")

def create_visualization(mask_analysis, unique_values):
    """Crea visualizzazione delle classi trovate"""
    
    # Colori per le classi
    colors = [
        [0, 0, 0],      # Classe 0: Nero (Sfondo)
        [255, 0, 0],    # Classe 1: Rosso (Alite)
        [0, 255, 0],    # Classe 2: Verde (Belite)
        [0, 0, 255],    # Classe 3: Blu
        [255, 255, 0],  # Classe 4: Giallo
        [255, 0, 255],  # Classe 5: Magenta
        [0, 255, 255],  # Classe 6: Ciano
        [255, 128, 0],  # Classe 7: Arancione
        [128, 0, 128],  # Classe 8: Viola
        [0, 128, 128],  # Classe 9: Teal
        [128, 128, 0],  # Classe 10: Olive
    ]
    
    # Mostra le prime 6 maschere come esempio
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    mask_files = list(mask_analysis.keys())[:6]
    
    for i, mask_file in enumerate(mask_files):
        if i >= 6:
            break
            
        mask_path = os.path.join("../images/Maschere/", mask_file)
        
        # Carica maschera
        mask = None
        try:
            mask = tifffile.imread(mask_path)
        except:
            try:
                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            except:
                mask = np.array(Image.open(mask_path))
        
        # Crea maschera colorata
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        
        for value in unique_values:
            if value < len(colors):
                colored_mask[mask == value] = colors[value]
        
        # Mostra maschera colorata
        axes[i, 0].imshow(colored_mask)
        axes[i, 0].set_title(f"Maschera: {mask_file[:20]}...")
        axes[i, 0].axis('off')
        
        # Mostra maschera originale (grayscale)
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title("Originale (Grayscale)")
        axes[i, 1].axis('off')
        
        # Mostra distribuzione dei valori
        unique_vals = mask_analysis[mask_file]['unique_values']
        value_counts = mask_analysis[mask_file]['value_counts']
        
        values = sorted(unique_vals)
        counts = [value_counts[v] for v in values]
        
        axes[i, 2].bar(values, counts, color=[np.array(colors[v])/255.0 for v in values])
        axes[i, 2].set_title("Distribuzione Valori")
        axes[i, 2].set_xlabel("Classe")
        axes[i, 2].set_ylabel("Numero Pixel")
        
        # Mostra informazioni testuali
        info_text = f"Valori: {unique_vals}\nShape: {mask.shape}\nRange: {np.min(mask)}-{np.max(mask)}"
        axes[i, 3].text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[i, 3].set_title("Informazioni")
        axes[i, 3].axis('off')
    
    plt.suptitle("Analisi Dettagliata Maschere - Prime 6 maschere", fontsize=16)
    plt.tight_layout()
    plt.savefig("mask_analysis_visualization.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Visualizzazione salvata in: mask_analysis_visualization.png")

def suggest_solutions(num_classes_found):
    """Suggerisce soluzioni basate sul numero di classi trovate"""
    
    print(f"\n💡 SOLUZIONI SUGGERITE:")
    print("=" * 50)
    
    if num_classes_found == 3:
        print("✅ Numero di classi corretto!")
        print("   Il problema potrebbe essere altrove.")
        
    elif num_classes_found > 3:
        print(f"⚠️  Trovate {num_classes_found} classi invece di 3")
        print("\n🎯 SOLUZIONI:")
        print("1. RIMAPPATURA: Mappa le classi extra alle 3 principali")
        print("2. AGGIORNAMENTO MODELLO: Cambia num_classes nel modello")
        print("3. ANALISI: Identifica cosa rappresentano le classi extra")
        
        print(f"\n🛠️ IMPLEMENTAZIONE RAPIDA:")
        print("   Modifica il data_loader.py per rimappare le classi:")
        print("   class 0-2: mantieni")
        print("   class 3+: rimappa a 0 (sfondo)")
        
    else:
        print(f"❌ Solo {num_classes_found} classi trovate")
        print("   Verifica la creazione delle maschere")

if __name__ == "__main__":
    print("🚀 AVVIO ANALISI COMPLETA MASCHERE")
    print("=" * 60)
    
    try:
        mask_analysis, unique_values = analyze_all_masks()
        suggest_solutions(len(unique_values))
        
        print(f"\n✅ ANALISI COMPLETATA!")
        print(f"📊 Trovate {len(unique_values)} classi: {unique_values}")
        
    except Exception as e:
        print(f"❌ Errore durante l'analisi: {e}")
        import traceback
        traceback.print_exc()

