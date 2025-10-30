"""
Script per analizzare la distribuzione dettagliata del dataset
Verifica quanti esperimenti ci sono per ogni temperatura e quante immagini per esperimento
"""
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from collections import Counter, defaultdict

def analyze_dataset_distribution():
    """Analizza la distribuzione completa del dataset"""
    
    print("🔍 ANALISI DETTAGLIATA DEL DATASET")
    print("=" * 70)
    
    # Carica CSV
    root_dir = Path(__file__).resolve().parents[2]
    csv_path = root_dir / "esperimenti.csv"
    
    if not csv_path.exists():
        print(f"❌ File CSV non trovato: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"📄 CSV caricato: {len(df)} esperimenti")
    print(f"📊 Colonne: {list(df.columns)}")
    
    # Carica dataset da Hugging Face
    print("\n📥 Caricamento dataset da Hugging Face...")
    ds = load_dataset("Nardellar/Esperimenti", split="train")
    print(f"✅ Dataset caricato: {len(ds)} immagini totali")
    
    # Mappa ID esperimento -> nome
    exp_id2name = {i: name for i, name in enumerate(ds.features["label"].names)}
    print(f"📋 Esperimenti nel dataset: {len(exp_id2name)}")
    
    # Analizza distribuzione per esperimento
    print("\n" + "=" * 70)
    print("📊 DISTRIBUZIONE IMMAGINI PER ESPERIMENTO")
    print("=" * 70)
    
    exp_counts = Counter(ds["label"])
    
    for exp_id, count in sorted(exp_counts.items()):
        exp_name = exp_id2name.get(exp_id, f"Unknown_{exp_id}")
        print(f"  {exp_name}: {count} immagini")
    
    # Analizza per attributo TEMPERATURA
    print("\n" + "=" * 70)
    print("📊 ANALISI PER TEMPERATURA")
    print("=" * 70)
    
    attr_map = df.set_index("ID")["temperatura"].to_dict()
    
    # Raggruppa per temperatura
    temp_distribution = defaultdict(lambda: {"experiments": [], "total_images": 0, "images_per_exp": []})
    
    for exp_id, count in exp_counts.items():
        exp_name = exp_id2name.get(exp_id, None)
        if exp_name:
            temp = attr_map.get(exp_name, None)
            if temp:
                temp_distribution[temp]["experiments"].append(exp_name)
                temp_distribution[temp]["total_images"] += count
                temp_distribution[temp]["images_per_exp"].append((exp_name, count))
    
    for temp in sorted(temp_distribution.keys()):
        data = temp_distribution[temp]
        print(f"\n🌡️  TEMPERATURA {temp}°C:")
        print(f"  📁 Numero esperimenti: {len(data['experiments'])}")
        print(f"  📊 Totale immagini: {data['total_images']}")
        print(f"  📈 Media immagini per esperimento: {data['total_images'] / len(data['experiments']):.1f}")
        print(f"  📋 Esperimenti:")
        for exp_name, count in sorted(data['images_per_exp']):
            print(f"     - {exp_name}: {count} immagini")
    
    # Analizza TUTTI gli attributi
    print("\n" + "=" * 70)
    print("📊 ANALISI PER TUTTI GLI ATTRIBUTI")
    print("=" * 70)
    
    attributes = [col for col in df.columns if col.lower() not in ["id", "esperimenti"]]
    
    for attribute in attributes:
        print(f"\n📌 ATTRIBUTO: {attribute.upper()}")
        attr_map = df.set_index("ID")[attribute].to_dict()
        
        attr_distribution = defaultdict(int)
        
        for exp_id, count in exp_counts.items():
            exp_name = exp_id2name.get(exp_id, None)
            if exp_name:
                attr_value = attr_map.get(exp_name, None)
                if attr_value:
                    attr_distribution[attr_value] += count
        
        for value in sorted(attr_distribution.keys()):
            count = attr_distribution[value]
            print(f"  {value}: {count} immagini")
    
    # Verifica split train/val
    print("\n" + "=" * 70)
    print("📊 SIMULAZIONE SPLIT TRAIN/VAL (80/20)")
    print("=" * 70)
    
    total_images = len(ds)
    train_size = int(total_images * 0.8)
    val_size = total_images - train_size
    
    print(f"📊 Totale immagini: {total_images}")
    print(f"🏋️  Training set (80%): {train_size} immagini")
    print(f"✅ Validation set (20%): {val_size} immagini")
    
    print("\n📌 Per temperatura (stima con split 80/20):")
    for temp in sorted(temp_distribution.keys()):
        total = temp_distribution[temp]["total_images"]
        train_est = int(total * 0.8)
        val_est = total - train_est
        print(f"  Temperatura {temp}°C:")
        print(f"    - Totale: {total} immagini")
        print(f"    - Training: ~{train_est} immagini")
        print(f"    - Validation: ~{val_est} immagini")
    
    # Verifica esperimenti con meno di 100 immagini
    print("\n" + "=" * 70)
    print("⚠️  ESPERIMENTI CON MENO DI 100 IMMAGINI")
    print("=" * 70)
    
    incomplete_experiments = []
    for exp_id, count in sorted(exp_counts.items()):
        if count < 100:
            exp_name = exp_id2name.get(exp_id, f"Unknown_{exp_id}")
            temp = attr_map.get(exp_name, "N/A")
            incomplete_experiments.append((exp_name, count, temp))
    
    if incomplete_experiments:
        print(f"📉 Trovati {len(incomplete_experiments)} esperimenti con meno di 100 immagini:")
        for exp_name, count, temp in incomplete_experiments:
            print(f"  - {exp_name}: {count} immagini (Temperatura: {temp})")
    else:
        print("✅ Tutti gli esperimenti hanno 100 immagini")
    
    # Verifica esperimenti con più di 100 immagini
    print("\n" + "=" * 70)
    print("📈 ESPERIMENTI CON PIÙ DI 100 IMMAGINI")
    print("=" * 70)
    
    extra_experiments = []
    for exp_id, count in sorted(exp_counts.items()):
        if count > 100:
            exp_name = exp_id2name.get(exp_id, f"Unknown_{exp_id}")
            temp = attr_map.get(exp_name, "N/A")
            extra_experiments.append((exp_name, count, temp))
    
    if extra_experiments:
        print(f"📈 Trovati {len(extra_experiments)} esperimenti con più di 100 immagini:")
        for exp_name, count, temp in extra_experiments:
            print(f"  - {exp_name}: {count} immagini (Temperatura: {temp})")
    else:
        print("✅ Nessun esperimento ha più di 100 immagini")
    
    # Statistiche finali
    print("\n" + "=" * 70)
    print("📊 STATISTICHE FINALI")
    print("=" * 70)
    
    counts_list = list(exp_counts.values())
    print(f"📊 Numero totale esperimenti: {len(exp_counts)}")
    print(f"📊 Totale immagini: {sum(counts_list)}")
    print(f"📈 Media immagini per esperimento: {sum(counts_list) / len(counts_list):.2f}")
    print(f"📉 Min immagini per esperimento: {min(counts_list)}")
    print(f"📈 Max immagini per esperimento: {max(counts_list)}")
    print(f"📊 Mediana immagini per esperimento: {sorted(counts_list)[len(counts_list)//2]}")

if __name__ == "__main__":
    analyze_dataset_distribution()
