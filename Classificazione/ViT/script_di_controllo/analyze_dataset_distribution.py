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
    
    # Analizza TUTTI gli attributi
    print("\n" + "=" * 70)
    print("📊 ANALISI PER TUTTI GLI ATTRIBUTI")
    print("=" * 70)
    
    attributes = [col for col in df.columns if col.lower() not in ["id", "esperimenti"]]
    
    for attribute in attributes:
        print(f"\n📌 ATTRIBUTO: {attribute.upper()}")
        attr_map = df.set_index("ID")[attribute].to_dict()
        
        attr_distribution = defaultdict(int)
        total_attr_images = 0

        def sort_key(value):
            try:
                return (0, float(value))
            except (TypeError, ValueError):
                return (1, str(value))

        for exp_id, count in exp_counts.items():
            exp_name = exp_id2name.get(exp_id, None)
            if exp_name:
                attr_value = attr_map.get(exp_name, None)
                # Mantiene anche classi con valore 0 (es. raffreddamento=0) e scarta solo valori mancanti.
                if attr_value is None:
                    continue
                if isinstance(attr_value, float) and pd.isna(attr_value):
                    continue
                attr_distribution[attr_value] += count
                total_attr_images += count

        for value in sorted(attr_distribution.keys(), key=sort_key):
            count = attr_distribution[value]
            percentage = (count / total_attr_images * 100) if total_attr_images else 0.0
            print(f"  {value}: {count} immagini ({percentage:.2f}%)")

if __name__ == "__main__":
    analyze_dataset_distribution()
