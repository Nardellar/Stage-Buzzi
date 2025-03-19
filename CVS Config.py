import pandas as pd

# Dati sperimentali in un dizionario
class_attributes = {
    "Esperimenti": [
        {"ID": "EXP01", "temperatura": 1400, "tempo": None, "rampa": None, "raffreddamento": 0},
        {"ID": "EXP02", "temperatura": 1400, "tempo": None, "rampa": None, "raffreddamento": 1},
        {"ID": "EXP03", "temperatura": 1400, "tempo": None, "rampa": None, "raffreddamento": None},
        {"ID": "EXP04", "temperatura": 1400, "tempo": None, "rampa": None, "raffreddamento": None},
        {"ID": "EXP05", "temperatura": 1500, "tempo": 45, "rampa": 40, "raffreddamento": 1},
        {"ID": "EXP06", "temperatura": 1300, "tempo": None, "rampa": None, "raffreddamento": None},
        {"ID": "EXP07", "temperatura": None, "tempo": None, "rampa": None, "raffreddamento": 1},
        {"ID": "EXP08", "temperatura": None, "tempo": None, "rampa": None, "raffreddamento": 0},
        {"ID": "EXP09", "temperatura": 1300, "tempo": None, "rampa": None, "raffreddamento": None},
        {"ID": "EXP10", "temperatura": 1500, "tempo": 45, "rampa": 10, "raffreddamento": 0},
        {"ID": "EXP11", "temperatura": 1500, "tempo": 15, "rampa": 40, "raffreddamento": 0},
        {"ID": "EXP12", "temperatura": 1300, "tempo": None, "rampa": None, "raffreddamento": None},
        {"ID": "EXP13", "temperatura": 1500, "tempo": 15, "rampa": 10, "raffreddamento": 1},
        {"ID": "EXP14", "temperatura": 1300, "tempo": None, "rampa": None, "raffreddamento": None}
    ]
}

# Creiamo il DataFrame a partire dal dizionario
df = pd.DataFrame(class_attributes["Esperimenti"])
# Salviamo il DataFrame normalizzato su un file CSV (opzionale, se desideri averlo su disco)
df.to_csv("esperimenti.csv", index=False)