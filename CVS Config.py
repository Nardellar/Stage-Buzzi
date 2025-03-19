import pandas as pd

# Dati sperimentali in un dizionario
class_attributes = {
    "Esperimenti": [
        {"ID": "EXP01", "Temperatura": 1400, "Tempo": None, "Rampa": None, "Raffreddamento": 0},
        {"ID": "EXP02", "Temperatura": 1400, "Tempo": None, "Rampa": None, "Raffreddamento": 1},
        {"ID": "EXP03", "Temperatura": 1400, "Tempo": None, "Rampa": None, "Raffreddamento": None},
        {"ID": "EXP04", "Temperatura": 1400, "Tempo": None, "Rampa": None, "Raffreddamento": None},
        {"ID": "EXP05", "Temperatura": 1500, "Tempo": 45, "Rampa": 40, "Raffreddamento": 1},
        {"ID": "EXP06", "Temperatura": 1300, "Tempo": None, "Rampa": None, "Raffreddamento": None},
        {"ID": "EXP07", "Temperatura": None, "Tempo": None, "Rampa": None, "Raffreddamento": 1},
        {"ID": "EXP08", "Temperatura": None, "Tempo": None, "Rampa": None, "Raffreddamento": 0},
        {"ID": "EXP09", "Temperatura": 1300, "Tempo": None, "Rampa": None, "Raffreddamento": None},
        {"ID": "EXP10", "Temperatura": 1500, "Tempo": 45, "Rampa": 10, "Raffreddamento": 0},
        {"ID": "EXP11", "Temperatura": 1500, "Tempo": 15, "Rampa": 40, "Raffreddamento": 0},
        {"ID": "EXP12", "Temperatura": 1300, "Tempo": None, "Rampa": None, "Raffreddamento": None},
        {"ID": "EXP13", "Temperatura": 1500, "Tempo": 15, "Rampa": 10, "Raffreddamento": 1},
        {"ID": "EXP14", "Temperatura": 1300, "Tempo": None, "Rampa": None, "Raffreddamento": None}
    ]
}

# Creiamo il DataFrame a partire dal dizionario
df = pd.DataFrame(class_attributes["Esperimenti"])
# Salviamo il DataFrame normalizzato su un file CSV (opzionale, se desideri averlo su disco)
df.to_csv("esperimenti.csv", index=False)