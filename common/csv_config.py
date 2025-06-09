import pandas as pd
from pathlib import Path

# Dati sperimentali in un dizionario
class_attributes = {
    "Esperimenti": [
        {
            "ID": "EXP01",
            "rampa": 25,
            "temperatura": 1400,
            "tempo": 30,
            "raffreddamento": 0,
            "tempo rampa": 18,
            "tempo totale": 48,
        },
        {
            "ID": "EXP02",
            "rampa": 25,
            "temperatura": 1400,
            "tempo": 30,
            "raffreddamento": 1,
            "tempo rampa": 18,
            "tempo totale": 48,
        },
        {
            "ID": "EXP03",
            "rampa": 25,
            "temperatura": 1400,
            "tempo": 30,
            "raffreddamento": 0,
            "tempo rampa": 18,
            "tempo totale": 48,
        },
        {
            "ID": "EXP04",
            "rampa": 25,
            "temperatura": 1400,
            "tempo": 30,
            "raffreddamento": 1,
            "tempo rampa": 18,
            "tempo totale": 48,
        },
        {
            "ID": "EXP05",
            "rampa": 40,
            "temperatura": 1500,
            "tempo": 45,
            "raffreddamento": 1,
            "tempo rampa": 13.75,
            "tempo totale": 58.75,
        },
        {
            "ID": "EXP06",
            "rampa": 10,
            "temperatura": 1300,
            "tempo": 15,
            "raffreddamento": 0,
            "tempo rampa": 35,
            "tempo totale": 50,
        },
        {
            "ID": "EXP07",
            "rampa": 25,
            "temperatura": 1400,
            "tempo": 30,
            "raffreddamento": 1,
            "tempo rampa": 18,
            "tempo totale": 48,
        },
        {
            "ID": "EXP08",
            "rampa": 25,
            "temperatura": 1400,
            "tempo": 30,
            "raffreddamento": 0,
            "tempo rampa": 18,
            "tempo totale": 48,
        },
        {
            "ID": "EXP09",
            "rampa": 40,
            "temperatura": 1300,
            "tempo": 15,
            "raffreddamento": 1,
            "tempo rampa": 8.75,
            "tempo totale": 23.75,
        },
        {
            "ID": "EXP10",
            "rampa": 10,
            "temperatura": 1500,
            "tempo": 45,
            "raffreddamento": 0,
            "tempo rampa": 55,
            "tempo totale": 100,
        },
        {
            "ID": "EXP11",
            "rampa": 40,
            "temperatura": 1500,
            "tempo": 15,
            "raffreddamento": 0,
            "tempo rampa": 13.75,
            "tempo totale": 28.75,
        },
        {
            "ID": "EXP12",
            "rampa": 10,
            "temperatura": 1300,
            "tempo": 45,
            "raffreddamento": 1,
            "tempo rampa": 35,
            "tempo totale": 80,
        },
        {
            "ID": "EXP13",
            "rampa": 10,
            "temperatura": 1500,
            "tempo": 15,
            "raffreddamento": 1,
            "tempo rampa": 55,
            "tempo totale": 70,
        },
        {
            "ID": "EXP14",
            "rampa": 40,
            "temperatura": 1300,
            "tempo": 45,
            "raffreddamento": 0,
            "tempo rampa": 8.75,
            "tempo totale": 53.75,
        },
    ]
}


def create_csv(csv_path: str | Path | None = None) -> Path:
    """Generate the ``esperimenti.csv`` file.

    Parameters
    ----------
    csv_path: str | Path | None, optional
        Destination path for the CSV.  If ``None`` the file is written in the
        project root.

    Returns
    -------
    pathlib.Path
        The full path of the generated CSV file.
    """

    df = pd.DataFrame(class_attributes["Esperimenti"])

    if csv_path is None:
        csv_path = Path(__file__).resolve().parent.parent / "esperimenti.csv"
    else:
        csv_path = Path(csv_path)

    df.to_csv(csv_path, index=False)
    return csv_path


if __name__ == "__main__":
    path = create_csv()
    print(f"✅ File CSV creato: {path}")
