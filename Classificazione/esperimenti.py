from pathlib import Path
import pandas as pd

from . import csv_config

# Root directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent
# Path to the single esperimenti table used across the project
CSV_FILE = BASE_DIR / "esperimenti.csv"


def get_dataframe() -> pd.DataFrame:
    """Return the esperimenti table as a :class:`pandas.DataFrame`.

    The CSV file is created automatically if it does not exist.
    """
    if not CSV_FILE.exists():
        csv_config.create_csv(CSV_FILE)
    return pd.read_csv(CSV_FILE)
