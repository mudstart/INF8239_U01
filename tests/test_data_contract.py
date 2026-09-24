import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "dataset.csv"

TARGET = "Air Quality"                            
REQUIRED = {TARGET, "Temperature", "Humidity"}    

def load_data():
    return pd.read_csv(DATA_PATH)

def test_dataset_is_not_empty():
    assert not load_data().empty

def test_required_columns_exist():
    assert REQUIRED <= set(load_data().columns)

def test_target_has_no_missing_and_two_classes():
    y = load_data()[TARGET]
    assert y.notna().all()
    assert y.nunique() >= 2