from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from starter.starter.ml.model import train_model
from starter.starter.train_model import read_data_into_df

dummy_arr = np.zeros(shape=(5, 2), dtype=np.uint8)


def test_model_type():
    model = train_model(dummy_arr, dummy_arr)

    assert isinstance(model, RandomForestClassifier)


def test_predictions_type():
    model = train_model(dummy_arr, dummy_arr)
    inference = model.predict(dummy_arr)

    assert isinstance(inference, np.ndarray)


def test_data_type():
    current_test_file_path = Path(__file__)

    # 2. Get the directory containing this test file using .parent
    test_dir = current_test_file_path.parent.parent.parent
    path = test_dir / "data/census.csv"
    data = read_data_into_df(path)

    assert isinstance(data, pd.DataFrame)
