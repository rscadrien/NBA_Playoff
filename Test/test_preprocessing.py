import pandas as pd
import numpy as np
import pytest
import joblib
from pathlib import Path
from Data_loading_preprocessing.preprocessing import scale_features  # Adjust import to your actual module path

# ---------- scale_features tests ----------

def test_scale_features_train_eval(tmp_path):
    df_train = pd.DataFrame({
        'A': [0, 5, 10],
        'B': [10, 15, 20],
        'C': ['X', 'Y', 'Z']  # Unused column
    })
    scaler_path = tmp_path / "scaler.pkl"

    # Train mode
    scaled_train = scale_features(df_train.copy(), ['A', 'B'], str(scaler_path), mode='train')
    
    assert np.allclose(scaled_train['A'], [0.0, 0.5, 1.0])
    assert np.allclose(scaled_train['B'], [0.0, 0.5, 1.0])
    assert scaler_path.exists()

    # Eval mode
    df_eval = pd.DataFrame({
        'A': [2.5, 7.5],
        'B': [12.5, 17.5],
        'C': ['foo', 'bar']
    })
    scaled_eval = scale_features(df_eval.copy(), ['A', 'B'], str(scaler_path), mode='eval')

    assert np.allclose(scaled_eval['A'], [0.25, 0.75])
    assert np.allclose(scaled_eval['B'], [0.25, 0.75])

def test_scale_features_invalid_mode():
    df = pd.DataFrame({'A': [0, 1], 'B': [2, 3]})
    with pytest.raises(ValueError, match="Mode must be either 'train' or 'eval'"):
        scale_features(df, ['A', 'B'], "scaler.pkl", mode='invalid')

def test_scale_features_missing_column(tmp_path):
    df = pd.DataFrame({'A': [0, 1]})  # Missing 'B'
    scaler_path = tmp_path / "scaler.pkl"

    with pytest.raises(KeyError):
        scale_features(df, ['A', 'B'], str(scaler_path), mode='train')

def test_scale_features_eval_invalid_scaler(tmp_path):
    df = pd.DataFrame({'A': [0, 1], 'B': [2, 3]})
    scaler_path = tmp_path / "missing_scaler.pkl"
    
    with pytest.raises(FileNotFoundError):
        scale_features(df, ['A', 'B'], str(scaler_path), mode='eval')