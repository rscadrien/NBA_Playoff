import pytest
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from Data_loading_preprocessing.feature_engineering import (
    create_targets,
    encode_conference,
    encode_playoff_results,
    drop_columns
)

# ---------- create_targets tests ----------

def test_create_targets():
    df = pd.DataFrame({
        'Playoff Outcome': [
            'NBA Champion',
            'NBA Final',
            'Conference Final',
            'Conference Semi-Final',
            'First Round'
        ]
    })

    result = create_targets(df)

    assert 'Champion' in result.columns
    assert list(result['Champion']) == [1, 0, 0, 0, 0]
    assert list(result['Finalist']) == [1, 1, 0, 0, 0]
    assert list(result['Conf_Finalist']) == [1, 1, 1, 0, 0]
    assert list(result['Conf_SemiFinalist']) == [1, 1, 1, 1, 0]
    assert 'Playoff Outcome' not in result.columns

def test_create_targets_with_unexpected_value():
    df = pd.DataFrame({'Playoff Outcome': ['Unknown']})
    result = create_targets(df)
    assert all(result[col].iloc[0] == 0 for col in result.columns)

# ---------- encode_conference tests ----------

def test_encode_conference_train_eval(tmp_path):
    df_train = pd.DataFrame({'Conference': ['Est', 'West', 'Est']})
    encoder_path = tmp_path / "encoder.pkl"
    
    # Train mode
    df_encoded_train = encode_conference(df_train.copy(), str(encoder_path), mode='train')
    assert list(df_encoded_train['Conference']) == [0.0, 1.0, 0.0]
    assert encoder_path.exists()

    # Eval mode
    df_eval = pd.DataFrame({'Conference': ['West', 'Est']})
    df_encoded_eval = encode_conference(df_eval, str(encoder_path), mode='eval')
    assert list(df_encoded_eval['Conference']) == [1.0, 0.0]

def test_encode_conference_invalid_mode():
    df = pd.DataFrame({'Conference': ['Est']})
    with pytest.raises(ValueError):
        encode_conference(df, "some_path.pkl", mode='invalid')

# ---------- encode_playoff_results tests ----------

def test_encode_playoff_results():
    df = pd.DataFrame({
        '2 seasons ago result': ['NBA Champion', 'Conference Final', 'Unknown'],
        'Last season result': ['NBA Final', 'First Round', 'No Playoff']
    })
    result = encode_playoff_results(df)

    assert list(result['2 seasons ago result (numeric)']) == [1.0, 0.25, np.nan]
    assert list(result['Last season result (numeric)']) == [0.5, 1/16, 0.0]
    assert '2 seasons ago result' not in result.columns
    assert 'Last season result' not in result.columns

# ---------- drop_columns tests ----------

def test_drop_columns():
    df = pd.DataFrame({
        'Season': [2022, 2023],
        'Team': ['A', 'B'],
        'Other': [1, 2]
    })
    result = drop_columns(df)
    assert 'Season' not in result.columns
    assert 'Team' not in result.columns
    assert 'Other' in result.columns

def test_drop_columns_missing():
    df = pd.DataFrame({'Other': [1, 2]})
    with pytest.raises(KeyError):
        drop_columns(df)