import numpy as np
import pandas as pd
import pytest
import joblib
from pathlib import Path
from Model.model import train_model, save_model  # Adjust import to your project structure
from sklearn.multioutput import ClassifierChain
from sklearn.ensemble import RandomForestClassifier

def make_toy_data():
    X = pd.DataFrame({
        "feat1": [1, 2, 3, 4],
        "feat2": [0.5, 0.3, 0.8, 0.1],
        "feat3": [10, 20, 10, 30]
    })
    y = pd.DataFrame({
        "Champion": [0, 1, 0, 0],
        "Finalist": [0, 1, 0, 1],
        "Conf_Finalist": [1, 1, 0, 1],
        "Conf_SemiFinalist": [1, 1, 1, 1]
    })
    return X, y

def test_train_model_output_type():
    X, y = make_toy_data()
    model = train_model(X, y)
    assert isinstance(model, ClassifierChain)
    assert isinstance(model.base_estimator, RandomForestClassifier)

def test_train_model_predict_shape():
    X, y = make_toy_data()
    model = train_model(X, y)
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape

def test_save_model(tmp_path):
    X, y = make_toy_data()
    model = train_model(X, y)
    save_path = tmp_path / "model.pkl"
    save_model(model, str(save_path))
    
    assert save_path.exists()
    
    # Load and test that it's still a ClassifierChain
    loaded_model = joblib.load(save_path)
    assert isinstance(loaded_model, ClassifierChain)
    assert loaded_model.predict(X).shape == y.shape