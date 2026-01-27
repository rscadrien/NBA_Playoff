import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
import numpy as np

def evaluate_model(model, X, Y):
    # Get cross-validated predictions
    Y_pred = cross_val_predict(model, X, Y, cv=5)
    rmse = np.sqrt(mean_squared_error(Y, Y_pred))
    r2 = r2_score(Y, Y_pred)
    print(f'Cross-validated RMSE: {rmse}')
    print(f'Cross-validated R^2: {r2}')
    