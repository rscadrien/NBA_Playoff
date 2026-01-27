from sklearn.ensemble import RandomForestRegressor
import joblib

def train_model_regression(X_train, y_train):
    rf_params = {'max_depth': 5, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 100}
    rf_model = RandomForestRegressor(**rf_params)
    rf_model.fit(X_train, y_train)
    return rf_model

def save_model(model, path: str):
    joblib.dump(model, path)