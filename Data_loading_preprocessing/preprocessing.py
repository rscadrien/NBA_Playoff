from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import joblib

def scale_features(df: pd.DataFrame, columns: list, scaler_path: str, mode: str = 'train') -> pd.DataFrame:
    if mode == 'train':
        scaler = MinMaxScaler()
        df[columns] = scaler.fit_transform(df[columns])
        joblib.dump(scaler, scaler_path)
    elif mode == 'eval':
        scaler = joblib.load(scaler_path)
        df[columns] = scaler.transform(df[columns])
    else:
        raise ValueError("Mode must be either 'train' or 'eval'")
    return df