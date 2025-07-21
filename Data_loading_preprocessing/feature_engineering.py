import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import joblib

def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    df['Champion'] = np.where(df['Playoff Outcome'] == 'NBA Champion', 1, 0)
    df['Finalist'] = np.where(df['Playoff Outcome'].isin(['NBA Champion', 'NBA Final']), 1, 0)
    df['Conf_Finalist'] = np.where(df['Playoff Outcome'].isin(['NBA Champion', 'NBA Final', 'Conference Final']), 1, 0)
    df['Conf_SemiFinalist'] = np.where(df['Playoff Outcome'].isin(['NBA Champion', 'NBA Final', 'Conference Final', 'Conference Semi-Final']), 1, 0)
    return df.drop(columns=['Playoff Outcome'])

def encode_conference(df: pd.DataFrame, encoder_path: str, mode: str = 'train') -> pd.DataFrame:
    if mode == 'train':
        encoder = OrdinalEncoder(categories=[['Est', 'West']])
        df['Conference'] = encoder.fit_transform(df[['Conference']])
        joblib.dump(encoder, encoder_path)
    elif mode == 'eval':
        encoder = joblib.load(encoder_path)
        df['Conference'] = encoder.transform(df[['Conference']])
    else:
        raise ValueError("Mode must be either 'train' or 'eval'")
    return df

def encode_playoff_results(df: pd.DataFrame) -> pd.DataFrame:
    result_mapping = {
        'No Playoff': 0,
        'First Round': 1/16,
        'Conference Semi-Final': 1/8,
        'Conference Final': 1/4,
        'NBA Final': 1/2,
        'NBA Champion': 1
    }
    df['2 seasons ago result (numeric)'] = df['2 seasons ago result'].map(result_mapping)
    df['Last season result (numeric)'] = df['Last season result'].map(result_mapping)
    return df.drop(columns=['2 seasons ago result', 'Last season result'])

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=['Season', 'Team'])