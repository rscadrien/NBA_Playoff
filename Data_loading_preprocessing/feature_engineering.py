import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import joblib

def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    result_mapping = {
        'No Playoff': 0,
        'First Round': 1/16,
        'Conference Semi-Final': 1/8,
        'Conference Final': 1/4,
        'NBA Final': 1/2,
        'NBA Champion': 1
    }
    df['Playoff Outcome (numeric)'] = df['Playoff Outcome'].map(result_mapping)
    return df.drop(columns=['Playoff Outcome'])

def encode_conference(df: pd.DataFrame, encoder_path: str, mode: str = 'train') -> pd.DataFrame:
    Conference_mapping ={
    'East': 0,
    'West': 1,
    }
    df['Conference'] = df['Conference'].map(Conference_mapping)    
    return df

def encode_playoff_results(df):
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