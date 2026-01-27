import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import joblib

def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    # Define the mapping
    result_mapping = {
        'No Playoff': 0,
        'First Round': 0.2,
        'Conference Semi-Final': 0.4,
        'Conference Final': 0.6,
        'NBA Final': 0.8,
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
        'First Round': 0.2,
        'Conference Semi-Final': 0.4,
        'Conference Final': 0.6,
        'NBA Final': 0.8,
        'NBA Champion': 1
    }
    df['2 seasons ago result (numeric)'] = df['2 seasons ago result'].map(result_mapping)
    df['Last season result (numeric)'] = df['Last season result'].map(result_mapping)
    return df.drop(columns=['2 seasons ago result', 'Last season result'])

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=['Season', 'Team'])