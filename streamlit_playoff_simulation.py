import joblib
import pandas as pd
import streamlit as st
import numpy as np
from Data_loading_preprocessing.feature_engineering import encode_conference, encode_playoff_results, drop_columns
from Data_loading_preprocessing.preprocessing import scale_features

# Streamlit UI
st.title("🏀 NBA Playoff Prediction")
if st.button("Predict Global Playoff Outcomes"):
    labels = ['Conf. Semi-Finalist', 'Conf. Finalist', 'NBA Finalist', 'NBA Champion']
    #Read the current NBA season data
    X_ini = pd.read_csv('./Data/Current_NBA_Season.csv')
    X = X_ini.copy()
    teams = X['Team']
    # Encode and scale input
    X = encode_conference(X, 'encoder_conference.joblib', mode='eval')
    X = encode_playoff_results(X)
    X = drop_columns(X)
    scaling_cols = ['Conf. Seed', 'NBA Seed', 'ORtg Rank', 'DRtg Rank']
    X = scale_features(X, scaling_cols, 'scaler_seed_rank.joblib', mode='eval')
    # Load model and predict
    model = joblib.load('NBA.joblib')
    y_prob = model.predict_proba(X)
    # Regularize the probabilities with a temperature scaling
    T=3
    prob_scaled = (y_prob**(1/T))/((y_prob**(1/T))+((1-y_prob)**(1/T)))
    #Convert  the results in a dataframe
    df = pd.DataFrame(prob_scaled, columns=labels, index=teams)
    df.index.name = "Team"
    # Display
    st.dataframe(df, use_container_width=True)