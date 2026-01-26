import joblib
import pandas as pd
import streamlit as st
import numpy as np
from Data_loading_preprocessing.feature_engineering import encode_conference, encode_playoff_results, drop_columns
from Data_loading_preprocessing.preprocessing import scale_features

# Streamlit UI
st.title("🏀 NBA Playoff Prediction")
#User selection of the sorted score
labels = ['Conf. Semi-Finalist', 'Conf. Finalist', 'NBA Finalist', 'NBA Champion']
# Regularize the probabilities with a temperature scaling
sort_col = st.selectbox("Select the score to sort the teams:", labels)
ascending = st.toggle("Ascending order", value=False)

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

T=3
prob_scaled = (y_prob**(1/T))/((y_prob**(1/T))+((1-y_prob)**(1/T)))

df = pd.DataFrame(prob_scaled, columns=labels, index=teams)
df.index.name = "Team"

#Sort dataframe based on user selection
df_sorted = df.sort_values(by=sort_col, ascending=ascending)
# Display
st.dataframe(df_sorted, use_container_width=True)