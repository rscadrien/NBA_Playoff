import joblib
import pandas as pd
import streamlit as st

from Data_loading_preprocessing.feature_engineering import encode_conference, encode_playoff_results
from Data_loading_preprocessing.preprocessing import scale_features

# Streamlit UI
st.title("🏀 NBA Playoff Prediction")

# User Inputs
conference = st.selectbox("Select the conference of the team:", ['East', 'West'])
season_record = st.number_input("What is the season record of the team?", min_value=0.0, max_value=1.0, step=0.01)
conf_seed = st.number_input("What is the seed of the team in its conference?", min_value=1, max_value=15, step=1)
nba_seed = st.number_input("What is the seed of the team in all the NBA?", min_value=1, max_value=30, step=1)
record_20 = st.number_input("What is the record of the team in the last 20 games?", min_value=0.0, max_value=1.0, step=0.01)
ortg_rank = st.number_input("What is the ranking of the team in offensive rating?", min_value=1, max_value=30, step=1)
drtg_rank = st.number_input("What is the ranking of the team in defensive rating?", min_value=1, max_value=30, step=1)

playoff_options = ['No Playoff', 'First Round', 'Conference Semi-Final', 'Conference Final', 'NBA Final', 'NBA Champion']
result_2_season = st.selectbox("What was the result of the team 2 seasons ago?", playoff_options)
result_last_season = st.selectbox("What was the result of the team last season?", playoff_options)

# Prediction Button
if st.button("Predict Playoff Outcome"):
    # Create input DataFrame
    X = pd.DataFrame([{
        'Conference': conference,
        'Season record': season_record,
        'Conf. Seed': conf_seed,
        'NBA Seed': nba_seed,
        'Last 20 Games record': record_20,
        'ORtg Rank': ortg_rank,
        'DRtg Rank': drtg_rank,
        '2 seasons ago result': result_2_season,
        'Last season result': result_last_season
    }])

    # Encode and scale input
    X = encode_conference(X, 'encoder_conference.joblib', mode='eval')
    X = encode_playoff_results(X)
    scaling_cols = ['Conf. Seed', 'NBA Seed', 'ORtg Rank', 'DRtg Rank']
    X = scale_features(X, scaling_cols, 'scaler_seed_rank.joblib', mode='eval')

    # Load model and predict
    model = joblib.load('NBA.joblib')
    st.write(f"Model type: {type(model)}")

    # Print each classifier in the chain
    for i, clf in enumerate(model.estimators_):
        st.write(f"Classifier #{i}: {type(clf)}")
        if hasattr(clf, "classes_"):
            st.write(f"Classifier #{i} is fitted. Classes: {clf.classes_}")
        else:
            st.error(f"Classifier #{i} is NOT fitted properly.")
    y_prob = model.predict_proba(X)

    # Display probabilities
    st.subheader("Prediction Probabilities:")
    labels = ['Conf. Semi-Finalist', 'Conf. Finalist', 'NBA Finalist', 'NBA Champion']
    for i, label in enumerate(labels):
        st.write(f"**{label}**: {y_prob[0][i]:.3f}")