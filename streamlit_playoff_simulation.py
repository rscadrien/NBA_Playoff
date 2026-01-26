import joblib
import pandas as pd
import streamlit as st
import numpy as np
from Data_loading_preprocessing.feature_engineering import encode_conference, encode_playoff_results, drop_columns
from Data_loading_preprocessing.preprocessing import scale_features

st.title("🏀 NBA Playoff Prediction")

# ---------- Load data and model once ----------
@st.cache_data
def load_data_and_model():
    X_ini = pd.read_csv('./Data/Current_NBA_Season.csv')
    model = joblib.load('NBA.joblib')
    return X_ini, model

X_ini, model = load_data_and_model()
teams = X_ini['Team'].tolist()

# ---------- Button 1: Predict Global Playoff Outcomes ----------
st.subheader("Predict Global Playoff outcomes for all NBA teams:")
if st.button("Predict Global Playoff Outcomes"):
    X = X_ini.copy()
    # Encode and scale
    X = encode_conference(X, 'encoder_conference.joblib', mode='eval')
    X = encode_playoff_results(X)
    X = drop_columns(X)
    scaling_cols = ['Conf. Seed', 'NBA Seed', 'ORtg Rank', 'DRtg Rank']
    X = scale_features(X, scaling_cols, 'scaler_seed_rank.joblib', mode='eval')
    
    # Predict probabilities
    y_prob = model.predict_proba(X)
    
    # Store in session_state for later
    st.session_state['y_prob'] = y_prob
    st.session_state['X_ini'] = X_ini

    # Temperature scaling
    T = 2
    prob_scaled = (y_prob**(1/T))/((y_prob**(1/T)) + ((1-y_prob)**(1/T)))
    
    # Display results
    labels = ['Conf. Semi-Finalist', 'Conf. Finalist', 'NBA Finalist', 'NBA Champion']
    df = pd.DataFrame(prob_scaled, columns=labels, index=teams)
    df.index.name = "Team"
    st.dataframe(df, use_container_width=True)

# ---------- Button 2: Run Playoff Simulations ----------
st.subheader("Run Playoff simulations:")
T = st.number_input("Upset factor (higher = more upsets)", min_value=1.0, max_value=4.0, step=0.5, value=2.0)
N = st.number_input("Number of simulations to run", min_value=10, max_value=1000, step=1, value=10)

# Initialize session state for simulations if not already
if 'all_simulations' not in st.session_state:
    st.session_state['all_simulations'] = []
if 'Number_championships' not in st.session_state:
    st.session_state['Number_championships'] = {team: 0 for team in teams}

if st.button("Run Playoff Simulations"):
    if 'y_prob' not in st.session_state:
        st.error("Please first calculate the playoff probabilities!")
    else:
        y_prob = st.session_state['y_prob']
        X_ini = st.session_state['X_ini']

        # Define brackets
        East_numbers = list(range(10))
        West_numbers = list(range(10, 20))

        # Helper function: simulate a single conference and return all round winners
        def simulate_conference(numbers):
            rounds = {}

            # First round
            winners_round1 = []
            matchups = [[0,7],[1,6],[2,5],[3,4]]
            for i,j in matchups:
                p_i = y_prob[numbers[i]][0]
                p_j = y_prob[numbers[j]][0]
                p_i_scaled = (p_i**(1/T))/((p_i**(1/T))+((1-p_i)**(1/T)))
                p_j_scaled = (p_j**(1/T))/((p_j**(1/T))+((1-p_j)**(1/T)))
                total = p_i_scaled + p_j_scaled
                winner = np.random.choice([numbers[i], numbers[j]], p=[p_i_scaled/total, p_j_scaled/total])
                winners_round1.append(winner)
            rounds['First Round'] = winners_round1

            # Semi-finals
            winners_round2 = []
            semi_matchups = [[0,3],[1,2]]
            for i,j in semi_matchups:
                p_i = y_prob[winners_round1[i]][1]
                p_j = y_prob[winners_round1[j]][1]
                p_i_scaled = (p_i**(1/T))/((p_i**(1/T))+((1-p_i)**(1/T)))
                p_j_scaled = (p_j**(1/T))/((p_j**(1/T))+((1-p_j)**(1/T)))
                total = p_i_scaled + p_j_scaled
                winner = np.random.choice([winners_round1[i], winners_round1[j]], p=[p_i_scaled/total, p_j_scaled/total])
                winners_round2.append(winner)
            rounds['Semi-Finals'] = winners_round2

            # Conference final
            p_i = y_prob[winners_round2[0]][2]
            p_j = y_prob[winners_round2[1]][2]
            p_i_scaled = (p_i**(1/T))/((p_i**(1/T))+((1-p_i)**(1/T)))
            p_j_scaled = (p_j**(1/T))/((p_j**(1/T))+((1-p_j)**(1/T)))
            total = p_i_scaled + p_j_scaled
            winner = np.random.choice([winners_round2[0], winners_round2[1]], p=[p_i_scaled/total, p_j_scaled/total])
            rounds['Conference Final'] = winner

            return rounds

        # Run simulations
        all_simulations = []
        Number_championships = {team: 0 for team in teams}  # reset counts for this run

        for sim in range(N):
            sim_result = {}

            # Simulate conferences
            winners_East = simulate_conference(East_numbers)
            winners_West = simulate_conference(West_numbers)
            sim_result['East'] = winners_East
            sim_result['West'] = winners_West

            # NBA Final
            winner_East = winners_East['Conference Final']
            winner_West = winners_West['Conference Final']
            p_E = y_prob[winner_East][3]
            p_W = y_prob[winner_West][3]
            p_E_scaled = (p_E**(1/T))/((p_E**(1/T))+((1-p_E)**(1/T)))
            p_W_scaled = (p_W**(1/T))/((p_W**(1/T))+((1-p_W)**(1/T)))
            total = p_E_scaled + p_W_scaled
            winner_NBA = np.random.choice([winner_East, winner_West], p=[p_E_scaled/total, p_W_scaled/total])
            sim_result['NBA Final'] = (winner_East, winner_West, winner_NBA)

            Number_championships[X_ini['Team'][winner_NBA]] += 1
            all_simulations.append(sim_result)

        # Save results in session state
        st.session_state['all_simulations'] = all_simulations
        st.session_state['Number_championships'] = Number_championships

        # Display championship summary
        st.subheader("🏆 NBA Championship Results after Simulations:")
        Number_championships_sorted = dict(sorted(Number_championships.items(), key=lambda item: item[1], reverse=True))
        for team, wins in Number_championships_sorted.items():
            st.write(f"**{team}**: {wins} championships ({(wins/N)*100:.2f}%)")

# ---------- Select a simulation to view the bracket ----------
if 'all_simulations' in st.session_state and st.session_state['all_simulations']:
    sim_number = st.number_input(
        "Choose simulation number to see the bracket",
        min_value=1,
        max_value=len(st.session_state['all_simulations']),
        step=1
    )

    if sim_number:
        sim_result = st.session_state['all_simulations'][sim_number - 1]
        st.write("### 🏟️ East Conference")
        st.write("First round")
        st.write(f"Matchup 1: {X_ini['Team'][East_numbers[0]]} vs {X_ini['Team'][East_numbers[7]]}")
        st.write(f" Winner -> {X_ini['Team'][sim_result['East']['First Round'][0]]}")
        st.write(f"Matchup 2: {X_ini['Team'][East_numbers[1]]} vs {X_ini['Team'][East_numbers[6]]} ")
        st.write(f" Winner -> {X_ini['Team'][sim_result['East']['First Round'][1]]}")
        st.write(f"Matchup 3: {X_ini['Team'][East_numbers[2]]} vs {X_ini['Team'][East_numbers[5]]}")
        st.write(f" Winner -> {X_ini['Team'][sim_result['East']['First Round'][2]]}")
        st.write(f"Matchup 4: {X_ini['Team'][East_numbers[3]]} vs {X_ini['Team'][East_numbers[4]]}")
        st.write(f" Winner -> {X_ini['Team'][sim_result['East']['First Round'][3]]}")
        st.write("Conference Semi-Final")
        st.write(f"Matchup 1: {X_ini['Team'][sim_result['East']['First Round'][0]]} vs {X_ini['Team'][sim_result['East']['First Round'][3]]}")
        st.write(f" Winner -> {X_ini['Team'][sim_result['East']['Semi-Finals'][0]]}")
        st.write(f"Matchup 2: {X_ini['Team'][sim_result['East']['First Round'][1]]} vs {X_ini['Team'][sim_result['East']['First Round'][2]]}")
        st.write(f" Winner -> {X_ini['Team'][sim_result['East']['Semi-Finals'][1]]}")
        st.write("Conference Final")
        st.write(f"Matchup: {X_ini['Team'][sim_result['East']['Semi-Finals'][0]]} vs {X_ini['Team'][sim_result['East']['Semi-Finals'][1]]}")
        st.write(f" Winner -> {X_ini['Team'][sim_result['East']['Conference Final']]}")
        st.write("### 🏟️ West Conference")
        st.write(sim_result['West'])
        st.write("### 🏆 NBA Final")
        st.write(f"{X_ini['Team'][sim_result['NBA Final'][0]]} vs {X_ini['Team'][sim_result['NBA Final'][1]]} -> Winner: {X_ini['Team'][sim_result['NBA Final'][2]]}")