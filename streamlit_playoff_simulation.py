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
    T = 3
    prob_scaled = (y_prob**(1/T))/((y_prob**(1/T)) + ((1-y_prob)**(1/T)))
    
    # Display results
    labels = ['Conf. Semi-Finalist', 'Conf. Finalist', 'NBA Finalist', 'NBA Champion']
    df = pd.DataFrame(prob_scaled, columns=labels, index=teams)
    df.index.name = "Team"
    st.dataframe(df, use_container_width=True)

# ---------- Button 2: Run Playoff Simulations ----------
st.subheader("Run Playoff simulations:")
T = st.number_input("Upset factor (higher = more upsets)", min_value=1.0, max_value=10.0, step=0.5, value=1.0)
N = st.number_input("Number of simulations to run", min_value=10, max_value=1000, step=1, value=10)
if st.button("Run Playoff Simulations"):
    if 'y_prob' not in st.session_state:
        st.error("Please first calculate the playoff probabilities!")
    else:
        y_prob = st.session_state['y_prob']
        X_ini = st.session_state['X_ini']


        # Define brackets
        East_numbers = list(range(10))
        West_numbers = list(range(10, 20))
        conferences = {'East': East_numbers, 'West': West_numbers}
        
        def simulate_conference(conf_name, numbers):
            st.write(f"🏟️ {conf_name} Conference")
            
            # First round
            winners_round1 = []
            matchups = [[0,7],[1,6],[2,5],[3,4]]  # 1v8, 2v7, 3v6, 4v5
            for i,j in matchups:
                p_i = y_prob[numbers[i]][0]
                p_j = y_prob[numbers[j]][0]
                # Temperature scaling
                p_i_scaled = (p_i**(1/T))/((p_i**(1/T))+((1-p_i)**(1/T)))
                p_j_scaled = (p_j**(1/T))/((p_j**(1/T))+((1-p_j)**(1/T)))
                # Renormalize
                total = p_i_scaled + p_j_scaled
                probs = [p_i_scaled/total, p_j_scaled/total]
                winner = np.random.choice([numbers[i], numbers[j]], p=probs)
                # st.write(f"{X_ini['Team'][numbers[i]]} vs {X_ini['Team'][numbers[j]]}")
                # st.write(f"Probabilities: {X_ini['Team'][numbers[i]]}: {probs[0]:.3f}, {X_ini['Team'][numbers[j]]}: {probs[1]:.3f}")
                # st.write(f"Winner: {X_ini['Team'][winner]}")
                winners_round1.append(winner)
            
            # Semi-finals
            winners_round2 = []
            semi_matchups = [[0,3],[1,2]]  # 1/8 winner vs 4/5 winner, 2/7 winner vs 3/6 winner
            for i,j in semi_matchups:
                p_i = y_prob[winners_round1[i]][1]
                p_j = y_prob[winners_round1[j]][1]
                p_i_scaled = (p_i**(1/T))/((p_i**(1/T))+((1-p_i)**(1/T)))
                p_j_scaled = (p_j**(1/T))/((p_j**(1/T))+((1-p_j)**(1/T)))
                total = p_i_scaled + p_j_scaled
                probs = [p_i_scaled/total, p_j_scaled/total]
                winner = np.random.choice([winners_round1[i], winners_round1[j]], p=probs)
                # st.write(f"Semi-Final: {X_ini['Team'][winners_round1[i]]} vs {X_ini['Team'][winners_round1[j]]}")
                # st.write(f"Probabilities: {X_ini['Team'][winners_round1[i]]}: {probs[0]:.3f}, {X_ini['Team'][winners_round1[j]]}: {probs[1]:.3f}")
                # st.write(f"Winner: {X_ini['Team'][winner]}")
                winners_round2.append(winner)
            
            # Conference Final
            p_i = y_prob[winners_round2[0]][2]
            p_j = y_prob[winners_round2[1]][2]
            p_i_scaled = (p_i**(1/T))/((p_i**(1/T))+((1-p_i)**(1/T)))
            p_j_scaled = (p_j**(1/T))/((p_j**(1/T))+((1-p_j)**(1/T)))
            total = p_i_scaled + p_j_scaled
            probs = [p_i_scaled/total, p_j_scaled/total]
            winner = np.random.choice([winners_round2[0], winners_round2[1]], p=probs)
            # st.write(f"Conference final: {X_ini['Team'][winners_round2[0]]} vs {X_ini['Team'][winners_round2[1]]}")
            # st.write(f"Probabilities: {X_ini['Team'][winners_round2[0]]}: {probs[0]:.3f}, {X_ini['Team'][winners_round2[1]]}: {probs[1]:.3f}")
            # st.write(f"Conference Final Winner: {X_ini['Team'][winner]}")
            return winner
        
        # Simulate both conferences
        Number_championships = {team:0 for team in teams}
        for sim in range(N):
            winner_East = simulate_conference("East", East_numbers)
            winner_West = simulate_conference("West", West_numbers)
        
            # NBA Final
            p_E = y_prob[winner_East][3]
            p_W = y_prob[winner_West][3]
            p_E_scaled = (p_E**(1/T))/((p_E**(1/T))+((1-p_E)**(1/T)))
            p_W_scaled = (p_W**(1/T))/((p_W**(1/T))+((1-p_W)**(1/T)))
            total = p_E_scaled + p_W_scaled
            probs = [p_E_scaled/total, p_W_scaled/total]
            winner_NBA = np.random.choice([winner_East, winner_West], p=probs)
            Number_championships[X_ini['Team'][winner_NBA]] += 1
        
        st.subheader("🏆 NBA Championship Results after Simulations:")
        for team, wins in Number_championships.items():
            st.write(f"**{team}**: {wins} championships ({(wins/N)*100:.2f}%)")

#            st.write(f"🏆 NBA Final: {X_ini['Team'][winner_East]} vs {X_ini['Team'][winner_West]}  -> Winner : {X_ini['Team'][winner_NBA]}")
#            st.write(f"Probabilities: {X_ini['Team'][winner_East]}: {probs[0]:.3f}, {X_ini['Team'][winner_West]}: {probs[1]:.3f}")
#            st.write(f"🏆 **NBA Champion: {X_ini['Team'][winner_NBA]}**")