import joblib
import pandas as pd
import streamlit as st
import numpy as np
import random
from Data_loading_preprocessing.feature_engineering import encode_conference, encode_playoff_results, drop_columns
from Data_loading_preprocessing.preprocessing import scale_features
from display_bracket import display_bracket

st.title("🏀 NBA Playoff Simulator powered by ML/AI")
st.markdown("""
This app uses a machine learning model trained on 40 years of NBA playoff data to estimate the playoff strength of each team in the current season.

The model assigns a numerical strength score to each team, which is then converted into probabilities of winning a playoff series between any two teams. Using these probabilities, the app can simulate the entire NBA playoffs multiple times, accounting for upsets and randomness.

You can explore the predicted playoff outcomes, visualize potential brackets, and see the likelihood of each team reaching each stage—from the Conference Semi-Finals all the way to NBA Champion.

Adjust the Upset Factor to make surprises more or less likely, and run multiple simulations to get a robust view of playoff dynamics.

### How to use this app:
1. Edit the current NBA season data if needed.
2. Adjust the **Upset factor** and **Number of simulations**, then click **Run Playoff Simulations**.
3. Choose a team and round to view an example bracket where that team reaches the selected round.

You can find more details on the model in the medium article: https://medium.com/@adriendevolder/nba-playoff-simulator-powered-by-machine-learning-7b6ac62b7198
""")
# ---------- Load data and model once ----------
@st.cache_data
def load_data_and_model():
    X_ini = pd.read_csv('./Data/Current_NBA_Season.csv')
    model = joblib.load('NBA.joblib')
    return X_ini, model
st.write("Current NBA Teams Data:")
X_ini, model = load_data_and_model()
teams = X_ini['Team'].tolist()
st.session_state['X_ini'] = X_ini
st.subheader("Current NBA season data")

edited_df = st.data_editor(
    st.session_state["X_ini"],
    use_container_width=True,
    num_rows="fixed",   # or "dynamic" if you want row add/delete
)
st.session_state['X_ini'] = edited_df
X = X_ini.copy()
# Encode and scale
X = encode_conference(X, 'encoder_conference.joblib', mode='eval')
X = encode_playoff_results(X)
X = drop_columns(X)
scaling_cols = ['Conf. Seed', 'NBA Seed', 'ORtg Rank', 'DRtg Rank']
X = scale_features(X, scaling_cols, 'scaler_seed_rank.joblib', mode='eval')
    
# Predict probabilities
y = model.predict(X)
    
# Store in session_state for later
st.session_state['y'] = y
st.session_state['X_ini'] = X_ini

# ---------- Run Playoff Simulations ----------
st.subheader("Run Playoff simulations:")
T = st.number_input("Upset factor (higher = more upsets)", min_value=0.0, max_value=5.0, step=0.1, value=0.8,
                    help="0 = no upsets, 1 = some randomness, higher values = more likely upsets")
N = st.number_input("Number of simulations to run", min_value=100, max_value=100000, step=1, value=1000)

if st.button("Run Playoff Simulations"):
    # Initialize session state for simulations if not already
    if 'all_simulations' not in st.session_state:
        st.session_state['all_simulations'] = []
    if 'Number_championships' not in st.session_state:
        st.session_state['Number_championships'] = {team: 0 for team in teams}

    if 'y' not in st.session_state:
        st.error("Please first calculate the playoff probabilities!")
    else:
        y = st.session_state['y']
        X_ini = st.session_state['X_ini']

        # Define brackets
        East_numbers = list(range(10))
        West_numbers = list(range(10, 20))
        st.session_state['East_numbers'] = East_numbers
        st.session_state['West_numbers'] = West_numbers

        # Helper function: simulate a single conference and return all round winners
        def simulate_conference(numbers):
            rounds = {}
            #Playin first round
            winner_playin = []
            loser_playin = []
            matchups_playin = [[6,7],[8,9]]
            for i,j in matchups_playin:
                p_i = y[numbers[i]]
                p_j = y[numbers[j]]
                EPS = 1e-6
                p_i = np.clip(p_i, EPS, 1-EPS)
                p_j = np.clip(p_j, EPS, 1-EPS)
                p_i_scaled = (p_i**(1/T))/((p_i**(1/T))+((1-p_i)**(1/T)))
                p_j_scaled = (p_j**(1/T))/((p_j**(1/T))+((1-p_j)**(1/T)))
                total = p_i_scaled + p_j_scaled
                winner = np.random.choice([numbers[i], numbers[j]], p=[p_i_scaled/total, p_j_scaled/total])
                loser = numbers[j] if winner == numbers[i] else numbers[i]
                winner_playin.append(winner)
                loser_playin.append(loser)
            #Playin third game for 8th seed
            p_i = y[loser_playin[0]]
            p_j = y[winner_playin[1]]
            EPS = 1e-6
            p_i = np.clip(p_i, EPS, 1-EPS)
            p_j = np.clip(p_j, EPS, 1-EPS)
            p_i_scaled = (p_i**(1/T))/((p_i**(1/T))+((1-p_i)**(1/T)))
            p_j_scaled = (p_j**(1/T))/((p_j**(1/T))+((1-p_j)**(1/T)))
            total = p_i_scaled + p_j_scaled
            winner = np.random.choice([loser_playin[0], winner_playin[1]], p=[p_i_scaled/total, p_j_scaled/total])
            winner_playin.append(winner)
            rounds['Playin'] = winner_playin

            
            # Update numbers for first round
            new_numbers = numbers.copy()
            new_numbers[6] = winner_playin[0]
            new_numbers[7] = winner_playin[2]
            # First round
            winners_round1 = []
            matchups = [[0,7],[1,6],[2,5],[3,4]]
            for i,j in matchups:
                p_i = y[new_numbers[i]]
                p_j = y[new_numbers[j]]
                EPS = 1e-6
                p_i = np.clip(p_i, EPS, 1 - EPS)
                p_j = np.clip(p_j, EPS, 1 - EPS)
                p_i_scaled = (p_i**(1/T))/((p_i**(1/T))+((1-p_i)**(1/T)))
                p_j_scaled = (p_j**(1/T))/((p_j**(1/T))+((1-p_j)**(1/T)))
                total = p_i_scaled + p_j_scaled
                winner = np.random.choice([new_numbers[i], new_numbers[j]], p=[p_i_scaled/total, p_j_scaled/total])
                winners_round1.append(winner)
            rounds['First Round'] = winners_round1

            # Semi-finals
            winners_round2 = []
            semi_matchups = [[0,3],[1,2]]
            for i,j in semi_matchups:
                p_i = y[winners_round1[i]]
                p_j = y[winners_round1[j]]
                EPS = 1e-6
                p_i = np.clip(p_i, EPS, 1 - EPS)
                p_j = np.clip(p_j, EPS, 1 - EPS)
                p_i_scaled = (p_i**(1/T))/((p_i**(1/T))+((1-p_i)**(1/T)))
                p_j_scaled = (p_j**(1/T))/((p_j**(1/T))+((1-p_j)**(1/T)))
                total = p_i_scaled + p_j_scaled
                winner = np.random.choice([winners_round1[i], winners_round1[j]], p=[p_i_scaled/total, p_j_scaled/total])
                winners_round2.append(winner)
            rounds['Semi-Finals'] = winners_round2

            # Conference final
            p_i = y[winners_round2[0]]
            p_j = y[winners_round2[1]]
            EPS = 1e-6
            p_i = np.clip(p_i, EPS, 1 - EPS)
            p_j = np.clip(p_j, EPS, 1 - EPS)
            p_i_scaled = (p_i**(1/T))/((p_i**(1/T))+((1-p_i)**(1/T)))
            p_j_scaled = (p_j**(1/T))/((p_j**(1/T))+((1-p_j)**(1/T)))
            total = p_i_scaled + p_j_scaled
            winner = np.random.choice([winners_round2[0], winners_round2[1]], p=[p_i_scaled/total, p_j_scaled/total])
            rounds['Conference Final'] = winner

            return rounds

        # Run simulations
        all_simulations = []
        # Initialize progression counters
        progress_counts = {
            team: {
                "Conference Semi-Final": 0,
                "Conference Final": 0,
                "NBA Final": 0,
                "NBA Champion": 0
            }
            for team in teams
        }



        for sim in range(N):
            sim_result = {}

            # Simulate conferences
            winners_East = simulate_conference(East_numbers)
            winners_West = simulate_conference(West_numbers)
            #Count progressions
            for t in winners_East['First Round']:
                progress_counts[X_ini['Team'][t]]["Conference Semi-Final"] += 1

            for t in winners_West['First Round']:
                progress_counts[X_ini['Team'][t]]["Conference Semi-Final"] += 1
            for t in winners_East['Semi-Finals']:
                progress_counts[X_ini['Team'][t]]["Conference Final"] += 1

            for t in winners_West['Semi-Finals']:
                progress_counts[X_ini['Team'][t]]["Conference Final"] += 1

            sim_result['East'] = winners_East
            sim_result['West'] = winners_West

            # NBA Final
            winner_East = winners_East['Conference Final']
            winner_West = winners_West['Conference Final']
            progress_counts[X_ini['Team'][winner_East]]["NBA Final"] += 1
            progress_counts[X_ini['Team'][winner_West]]["NBA Final"] += 1
            p_E = y[winner_East]
            p_W = y[winner_West]
            EPS = 1e-6
            p_E = np.clip(p_E, EPS, 1 - EPS)
            p_W = np.clip(p_W, EPS, 1 - EPS)
            p_E_scaled = (p_E**(1/T))/((p_E**(1/T))+((1-p_E)**(1/T)))
            p_W_scaled = (p_W**(1/T))/((p_W**(1/T))+((1-p_W)**(1/T)))
            total = p_E_scaled + p_W_scaled
            winner_NBA = np.random.choice([winner_East, winner_West], p=[p_E_scaled/total, p_W_scaled/total])
            sim_result['NBA Final'] = (winner_East, winner_West, winner_NBA)

            progress_counts[X_ini['Team'][winner_NBA]]["NBA Champion"] += 1
            all_simulations.append(sim_result)

        # Save results in session state
        st.session_state['all_simulations'] = all_simulations
        st.session_state['progress_counts'] = progress_counts
        st.session_state['T'] = T

# Display championship summary if it exists
if 'progress_counts' in st.session_state:
    st.subheader("📊 Playoff Progression Summary")

    df_progress = pd.DataFrame.from_dict(
        st.session_state['progress_counts'],
        orient='index'
    )

    df_progress = df_progress.sort_values(
        by="NBA Champion",
        ascending=False
    )
    df_progress_pct = df_progress / N
    st.dataframe(df_progress_pct.style.format("{:.2%}"))

    

    st.markdown("""
    **Interpretation:**
    Values correspond to the percentage (out of all simulations)
      a team reached each playoff stage.
    """)

# ---------- Select a simulation to view the bracket ----------

if 'all_simulations' in st.session_state and st.session_state['all_simulations']:
    st.subheader("🔍 Show a bracket where a team reaches a specific round:")

    team_choice = st.selectbox("Choose a team", X_ini['Team'].tolist())

    round_choice = st.selectbox(
        "Choose the round",
        ["Conference Semi-Final", "Conference Final", "NBA Final", "NBA Champion"]
    )

    valid_sims = []
    team_index = X_ini[X_ini['Team'] == team_choice].index[0]
    for idx, sim_result in enumerate(st.session_state['all_simulations']):
        if round_choice == "Conference Semi-Final":
            if team_index in sim_result['East']['First Round'] or team_index in sim_result['West']['First Round']:
                valid_sims.append(idx + 1)
        elif round_choice == "Conference Final":
            if team_index in sim_result['East']['Semi-Finals'] or team_index in sim_result['West']['Semi-Finals']:
                valid_sims.append(idx + 1)
        elif round_choice == "NBA Final":
            if team_index == sim_result['East']['Conference Final'] or team_index == sim_result['West']['Conference Final']:
                valid_sims.append(idx + 1)
        elif round_choice == "NBA Champion":
            if team_index == sim_result['NBA Final'][2]:
                valid_sims.append(idx + 1)

    if valid_sims:
        sim_index = random.choice(valid_sims)
        st.markdown(f"### Example bracket where **{team_choice}** reaches **{round_choice}** (Simulation #{sim_index}):")
        sim_number = sim_index
        East_numbers = st.session_state['East_numbers']
        West_numbers = st.session_state['West_numbers']
        X_ini = st.session_state['X_ini']
        y = st.session_state['y']
        T = st.session_state['T']
        sim_result = st.session_state['all_simulations'][sim_number - 1]
        display_bracket(sim_result, X_ini,y, T, East_numbers, West_numbers)
    else:
        st.markdown(f"⚠️ No simulations found where **{team_choice}** reaches **{round_choice}**.")