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
    # Playoff Simulation Button
    st.title("🏀 NBA Playoff Simulation")
    # User Inputs
    T = st.number_input("What is the upset factor?", min_value=1.0, max_value=10.0, step=0.5)
    # N = st.number_input("How many simulations to run?", min_value=10, max_value=1000, step=10)

    if st.button("Run Playoff Simulations"):
        Brackets =[[1,8],[2,7],[3,6],[4,5]]
        East_conference_number=[0,1,2,3,4,5,6,7,8,9]
        West_conference_number=[10,11,12,13,14,15,16,17,18,19]
        st.write("East conference brackets")
        st.write("Confrontation first round")
        st.write("1 vs 8")
        team_1 = East_conference_number[0]
        team_8 = East_conference_number[7]
        p_1 = y_prob[team_1][0]
        p_8 = y_prob[team_8][0]
        p_1_scaled = (p_1**(1/T))/((p_1**(1/T))+((1-p_1)**(1/T)))
        p_8_scaled = (p_8**(1/T))/((p_8**(1/T))+((1-p_8)**(1/T)))

        #Renormalize probabilities
        total = p_1_scaled + p_8_scaled
        p_1_final = p_1_scaled / total
        p_8_final = p_8_scaled / total
        #Draw the winner
        winner_18 = np.random.choice([team_1, team_8], p=[p_1_final, p_8_final])
        st.write(f"Winner: {X_ini['Team'][winner_18]}")

        st.write("2 vs 7")
        team_2 = East_conference_number[1]
        team_7 = East_conference_number[6]
        p_2 = y_prob[team_2][0]
        p_7 = y_prob[team_7][0]
        p_2_scaled = (p_2**(1/T))/((p_2**(1/T))+((1-p_2)**(1/T)))
        p_7_scaled = (p_7**(1/T))/((p_7**(1/T))+((1-p_7)**(1/T)))
        #Renormalize probabilities
        total = p_2_scaled + p_7_scaled
        p_2_final = p_2_scaled / total
        p_7_final = p_7_scaled / total
        #Draw the winner
        winner_27 = np.random.choice([team_2, team_7], p=[p_2_final, p_7_final])
        st.write(f"Winner: {X_ini['Team'][winner_27]}")
        st.write("3 vs 6")
        team_3 = East_conference_number[2]
        team_6 = East_conference_number[5]
        p_3 = y_prob[team_3][0]
        p_6 = y_prob[team_6][0]
        p_3_scaled = (p_3**(1/T))/((p_3**(1/T))+((1-p_3)**(1/T)))
        p_6_scaled = (p_6**(1/T))/((p_6**(1/T))+((1-p_6)**(1/T)))
        #Renormalize probabilities
        total = p_3_scaled + p_6_scaled
        p_3_final = p_3_scaled / total
        p_6_final = p_6_scaled / total
        #Draw the winner
        winner_36 = np.random.choice([team_3, team_6], p=[p_3_final, p_6_final])
        st.write(f"Winner: {X_ini['Team'][winner_36]}")
        st.write("4 vs 5")
        team_4 = East_conference_number[3]
        team_5 = East_conference_number[4]
        p_4 = y_prob[team_4][0]
        p_5 = y_prob[team_5][0]
        p_4_scaled = (p_4**(1/T))/((p_4**(1/T))+((1-p_4)**(1/T)))
        p_5_scaled = (p_5**(1/T))/((p_5**(1/T))+((1-p_5)**(1/T)))
        #Renormalize probabilities
        total = p_4_scaled + p_5_scaled
        p_4_final = p_4_scaled / total
        p_5_final = p_5_scaled / total
        #Draw the winner
        winner_45 = np.random.choice([team_4, team_5], p=[p_4_final, p_5_final])
        st.write(f"Winner: {X_ini['Team'][winner_45]}")

        st.write("Confrontation Semi-Final")
        st.write("1/8 winner vs 4/5 winner")
        team_A = winner_18
        team_B = winner_45
        p_A = y_prob[team_A][1]
        p_B = y_prob[team_B][1]
        p_A_scaled = (p_A**(1/T))/((p_A**(1/T))+((1-p_A)**(1/T)))
        p_B_scaled = (p_B**(1/T))/((p_B**(1/T))+((1-p_B)**(1/T)))
        #Renormalize probabilities
        total = p_A_scaled + p_B_scaled
        p_A_final = p_A_scaled / total
        p_B_final = p_B_scaled / total
        #Draw the winner
        winner_AB = np.random.choice([team_A, team_B], p=[p_A_final, p_B_final])
        st.write(f"Semi-Final Winner: {X_ini['Team'][winner_AB]}")
        st.write("2/7 winner vs 3/6 winner")
        team_C = winner_27
        team_D = winner_36
        p_C = y_prob[team_C][1]
        p_D = y_prob[team_D][1]
        p_C_scaled = (p_C**(1/T))/((p_C**(1/T))+((1-p_C)**(1/T)))
        p_D_scaled = (p_D**(1/T))/((p_D**(1/T))+((1-p_D)**(1/T)))
        #Renormalize probabilities
        total = p_C_scaled + p_D_scaled
        p_C_final = p_C_scaled / total
        p_D_final = p_D_scaled / total
        #Draw the winner
        winner_CD = np.random.choice([team_C, team_D], p=[p_C_final, p_D_final])
        st.write(f"Semi-Final Winner: {X_ini['Team'][winner_CD]}")
        st.write("Conference Final")
        team_E = winner_AB
        team_F = winner_CD
        p_E = y_prob[team_E][2]
        p_F = y_prob[team_F][2]
        p_E_scaled = (p_E**(1/T))/((p_E**(1/T))+((1-p_E)**(1/T)))
        p_F_scaled = (p_F**(1/T))/((p_F**(1/T))+((1-p_F)**(1/T)))
        #Renormalize probabilities
        total = p_E_scaled + p_F_scaled
        p_E_final = p_E_scaled / total
        p_F_final = p_F_scaled / total
        #Draw the winner
        winner_East = np.random.choice([team_E, team_F], p=[p_E_final, p_F_final])
        st.write(f"Conference Final Winner: {X_ini['Team'][winner_East]}")

        st.write("West conference brackets")
        st.write("Confrontation first round")
        st.write("1 vs 8")
        team_1 = West_conference_number[0]
        team_8 = West_conference_number[7]
        p_1 = y_prob[team_1][0]
        p_8 = y_prob[team_8][0]
        p_1_scaled = (p_1**(1/T))/((p_1**(1/T))+((1-p_1)**(1/T)))
        p_8_scaled = (p_8**(1/T))/((p_8**(1/T))+((1-p_8)**(1/T)))
        #Renormalize probabilities
        total = p_1_scaled + p_8_scaled
        p_1_final = p_1_scaled / total
        p_8_final = p_8_scaled / total
        #Draw the winner
        winner_18 = np.random.choice([team_1, team_8], p=[p_1_final, p_8_final])
        st.write(f"Winner: {X_ini['Team'][winner_18]}")
        st.write("2 vs 7")
        team_2 = West_conference_number[1]
        team_7 = West_conference_number[6]
        p_2 = y_prob[team_2][0]
        p_7 = y_prob[team_7][0]
        p_2_scaled = (p_2**(1/T))/((p_2**(1/T))+((1-p_2)**(1/T)))
        p_7_scaled = (p_7**(1/T))/((p_7**(1/T))+((1-p_7)**(1/T)))
        #Renormalize probabilities
        total = p_2_scaled + p_7_scaled
        p_2_final = p_2_scaled / total
        p_7_final = p_7_scaled / total
        #Draw the winner
        winner_27 = np.random.choice([team_2, team_7], p=[p_2_final, p_7_final])
        st.write(f"Winner: {X_ini['Team'][winner_27]}")
        st.write("3 vs 6")
        team_3 = West_conference_number[2]
        team_6 = West_conference_number[5]
        p_3 = y_prob[team_3][0]
        p_6 = y_prob[team_6][0]
        p_3_scaled = (p_3**(1/T))/((p_3**(1/T))+((1-p_3)**(1/T)))
        p_6_scaled = (p_6**(1/T))/((p_6**(1/T))+((1-p_6)**(1/T)))
        #Renormalize probabilities
        total = p_3_scaled + p_6_scaled
        p_3_final = p_3_scaled / total
        p_6_final = p_6_scaled / total
        #Draw the winner
        winner_36 = np.random.choice([team_3, team_6], p=[p_3_final, p_6_final])
        st.write(f"Winner: {X_ini['Team'][winner_36]}")
        st.write("4 vs 5")
        team_4 = West_conference_number[3]
        team_5 = West_conference_number[4]
        p_4 = y_prob[team_4][0]
        p_5 = y_prob[team_5][0]
        p_4_scaled = (p_4**(1/T))/((p_4**(1/T))+((1-p_4)**(1/T)))
        p_5_scaled = (p_5**(1/T))/((p_5**(1/T))+((1-p_5)**(1/T)))
        #Renormalize probabilities
        total = p_4_scaled + p_5_scaled
        p_4_final = p_4_scaled / total
        p_5_final = p_5_scaled / total
        #Draw the winner
        winner_45 = np.random.choice([team_4, team_5], p=[p_4_final, p_5_final])
        st.write(f"Winner: {X_ini['Team'][winner_45]}")
        # st.write("Confrontation Semi-Final")
        # st.write("1/8 winner vs 4/5 winner")
        team_A = winner_18
        team_B = winner_45
        p_A = y_prob[team_A][1]
        p_B = y_prob[team_B][1]
        p_A_scaled = (p_A**(1/T))/((p_A**(1/T))+((1-p_A)**(1/T)))
        p_B_scaled = (p_B**(1/T))/((p_B**(1/T))+((1-p_B)**(1/T)))
        #Renormalize probabilities
        total = p_A_scaled + p_B_scaled
        p_A_final = p_A_scaled / total
        p_B_final = p_B_scaled / total
        #Draw the winner
        winner_AB = np.random.choice([team_A, team_B], p=[p_A_final, p_B_final])
        st.write(f"Semi-Final Winner: {X_ini['Team'][winner_AB]}")
        st.write("2/7 winner vs 3/6 winner")
        team_C = winner_27
        team_D = winner_36
        p_C = y_prob[team_C][1]
        p_D = y_prob[team_D][1]
        p_C_scaled = (p_C**(1/T))/((p_C**(1/T))+((1-p_C)**(1/T)))
        p_D_scaled = (p_D**(1/T))/((p_D**(1/T))+((1-p_D)**(1/T)))
        #Renormalize probabilities
        total = p_C_scaled + p_D_scaled
        p_C_final = p_C_scaled / total
        p_D_final = p_D_scaled / total
        #Draw the winner
        winner_CD = np.random.choice([team_C, team_D], p=[p_C_final, p_D_final])
        st.write(f"Semi-Final Winner: {X_ini['Team'][winner_CD]}")
        st.write("Conference Final")
        team_E = winner_AB
        team_F = winner_CD
        p_E = y_prob[team_E][2]
        p_F = y_prob[team_F][2]
        p_E_scaled = (p_E**(1/T))/((p_E**(1/T))+((1-p_E)**(1/T)))
        p_F_scaled = (p_F**(1/T))/((p_F**(1/T))+((1-p_F)**(1/T)))
        #Renormalize probabilities
        total = p_E_scaled + p_F_scaled
        p_E_final = p_E_scaled / total
        p_F_final = p_F_scaled / total
        #Draw the winner
        winner_West = np.random.choice([team_E, team_F], p=[p_E_final, p_F_final])
        st.write(f"Conference Final Winner: {X_ini['Team'][winner_West]}")
        st.write("NBA Final")
        team_G = winner_East
        team_H = winner_West
        p_G = y_prob[team_G][3]
        p_H = y_prob[team_H][3]
        p_G_scaled = (p_G**(1/T))/((p_G**(1/T))+((1-p_G)**(1/T)))
        p_H_scaled = (p_H**(1/T))/((p_H**(1/T))+((1-p_H)**(1/T)))
        #Renormalize probabilities
        total = p_G_scaled + p_H_scaled
        p_G_final = p_G_scaled / total
        p_H_final = p_H_scaled / total
        #Draw the winner
        winner_NBA = np.random.choice([team_G, team_H], p=[p_G_final, p_H_final])
        st.write(f"NBA Champion: {X_ini['Team'][winner_NBA]}")



    


    


