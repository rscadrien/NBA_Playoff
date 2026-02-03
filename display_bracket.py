import streamlit as st
import numpy as np
blue = "#87CEEB"
TEAM_COLORS = {
    "Detroit Pistons": "red",
    "Boston Celtics": "green",
    "Toronto Raptors": "red",
    "New York Knicks": "orange",
    "Cleveland Cavaliers": "red",
    "Philadelphia 76ers": blue,
    "Miami Heat": "red",
    "Orlando Magic": blue,
    "Chicago Bulls": "red",
    "Atlanta Hawks": "red",
    "Oklahoma City Thunder": blue,
    "San Antonio Spurs": "gray",
    "Denver Nuggets": blue,
    "Houston Rockets": "red",
    "Los Angeles Lakers": "yellow",
    "Phoenix Suns": "orange",
    "Minesota Timberwolves": blue,
    "Golden State Warriors": blue,
    "Portland Trail Blazers": "red",
    "Los Angeles Clippers": "red",
}

def display_bracket(sim_result, X_ini, y, T, East_numbers, West_numbers):
    """
    Display the NBA bracket for a given simulation.
    sim_result: one simulation from all_simulations
    X_ini: DataFrame with team info
    East_numbers, West_numbers: list of indices for East/West teams
    """
    def matchup_prob(team_idx1, team_idx2):
        # Compute scaled probability for team_idx1 to win
        p1 = y[team_idx1]
        p2 = y[team_idx2]
        EPS = 1e-6
        p1 = np.clip(p1, EPS, 1 - EPS)
        p2 = np.clip(p2, EPS, 1 - EPS)
        p1_scaled = (p1**(1/T)) / ((p1**(1/T)) + ((1-p1)**(1/T)))
        p2_scaled = (p2**(1/T)) / ((p2**(1/T)) + ((1-p2)**(1/T)))
        total = p1_scaled + p2_scaled
        return p1_scaled/total, p2_scaled/total

    def print_matchup(team1_idx, team2_idx, winner_idx):
        team1_name = X_ini['Team'][team1_idx]
        team2_name = X_ini['Team'][team2_idx]
        winner_name = X_ini['Team'][winner_idx]

        # Get colors (default to black if team not in dict)
        color1 = TEAM_COLORS.get(team1_name, "black")
        color2 = TEAM_COLORS.get(team2_name, "black")
        color_winner = TEAM_COLORS.get(winner_name, "black")

        p1_prob, p2_prob = matchup_prob(team1_idx, team2_idx)

        # Display with color
        st.markdown(
            f"<span style='color:{color1}'>{team1_name}</span> "
            f"({p1_prob:.1%}) vs "
            f"<span style='color:{color2}'>{team2_name}</span> "
            f"({p2_prob:.1%}) -> Winner: "
            f"<span style='color:{color_winner}; font-weight:bold'>{winner_name}</span>",
            unsafe_allow_html=True
        )
    
    # ---------- East Conference ----------
    st.write("## 🏟️ East Conference")
    st.write("### Playin###")
    playin = sim_result['East']['Playin']
    for i,j,k in zip([6,8], [7,9], playin[0:2]):
        print_matchup(East_numbers[i], East_numbers[j], k)
    
    loser = East_numbers[7] if playin[0] == East_numbers[i] else East_numbers[i]
    print_matchup(loser, playin[1], playin[2])

    # Update numbers for first round
    new_east_numbers = East_numbers.copy()
    new_east_numbers[6] = playin[0]
    new_east_numbers[7] = playin[2]
    

    st.write("### First Round")
    first_round = sim_result['East']['First Round']
    matchups = [[0,7],[1,6],[2,5],[3,4]]
    for i,j,k in zip([0,1,2,3], [7,6,5,4], first_round):
        print_matchup(new_east_numbers[i], new_east_numbers[j], k)

    st.write("### Conference Semi-Final")
    semi = sim_result['East']['Semi-Finals']
    print_matchup(first_round[0], first_round[3], semi[0])
    print_matchup(first_round[1], first_round[2], semi[1])

    st.write("### Conference Final")
    conf_final = sim_result['East']['Conference Final']
    print_matchup(semi[0], semi[1], conf_final)

    # ---------- West Conference ----------
    st.write("## 🏟️ West Conference")
    st.write("### Playin###")
    playin = sim_result['West']['Playin']
    for i,j,k in zip([6,8], [7,9], playin[0:2]):
        print_matchup(West_numbers[i], West_numbers[j], k)
    
    loser = West_numbers[7] if playin[0] == West_numbers[i] else West_numbers[i]
    print_matchup(loser, playin[1], playin[2])

    # Update numbers for first round
    new_west_numbers = West_numbers.copy()
    new_west_numbers[6] = playin[0]
    new_west_numbers[7] = playin[2]

    st.write("### First Round")
    first_round = sim_result['West']['First Round']
    matchups = [[0,7],[1,6],[2,5],[3,4]]
    for i,j,k in zip([0,1,2,3], [7,6,5,4], first_round):
        print_matchup(new_west_numbers[i], new_west_numbers[j], k)

    st.write("### Conference Semi-Final")
    semi = sim_result['West']['Semi-Finals']
    print_matchup(first_round[0], first_round[3], semi[0])
    print_matchup(first_round[1], first_round[2], semi[1])

    st.write("### Conference Final")
    conf_final = sim_result['West']['Conference Final']
    print_matchup(semi[0], semi[1], conf_final)

    # ---------- NBA Final ----------
    st.write("## 🏆 NBA Final")
    final = sim_result['NBA Final']
    print_matchup(final[0], final[1], final[2])