import streamlit as st
TEAM_COLORS = {
    "Detroit Pistons": "red",
    "Boston Celtics": "green",
    "Toronto Raptors": "red",
    "New York Knicks": "orange",
    "Cleveland Cavaliers": "red",
    "Philadelphia 76ers": "blue",
    "Miami Heat": "red",
    "Orlando Magic": "blue",
    "Chicago Bulls": "red",
    "Atlanta Hawks": "red",
    "Oklahoma City Thunder": "blue",
    "San Antonio Spurs": "gray",
    "Denver Nuggets": "blue",
    "Houston Rockets": "red",
    "Los Angeles Lakers": "yellow",
    "Phoenix Suns": "orange",
    "Minesota Timberwolves": "blue",
    "Golden State Warriors": "blue",
    "Portland Trail Blazers": "red",
    "Los Angeles Clippers": "red",
}

def display_bracket(sim_result, X_ini, East_numbers, West_numbers):
    """
    Display the NBA bracket for a given simulation.
    sim_result: one simulation from all_simulations
    X_ini: DataFrame with team info
    East_numbers, West_numbers: list of indices for East/West teams
    """
    def print_matchup(team1_idx, team2_idx, winner_idx):
        team1_name = X_ini['Team'][team1_idx]
        team2_name = X_ini['Team'][team2_idx]
        winner_name = X_ini['Team'][winner_idx]

        # Get colors (default to black if team not in dict)
        color1 = TEAM_COLORS.get(team1_name, "black")
        color2 = TEAM_COLORS.get(team2_name, "black")
        color_winner = TEAM_COLORS.get(winner_name, "black")

        # Display with color
        st.markdown(
            f"<span style='color:{color1}'>{team1_name}</span> vs "
            f"<span style='color:{color2}'>{team2_name}</span> -> Winner: "
            f"<span style='color:{color_winner}; font-weight:bold'>{winner_name}</span>",
            unsafe_allow_html=True
        )
    
    # ---------- East Conference ----------
    st.write("## 🏟️ East Conference")
    st.write("### First Round")
    first_round = sim_result['East']['First Round']
    matchups = [[0,7],[1,6],[2,5],[3,4]]
    for i,j,k in zip([0,1,2,3], [7,6,5,4], first_round):
        print_matchup(East_numbers[i], East_numbers[j], k)

    st.write("### Conference Semi-Final")
    semi = sim_result['East']['Semi-Finals']
    print_matchup(first_round[0], first_round[3], semi[0])
    print_matchup(first_round[1], first_round[2], semi[1])

    st.write("### Conference Final")
    conf_final = sim_result['East']['Conference Final']
    print_matchup(semi[0], semi[1], conf_final)

    # ---------- West Conference ----------
    st.write("## 🏟️ West Conference")
    st.write("### First Round")
    first_round = sim_result['West']['First Round']
    matchups = [[0,7],[1,6],[2,5],[3,4]]
    for i,j,k in zip([0,1,2,3], [7,6,5,4], first_round):
        print_matchup(West_numbers[i], West_numbers[j], k)

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