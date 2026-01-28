import streamlit as st
def display_bracket(sim_result, X_ini, East_numbers, West_numbers):
    """
    Display the NBA bracket for a given simulation.
    sim_result: one simulation from all_simulations
    X_ini: DataFrame with team info
    East_numbers, West_numbers: list of indices for East/West teams
    """
    def print_matchup(team1_idx, team2_idx, winner_idx):
        st.write(f"{X_ini['Team'][team1_idx]} vs {X_ini['Team'][team2_idx]} -> Winner: {X_ini['Team'][winner_idx]}")

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
    st.write("### 🏆 NBA Final")
    final = sim_result['NBA Final']
    print_matchup(final[0], final[1], final[2])