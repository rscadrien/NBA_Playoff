import matplotlib.pyplot as plt

def draw_nba_bracket(X_ini, sim_result, East_numbers, West_numbers):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')  # hide axes

    # --- Parameters ---
    rounds = ['1st Round', 'Semi-Finals', 'Conference Final', 'NBA Final']
    x_spacing = 3
    y_spacing = 1.5
    box_width = 2.5
    box_height = 0.6

    # --- WEST Conference ---
    west_y = [i * y_spacing for i in range(4)]
    west_x = 0

    # First Round
    for i in range(4):
        y = west_y[i]
        team1 = X_ini['Team'][West_numbers[i]] 
        team2 = X_ini['Team'][West_numbers[7-i]]
        winner = X_ini['Team'][sim_result['West']['First Round'][i]]
        ax.text(west_x, y, f"{team1}\nvs\n{team2}", ha='left', va='center', fontsize=10, bbox=dict(facecolor='lightblue', alpha=0.3))
        ax.text(west_x + 1.5 * box_width, y, f"{winner}", ha='left', va='center', fontsize=10, fontweight='bold', color='green')

    # Semi-Finals
    semi_y = [(west_y[0]+west_y[3])/2, (west_y[1]+west_y[2])/2]
    semi_winners = sim_result['West']['Semi-Finals']
    for i in range(2):
        y = semi_y[i]
        t1 = X_ini['Team'][sim_result['West']['First Round'][0 if i==0 else 1]]
        t2 = X_ini['Team'][sim_result['West']['First Round'][3 if i==0 else 2]]
        winner = X_ini['Team'][semi_winners[i]]
        ax.text(west_x + x_spacing, y, f"{winner}", ha='left', va='center', fontsize=10, fontweight='bold', color='green')
        # connecting lines
        ax.plot([west_x + box_width*0.8, west_x + x_spacing*0.1], [west_y[i*2], y], color='black')
        ax.plot([west_x + box_width*0.8, west_x + x_spacing*0.1], [west_y[3-i*2], y], color='black')

    # Conference Final
    conf_final_y = (semi_y[0]+semi_y[1])/2
    west_conf_winner = sim_result['West']['Conference Final']
    ax.text(west_x + x_spacing*2, conf_final_y, X_ini['Team'][west_conf_winner], ha='left', va='center', fontsize=10, fontweight='bold', color='green')
    # connect semi-finals
    for y in semi_y:
        ax.plot([west_x + x_spacing + box_width*0.8, west_x + x_spacing*2*0.1], [y, conf_final_y], color='black')

    # --- EAST Conference ---
    east_y = [i * y_spacing for i in range(4)]
    east_x = x_spacing*6

    # First Round
    for i in range(4):
        y = east_y[3-i]  # flip to mirror west
        team1 = X_ini['Team'][East_numbers[i]] 
        team2 = X_ini['Team'][East_numbers[7-i]]
        winner = X_ini['Team'][sim_result['East']['First Round'][i]]
        ax.text(east_x, y, f"{team1}\nvs\n{team2}", ha='right', va='center', fontsize=10, bbox=dict(facecolor='lightpink', alpha=0.3))
        ax.text(east_x - box_width, y, f"{winner}", ha='right', va='center', fontsize=10, fontweight='bold', color='green')

    # Semi-Finals
    semi_y = [(east_y[3]+east_y[0])/2, (east_y[2]+east_y[1])/2]
    semi_winners = sim_result['East']['Semi-Finals']
    for i in range(2):
        y = semi_y[i]
        winner = X_ini['Team'][semi_winners[i]]
        ax.text(east_x - x_spacing, y, f"{winner}", ha='right', va='center', fontsize=10, fontweight='bold', color='green')
        ax.plot([east_x - box_width*0.8, east_x - x_spacing*0.1], [east_y[3-i*2], y], color='black')
        ax.plot([east_x - box_width*0.8, east_x - x_spacing*0.1], [east_y[i*2], y], color='black')

    # Conference Final
    conf_final_y = (semi_y[0]+semi_y[1])/2
    east_conf_winner = sim_result['East']['Conference Final']
    ax.text(east_x - x_spacing*2, conf_final_y, X_ini['Team'][east_conf_winner], ha='right', va='center', fontsize=10, fontweight='bold', color='green')
    for y in semi_y:
        ax.plot([east_x - x_spacing - box_width*0.8, east_x - x_spacing*2*0.1], [y, conf_final_y], color='black')

    # --- NBA Final ---
    nba_x = x_spacing*3
    nba_y = (west_y[1]+east_y[1])/2
    nba_winner = sim_result['NBA Final'][2]
    ax.text(nba_x, nba_y, f"🏆 {X_ini['Team'][nba_winner]}", ha='center', va='center', fontsize=12, fontweight='bold', color='gold')
    # connect conference finals
    ax.plot([west_x + x_spacing*2, nba_x - 0.5], [ ( (west_y[0]+west_y[3])/2 + (west_y[1]+west_y[2])/2)/2, nba_y], color='black')
    ax.plot([east_x - x_spacing*2, nba_x + 0.5], [ ( (east_y[0]+east_y[3])/2 + (east_y[1]+east_y[2])/2)/2, nba_y], color='black')

    return fig