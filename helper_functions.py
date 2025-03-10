import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from urllib.request import urlopen
from fuzzywuzzy import process
from IPython.display import Image as ImageDisplay
import pandas as pd
import torch
from sklearn.base import BaseEstimator

logos = pd.read_csv('logos.csv')
logo_mapper = pd.read_csv('ncaa_name_mapper.csv')

# Take in their data and model and run it through our more specific functions based on the model type
def simulate_tournament(test_data, model):

    if isinstance(model, torch.nn.Module):
        return simulate_tournament_pytorch(test_data, model)
    elif isinstance(model, BaseEstimator):
        return simulate_tournament_sklearn(test_data, model)
    else:
        raise TypeError("Unsupported model type. Must be a PyTorch or scikit-learn model.")

# Full bracket simulating function
def simulate_tournament_sklearn(test_data, model):
    
    if not isinstance(test_data, pd.DataFrame):
        raise TypeError("Please pass data as a pandas DataFrame!")
    
    # Custom sort functions to properly order matchups for the next round
    custom_sort1 = ['W', 'X', 'Y', 'Z']
    custom_sort2 = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
    
    region_order = {region: i for i, region in enumerate(custom_sort1)}
    seed_order = {seed: i for i, seed in enumerate(custom_sort2)}
    
    # These are the colnames that aren't prefixed by TeamA/TeamB but still may be
    # relevant predictors so I explicitly say them here
    # MAYBE THIS IS WHERE WE CAN ADD IN THEIR FEATURE ENGINEERED STUFF AS A FUNC ARG
    extra_cols_a = ['TotG', 'TotW', 'TotL', 'NeutralG', 'WinPct']
    extra_cols_b = ['TotG.1', 'TotW.1', 'TotL.1', 'NeutralG.1', 'WinPct.1']
    
    # Filter data to just R1
    # THIS IS USELESS IN THE FUTURE JUST DOING BC WE ARE TESTING ON DATA w/ FULL ROUND INFO!
    test_data = test_data[test_data['Round'] == 1]
    
    # Define R1 matchups
    inits = list(test_data[['TeamA', 'RegionTeamA', 'SeedTeamA']].itertuples(index=False, name=None)) + \
            list(test_data[['TeamB', 'RegionTeamB', 'SeedTeamB']].itertuples(index=False, name=None))
    
    inits = sorted(inits, key=lambda x: (region_order[x[1]], seed_order[x[2]]))
    
    # Make our list of round by round preds to spit out
    to_return = []
    
    # Start with R1 (0 is first 4 which we dc about)
    current_round = 1
    
    # Run until we do the finals
    while current_round <= 6:
        # Filter to just current round
        round_data = test_data[test_data['Round'] == current_round]
        
        # If somehow no data, end
        if round_data.empty:
            break

        # Get our predictors and the teams by row
        round_X = round_data.drop(columns=['RegionTeamA', 'RegionTeamB', 'TeamA', 'TeamB', 'ResultTeamA'])
        round_teams = round_data[['TeamA', 'TeamB', 'RegionTeamA', 'RegionTeamB', 'SeedTeamA', 'SeedTeamB']]
                
        # Predict winners using the model you made
        predictions = model.predict(round_X)
            
        
        # Get the winner names
        winners = [(row.TeamA if pred == 1 else row.TeamB, row.RegionTeamA if pred == 1 else row.RegionTeamB, \
                    row.SeedTeamA if pred == 1 else row.SeedTeamB) for row, pred in zip(round_teams.itertuples(), predictions)]
        
        # Sort the winners to properly assigned next round matchups
        winners = sorted(winners, key=lambda x: (region_order[x[1]], seed_order[x[2]]))
        
        # Create new matchups for the next round
        next_round_data = []
        for i in range(0, len(winners), 2):
            if i + 1 < len(winners):
                # Get the data we need to pass to the next round
                next_team_A = winners[i][0]
                seed_team_A = winners[i][2]
                next_team_B = winners[i + 1][0]
                seed_team_B = winners[i + 1][2]
                
                # Get stats for each team
                team_A_stats = test_data[(test_data['TeamA'] == next_team_A) | (test_data['TeamB'] == next_team_A)].\
                    drop(columns=['TeamA', 'TeamB', 'ResultTeamA', 'SeedTeamA', 'SeedTeamB']).iloc[-1]
                team_B_stats = test_data[(test_data['TeamA'] == next_team_B) | (test_data['TeamB'] == next_team_B)].\
                    drop(columns=['TeamA', 'TeamB', 'ResultTeamA', 'SeedTeamA', 'SeedTeamB']).iloc[-1]

            
                # Get only relevant columns
                # WE MAY NEED TO CHANGE THIS BASED ON FEATURE SELECTION/ENGINEERING
                team_A_stats = team_A_stats[team_A_stats.index.str.contains('TeamA') | team_A_stats.index.isin(extra_cols_a)]
                team_B_stats = team_B_stats[team_B_stats.index.str.contains('TeamB') | team_B_stats.index.isin(extra_cols_b)]
                
                matchup = pd.concat([pd.Series({'TeamA': next_team_A, 'TeamB': next_team_B, \
                                      'Round': current_round + 1, 'SeedTeamA': seed_team_A, 'SeedTeamB': seed_team_B }), \
                                      team_A_stats, team_B_stats], axis=0)
                next_round_data.append(matchup)

                    
        # Convert the next round data to DF and append to our running new data
        test_data = pd.concat([test_data, pd.DataFrame(next_round_data)], ignore_index=True)
        
        # Add our rounds winners to our return var
        to_return.append(winners)
        
        # Move to next round
        current_round += 1

    # Add the teams from R1 (predefined by seeding) and return
    penultimate = [inits] + to_return
    return [[tuple(sublist[j:j+2]) if j+1 < len(sublist) else (sublist[j],) 
           for j in range(0, len(sublist), 2)] for sublist in penultimate]

def simulate_tournament_pytorch(test_data, model, device='cpu'):
    
    if not isinstance(test_data, pd.DataFrame):
        raise TypeError("Please pass data as a pandas DataFrame!")
    
    # Custom sort functions to properly order matchups for the next round
    custom_sort1 = ['W', 'X', 'Y', 'Z']
    custom_sort2 = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
    
    region_order = {region: i for i, region in enumerate(custom_sort1)}
    seed_order = {seed: i for i, seed in enumerate(custom_sort2)}
    
    extra_cols_a = ['TotG', 'TotW', 'TotL', 'NeutralG', 'WinPct']
    extra_cols_b = ['TotG.1', 'TotW.1', 'TotL.1', 'NeutralG.1', 'WinPct.1']
    
    test_data = test_data[test_data['Round'] == 1]
    
    inits = list(test_data[['TeamA', 'RegionTeamA', 'SeedTeamA']].itertuples(index=False, name=None)) + \
            list(test_data[['TeamB', 'RegionTeamB', 'SeedTeamB']].itertuples(index=False, name=None))
    
    inits = sorted(inits, key=lambda x: (region_order[x[1]], seed_order[x[2]]))
    
    to_return = []
    current_round = 1
    
    model.to(device)
    model.eval()
    
    while current_round <= 5:
        round_data = test_data[test_data['Round'] == current_round]
        
        if round_data.empty:
            break

        round_X = round_data.drop(columns=['RegionTeamA', 'RegionTeamB', 'TeamA', 'TeamB', 'ResultTeamA'])
        round_teams = round_data[['TeamA', 'TeamB', 'RegionTeamA', 'RegionTeamB', 'SeedTeamA', 'SeedTeamB']]
        
        with torch.no_grad():
            inputs = torch.tensor(round_X.values, dtype=torch.float32, device=device)
            predictions = model(inputs).squeeze().cpu().numpy()
                    
        winners = [(row.TeamA if pred > 0.5 else row.TeamB, row.RegionTeamA if pred > 0.5 else row.RegionTeamB, \
                    row.SeedTeamA if pred > 0.5 else row.SeedTeamB) for row, pred in zip(round_teams.itertuples(), predictions)]
        
        winners = sorted(winners, key=lambda x: (region_order[x[1]], seed_order[x[2]]))
        
        next_round_data = []
        for i in range(0, len(winners), 2):
            if i + 1 < len(winners):
                next_team_A, seed_team_A = winners[i][0], winners[i][2]
                next_team_B, seed_team_B = winners[i + 1][0], winners[i + 1][2]
                
                team_A_stats = test_data[(test_data['TeamA'] == next_team_A) | (test_data['TeamB'] == next_team_A)].\
                    drop(columns=['TeamA', 'TeamB', 'ResultTeamA', 'SeedTeamA', 'SeedTeamB']).iloc[-1]
                team_B_stats = test_data[(test_data['TeamA'] == next_team_B) | (test_data['TeamB'] == next_team_B)].\
                    drop(columns=['TeamA', 'TeamB', 'ResultTeamA', 'SeedTeamA', 'SeedTeamB']).iloc[-1]
                
                team_A_stats = team_A_stats[team_A_stats.index.str.contains('TeamA') | team_A_stats.index.isin(extra_cols_a)]
                team_B_stats = team_B_stats[team_B_stats.index.str.contains('TeamB') | team_B_stats.index.isin(extra_cols_b)]
                
                matchup = pd.concat([pd.Series({'TeamA': next_team_A, 'TeamB': next_team_B, \
                                      'Round': current_round + 1, 'SeedTeamA': seed_team_A, 'SeedTeamB': seed_team_B }), \
                                      team_A_stats, team_B_stats], axis=0)
                next_round_data.append(matchup)
        
        test_data = pd.concat([test_data, pd.DataFrame(next_round_data)], ignore_index=True)
        to_return.append(winners)
        current_round += 1
    
    penultimate = [inits] + to_return
    return [[tuple(sublist[j:j+2]) if j+1 < len(sublist) else (sublist[j],) 
           for j in range(0, len(sublist), 2)] for sublist in penultimate]

# Define the specific plot parameters for the output bracket image
def get_plot_params():
    # Define base recentangle params
    rect_width = 5
    rect_height = 1

    # Gap sizes for each rounds
    gaps = {16: rect_height/2, 8: rect_height*2, 4: rect_height*5, 2: rect_height*11, 1: rect_height*16}

    # Extra base height for each round
    y_off_add = {16: 0, 8: 0.75, 4: 2.25, 2: 5.25, 1: 11.25}
    
    return rect_width, rect_height, gaps, y_off_add

# Draw individual rectangle column (side of the bracket per round)
def draw_column(x_offset, preds, num_rectangles, ax, text_color='black'):
    
    # Grab out the plot params
    rect_width, rect_height, gaps, y_off_add = get_plot_params()
    
    # Change font to be similar to ESPN
    plt.rcParams.update({
        "font.family": "Arial",
    })
        
    # Add option for user to make plot dark mode
    if text_color == 'black':
        plt.rcParams.update({
            "lines.color": "black",
            "patch.edgecolor": "black",
            "text.color": text_color,
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "grid.color": "black",
            "figure.facecolor": "white",
            "figure.edgecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white"
        })
        face_color = 'white'
    else:
        plt.rcParams.update({
            "lines.color": "white",
            "patch.edgecolor": "white",
            "text.color": text_color,
            "axes.facecolor": "black",
            "axes.edgecolor": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "grid.color": "white",
            "figure.facecolor": "black",
            "figure.edgecolor": "black",
            "savefig.facecolor": "black",
            "savefig.edgecolor": "black"
        })
        face_color = 'black'
    
    # Copy and reverse our original list (for plotting reasons)
    preds_copy = preds.copy()[::-1]
    
    # Iterate over each rectangle in the column
    for i in range(num_rectangles):
        
        # Set the proper spacing amount and y offset amount
        gap = gaps[num_rectangles]
        y_offset = i * (rect_height + gap) + y_off_add[num_rectangles]
        
        # Add the rectangle to the plot to store matchup data
        rect = plt.Rectangle((x_offset, y_offset), rect_width, rect_height, edgecolor=text_color, facecolor=face_color)
        ax.add_patch(rect)
       
        # Left columns
        if x_offset < 0:
            # As long as not semis or later, draw connecting line to next round
            if num_rectangles > 1:
                ax.plot([x_offset, x_offset + 3*rect_width/2], [y_offset + rect_height / 2] * 2, color=text_color, linewidth=1)
            else:
                ax.plot([x_offset, x_offset + 2*rect_width], [y_offset + rect_height / 2] * 2, color=text_color, linewidth=1)
                
            # Direction of connecting "elbow" depends on if matchup is below or above middle line
            if i % 2 == 1 and num_rectangles > 1:
                ax.plot([x_offset + 3*rect_width/2] * 2, [y_offset + rect_height / 2, y_offset - gap/2 + rect_height/2], color=text_color, linewidth=1)
            elif i % 2 == 0 and num_rectangles > 1:
                ax.plot([x_offset + 3*rect_width/2] * 2, [y_offset + rect_height / 2, y_offset + gap/2 + rect_height/2], color=text_color, linewidth=1)
        
        # Right columns
        elif x_offset > 0:
            # As long as not semis or later, draw connecting line to next round
            if num_rectangles > 1:
                ax.plot([x_offset - rect_width/2, x_offset + rect_width], [y_offset + rect_height / 2] * 2, color=text_color, linewidth=1)
            else:
                ax.plot([x_offset - rect_width, x_offset + rect_width], [y_offset + rect_height / 2] * 2, color=text_color, linewidth=1)
            
            # Direction of connecting "elbow" depends on if matchup is below or above middle line
            if i % 2 == 1 and num_rectangles > 1:
                ax.plot([x_offset - rect_width/2] * 2, [y_offset + rect_height / 2, y_offset - gap/2 + rect_height/2], color=text_color, linewidth=1)
            elif i % 2 == 0 and num_rectangles > 1:
                ax.plot([x_offset - rect_width/2] * 2, [y_offset + rect_height / 2, y_offset + gap/2 + rect_height/2], color=text_color, linewidth=1)
        
        # Center (finals)
        else:
            ax.plot([x_offset, x_offset + rect_width], [y_offset + rect_height / 2] * 2, color=text_color, linewidth=1)
        
        
        # Get the team ames for the matchup
        team1_text = f"({preds_copy[i][0][2]}) {preds_copy[i][0][0]}"
        team1_short_name = preds_copy[i][0][0]
                
        team2_text = f"({preds_copy[i][1][2]}) {preds_copy[i][1][0]}"
        team2_short_name = preds_copy[i][1][0]
        
        # Set specific params for logo placements
        desired_height = 9
        within_box_offset = 0.2
        img_left_offset = 0.5
        
        # Add team names (as long as > 1 team left)
        if len(preds_copy) > 0 and len(preds_copy[i]) > 1:
            
            # Get the ESPN-style name for the team so we can get the logo image
            team1_full_name = logo_mapper[logo_mapper['short_name'] == team1_short_name]['full_name'].iloc[0]
            team2_full_name = logo_mapper[logo_mapper['short_name'] == team2_short_name]['full_name'].iloc[0]

            team1_best_match = process.extractOne(team1_full_name, logos['Image Alt'])
            team2_best_match = process.extractOne(team2_full_name, logos['Image Alt'])
            
            # If left side or finals:
            if x_offset <= 0:
                
                # Put int the team names
                ax.text(
                    x_offset + rect_width / 3, 
                    y_offset + (3/4) * rect_height, 
                    team1_text, 
                    fontsize=5 if len(team1_text) > 15 else (7.5 if len(team1_text) < 10 else 6), 
                    ha='left', 
                    va='center',
                    color=text_color
                )
                ax.text(
                    x_offset + rect_width / 3, 
                    y_offset + (1/4) * rect_height, 
                    team2_text, 
                    fontsize=5 if len(team2_text) > 15 else (7.5 if len(team2_text) < 10 else 6), 
                    ha='left', 
                    va='center',
                    color=text_color
                )
                
                # Get the images from our predefined logos and add them to the plot
                img1 = Image.open(urlopen(logos[logos['Image Alt'] == team1_best_match[0]]['Image Src'].iloc[0].replace("80", "200")))
                img1 = np.array(img1)

                height, width, _ = img1.shape
                aspect_ratio = width / height

                # Calculate the corresponding width to maintain aspect ratio
                desired_width = desired_height * aspect_ratio

                # Create OffsetImage with the calculated size
                imagebox1 = OffsetImage(img1, zoom=desired_height / height)

                ab1 = AnnotationBbox(imagebox1, 
                                    (x_offset + img_left_offset, y_offset + rect_height - within_box_offset),
                                    frameon=False, 
                                    xycoords='data', 
                                    boxcoords="data")
                ax.add_artist(ab1)

                img2 = Image.open(urlopen(logos[logos['Image Alt'] == team2_best_match[0]]['Image Src'].iloc[0].replace("80", "200")))
                img2 = np.array(img2)
                
                height, width, _ = img2.shape
                aspect_ratio = width / height

                # Calculate the corresponding width to maintain aspect ratio
                desired_width = desired_height * aspect_ratio

                # Create OffsetImage with the calculated size
                imagebox2 = OffsetImage(img2, zoom=desired_height / height)

                ab2 = AnnotationBbox(imagebox2, 
                                    (x_offset + img_left_offset, y_offset + within_box_offset),
                                    frameon=False, 
                                    xycoords='data', 
                                    boxcoords="data")
                ax.add_artist(ab2)
                
            # If right side:
            else:
                
                # Put int the team names
                ax.text(
                    x_offset, 
                    y_offset + (3/4) * rect_height, 
                    team1_text, 
                    fontsize=5 if len(team1_text) > 15 else (7.5 if len(team1_text) < 10 else 6), 
                    ha='left', 
                    va='center',
                    color=text_color
                )
                ax.text(
                    x_offset, 
                    y_offset + (1/4) * rect_height, 
                    team2_text, 
                    fontsize=5 if len(team2_text) > 15 else (7.5 if len(team2_text) < 10 else 6), 
                    ha='left', 
                    va='center',
                    color=text_color
                )
                
                # Get the images from our predefined logos and add them to the plot
                img1 = Image.open(urlopen(logos[logos['Image Alt'] == team1_best_match[0]]['Image Src'].iloc[0].replace("80", "200")))
                img1 = np.array(img1)

                height, width, _ = img1.shape
                aspect_ratio = width / height

                # Calculate the corresponding width to maintain aspect ratio
                desired_width = desired_height * aspect_ratio

                # Create OffsetImage with the calculated size
                imagebox1 = OffsetImage(img1, zoom=desired_height / height)


                ab1 = AnnotationBbox(imagebox1, 
                                    (x_offset + 4.25, y_offset + rect_height - within_box_offset),
                                    frameon=False, 
                                    xycoords='data', 
                                    boxcoords="data")
                ax.add_artist(ab1)

                img2 = Image.open(urlopen(logos[logos['Image Alt'] == team2_best_match[0]]['Image Src'].iloc[0].replace("80", "200")))
                img2 = np.array(img2)
                
                height, width, _ = img2.shape
                aspect_ratio = width / height

                # Calculate the corresponding width to maintain aspect ratio
                desired_width = desired_height * aspect_ratio

                # Create OffsetImage with the calculated size
                imagebox2 = OffsetImage(img2, zoom=desired_height / height)

                ab2 = AnnotationBbox(imagebox2, 
                                    (x_offset + 4.25, y_offset + within_box_offset),
                                    frameon=False, 
                                    xycoords='data', 
                                    boxcoords="data")
                ax.add_artist(ab2)

# Draw the bracket!
def draw_bracket(plot_feeder, name="My", text_color='black'):
    # Initialize the figure
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Make font Arial
    plt.rcParams.update({
        "font.family": "Arial",
    })
    
    # Allow user to make plot dark mode if dsired
    if text_color == 'black':
        plt.rcParams.update({
            "lines.color": "black",
            "patch.edgecolor": "black",
            "text.color": text_color,
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "grid.color": "black",
            "figure.facecolor": "white",
            "figure.edgecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white"
        })
    else:
        plt.rcParams.update({
        "lines.color": "white",
        "patch.edgecolor": "white",
        "text.color": text_color,
        "axes.facecolor": "black",
        "axes.edgecolor": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "grid.color": "white",
        "figure.facecolor": "black",
        "figure.edgecolor": "black",
        "savefig.facecolor": "black",
        "savefig.edgecolor": "black"
    })

    # Specify the locations and number of rects needed per col
    dists = [-30, -25, -20, -15, -10, 0, 10, 15, 20, 25, 30]
    n_cols = [16, 8, 4, 2, 1, 1, 1, 2, 4, 8, 16]
    
    # Round indices to get the proper team names out
    rounds = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]

    # Iterate over and draw each col
    for i in range(0, len(dists)):
        if i <= 5:
            draw_column(dists[i], plot_feeder[rounds[i]][:n_cols[i]], n_cols[i], ax, text_color)
        else:
            draw_column(dists[i], plot_feeder[rounds[i]][n_cols[i]:], n_cols[i], ax, text_color)
         
    # Get out rect height for plotting
    _, rect_height_help, _, _ = get_plot_params()

    # Set axis limits and remove axes
    ax.set_xlim(-30, 35)
    
    # Hard coded based on initial params but we can change later if we need to
    ax.set_ylim(-1, 16 * (rect_height_help + 0.5))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    # Set the plot name properly
    if name != "My":
        name += "'s"
    
    plt.title(f"{name} 2025 March Tournament Bracket".upper())
    
    # Add in special logo and text for their predicted winner!
    ax.text(2.5, 21, "My 2025 Champion", ha='center', weight='bold')
    
    champ_short_name = plot_feeder[-1][0][0][0]
    
    champ_full_name = logo_mapper[logo_mapper['short_name'] == champ_short_name]['full_name'].iloc[0]
    champ_best_match = process.extractOne(champ_full_name, logos['Image Alt'])
    
    champ = Image.open(urlopen(logos[logos['Image Alt'] == champ_best_match[0]]['Image Src'].iloc[0].replace("80", "200")))
    champ = np.array(champ)

    imagebox = OffsetImage(champ, zoom=0.7)


    ab = AnnotationBbox(imagebox, 
                        (2.5, 17),
                        frameon=False, 
                        xycoords='data', 
                        boxcoords="data")  # Adjust as needed for positioning
    ax.add_artist(ab)
    
    plt.savefig('mybracket.png', bbox_inches='tight')
    plt.close()
    
def display_bracket(filename='mybracket.png'):
    display(ImageDisplay(filename)) 
