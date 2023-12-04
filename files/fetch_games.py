import os
import json
import pandas as pd
from datetime import datetime, timedelta
from nba_api.live.nba.endpoints import scoreboard
import warnings

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

try:
    # Get Live NBA schedule and process
    print("Fetching live NBA data...")

    # Today's Score Board
    games = scoreboard.ScoreBoard()

    # Get the dictionary representation of the data
    games_data = games.get_dict()

    # Extract the 'games' key from the dictionary
    games_list = games_data['scoreboard']['games']

    # Convert the list of games to a DataFrame
    games_df = pd.DataFrame(games_list)

    # Check if games_df is empty
    if games_df.empty:
        print("No games today")
        # You can exit the script or add any other logic you need
        exit()

    # Continue with the rest of your code

except Exception as e:
    print(f"An error occurred: {e}")

#Clean NBA dataframe
print("Cleaning live data...")

# Drop specified columns and any columns with null values
columns_to_drop = ['gameCode','gameStatusText','gameId','gameStatus', 'period', 'gameClock', 'gameEt', 'regulationPeriods', 'ifNecessary', 'seriesGameNumber', 'seriesText', 'seriesConference', 'poRoundDesc', 'gameSubtype', 'gameLeaders', 'pbOdds']
games_df = games_df.drop(columns=columns_to_drop).dropna(axis=1, how='all')

# Extract 'teamName' from 'homeTeam' and 'awayTeam'
games_df['homeTeam'] = games_df['homeTeam'].apply(lambda x: x['teamName'] if isinstance(x, dict) and 'teamName' in x else None)
games_df['awayTeam'] = games_df['awayTeam'].apply(lambda x: x['teamName'] if isinstance(x, dict) and 'teamName' in x else None)

# Read abbreviations from the JSON file into a dictionary
json_path = os.path.join('datasets', 'teams.json')
file = os.path.join(os.path.dirname(os.getcwd()), json_path)
with open(file, 'r') as file:
    team_name_to_abbr = json.load(file)

# Function to replace team names with abbreviations
def replace_team_with_abbr(team_name):
    for abbr, full_name in team_name_to_abbr.items():
        if team_name in full_name:
            return abbr
    return team_name

# Replace team names with abbreviations in the homeTeam and awayTeam columns
games_df['homeTeam'] = games_df['homeTeam'].apply(replace_team_with_abbr)
games_df['awayTeam'] = games_df['awayTeam'].apply(replace_team_with_abbr)

# Split 'gameTimeUTC' into 'date' and 'time' manually
games_df['date'] = games_df['gameTimeUTC'].str.split('T').str[0]
games_df['time'] = games_df['gameTimeUTC'].str.split('T').str[1].str[:-1]  # Remove the 'Z' at the end

# Convert 'time' column to PST in AM/PM format
games_df['time'] = pd.to_datetime(games_df['time'], format='%H:%M:%S') - pd.Timedelta(hours=8)
games_df['time'] = games_df['time'].dt.strftime('%I:%M %p').str.lstrip('0')  # Remove leading zero

# Drop the 'gameTimeUTC' column
games_df = games_df.drop('gameTimeUTC', axis=1)

# Specify the date and time format
date_time_format = "%Y-%m-%d %I:%M %p"

# Combine 'date' and 'time' columns and convert to datetime
games_df['datetime'] = pd.to_datetime(games_df['date'] + ' ' + games_df['time'], format=date_time_format)

# Sort DataFrame by date and time
games_df = games_df.sort_values(by=['datetime'])

# Reset the index
games_df = games_df.reset_index(drop=True)

# Drop the 'datetime' column if you don't need it in the final CSV
games_df = games_df.drop(columns=['datetime'])

# Specify the relative path
relative_path = 'datasets'
save_path = os.path.join(os.path.dirname(os.getcwd()), relative_path)
# Save the sorted and reset index DataFrame to a CSV file in the 'datasets' folder
games_df.to_csv(os.path.join(save_path, 'games.csv'), index=False)

# Specify the relative paths to the CSV files
games_path = 'datasets/games.csv' # Path to save games data
processed_data_path = 'datasets/processed_data.csv' # Path to our feature data frame
read_file = os.path.join(os.path.dirname(os.getcwd()), processed_data_path)
game_file = os.path.join(os.path.dirname(os.getcwd()), games_path)

# Read the processed data CSV file to get column headers
processed_data_df = pd.read_csv(read_file)
column_headers = processed_data_df.columns.tolist()

# Create an empty DataFrame with the column headers
predict_df = pd.DataFrame(columns=column_headers)

# Read the games.csv file
games_df = pd.read_csv(game_file)

# Add column headers from processed_df to dataframe
print("Adding most recent averages...")
for index, row in games_df.iterrows():
    # Create a row for the home team
    home_row = pd.Series(index=column_headers)
    home_row['team_x'] = row['homeTeam']
    home_row['team_opp_next_x'] = row['awayTeam']
    home_row['team_y'] = row['awayTeam']
    home_row['team_opp_next_y'] = row['homeTeam']
    home_row['home_next'] = 1
    home_row['date_next'] = row['date']
    
    # Use df.loc to append the row to predict_df
    predict_df.loc[len(predict_df)] = home_row

    # Create a row for the away team
    away_row = pd.Series(index=column_headers)
    away_row['team_x'] = row['awayTeam']
    away_row['team_opp_next_x'] = row['homeTeam']
    away_row['home_next'] = 0
    away_row['team_y'] = row['homeTeam']
    away_row['team_opp_next_y'] = row['awayTeam']
    away_row['date_next'] = row['date']

    # Use df.loc to append the row to predict_df
    predict_df.loc[len(predict_df)] = away_row
    
    # Select columns containing 'season'
    season_columns = predict_df.filter(like='season')

    # Set all cells in the selected columns to 2023
    predict_df.loc[len(predict_df) - 1, season_columns.columns] = 2023
    predict_df.loc[len(predict_df) - 2, season_columns.columns] = 2023

# Add rolling averages to NBA teams
for index, row in predict_df.iterrows():
    # Iterate through processed_data_df starting from the last row for team_x
    found_x = False
    for _, team_x_row in processed_data_df[processed_data_df['team_x'].isin([row['team_x'], 'BNK', 'BRK'])].iloc[::-1].iterrows():
        # Find the columns with '10_x' suffix not including 'opp_10_x'
        columns_to_fill_x = team_x_row.index[team_x_row.index.str.endswith('10_x') & ~team_x_row.index.str.endswith('opp_10_x')]

        # Switch 'BRK' to 'BNK'
        if team_x_row['team_x'] == 'BRK':
            team_x_row['team_x'] = 'BNK'

        # Fill in the corresponding columns in predict_df for team_x
        for col_x in columns_to_fill_x:
            predict_df.at[index, col_x] = team_x_row[col_x]
        found_x = True
        break

    # If no matching row is found for team_x, you can handle this case accordingly
    if not found_x:
        print(f"No match found for team_x: {row['team_x']}")

    # Repeat the process for team_y
    found_y = False
    for _, team_y_row in processed_data_df[processed_data_df['team_y'].isin([row['team_y'], 'BNK', 'BRK'])].iloc[::-1].iterrows():
        # Find the columns with '10_y' suffix not including 'opp_10_y'
        columns_to_fill_y = team_y_row.index[team_y_row.index.str.endswith('10_y') & ~team_y_row.index.str.endswith('opp_10_y')]

        # Switch 'BRK' to 'BNK'
        if team_y_row['team_y'] == 'BRK':
            team_y_row['team_y'] = 'BNK'

        # Fill in the corresponding columns in predict_df for team_y
        for col_y in columns_to_fill_y:
            predict_df.at[index, col_y] = team_y_row[col_y]
        found_y = True
        break

    # If no matching row is found for team_y, you can handle this case accordingly
    if not found_y:
        print(f"No match found for team_y: {row['team_y']}")
        
    
# Drop certain columns from predict_df
columns_to_drop = ['season_10_x', 'season_10_y', 'team_10_x', 'team_10_y', 'won_10_y']
predict_df = predict_df.drop(columns=columns_to_drop, errors='ignore')

# Drop columns with NaN values
predict_df = predict_df.dropna(axis=1, how='all')

# Save the resulting DataFrame to a new CSV file
predict_data_path = 'datasets/predict.csv'
predict_file = os.path.join(os.path.dirname(os.getcwd()), predict_data_path)
predict_df.to_csv(predict_file, index=False)
print("Data to predict on saved to predict.csv")

