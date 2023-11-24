#Import Libraries
import csv
import pandas as pd
import os
import torch
from sklearn.preprocessing import MinMaxScaler

#Read csv and load into a dataframe
relative_path = 'datasets/nba_games.csv'
read_file = os.path.join(os.getcwd(), relative_path)
df = pd.read_csv(read_file, index_col = 0)

#Sort by date and drop irrelevant columns
df = df.sort_values("date")
df = df.reset_index(drop=True)
del df["mp.1"]
del df["mp_opp.1"]
del df["index_opp"]

# Add a target column: "Whether team won next game"
def add_target(group):
    group = pd.concat([group, group["won"].shift(-1).rename("target")], axis=1)
    return group

df = df.groupby("team", group_keys=False).apply(add_target)
df.loc[pd.isnull(df["target"]), "target"] = 2
df["target"] = df["target"].astype(int, errors="ignore")

#Create copy dataframe without null values
nulls = pd.isnull(df).sum()
nulls = nulls[nulls > 0]
valid_columns = df.columns[~df.columns.isin(nulls.index)]
df = df[valid_columns].copy()

# Convert boolean column 'won' to binary (0 and 1)
df['won'] = df['won'].astype(int)

#Select stat columns
removed_columns = ["season", "date", "won", "target", "team", "team_opp"]
selected_columns = df.columns[~df.columns.isin(removed_columns)]

#Scale stat columns
scaler = MinMaxScaler()
df[selected_columns] = scaler.fit_transform(df[selected_columns])

#Calculate rolling averages of stat columns
rolling = df[list(selected_columns) + ["won", "team", "season"]]

def find_team_averages(team):
    # Apply rolling mean only to numeric columns
    team[selected_columns] = team[selected_columns].rolling(10).mean()
    return team

# Group by team and season, then apply the rolling average function
rolling = rolling.groupby(["team", "season"], group_keys=False).apply(find_team_averages)
rolling_cols = [f"{col}_10" for col in rolling.columns]
rolling.columns = rolling_cols

#Concatenate new columns back into dataframe, dropping null rows and resetting index
df = pd.concat([df,rolling], axis = 1)
df = df.dropna()
df = df.reset_index(drop=True)

#Add future game data to columns
def shift_col(team, col_name):
    next_col = team[col_name].shift(-1)
    return next_col

def add_col(df, col_name):
    return df.groupby("team", group_keys=False).apply(lambda x: shift_col(x, col_name))

df["home_next"] = add_col(df, "home")
df["team_opp_next"] = add_col(df, "team_opp")
df["date_next"] = add_col(df, "date")

