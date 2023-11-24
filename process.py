import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

class NBADataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self._load_data()

    def _load_data(self):
        df = pd.read_csv(self.file_path, index_col=0)
        return df

    def _add_target(self, group):
        group = pd.concat([group, group["won"].shift(-1).rename("target")], axis=1)
        return group

    def _scale_data(self, df, selected_columns):
        scaler = MinMaxScaler()
        df[selected_columns] = scaler.fit_transform(df[selected_columns])
        return df

    def _calculate_rolling_averages(self, df, selected_columns):
        rolling = df[list(selected_columns) + ["won", "team", "season"]]
        
        def find_team_averages(team):
            team[selected_columns] = team[selected_columns].rolling(10).mean()
            return team

        rolling = rolling.groupby(["team", "season"], group_keys=False).apply(find_team_averages)
        rolling_cols = [f"{col}_10" for col in rolling.columns]
        rolling.columns = rolling_cols

        return rolling

    def _add_future_game_data(self, df):
        def shift_col(team, col_name):
            next_col = team[col_name].shift(-1)
            return next_col

        df["home_next"] = self._add_col(df, "home")
        df["team_opp_next"] = self._add_col(df, "team_opp")
        df["date_next"] = self._add_col(df, "date")

        return df

    def _add_col(self, df, col_name):
        return df.groupby("team", group_keys=False).apply(lambda x: self._shift_col(x, col_name))

    def _shift_col(self, team, col_name):
        next_col = team[col_name].shift(-1)
        return next_col

    def prepare_dataset(self):
        # Sort by date and drop irrelevant columns
        self.df = self.df.sort_values("date")
        self.df = self.df.reset_index(drop=True)
        del self.df["mp.1"]
        del self.df["mp_opp.1"]
        del self.df["index_opp"]

        #Add a target column (Whether or not team won next game)
        self.df = self.df.groupby("team", group_keys=False).apply(self._add_target)
        self.df.loc[pd.isnull(self.df["target"]), "target"] = 2
        self.df["target"] = self.df["target"].astype(int, errors="ignore")
        
        #Create copy dataframe without null values
        nulls = pd.isnull(self.df).sum()
        nulls = nulls[nulls > 0]
        valid_columns = self.df.columns[~self.df.columns.isin(nulls.index)]
        self.df = self.df[valid_columns].copy()
        
        #Convert boolean column to binary
        self.df['won'] = self.df['won'].astype(int)
        
        #Scale stat columns
        removed_columns = ["season", "date", "won", "target", "team", "team_opp"]
        selected_columns = self.df.columns[~self.df.columns.isin(removed_columns)]
        self.df = self._scale_data(self.df, selected_columns)
        rolling = self._calculate_rolling_averages(self.df, selected_columns)
        
        #Concatenate new columns back into dataframe, dropping null and resetting index
        self.df = pd.concat([self.df, rolling], axis=1)
        self.df = self.df.dropna()
        self.df = self.df.reset_index(drop=True)

        #Add future game data to columns
        self.df = self._add_future_game_data(self.df)

        return self.df
    
    def _extract_features(self, df):
        # Extract features for team_x and team_opp_next_x
        team_x_cols = [col for col in df.columns if '_10_x' in col and 'opp' not in col]
        team_opp_next_x_cols = [col for col in df.columns if 'opp_10_x' in col]

        # Concatenate features and rolling averages, including home_next
        features_columns = team_x_cols + team_opp_next_x_cols + ["home_next"]
        features_df = df[features_columns].copy()

        # Scale the features
        scaler = MinMaxScaler()
        features_df[features_columns] = scaler.fit_transform(features_df[features_columns])

        return features_df

class NBADataset(Dataset):
    def __init__(self, features, target):
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.target[idx], dtype=torch.float32)
        return x, y