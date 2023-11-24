import os
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from model import Net, NBADataset
from process import NBADataProcessor

# Specify the relative path to the dataset
relative_path = 'datasets/nba_games.csv'
read_file = os.path.join(os.getcwd(), relative_path)

# Initialize NBADataProcessor and prepare the dataset
data_processor = NBADataProcessor(file_path=read_file)
processed_df = data_processor.prepare_dataset()

# Extract features and target
features_columns = ["team_rolling_10", "team_opp_rolling_10", "home_next"]
target_column = "target"

features = processed_df[features_columns].values
target = processed_df[target_column].values

# Define dataset and dataloader
dataset = NBADataset(features, target)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
