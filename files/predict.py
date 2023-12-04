import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch.optim as optim
from process import NBADataProcessor, NBADataset
from model import SimpleNet

# Specify the path to the prediction dataset
print("Loading data...")
predict_path = 'datasets/predict.csv'
predict_file = os.path.join(os.path.dirname(os.getcwd()), predict_path)

# Load the prediction data
predict_df = pd.read_csv(predict_file)

# Add a target column with all zeros
predict_df['target'] = 0

# Extract features from the prediction dataset
print("Extracting features...")
removed_columns = list(predict_df.columns[predict_df.dtypes == "object"])
selected_columns = predict_df.columns[~predict_df.columns.isin(removed_columns)]

# Exclude columns with specific words
excluded_words = ["season", "date", "won", "target", "team", "team_opp"]
features_columns = [col for col in selected_columns if not any(word in col for word in excluded_words)]

predict_features = predict_df[features_columns].values
predict_target = predict_df['target'].values  # Assuming 'target' is the name of your target column

# Create a DataLoader for the prediction data
predict_dataset = NBADataset(predict_features, target=predict_target)
predict_dataloader = DataLoader(predict_dataset, batch_size=32, shuffle=False)

# Initialize the model
input_size = len(features_columns)
net = SimpleNet(input_size=input_size)

# Load the saved model state dictionary
model_name = 'model_state_dict.pth'
net.load_state_dict(torch.load(model_name), strict=False)
net.eval()  # Set the model to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Make predictions
predictions = []
print("Making predictions...")

with torch.no_grad():
    for inputs, labels in predict_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        probabilities = outputs.cpu().numpy()  # Get probabilities as NumPy array
        predictions.extend(probabilities)

# Add the probabilities to the original prediction dataframe
predict_df['predicted_probability'] = predictions
predict_df['predicted_result'] = ['Win' if prob > 0.5 else 'Loss' for prob in predictions]

# Display the DataFrame with predicted results
result_df = predict_df[['date_next', 'team_x', 'team_opp_next_x', 'predicted_result', 'predicted_probability', 'season']]

# Create an empty DataFrame to store the game information
game_df = pd.DataFrame(columns=['Team vs Team_Opp', 'Winner', 'Loser', 'Season', 'Date'])

# Initialize variables to keep track of the current game
current_game = 1
current_winner = None

# Loop through the result_df and populate the game_df
for index, row in result_df.iterrows():
    probability = row['predicted_probability'][0]  # Extracting the scalar value

    # Check if it's the first row of a new game
    if index == 0 or row['predicted_result'] == 'Win':
        current_winner = row['team_x']
        game_df.loc[current_game - 1] = {
            'Team vs Team_Opp': f"{row['team_x']} vs {row['team_opp_next_x']}",
            'Winner': "",
            'Loser': "",
            'Season': row['season'],
            'Date': row['date_next']
        }

    # Check if the team won, and only add to the DataFrame if they did
    if row['predicted_result'] == 'Win':
        # Populate the 'Winner' and 'Loser' columns based on the second row of the game
        game_df.loc[current_game - 1, 'Winner'] = f"{current_winner} wins with {probability * 100:.2f}% probability"
        game_df.loc[current_game - 1, 'Loser'] = f"{row['team_opp_next_x']} loses with {100 - probability * 100:.2f}% probability"
        current_game += 1

predictions_path = 'datasets/predictions.csv'
predict_file = os.path.join(os.path.dirname(os.getcwd()), predictions_path)
# Save the resulting DataFrame to a new CSV file
game_df.to_csv(predict_file, index=False)
print("Predictions saved to predictions.csv")
