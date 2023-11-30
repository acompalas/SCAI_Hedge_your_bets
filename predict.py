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
predict_path = 'datasets/predict.csv'
predict_file = os.path.join(os.getcwd(), predict_path)

# Load the prediction data
predict_df = pd.read_csv(predict_file)

# Extract features from the prediction dataset
removed_columns = list(predict_df.columns[predict_df.dtypes == "object"])
selected_columns = predict_df.columns[~predict_df.columns.isin(removed_columns)]

# Exclude columns with specific words
excluded_words = ["season", "date", "won", "target", "team", "team_opp"]
features_columns = [col for col in selected_columns if not any(word in col for word in excluded_words)]

predict_features = predict_df[features_columns].values

# Create a DataLoader for the prediction data
predict_dataset = NBADataset(predict_features, target=None)  # Set target to None
predict_dataloader = DataLoader(predict_dataset, batch_size=32, shuffle=False)

# Initialize the model
input_size = len(features_columns)
net = SimpleNet(input_size=input_size)

# Load the saved model state dictionary
model_name = 'model_state_dict.pth'
net.load_state_dict(torch.load(model_name), strict=False)
net.eval()  # Set the model to evaluation mode

# Make predictions
predictions = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    for inputs,labels in predict_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        predictions = torch.round(outputs)

        predictions.extend(predictions.cpu().numpy())

# Add the predictions and probabilities to the original prediction dataframe
predict_df['predicted_probability'] = predictions
predict_df['predicted_result'] = ['Win' if pred > 0.5 else 'Loss' for pred in predictions]

# Display the DataFrame with predicted results
result_df = predict_df[['date', 'team', 'team_opp', 'predicted_result', 'predicted_probability']]

# Print team matchups and predicted results with probabilities
for index, row in result_df.iterrows():
    print(f"{row['team']} vs {row['team_opp']}: {row['predicted_result']} with {row['predicted_probability'] * 100:.2f}% probability")
