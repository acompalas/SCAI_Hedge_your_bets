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

print("Program start ...")
# Specify the relative path to the dataset
relative_path = 'datasets/nba_games.csv'
read_file = os.path.join(os.path.dirname(os.getcwd()), relative_path)

# Initialize NBADataProcessor and prepare the dataset
print("Processing Data ...")
data_processor = NBADataProcessor(file_path=read_file)
processed_df = data_processor.prepare_dataset()

# Specify the relative path to save the processed data
output_path = 'datasets/processed_data.csv'
write_file = os.path.join(os.path.dirname(os.getcwd()), output_path)

# Save the processed DataFrame to a CSV file
processed_df.to_csv(write_file, index=False)
print(f"Processed data saved to: {output_path}")

#Load Training Data
train_dataloader, test_dataloader, features_columns, features_df = data_processor.load_training_data()

# Initialize the model
input_size = len(features_columns)
net = SimpleNet(input_size=input_size)

# Print the number of features used to initialize the model
print(f"Trained on {input_size} features")

# Train the model
print("Training model ...")
# Train the model
loss_values, accuracy_values, val_loss_values, val_accuracy_values = net.train_model(train_dataloader, test_dataloader, num_epochs=20, lr=0.001)

#Test model
print("Testing Model ...")
# Plot loss and accuracy
# Plot loss and accuracy
net.plot_loss(loss_values, val_loss_values)
net.plot_accuracy(accuracy_values, val_accuracy_values)

# Perform backtesting
net.backtest(features_df, features_columns)

# Save Model in the current working directory
model_name = 'model_state_dict.pth'
torch.save(net.state_dict(), model_name)
print(f"Model saved at {model_name}")