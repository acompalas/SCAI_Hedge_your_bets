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
read_file = os.path.join(os.getcwd(), relative_path)

# Initialize NBADataProcessor and prepare the dataset
print("Processing Data ...")
data_processor = NBADataProcessor(file_path=read_file)
processed_df = data_processor.prepare_dataset()

# # Load the data
print("Loading training data...")
train_dataloader, features_columns, features_df = data_processor.load_training_data()

# Initialize the model
input_size = len(features_columns)
net = SimpleNet(input_size=input_size)

# Train the model
print("Training model ...")
loss_values, accuracy_values = net.train_model(train_dataloader, num_epochs=30, lr=0.001)

#Test model
print("Testing Model ...")
# Plot loss and accuracy
net.plot_loss(loss_values)
net.plot_accuracy(accuracy_values)

# Perform backtesting
net.backtest(features_df, features_columns)