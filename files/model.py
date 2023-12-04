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
from torch.optim.lr_scheduler import ReduceLROnPlateau

class SimpleNet(nn.Module):
    def __init__(self, input_size):
        super(SimpleNet, self).__init__()
        #self.batch_norm1 = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        #x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
    def train_model(self, train_dataloader,test_dataloader, num_epochs=20, lr=0.001):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        loss_values = []  # To store the loss for each epoch
        accuracy_values = []  # To store the accuracy for each epoch
        val_loss_values = []
        val_accuracy_values = []
        
        # Define a scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5, verbose=False)

        for epoch in range(num_epochs):
            self.train()
            total_correct = 0
            total_examples = 0
            epoch_loss = 0.0

            for inputs, labels in train_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = self(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))

                # Calculate accuracy
                predictions = (outputs > 0.5).float()
                correct = (predictions == labels.unsqueeze(1)).float().sum()
                total_correct += correct
                total_examples += labels.size(0)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Validation loop
            self.eval()  # Set the model to evaluation mode

            val_correct = 0
            val_total = 0
            val_loss = 0.0

            with torch.no_grad():
                for val_inputs, val_labels in test_dataloader:
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

                    val_outputs = self(val_inputs)
                    val_loss += criterion(val_outputs, val_labels.unsqueeze(1)).item()

                    # Calculate accuracy
                    val_predictions = (val_outputs > 0.5).float()
                    val_correct += (val_predictions == val_labels.unsqueeze(1)).float().sum()
                    val_total += val_labels.size(0)

            val_accuracy = (val_correct / val_total) * 100.0
            val_average_loss = val_loss / len(test_dataloader)
            val_loss_values.append(val_average_loss)
            val_accuracy_values.append(val_accuracy)

            scheduler.step(val_average_loss)  # Update the scheduler based on the validation loss

            # Calculate accuracy and loss for the epoch
            accuracy = (total_correct / total_examples) * 100.0
            average_loss = epoch_loss / len(train_dataloader)

            # Store loss and accuracy values
            loss_values.append(average_loss)
            accuracy_values.append(accuracy)

            # Print the loss, accuracy, and validation metrics for each epoch
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%, Validation Loss: {val_average_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        return loss_values, accuracy_values, val_loss_values, val_accuracy_values
    
    def plot_loss(self, loss_values, val_loss_values, title="Training Loss"):
        plt.plot(loss_values, marker='o', label='Training Loss')
        plt.plot(val_loss_values, marker='o', label='Validation Loss', color = 'green')
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    def plot_accuracy(self, accuracy_values, val_accuracy_values, title="Training Accuracy"):
        plt.plot(accuracy_values, marker='o')
        plt.plot(val_accuracy_values, marker='o', color='green')
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.show()

    def backtest(self, data, predictors, start=0, step=1):
        seasons = sorted(data["season"].unique())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i in range(start, len(seasons), step):
            season = seasons[i]
            test_df = data[data["season"] == season]

            target_column = "target"

            # Extract features
            test_features = test_df[predictors].values
            test_target = test_df[target_column].values

            test_dataset = NBADataset(test_features, test_target)
            test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            self.eval()  # Set the model to evaluation mode

            all_labels = []
            all_predictions = []

            with torch.no_grad():
                correct = 0
                total = 0

                for inputs, labels in test_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = self(inputs)
                    predicted = torch.round(outputs)

                    total += labels.size(0)
                    correct += (predicted == labels.unsqueeze(1)).sum().item()

                    # Collect labels and predictions for confusion matrix
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy().flatten())

            accuracy = correct / total
            print(f"Season {season} Accuracy: {accuracy * 100:.2f}%")

            # Create confusion matrix
            conf_matrix = confusion_matrix(all_labels, all_predictions)

            # Plot confusion matrix using seaborn
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
            plt.title(f"Confusion Matrix - Season {season}")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.show()
    