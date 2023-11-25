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

class SimpleNet(nn.Module):
    def __init__(self, input_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, 64)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    
    def train_model(self, train_dataloader, num_epochs=30, lr=0.001):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)

        loss_values = []  # To store the loss for each epoch
        accuracy_values = []  # To store the accuracy for each epoch

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

            # Learning rate scheduling
            if epoch % 5 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.9  # Reduce learning rate by 10% every 5 epochs

            # Calculate accuracy and loss for the epoch
            accuracy = (total_correct / total_examples) * 100.0
            average_loss = epoch_loss / len(train_dataloader)

            # Store loss and accuracy values
            loss_values.append(average_loss)
            accuracy_values.append(accuracy)

            # Print the loss and accuracy for each epoch
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")

        return loss_values, accuracy_values
    
    def plot_loss(self, loss_values, title="Training Loss"):
        plt.plot(loss_values, marker='o')
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    def plot_accuracy(self, accuracy_values, title="Training Accuracy"):
        plt.plot(accuracy_values, marker='o', color='green')
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
    