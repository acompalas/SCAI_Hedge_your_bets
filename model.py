import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

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
    

# Set the device for PyTorch (assuming GPU is available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the seasons range (2016 - 2022)
seasons_range = range(2016, 2023)

# Loop through each season
for cutoff_season in seasons_range:
    # Divide the data into training and testing sets
    train_df = full[full["season"] < cutoff_season]
    test_df = full[full["season"] == cutoff_season]

    # Extract features and target for training and testing sets
    train_features = train_df[features_columns].values
    train_target = train_df[target_column].values

    test_features = test_df[features_columns].values
    test_target = test_df[target_column].values

    # Define datasets and dataloaders for training and testing
    train_dataset = NBADataset(train_features, train_target)
    test_dataset = NBADataset(test_features, test_target)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize and train the model
    net = Net()  # replace with the instantiation of your neural network
    net.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10

    for epoch in range(num_epochs):
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))

            loss.backward()
            optimizer.step()

    # Evaluate the model on the test set
    net.eval()
    with torch.no_grad():
        test_inputs, test_labels = test_dataloader
        test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

        test_outputs = net(test_inputs)
        test_loss = criterion(test_outputs, test_labels.unsqueeze(1))

    print(f"Season: {cutoff_season}, Test Loss: {test_loss.item():.4f}")
