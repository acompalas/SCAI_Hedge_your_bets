import os
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from model import Net
from process import NBADataProcessor, NBADataset

# Specify the relative path to the dataset
relative_path = 'datasets/nba_games.csv'
read_file = os.path.join(os.getcwd(), relative_path)

# Initialize NBADataProcessor and prepare the dataset
data_processor = NBADataProcessor(file_path=read_file)
processed_df = data_processor.prepare_dataset()

# Extract features and target using the _extract_features method
features_df = data_processor._extract_features(processed_df)

# Define features and target
features_columns = features_df.columns.tolist()
target_column = "target"

features = features_df.values
target = processed_df[target_column].values

# Define dataset and dataloader
dataset = NBADataset(features, target)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize and train the model
input_size = len(features_columns)
net = Net(input_size=input_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Training loop
num_epochs = 30

for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))

        loss.backward()
        optimizer.step()

    # Print the loss for each epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

# Save the trained model
torch.save(net.state_dict(), "trained_model.pth")
