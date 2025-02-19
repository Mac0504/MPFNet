# main.py
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from CA-I3D import CAI3D
import os
import numpy as np

# 1. Define dataset selection variables
datasets = {
    'SMIC': 'raw_samples/SMIC',
    'CASME_II': 'raw_samples/CASME_II',
    'SAMM': 'raw_samples/SAMM',
    'MEGC2019_CD': 'raw_samples/MEGC2019_CD'
}

# Select the dataset to use (you can change this based on your preference)
selected_dataset = 'SMIC'  # Choose 'SMIC', 'CASME_II', 'SAMM', or 'MEGC2019_CD'

# 2. Define dataset loader (assuming the datasets are preprocessed into feature vectors)
class ExpressionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = self.load_data()

    def load_data(self):
        # Load dataset (assumed that the features are pre-processed into numpy arrays)
        # Example: Load stored feature vectors and labels
        data = np.load(os.path.join(self.dataset_path, 'features.npy'))
        labels = np.load(os.path.join(self.dataset_path, 'labels.npy'))
        return list(zip(data, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature, label = self.data[idx]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# 3. Define Triplet Loss function
class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Compute distances between anchor-positive and anchor-negative pairs
        positive_distance = torch.norm(anchor - positive, p=2, dim=1)
        negative_distance = torch.norm(anchor - negative, p=2, dim=1)
        
        # Compute the triplet loss (with margin)
        loss = torch.mean(torch.clamp(positive_distance - negative_distance + self.margin, min=0))
        return loss

# 4. Define the training function
def train(model, train_loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0

    for anchor, positive, negative in train_loader:
        # Move tensors to the selected device (GPU or CPU)
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass: pass anchor, positive, and negative through the model
        anchor_output = model(anchor)
        positive_output = model(positive)
        negative_output = model(negative)

        # Calculate the triplet loss
        loss = loss_fn(anchor_output, positive_output, negative_output)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Return the average loss for this epoch
    return running_loss / len(train_loader)

# 5. Define the model saving function
def save_model(model, path):
    # Save the model's parameters (weights)
    torch.save(model.state_dict(), path)

# 6. Main program
if __name__ == "__main__":
    # Select the dataset path based on the chosen dataset
    dataset_path = datasets[selected_dataset]
    # Load the training dataset
    train_dataset = ExpressionDataset(dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize the model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CAI3D().to(device)  # Use CAI3D model (defined in CA-I3D.py)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = TripletLoss(margin=1.0)

    # Train the model for a given number of epochs
    num_epochs = 10
    for epoch in range(num_epochs):
        loss = train(model, train_loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    # Save the trained model parameters
    save_model(model, 'models/PLTN.pth')
    print("Model saved to 'models/PLTN.pth'")
