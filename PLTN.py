import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import glob
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from CA_I3D import CAI3D
import random


# TripletLoss definition
class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin  # The margin between the positive and negative pair distances
    
    def forward(self, anchor, positive, negative):
        # Compute the distance between the anchor and positive sample
        positive_distance = torch.norm(anchor - positive, p=2, dim=1)
        
        # Compute the distance between the anchor and negative sample
        negative_distance = torch.norm(anchor - negative, p=2, dim=1)
        
        # Compute the triplet loss with margin
        loss = torch.mean(torch.clamp(positive_distance - negative_distance + self.margin, min=0))
        
        return loss

# Define Dataset class
class MEFeaturesDataset(Dataset):
    def __init__(self, dataset_name, root_dir, num_classes=3, num_shots=5, transform=None):
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.num_classes = num_classes  # Number of classes for the task (3-way or 5-way)
        self.num_shots = num_shots  # Number of samples per class (5-shot)
        self.transform = transform
        self.data = []
        self.labels = []

        self._load_data()

    def _load_data(self):
        # Load dataset based on the dataset name
        if self.dataset_name == 'SMIC':
            smic_dir = os.path.join(self.root_dir, 'ME_features', 'SMIC')
            flow_feature_dir = os.path.join(smic_dir, 'flow_feature')
            frame_diff_feature_dir = os.path.join(smic_dir, 'frame_diff_feature')

            flow_features = glob.glob(os.path.join(flow_feature_dir, '*.npy'))
            frame_diff_features = glob.glob(os.path.join(frame_diff_feature_dir, '*.npy'))

            for flow_file, frame_diff_file in zip(flow_features, frame_diff_features):
                flow_feature = np.load(flow_file)
                frame_diff_feature = np.load(frame_diff_file)

                # Concatenate features along the channel dimension
                combined_feature = np.concatenate((flow_feature, frame_diff_feature), axis=-1)

                label = int(flow_file.split('_')[0])  # Extract label from filename

                self.data.append(combined_feature)
                self.labels.append(label)

        else:
            # For other datasets, directly load feature files
            dataset_dir = os.path.join(self.root_dir, 'ME_features', self.dataset_name)
            feature_files = glob.glob(os.path.join(dataset_dir, '*.npy'))

            for feature_file in feature_files:
                feature = np.load(feature_file)
                label = int(feature_file.split('_')[0])  # Extract label from filename
                self.data.append(feature)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            feature = self.transform(feature)

        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# Define training function
def train(model, train_loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0

    for anchor, positive, negative in train_loader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        optimizer.zero_grad()

        anchor_output = model(anchor)
        positive_output = model(positive)
        negative_output = model(negative)

        loss = loss_fn(anchor_output, positive_output, negative_output)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

# Define model saving function
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Main program
if __name__ == "__main__":
    dataset_name = 'SMIC'  # Choose dataset
    root_dir = 'ME_features' 
    train_dataset = MEFeaturesDataset(dataset_name=dataset_name, root_dir=root_dir, num_classes=3, num_shots=5)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)  # Updated batch size to 128

    # Initialize model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CAI3D().to(device)

    # SGD optimizer with momentum, learning rate, and weight decay
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)  # Updated optimizer

    loss_fn = TripletLoss(margin=1.0)

    # Train the model
    num_epochs = 60  # Updated number of epochs
    for epoch in range(num_epochs):
        loss = train(model, train_loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    # Save the model
    save_model(model, 'models/PLTN.pth')
    print("Model saved to 'models/PLTN.pth'")
