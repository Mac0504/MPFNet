# main.py
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from CA-I3D import CAI3D
import cv2
import numpy as np

# 1. Define dataset paths
datasets = {
    'SMIC': '/path/to/sample_balanced_motion_amplified_ME_dataset/SMIC',
    'CASME_II': '/path/to/sample_balanced_motion_amplified_ME_dataset/CASME_II',
    'SAMM': '/path/to/sample_balanced_motion_amplified_ME_dataset/SAMM',
    'MEGC2019_CD': '/path/to/sample_balanced_motion_amplified_ME_dataset/MEGC2019_CD'
}

# Select the dataset to use (choose 'SMIC', 'CASME_II', 'SAMM', or 'MEGC2019_CD')
selected_dataset = 'SMIC'

# 2. Dataset loading and preprocessing
class ExpressionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = self.load_data()

    def load_data(self):
        data = []
        # Load all frames for each sequence in the dataset
        for label_folder in os.listdir(self.dataset_path):
            label_folder_path = os.path.join(self.dataset_path, label_folder)
            if os.path.isdir(label_folder_path):
                for seq_folder in os.listdir(label_folder_path):
                    seq_folder_path = os.path.join(label_folder_path, seq_folder)
                    if os.path.isdir(seq_folder_path):
                        frames = []
                        for frame_file in sorted(os.listdir(seq_folder_path)):
                            frame_path = os.path.join(seq_folder_path, frame_file)
                            frame = cv2.imread(frame_path)
                            frames.append(frame)
                        data.append((np.array(frames), label_folder))  # Save frames and corresponding label
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frames, label = self.data[idx]
        frames_tensor = [torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) for frame in frames]  # Convert to tensor (C, H, W)
        label_tensor = torch.tensor(int(label))  # Convert label to integer if it's categorical
        return torch.stack(frames_tensor), label_tensor

# 3. Cross-Entropy Loss function (already provided by PyTorch)
# CrossEntropyLoss expects the output of the model to be logits (raw scores for each class).
# The target is expected to be a tensor of class indices (integers).
# We use the default cross-entropy loss function from PyTorch.
class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, output, target):
        return F.cross_entropy(output, target)

# 4. Model training function
def train(model, train_loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0

    for frames, labels in train_loader:
        frames, labels = frames.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass through the model (input: 11 frames)
        output = model(frames)

        # Compute loss
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

# 5. Save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)

# 6. Main training loop
if __name__ == "__main__":
    # Dataset selection and DataLoader setup
    dataset_path = datasets[selected_dataset]
    train_dataset = ExpressionDataset(dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CAI3D().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = CrossEntropyLoss()

    # Train the model for a specified number of epochs
    num_epochs = 10
    for epoch in range(num_epochs):
        loss = train(model, train_loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    # Save the trained model
    save_model(model, 'models/PLSM.pth')
    print("Model saved to 'models/PLSM.pth'")
