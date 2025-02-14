import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm  # For progress bar

# Define the Coordinate Attention (CA) Module
class CoordinateAttention(nn.Module):
    def __init__(self, in_channels):
        super(CoordinateAttention, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(in_channels, in_channels, kernel_size=1, padding=0)
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(in_channels, in_channels, kernel_size=1, padding=0)
        )

    def forward(self, x):
        # Compute channel attention
        channel_att = self.channel_attention(x)
        channel_att = torch.sigmoid(channel_att)

        # Compute spatial attention
        spatial_att = self.spatial_attention(x)
        spatial_att = torch.sigmoid(spatial_att)

        # Apply both channel and spatial attention
        out = x * channel_att * spatial_att
        return out

# Define the 3D Inception Block with CA
class Inception3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Inception3DBlock, self).__init__()
        self.conv1x1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv3x3x3 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
        
        self.ca = CoordinateAttention(out_channels)

    def forward(self, x):
        # Apply 1x1x1 convolution
        conv1x1x1 = self.conv1x1x1(x)
        # Apply 3x3x3 convolution
        conv3x3x3 = self.conv3x3x3(x)
        # Apply max pooling
        pool = self.pool(x)
        
        # Concatenate all outputs
        out = torch.cat([conv1x1x1, conv3x3x3, pool], dim=1)
        # Apply Coordinate Attention
        out = self.ca(out)
        return out

# Define the CA-I3D model
class CAI3D(nn.Module):
    def __init__(self, num_classes):
        super(CAI3D, self).__init__()

        # Initial 3x3x3 Convolution
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        
        # First 3D CA-Inception v1 Block
        self.inception1 = Inception3DBlock(64, 64)
        
        # MaxPooling
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Second 3D CA-Inception v1 Block
        self.inception2 = Inception3DBlock(192, 128)
        
        # MaxPooling
        self.pool2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Third 3D CA-Inception v1 Block
        self.inception3 = Inception3DBlock(256, 256)

        # Fully Connected Layer for final classification
        self.fc = nn.Linear(256 * 4 * 4 * 4, num_classes)

    def forward(self, x):
        # Initial convolution
        x = F.relu(self.conv1(x))

        # First Inception Block
        x = self.inception1(x)
        
        # MaxPooling
        x = self.pool1(x)

        # Second Inception Block
        x = self.inception2(x)

        # MaxPooling
        x = self.pool2(x)

        # Third Inception Block
        x = self.inception3(x)

        # Flatten and pass to fully connected layer
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)

        return x

# Define the training function
def train(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1} - Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    return running_loss / len(dataloader)

# Define the testing function
def test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return running_loss / len(dataloader), accuracy

# Define learning rate scheduler (StepLR)
def get_scheduler(optimizer):
    return StepLR(optimizer, step_size=5, gamma=0.7)

# Create model instance
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CAI3D(num_classes=10).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create learning rate scheduler
scheduler = get_scheduler(optimizer)

# Training and testing the model
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, criterion, optimizer, device, epoch)
    test_loss, accuracy = test(model, test_dataloader, criterion, device)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    # Step the learning rate scheduler
    scheduler.step()

    # Save the model checkpoint after every epoch
    if (epoch + 1) % 5 == 0:
        checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

# Save the final model
torch.save(model.state_dict(), 'ca_i3d_final.pth')
