import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.optim import SGD
from CA_I3D import CAI3D  

class MEFeaturesDataset(Dataset):
    def __init__(self, dataset_name, root_dir, transform=None):
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        
        self._load_data()

    def _load_data(self):
        if self.dataset_name == 'SMIC':
            smic_dir = os.path.join(self.root_dir, 'motion_amplified_ME_features', 'SMIC')
            flow_feature_dir = os.path.join(smic_dir, 'flow_feature')
            frame_diff_feature_dir = os.path.join(smic_dir, 'frame_diff_feature')
            
            flow_features = glob.glob(os.path.join(flow_feature_dir, '*.npy'))
            frame_diff_features = glob.glob(os.path.join(frame_diff_feature_dir, '*.npy'))
            
            for flow_file, frame_diff_file in zip(flow_features, frame_diff_features):
                flow_feature = np.load(flow_file)
                frame_diff_feature = np.load(frame_diff_file)
                
                # Concatenate along the channel dimension 
                combined_feature = np.concatenate((flow_feature, frame_diff_feature), axis=-1)
                
                label = int(flow_file.split('_')[0])  
                self.data.append(combined_feature)
                self.labels.append(label)
        
        # For other datasets like 'CASME II', 'SAMM', 'MEGC2019-CD'
        elif self.dataset_name in ['CASME II', 'SAMM', 'MEGC2019-CD']:
            dataset_dir = os.path.join(self.root_dir, 'motion_amplified_ME_features', self.dataset_name)
            feature_files = glob.glob(os.path.join(dataset_dir, '*.npy'))
            
            for feature_file in feature_files:
                feature = np.load(feature_file)
                label = int(feature_file.split('_')[0])  
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


# Function to save the model
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to '{filepath}'")

# Training setup
def train_model(dataset_name, root_dir, num_classes=3, batch_size=32, epochs=80, initial_lr=0.001, lr_decay_epoch=10):
    # Create dataset and dataloaders
    train_dataset = MEFeaturesDataset(dataset_name=dataset_name, root_dir=root_dir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CAI3D(num_classes=num_classes).to(device)

    # Define loss function and optimizer (SGD with momentum and weight decay)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=5e-4)

    # Learning rate scheduler to decay the learning rate every 10 epochs by a factor of 10
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_epoch, gamma=0.1)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Update weights
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

        # Decay learning rate at every 10 epochs
        scheduler.step()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_preds / total_preds

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save the trained model
    save_model(model, 'models/PLSM.pth')  

# Example usage
dataset_name = 'SMIC'  
root_dir = 'motion_amplified_ME_features'  
train_model(dataset_name, root_dir)
