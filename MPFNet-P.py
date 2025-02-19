import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, Dataset
import numpy as np
from CA_I3D import CAI3D 
import cv2 
import glob


# Dataset class for loading and processing data
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
            smic_dir = os.path.join(self.root_dir, 'ME_features', 'SMIC')
            flow_feature_dir = os.path.join(smic_dir, 'flow_feature')
            frame_diff_feature_dir = os.path.join(smic_dir, 'frame_diff_feature')
            
            flow_features = glob.glob(os.path.join(flow_feature_dir, '*.npy'))  # Assuming numpy files
            frame_diff_features = glob.glob(os.path.join(frame_diff_feature_dir, '*.npy'))
            
            for flow_file, frame_diff_file in zip(flow_features, frame_diff_features):
                flow_feature = np.load(flow_file)
                frame_diff_feature = np.load(frame_diff_file)
                
                # Concatenate along the channel dimension (assuming the shape is (height, width, time, channels))
                combined_feature = np.concatenate((flow_feature, frame_diff_feature), axis=-1)
                
                label = int(flow_file.split('_')[0])  # Extract label from filename, adjust this as needed
                
                self.data.append(combined_feature)
                self.labels.append(label)
        
        # Similarly, handle loading for other datasets like 'CASME II', 'SAMM', 'MEGC2019-CD'
        # For simplicity, assuming the same structure for all datasets (modify if needed)
        else:
            dataset_dir = os.path.join(self.root_dir, 'ME_features', self.dataset_name)
            feature_files = glob.glob(os.path.join(dataset_dir, '*.npy'))  # Assuming numpy files
            
            for feature_file in feature_files:
                feature = np.load(feature_file)
                label = int(feature_file.split('_')[0])  # Extract label from filename (adjust as needed)
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


# Meta-Learning Module: It uses GFE and AFE for feature extraction and computes similarity for classification
class MetaLearningModule(nn.Module):
    def __init__(self, gfe_model_path, afe_model_path, input_dim, output_dim, gamma=0.1):
        super(MetaLearningModule, self).__init__()
        
        # Load GFE and AFE models (using CAI3D class for both)
        self.gfe = CAI3D(input_dim=input_dim, output_dim=output_dim)  # GFE feature extractor
        self.afe = CAI3D(input_dim=input_dim, output_dim=output_dim)  # AFE feature extractor
        
        # Load pre-trained model weights for GFE and AFE
        self.gfe.load_state_dict(torch.load(gfe_model_path))
        self.afe.load_state_dict(torch.load(afe_model_path))
        
        # Set gamma to control the influence of AFE's similarity score
        self.gamma = gamma
    
    def forward(self, support_set, query_set):
        # GFE feature extraction
        support_gfe_features = self.gfe(support_set)  # Extract features from support set using GFE
        query_gfe_features = self.gfe(query_set)  # Extract features from query set using GFE
        wc_G = support_gfe_features.mean(dim=0)  # Compute average feature vector for each class in support set
        d_GFE = cosine_similarity(query_gfe_features.detach().cpu().numpy(), wc_G.detach().cpu().numpy())  # Compute cosine similarity
        
        # AFE feature extraction
        support_afe_features = self.afe(support_set)  # Extract features from support set using AFE
        query_afe_features = self.afe(query_set)  # Extract features from query set using AFE
        wc_A = support_afe_features.mean(dim=0)  # Compute average feature vector for each class in support set
        d_AFE = cosine_similarity(query_afe_features.detach().cpu().numpy(), wc_A.detach().cpu().numpy())  # Compute cosine similarity
        
        # Fusion of GFE and AFE similarities with weighted sum
        d_sum = d_GFE + self.gamma * d_AFE  # Weighted sum of the similarities
        return d_sum


# Classification Module: Takes fused similarity features and classifies them
class ClassificationModule(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationModule, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)  # Fully connected layer for classification
    
    def forward(self, x):
        return torch.softmax(self.fc(x), dim=1)  # Softmax to get probability distribution over classes


# Loss function (Cross-Entropy Loss) for training
criterion = nn.CrossEntropyLoss()

# Optimizer (Adam optimizer for model training)
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop function
def train(model, train_loader, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for support_set, query_set, labels in train_loader:
            optimizer.zero_grad()

            # Get the fused similarity vector from the MetaLearningModule
            d_sum = model(support_set, query_set)

            # Classify using the ClassificationModule
            output = ClassificationModule(d_sum)

            # Compute loss (Cross-Entropy)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}")


# Input feature dimensions (128x128x5x10)
input_dim = 128 * 128 * 5 * 10  # Flattened feature size
output_dim = 128  # Output dimensionality of the feature space
num_classes = 3  # Number of emotion classes (can be adjusted to 5 classes)

# Define paths to pre-trained models for GFE and AFE
gfe_model_path = "models/PLTN.pth"
afe_model_path = "models/PLSM.pth"

# Define the dataset to use
dataset_name = 'SMIC'  # Change to 'CASME II', 'SAMM', or 'MEGC2019-CD' as needed
root_dir = 'ME_features'  # Path to your ME_features directory

# Initialize the dataset and DataLoader
dataset = MEFeaturesDataset(dataset_name=dataset_name, root_dir=root_dir)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the model
model = MetaLearningModule(gfe_model_path, afe_model_path, input_dim, output_dim)

# Start training
train(model, train_loader)

# predicted_class = predict(model, support_set, query_set)
