import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import glob
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from CA_I3D import CAI3D  # Assuming CAI3D class is in CA-I3D.py
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

# Dataset class to load and preprocess the data
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
        if self.dataset_name == 'SMIC':
            smic_dir = os.path.join(self.root_dir, 'ME_features', 'SMIC')
            flow_feature_dir = os.path.join(smic_dir, 'flow_feature')
            frame_diff_feature_dir = os.path.join(smic_dir, 'frame_diff_feature')
            
            flow_features = glob.glob(os.path.join(flow_feature_dir, '*.npy'))
            frame_diff_features = glob.glob(os.path.join(frame_diff_feature_dir, '*.npy'))
            
            for flow_file, frame_diff_file in zip(flow_features, frame_diff_features):
                flow_feature = np.load(flow_file)
                frame_diff_feature = np.load(frame_diff_file)
                
                # Concatenate along the channel dimension 
                combined_feature = np.concatenate((flow_feature, frame_diff_feature), axis=-1)
                
                label = int(flow_file.split('_')[0])  # Extract label from filename
                
                self.data.append(combined_feature)
                self.labels.append(label)
        
        # For other datasets like 'CASME II', 'SAMM', 'MEGC2019-CD'
        else:
            dataset_dir = os.path.join(self.root_dir, 'ME_features', self.dataset_name)
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

    def create_task(self):
        """
        Create a task for 3-way 5-shot or 5-way 5-shot learning.
        Returns:
            support_set, query_set, support_labels, query_labels
        """
        # Randomly select `self.num_classes` classes for the task
        unique_classes = list(set(self.labels))
        selected_classes = random.sample(unique_classes, self.num_classes)

        support_set = []
        query_set = []
        support_labels = []
        query_labels = []

        # For each selected class, select `self.num_shots` for the support set and 1 for the query set
        for class_label in selected_classes:
            # Get all the sample indices corresponding to the class
            class_indices = [i for i, label in enumerate(self.labels) if label == class_label]
            # Randomly sample `num_shots + 1` samples (5 shots + 1 query sample)
            selected_samples = random.sample(class_indices, self.num_shots + 1)

            # First `num_shots` samples go to the support set, last one to the query set
            support_set.extend([self.data[i] for i in selected_samples[:-1]])
            query_set.extend([self.data[i] for i in selected_samples[-1:]])
            support_labels.extend([self.labels[i] for i in selected_samples[:-1]])
            query_labels.extend([self.labels[i] for i in selected_samples[-1:]])

        support_set = torch.stack(support_set)
        query_set = torch.stack(query_set)
        support_labels = torch.tensor(support_labels)
        query_labels = torch.tensor(query_labels)

        return support_set, query_set, support_labels, query_labels


# Meta-Learning Module: It uses GFE and AFE for feature extraction and computes similarity for classification
class MetaLearningModule(nn.Module):
    def __init__(self, gfe_model_path, afe_model_path, input_dim, output_dim, gamma=0.1):
        super(MetaLearningModule, self).__init__()
        
        # Load GFE and AFE models (using CAI3D class for both)
        self.gfe = CAI3D(input_dim=input_dim, output_dim=output_dim)  # GFE feature extractor
        self.afe = CAI3D(input_dim=input_dim + output_dim, output_dim=output_dim)  # AFE feature extractor
        
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

# Optimizer (SGD with momentum and weight decay)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)


# Training loop function
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

    return running_loss / len(train_loader)


# Model saving function
def save_model(model, path):
    torch.save(model.state_dict(), path)


# Main program
if __name__ == "__main__":
    # Select the dataset path based on the chosen dataset
    selected_dataset = 'SMIC'  # Can choose 'SMIC', 'CASME_II', 'SAMM', or 'MEGC2019_CD'
    dataset_path = 'raw_samples/{}'.format(selected_dataset)

    # Load the training dataset
    train_dataset = MEFeaturesDataset(dataset_name=selected_dataset, root_dir=dataset_path, num_classes=3, num_shots=5)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Initialize the model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetaLearningModule(gfe_model_path='models/PLTN.pth', afe_model_path='models/PLSM.pth', input_dim=128*128*5*10, output_dim=128).to(device)
    loss_fn = TripletLoss(margin=1.0)

    # Train the model for 60 epochs
    num_epochs = 60
    for epoch in range(num_epochs):
        loss = train(model, train_loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    # Save the trained model
    save_model(model, 'models/PLTN.pth')
    print("Model saved to 'models/PLTN.pth'")
