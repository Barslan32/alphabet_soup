import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch import nn
import os

charz = ['B', 'U', 'D', 'R', 'K', 'A', 'E', '6', 'N']

dataset_name = "dataset_15k"

class CustomDataset(Dataset):
    def __init__(self, csv_file, charz, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with image paths and labels.
            charz (list): List of characters to include in the dataset.
            transform (callable, optional): Optional transform to apply to samples.
        """
        self.data = pd.read_csv(csv_file)
        self.charz = charz
        self.char_to_idx = {char: idx for idx, char in enumerate(charz)}  # Map chars to indices
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        # Load image
        image = Image.open(os.path.join(dataset_name, img_path)).convert("L")  # Grayscale
        
        if self.transform:
            image = self.transform(image)
        
        # Map label to index
        label_idx = self.char_to_idx[label]
        
        return torch.tensor(np.array(image, dtype=np.float32) / 255.0).unsqueeze(0), label_idx

class EMNISTCNN(nn.Module):
    def __init__(self, num_classes=9):  # Only 9 classes
        super(EMNISTCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Define transformations (if needed)
transform = None  # You can add augmentation transforms here

# Create dataset and dataloader
dataset = CustomDataset(csv_file=dataset_name+"/labels.csv", charz=charz, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss function, and optimizer
model = EMNISTCNN(num_classes=len(charz))  # Only 9 classes
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):  # Train for 10 epochs
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader):.4f}")


torch.save(model.state_dict(), "custom_charz.pth")

# Reload the trained model
model = EMNISTCNN(num_classes=len(charz))
model.load_state_dict(torch.load("custom_charz.pth"))
model.eval()

# Predict function
#def predict_letter_from_cell(cell, model, charz):
#    char_to_idx = {idx: char for idx, char in enumerate(charz)}
#    image_tensor = preprocess_cell(cell)  # Preprocess the cell
#    with torch.no_grad():
#        output = model(image_tensor)
#        _, predicted = torch.max(output, 1)
#    return char_to_idx[predicted.item()]

# Create test dataset and dataloader
test_dataset = CustomDataset(csv_file=dataset_name+"/labels.csv", charz=['B', 'U', 'D', 'R', 'K', 'A', 'E', '6', 'N'])
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy

# Test the model
evaluate_model(model, test_loader)
