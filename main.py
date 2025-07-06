#!/usr/bin/env python
# coding: utf-8

# In[41]:


import sys
import logging

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd


# In[42]:


batch_size = 8
bs = batch_size
epochs = 50  # Reduced number of training epochs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logging.info(f"Batch size: {batch_size}, Epochs: {epochs}")


# In[43]:


device = "mps" if torch.backends.mps.is_available() else "cpu"
if device == "mps":
    logging.info("Using Apple Silicon GPU (MPS) for training.")


# In[44]:


df = pd.read_csv("iris.csv")
df["variety_enc"] = df["variety"].astype("category").cat.codes
df.columns = df.columns.str.replace(".", "_").str.lower()
df.info()


# In[45]:


class IrisDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        # Convert to float64 array first, then to tensor
        features = torch.tensor(
            row[
                ["sepal_length", "sepal_width", "petal_length", "petal_width"]
            ].values.astype("float64"),
            dtype=torch.float32,
        )
        label = torch.tensor(row["variety_enc"], dtype=torch.long)
        return features, label


# First split: 80% train+val, 20% test
train_val_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_val_df.index)

# Second split: from the 80%, take 75% for train (60% of total) and 25% for val (20% of total)
train_df = train_val_df.sample(frac=0.75, random_state=42)
val_df = train_val_df.drop(train_df.index)

logging.info(
    f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}"
)

# Create the datasets and dataloaders
iris_train_ds = IrisDataset(train_df)
train_loader = DataLoader(iris_train_ds, batch_size=bs, shuffle=True)

iris_val_ds = IrisDataset(val_df)
val_loader = DataLoader(iris_val_ds, batch_size=bs, shuffle=False)

iris_test_ds = IrisDataset(test_df)
test_loader = DataLoader(iris_test_ds, batch_size=bs, shuffle=False)


# In[46]:


class IrisClassifier(nn.Module):
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Dropout(0.01),  # Dropout with 1% probability
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        return self.net(x)


model = IrisClassifier().to(device)
logging.info(f"Model architecture:\n{model}")

criterion = nn.CrossEntropyLoss()

# Update optimizer to reduce weight decay
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Update learning rate scheduler to increase step_size
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


# Refactor accuracy calculation into a reusable function
def calculate_accuracy(loader, model, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


# In[48]:


# Ensure dataset objects are properly sized
train_dataset_size = len(iris_train_ds)
val_dataset_size = len(iris_val_ds)
test_dataset_size = len(iris_test_ds)

# Update the calculations to use these sizes
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)
        correct_train += (outputs.argmax(dim=1) == labels).sum().item()
        total_train += labels.size(0)

    epoch_loss = running_loss / train_dataset_size
    accuracy_train = correct_train / total_train

    # Calculate validation accuracy using the refactored function
    accuracy_val = calculate_accuracy(val_loader, model, device)

    logging.info(
        f"Epoch [{epoch + 1}/{epochs}], Train-Loss: {epoch_loss:.4f}, Accuracy: {accuracy_train:.4f}, Val-Accuracy: {accuracy_val:.4f}"
    )

    # Step the scheduler
    scheduler.step()

# Test phase
accuracy_test = calculate_accuracy(test_loader, model, device)
logging.info(f"Test Accuracy: {accuracy_test:.4f}")
