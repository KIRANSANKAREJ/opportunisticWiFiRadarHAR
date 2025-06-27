import torch
import torch.nn as nn
import torch.nn.functional as F
from testData import getData
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class PWRDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.labels = LabelEncoder().fit_transform(dataframe['activity'].values)
        self.inputs = []

        for i, row in dataframe.iterrows():
            # Channelifying
            sample = np.hstack((row['PWR_ch1'], row['PWR_ch2'], row['PWR_ch3']))  
            sample = sample.T  # shape: (3, 200)
            self.inputs.append(torch.tensor(sample, dtype=torch.float32))

        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


class BasicPWRNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=5)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(16 * 98, 64)
        self.fc2 = nn.Linear(64, num_classes)


    def forward(self, x):  # x: (batch_size, 3, 200)
        x = self.pool(F.relu(self.conv1(x)))  # (batch_size, 16, 98)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(model, train_loader, optimizer, criterion, epochs=20, device="cuda"):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save model after training
    torch.save(model.state_dict(), "basic_pvr_model.pt")
    print("Model saved to 'basic_pvr_model.pt'")


def evaluate_model(model, test_loader, label_names, device="cuda"):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nEvaluation Report:")
    print(classification_report(all_labels, all_preds, target_names=label_names))
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(label_names)))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plotting
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix Heatmap")
    plt.tight_layout()
    plt.savefig("Confusion Matrix.png")
    plt.show()


# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', torch.cuda.get_device_name(0))

# Train-test split
try:
    print("Checking for existing train and test splits...")
    train_df = pd.read_pickle("train_df.pkl")
    test_df = pd.read_pickle("test_df.pkl")
    print("Loaded pre-existing pkl files...")
except:
    print("No pre-existing data found...")
    df = getData(force_reload=True)
    print("Data import complete, now starting train test split...")
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=2555304, 
        stratify=df['activity']
        )
    train_df.to_pickle("train_df.pkl")
    test_df.to_pickle("test_df.pkl")

train_dataset = PWRDataset(train_df)
test_dataset = PWRDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Model definition starting...")
model = BasicPWRNet(num_classes=len(np.unique(train_df['activity'])))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

try:
    print("Checking for previous training...")
    model.load_state_dict(torch.load("basic_pvr_model.pt", weights_only=True))
    print("State dictionary loaded...")
except:
    print("No saved model found, initializing training...")
    # Train
    train_model(model, train_loader, optimizer, criterion, epochs=10)

# Label names (for report)
label_names = sorted(train_df["activity"].unique())
# print(label_names)

# Evaluate
evaluate_model(model, test_loader, label_names)