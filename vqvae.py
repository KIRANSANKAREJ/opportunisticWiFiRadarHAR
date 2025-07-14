import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import time

class PWRDataset(Dataset):
    def __init__(self, pkl_path="pkl files/raw_data.pkl"):
        self.label_encoder = LabelEncoder()
        self.data, self.labels = self.load_vqvae_training_data(pkl_path)
        self.labels = self.label_encoder.fit_transform(self.labels)  # Encode string labels to int

    def load_vqvae_training_data(self, pkl_path):
        df = pd.read_pickle(pkl_path)
        data = []
        labels = []

        for _, row in df.iterrows():
            ch1 = np.array(row["PWR_ch1"]).flatten()
            ch2 = np.array(row["PWR_ch2"]).flatten()
            ch3 = np.array(row["PWR_ch3"]).flatten()

            sample = np.stack([ch1, ch2, ch3])  # (3, 200)
            if sample.shape != (3, 200):
                continue

            # Normalize to [-1, 1]
            sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample) + 1e-8)
            sample = (sample * 2) - 1

            data.append(sample)
            labels.append(row["activity"])  # string

        return np.array(data, dtype=np.float32), np.array(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def load_vqvae_training_data(pkl_path):
    df = pd.read_pickle(pkl_path)
    data, labels = [], []

    for _, row in df.iterrows():
        ch1 = np.array(row["PWR_ch1"]).flatten()
        ch2 = np.array(row["PWR_ch2"]).flatten()
        ch3 = np.array(row["PWR_ch3"]).flatten()

        sample = np.stack([ch1, ch2, ch3])
        if sample.shape != (3, 200):
            continue

        # sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample) + 1e-8)
        # sample = (sample * 2) - 1

        data.append(sample)
        labels.append(row["activity"])

    le = LabelEncoder()
    labels = le.fit_transform(labels)  # Converts to integers

    return np.array(data, dtype=np.float32), labels



class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z):
        # Flatten input: (B, D, T) → (B*T, D)
        z = z.permute(0, 2, 1).contiguous()  # (B, T, D)
        z_flattened = z.view(-1, self.embedding_dim)  # (B*T, D)

        # Compute distances (B*T, N)
        distances = (torch.sum(z_flattened ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(z_flattened, self.embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1)  # (B*T)
        quantized = self.embedding(encoding_indices).view(z.shape)  # (B, T, D)

        # Straight-through estimator
        quantized = quantized.permute(0, 2, 1).contiguous()  # (B, D, T)
        z = z.permute(0, 2, 1).contiguous()  # (B, D, T)

        # Losses
        vq_loss = F.mse_loss(quantized.detach(), z)
        commitment_loss = self.commitment_cost * F.mse_loss(quantized, z.detach())

        quantized = z + (quantized - z).detach()  # Straight-through trick
        return quantized, vq_loss, commitment_loss
    

class VQVAE(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64, latent_dim=64, num_embeddings=512):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1),  
            # (B, 64, 100)
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),  
            # (B, 64, 50)
            nn.ReLU(),
            nn.Conv1d(hidden_channels, latent_dim, kernel_size=4, stride=2, padding=1),  
            # (B, 64, 25)
        )

        # Quantizer
        self.vq = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, hidden_channels, kernel_size=4, stride=2, padding=1),  
            # (B, 64, 50)
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),  
            # (B, 64, 100)
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_channels, in_channels, kernel_size=4, stride=2, padding=1),  
            # (B, 3, 200)
        )

    def forward(self, x):
        z = self.encoder(x)
        # (B, 64, 25)
        quantized, vq_loss, commit_loss = self.vq(z)  
        # (B, 64, 25)
        x_recon = self.decoder(quantized)  
        # (B, 3, 200)
        return x_recon, vq_loss, commit_loss


def train_vqvae(model, dataloader, epochs=20, lr=1e-3, device="cuda"):
    start = time.time()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_recon, total_vq, total_commit = 0, 0, 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1:02d}", leave=False)
        for batch in pbar:
            inputs = batch[0].to(device)  # (B, 3, 200)
            optimizer.zero_grad()

            recon, vq_loss, commit_loss = model(inputs)
            recon_loss = F.mse_loss(recon, inputs)

            loss = recon_loss + vq_loss + commit_loss
            loss.backward()
            optimizer.step()

            total_recon += recon_loss.item()
            total_vq += vq_loss.item()
            total_commit += commit_loss.item()

        print(f"Epoch {epoch+1:02d} | Recon: {total_recon:.4f} | VQ: {total_vq:.4f} | Commit: {total_commit:.4f}")

    end = time.time()
    torch.save(model.state_dict(), "model pt files/vqvae_pwr.pt")
    print("Model saved to 'vqvae_pwr.pt'")
    print(f"Training took {(end-start)/60:.2f} minutes")

# Load dataset
dataset = PWRDataset("pkl files/raw_data.pkl")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
vqvae = VQVAE(in_channels=3)

# Train
train_vqvae(
    vqvae, 
    dataloader, 
    epochs=30, 
    lr=1e-3, 
    device="cuda" if torch.cuda.is_available() else "cpu")



