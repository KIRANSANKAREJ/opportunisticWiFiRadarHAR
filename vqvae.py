import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import time
import argparse
import os

class PWRDataset(Dataset):
    def __init__(self, pkl_path="pkl files/raw_data.pkl", channels_to_use=[0, 1, 2]):
        self.channels_to_use = channels_to_use
        self.data, self.labels = self.load_vqvae_training_data(pkl_path, channels_to_use)
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

    def load_vqvae_training_data(self, pkl_path, channels_to_use):
        df = pd.read_pickle(pkl_path)
        data = []
        labels = []

        for _, row in df.iterrows():
            ch_raw = [
                np.array(row["PWR_ch1"]).flatten(),
                np.array(row["PWR_ch2"]).flatten(),
                np.array(row["PWR_ch3"]).flatten()
            ]

            selected_channels = [ch_raw[i] for i in channels_to_use]
            sample = np.stack(selected_channels)

            if sample.shape[1] != 200:
                continue

            sample = (sample - sample.min()) / (sample.max() - sample.min() + 1e-8)
            sample = (sample * 2) - 1

            data.append(sample.astype(np.float32))
            labels.append(row["activity"])

        return np.array(data), labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx])
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
        z = z.permute(0, 2, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)

        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices).view(z.shape)

        quantized = quantized.permute(0, 2, 1).contiguous()
        z = z.permute(0, 2, 1).contiguous()

        vq_loss = F.mse_loss(quantized.detach(), z)
        commitment_loss = self.commitment_cost * F.mse_loss(quantized, z.detach())

        quantized = z + (quantized - z).detach()
        return quantized, vq_loss, commitment_loss


class VQVAE(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64, latent_dim=64, num_embeddings=512):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, latent_dim, kernel_size=4, stride=2, padding=1),
        )

        self.vq = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=latent_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_channels, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, commit_loss = self.vq(z)
        x_recon = self.decoder(quantized)
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
    print("[INFO] Model saved to 'vqvae_pwr.pt'")
    print(f"[INFO] Training took {(end-start)/60:.2f} minutes")


class VQClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


def run_classification_torch(encoder, dataset, device="cuda", batch_size=64, epochs=20, lr=1e-3):
    print("[INFO] Extracting encoder features...")

    encoder.eval()
    X, y = [], []

    for i in tqdm(range(len(dataset)), desc="Encoding dataset"):
        sample, label = dataset[i]
        sample = torch.tensor(sample).unsqueeze(0).to(device)  # (1, 3, 200)
        with torch.no_grad():
            z = encoder(sample)  # (1, latent_dim, T)
        z_flat = z.view(-1).cpu()
        X.append(z_flat)
        y.append(label)

    X = torch.stack(X)
    le = LabelEncoder()
    y = torch.tensor(le.fit_transform(y))

    print(f"[INFO] Encoded feature shape: {X.shape}")
    print(f"[INFO] Number of classes: {len(le.classes_)}")

    # Split into train/test
    n_train = int(0.8 * len(X))
    train_ds = TensorDataset(X[:n_train], y[:n_train])
    test_ds = TensorDataset(X[n_train:], y[n_train:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    clf = VQClassifier(input_dim=X.shape[1], num_classes=len(le.classes_)).to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print("[INFO] Training classifier...")
    for epoch in range(epochs):
        clf.train()
        total_loss, correct = 0, 0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = clf(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == yb).sum().item()
        acc = correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Train Acc={acc:.4f}")

    # Evaluation
    clf.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = clf(xb).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(yb.tolist())

    print("\nClassification Report:")
    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(all_labels, all_preds, target_names=[str(c) for c in le.classes_]))

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    import seaborn as sns
    import matplotlib.pyplot as plt
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (VQ-VAE Classifier)")
    plt.tight_layout()
    plt.savefig("output_images/vqvae_torch_classifier_confusion_matrix.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train or load VQ-VAE for PWR data.")
    parser.add_argument("--retrain", action="store_true", help="Force retraining the VQ-VAE model.")
    parser.add_argument("--freeze", action="store_true", help="Freeze encoder weights after training.")
    parser.add_argument("--classify", action="store_true", help="Train classifier on top of VQ-VAE embeddings.")
    parser.add_argument("--pkl_path", default="pkl files/raw_data.pkl", help="Path to pickled training data.")
    parser.add_argument("--model_path", default="model pt files/vqvae_pwr.pt", help="Path to save/load model.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    args = parser.parse_args()

    device = "cuda"

    # Load dataset
    channels = [0]
    dataset = PWRDataset(args.pkl_path, channels_to_use=channels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    model = VQVAE(in_channels=len(channels)).to(device)

    if os.path.exists(args.model_path) and not args.retrain:
        print(f"[INFO] Loading model from: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        print("[INFO] Training new VQ-VAE model...")
        train_vqvae(model, dataloader, epochs=args.epochs, lr=args.lr, device=device)

    if args.freeze or args.classify:
        for param in model.encoder.parameters():
            param.requires_grad = False

    if args.classify:
        print("\n==> [INFO] Starting classification task...")
        run_classification_torch(model.encoder, dataset, device=device)
        return

    print("[INFO] Pipeline complete.")


if __name__ == "__main__":
    main()




