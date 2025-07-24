import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=128, beta=1.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1, 1)

    def forward(self, z_e):
        B, C, H, W = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, C)
        distances = (
            torch.sum(z_e_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_e_flat, self.embedding.weight.t())
        )
        indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(indices).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss

        # Straight Through Estimator
        z_q = z_e + (z_q - z_e).detach()
        return z_q, codebook_loss, commitment_loss, vq_loss, indices

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1)
        )
    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(embedding_dim, 128, 3, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1)
        )
    def forward(self, x):
        return self.model(x)

    

class VQVAE(nn.Module):
    def __init__(self, in_channels=1, embedding_dim=128, num_embeddings=512, beta=0.25):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, beta)
        self.decoder = Decoder(embedding_dim)

    def forward(self, x):
        z_e = self.encoder(x)

        z_q, codebook_loss, commitment_loss, vq_loss, indices = self.quantizer(z_e)
        x_recon = self.decoder(z_q)
        recon_loss = F.mse_loss(x_recon, x)
        total_loss = recon_loss + vq_loss
        return {
            "recon_x": x_recon,
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "vq_loss": vq_loss,
            "z_q": z_q,
            "indices": indices
        }


def compute_perplexity(indices, num_embeddings):
    """
    indices: Tensor or array of shape (N,) — codebook indices used during a batch/epoch.
    num_embeddings: Total number of codebook vectors.
    """
    if isinstance(indices, torch.Tensor):
        indices = indices.detach().cpu().numpy()

    # Histogram of code usage
    counts = np.bincount(indices, minlength=num_embeddings).astype(np.float32)
    probs = counts / counts.sum()

    # Mask out zero probabilities to avoid log(0)
    nonzero_probs = probs[probs > 0]
    entropy = -np.sum(nonzero_probs * np.log(nonzero_probs + 1e-10))
    perplexity = np.exp(entropy)
    return perplexity


class ChunkImageDataset(Dataset):
    def __init__(self, chunks, transform=None):
        self.data = []

        for df in chunks:
            image = np.stack(df['PWR_ch1'].to_list(), axis=1)
            self.data.append(image.astype(np.float32))

        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]  # shape: (200, 80)
        img_tensor = torch.tensor(img, dtype=torch.float32)

        # Normalize per image
        mean = img_tensor.mean()
        std = img_tensor.std()
        if std < 1e-6:
            std = 1.0  # prevent division by zero

        img_tensor = (img_tensor - mean) / std
        return img_tensor.unsqueeze(0)  # (1, 200, 80)