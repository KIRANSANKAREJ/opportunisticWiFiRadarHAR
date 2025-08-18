import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import math


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=128, beta=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1, 1)

    def forward(self, z_e):
        B, C, H, W = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, C)

        # Compute distances to codebook vectors
        distances = (
            torch.sum(z_e_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_e_flat, self.embedding.weight.t())
        )

        indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(indices).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        codebook_loss   = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss

        # Straight-Through Estimator
        z_q = z_e + (z_q - z_e).detach()
        return z_q, codebook_loss, commitment_loss, vq_loss, indices

    def set_codebook(self, new_weights):
        with torch.no_grad():
            self.embedding.weight.copy_(new_weights)


class Encoder(nn.Module):
    def __init__(self, in_channels, embedding_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            # CHANGED: stride (2, 1) -> (2, 2) to downsample both H and W for 224x224
            nn.Conv2d(64, 128, kernel_size=4, stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            nn.Conv2d(128, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, embedding_dim, in_channels=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(embedding_dim, 128, kernel_size=3, stride=1, padding=1),  # 56×56
            nn.ReLU(),
            # CHANGED: stride (2, 1) -> (2, 2) : 56×56 -> 112×112
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=(2, 2), padding=1),
            nn.ReLU(),
            # CHANGED: stride (2, 2) stays (2, 2): 112×112 -> 224×224
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, in_channels, kernel_size=3, stride=1, padding=1)       # 224×224
        )

    def forward(self, x):
        return self.model(x)


class VQVAE(nn.Module):
    def __init__(self, in_channels=1, embedding_dim=64, num_embeddings=128, beta=0.25):
        super().__init__()
        self.in_channels = in_channels

        self.encoder = Encoder(in_channels, embedding_dim)
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            beta=beta
        )
        self.decoder = Decoder(embedding_dim, in_channels)

    def forward(self, x, quantize=True):
        input_was_channels_last = False
        if x.dim() == 4 and x.shape[1] != self.in_channels and x.shape[-1] == self.in_channels:
            input_was_channels_last = True
            x_bchw = x.permute(0, 3, 1, 2).contiguous()
        else:
            x_bchw = x

        z_e = self.encoder(x_bchw)

        if quantize:
            z_q, codebook_loss, commitment_loss, vq_loss, indices = self.quantizer(z_e)
        else:
            z_q = z_e
            indices = None
            codebook_loss   = torch.zeros((), device=z_e.device)
            commitment_loss = torch.zeros((), device=z_e.device)
            vq_loss         = torch.zeros((), device=z_e.device)

        x_recon_bchw = self.decoder(z_q)

        # Compute loss in BCHW space
        recon_loss = F.mse_loss(x_recon_bchw, x_bchw)
        total_loss = recon_loss + vq_loss

        # [NEW] Return reconstruction in the same layout as input
        if input_was_channels_last:
            x_recon = x_recon_bchw.permute(0, 2, 3, 1).contiguous()  # -> (B, H, W, C)
        else:
            x_recon = x_recon_bchw

        return {
            "recon_x": x_recon,
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "vq_loss": vq_loss,
            "z_q": z_q,
            "indices": indices,
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
    def __init__(self, chunks, in_channels=1, transform=None):
        assert in_channels in (1, 2, 3), "in_channels must be 1, 2, or 3"
        self.in_channels = in_channels
        self.data = []
        self.transform = transform

        cols = ["PWR_ch1"]
        if in_channels >= 2: cols.append("PWR_ch2")
        if in_channels >= 3: cols.append("PWR_ch3")

        for df in chunks:
            chans = []
            for c in cols:
                if c in df.columns:
                    # stack time along axis=1 to get (200, T)
                    arr = np.stack(df[c].to_list(), axis=1).astype(np.float32)
                else:
                    # fallback: zero channel if missing
                    if len(chans) > 0:
                        T = chans[0].shape[1]
                    else:
                        base = np.stack(df["PWR_ch1"].to_list(), axis=1).astype(np.float32)
                        T = base.shape[1]
                    arr = np.zeros((200, T), dtype=np.float32)
                chans.append(arr)

            # (C, 200, T)
            img = np.stack(chans, axis=0)
            self.data.append(img)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx])  # (C, 200, T), float32

        # per-sample, per-channel z-score
        mean = x.mean(dim=(1, 2), keepdim=True)
        std  = x.std(dim=(1, 2), keepdim=True, unbiased=False)
        x = (x - mean) / (std + 1e-8)                           
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if self.transform is not None:
            x = self.transform(x)

        return x