import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import math


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, decay=0.99, eps=1e-5, cosine_assign=False, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.eps = eps
        self.cosine_assign = cosine_assign
        self.beta = beta  # commitment weight

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0/num_embeddings, 1.0/num_embeddings)

        # EMA buffers
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", torch.zeros(num_embeddings, embedding_dim))

    @torch.no_grad()
    def _ema_update(self, z_e_flat, indices):
        # one-hot assignments
        N = z_e_flat.shape[0]
        one_hot = F.one_hot(indices, self.num_embeddings).type_as(z_e_flat)  # (N,K)

        # batch stats
        cluster_size = one_hot.sum(0)  # (K,)
        embed_sums = one_hot.t() @ z_e_flat  # (K,D)

        # ema updates
        self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.ema_w.mul_(self.decay).add_(embed_sums, alpha=1 - self.decay)

        # Laplace smoothing to avoid NaNs
        n = self.ema_cluster_size.sum()
        smoothed = (self.ema_cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n

        # normalize centers
        self.embedding.weight.data.copy_(self.ema_w / smoothed.unsqueeze(1))

    def forward(self, z_e):
        # z_e: (B,C,H,W)
        B, C, H, W = z_e.shape
        z = z_e.permute(0,2,3,1).contiguous().view(-1, C)  # (N, D)

        # assignment
        E = self.embedding.weight  # (K, D)

        if self.cosine_assign:
            z_n = F.normalize(z, dim=1)
            E_n = F.normalize(E, dim=1)
            indices = torch.argmax(z_n @ E_n.t(), dim=1)
        else:
            # squared L2 via dot trick
            z2 = (z*z).sum(dim=1, keepdim=True)    # (N,1)
            e2 = (E*E).sum(dim=1, keepdim=True).t()# (1,K)
            d = z2 + e2 - 2.0 * (z @ E.t())        # (N,K)
            indices = torch.argmin(d, dim=1)

        z_q = F.embedding(indices, E)             # (N, D)
        z_q = z_q.view(B, H, W, C).permute(0,3,1,2).contiguous()

        # EMA codebook update (no grad) using flattened z
        self._ema_update(z.detach().view(-1, C), indices.detach())

        # losses: commit only (codebook is updated via EMA)
        commit = self.beta * F.mse_loss(z_q.detach(), z_e)
        # straight-through
        z_q = z_e + (z_q - z_e).detach()

        vq_loss = commit   # keep key 'vq_loss' for your caller
        codebook_loss = torch.zeros((), device=z_e.device)
        commitment_loss = commit

        indices_out = indices.view(B, H, W)
        return z_q, codebook_loss, commitment_loss, vq_loss, indices_out



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

    def set_codebook(self, new_weights):
        with torch.no_grad():
            self.embedding.weight.copy_(new_weights)


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.1),  # light dropout early to not destroy low-level features

            nn.Conv2d(64, 128, kernel_size=4, stride=(2, 1), padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2),  # stronger dropout mid-level

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.3)   # heaviest dropout before quantization
        )

    def forward(self, x):
        return self.model(x)




class Decoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(embedding_dim, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),      # 50×15
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=(2, 1), padding=1), nn.ReLU(),   # 100×15
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=(2, 2), padding=1), nn.ReLU(),    # 200×30
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)                               # Final: 200×30
        )

    def forward(self, x):
        return self.model(x)

    

class VQVAE(nn.Module):
    def __init__(self, in_channels=1, embedding_dim=32, num_embeddings=512, beta=0.25,
                 decay=0.99, cosine_assign=True):
        super().__init__()
        self.encoder = Encoder(in_channels)          # outputs C_enc=128 in your file
        C_enc = 128                                  # <- last conv of Encoder uses 128

        # map encoder channels -> embedding_dim for the quantizer
        self.quant_proj = nn.Conv2d(C_enc, embedding_dim, kernel_size=1)
        self.quant_ln = nn.LayerNorm(embedding_dim)

        # EMA quantizer (make sure args are in the right slots!)
        self.quantizer = VectorQuantizerEMA(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            decay=decay,
            eps=1e-5,
            cosine_assign=cosine_assign,
            beta=beta
        )

        self.decoder = Decoder(embedding_dim)        # your Decoder already takes embedding_dim

    def forward(self, x, quantize=True):
        z_e = self.encoder(x)                        # (B, C_enc, H, W)
        z_e = self.quant_proj(z_e)                   # (B, embedding_dim, H, W)

        # apply LN over channel dim (B,C,H,W) → (BHW,C) → LN → back
        B, C, H, W = z_e.shape
        z_flat = z_e.permute(0, 2, 3, 1).reshape(-1, C)
        z_flat = self.quant_ln(z_flat)
        z_e = z_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        if quantize:
            z_q, codebook_loss, commitment_loss, vq_loss, indices = self.quantizer(z_e)
        else:
            # warm-up / no-quant path: return zeros as 0-D tensors and indices=None
            z_q = z_e
            indices = None
            codebook_loss   = torch.zeros((), device=z_e.device)
            commitment_loss = torch.zeros((), device=z_e.device)
            vq_loss         = torch.zeros((), device=z_e.device)

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