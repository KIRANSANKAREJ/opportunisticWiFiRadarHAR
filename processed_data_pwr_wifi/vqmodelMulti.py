import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64,  kernel_size=4, stride=2, padding=1),  # 224 -> 112
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128,         kernel_size=4, stride=2, padding=1),  # 112 -> 56
            nn.ReLU(inplace=True),
            nn.Conv2d(128, embedding_dim, kernel_size=3, stride=1, padding=1) # 56 -> 56
        )
    def forward(self, x): return self.net(x)


class Decoder(nn.Module):
    def __init__(self, embedding_dim, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embedding_dim, 128, kernel_size=3, stride=1, padding=1),  # 56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1),   # 56 -> 112
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1)  # 112 -> 224
        )
    def forward(self, x): return self.net(x)


class MultiVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=256, embedding_dim=64, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim
        self.beta = beta
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1, 1)

    def forward(self, z_e_list):
        """
        z_e_list: [z_e1, z_e2, ...] each [B, C, H, W] with same C/H/W
        Uses mean squared-Euclidean distance across branches to pick indices.
        """
        B, C, H, W = z_e_list[0].shape
        E = self.embedding.weight                     # [K,C]
        e_sq = torch.sum(E**2, dim=1)                 # [K]

        dist_sum = 0
        for z in z_e_list:
            zf = z.permute(0,2,3,1).reshape(-1, C)    # [BHW, C]
            dist = torch.sum(zf**2, dim=1, keepdim=True) + e_sq - 2 * (zf @ E.t())  # [BHW,K]
            dist_sum = dist_sum + dist

        distances = dist_sum / len(z_e_list)          # mean distance
        indices = torch.argmin(distances, dim=1)      # [BHW]

        z_q = self.embedding(indices).view(B, H, W, C).permute(0,3,1,2).contiguous()

        # losses: average over branches
        codebook_loss   = sum(F.mse_loss(z_q, z.detach()) for z in z_e_list) / len(z_e_list)
        commitment_loss = sum(F.mse_loss(z, z_q.detach()) for z in z_e_list) / len(z_e_list)
        vq_loss = codebook_loss + self.beta * commitment_loss
        return z_q, codebook_loss, commitment_loss, vq_loss, indices

    def set_codebook(self, new_weights):
        with torch.no_grad():
            self.embedding.weight.copy_(new_weights)


class VQVAE_Multi(nn.Module):
    def __init__(self, in_channels=3, embedding_dim=64, num_embeddings=256, beta=0.25):
        super().__init__()
        assert in_channels == 3, "set in_channels=3 for three branches"
        self.in_channels = in_channels

        self.encoders = nn.ModuleList([Encoder(1, embedding_dim) for _ in range(in_channels)])
        self.quantizer = MultiVectorQuantizer(num_embeddings=num_embeddings,
                                              embedding_dim=embedding_dim, beta=beta)
        self.decoders = nn.ModuleList([Decoder(embedding_dim, 1) for _ in range(in_channels)])

    def forward(self, x, quantize=True):
        # accept channels-last too
        if x.dim()==4 and x.shape[1]==self.in_channels:
            x_bchw = x
        elif x.dim()==4 and x.shape[-1]==self.in_channels:
            x_bchw = x.permute(0,3,1,2).contiguous()
        else:
            raise ValueError("Expected (B,3,H,W) or (B,H,W,3)")

        # split channels: [B,1,H,W] each
        xs = [x_bchw[:, i:i+1] for i in range(self.in_channels)]

        # encode
        z_es = [enc(xi) for enc, xi in zip(self.encoders, xs)]  # each [B,C,H',W']

        if quantize:
            z_q_shared, codebook_loss, commitment_loss, vq_loss, indices = self.quantizer(z_es)
            # straight-through per branch
            z_qs = [ze + (z_q_shared - ze).detach() for ze in z_es]
        else:
            z_q_shared = z_es[0]
            z_qs = z_es
            indices = None
            codebook_loss   = torch.zeros((), device=x_bchw.device)
            commitment_loss = torch.zeros((), device=x_bchw.device)
            vq_loss         = torch.zeros((), device=x_bchw.device)

        # decode per branch, then concat
        recons = [dec(zq) for dec, zq in zip(self.decoders, z_qs)]   # each [B,1,H,W]
        x_recon = torch.cat(recons, dim=1)                           # [B,2,H,W]

        # losses
        recon_loss = F.mse_loss(x_recon, x_bchw)
        total_loss = recon_loss + vq_loss

        return {
            "recon_x": x_recon,
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "vq_loss": vq_loss,
            "z_q": z_q_shared,
            "indices": indices,
        }
