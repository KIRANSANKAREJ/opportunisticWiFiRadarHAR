import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=128, beta=0.25):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Codebook of shape num_embeddings * embedding_dim
        self.embedding = nn.Embedding(
            self.num_embeddings,
            self.embedding_dim
            )
        
        # Initialize random embeddings between -1.0 and 1.0
        self.embedding.weight.data.uniform_(
            -1.0 / num_embeddings,
            1.0 / num_embeddings
            )
        
        # Initialize the weight for commitment loss
        self.beta = beta
        
    def forward(self, z_e):
        # z_e = (B, C, H, W) - output from encoder
        B, C, H, W = z_e.shape

        # Reshape to (B*H*W, C)
        # (B, H, W, C)
        z_e_flat = z_e.permute(0, 2, 3, 1)
        # (B*H*W, C)
        z_e_flat = z_e_flat.reshape(-1, C)

        # Embeddings = (num_embeddings, C) or (K, D)
        e = self.embedding.weight

        # Euclidean Distance
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e^T
        # Distances has shape (num_vectors, num_embeddings)
        # num_vectors = N = B*H*W
        distances = (
            z_e_flat.pow(2).sum(1, keepdim=True)
            + e.pow(2).sum(1)
        
        # Get the codebook vectors from indices
        # z_q has    - 2 * z_e_flat @ e.t()
        )
        # Find the closest e for each z
        # indices has shape (N, 1)
        # Each value is an index of codebook
        # Values ranging from 0 to len(num_embeddings)-1
        indices = torch.argmin(distances, dim=1)
        # shape (N, C) or (N, D)
        z_q = self.embedding(indices)

        # Reshape back to the correct format
        z_q = z_q.view(B, H, W, C)
        # Reshape to (B, C, H, W)
        z_q = z_q.permute(0, 3, 1, 2)
        # Make z_q contiguous in memory for further operations
        z_q = z_q.contiguous()

        # L_codebook = ||sg(ze) - zq||^2
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        # L_commitment = ||ze - sg(zq)||^2
        commitment_loss = F.mse_loss(z_q.detach(), z_e)

        # Total VQ loss
        vq_loss = codebook_loss + commitment_loss

        return z_q, vq_loss


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, embedding_dim):
        super(Decoder, self).__init__()
        
        self.model = nn.Sequential(
            # Same shape as input(50*20)
            nn.Conv2d(embedding_dim, 128, kernel_size=3, stride=1, padding=1),

            # 50*20 to 100*40 
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 100*40 to 200*80(Target)
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)
    

class VQVAE(nn.Module):
    def __init__(self, 
                in_channels, 
                embedding_dim = 128, 
                num_embeddings = 512, 
                beta = 0.25
                ):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = Decoder(embedding_dim=embedding_dim)

        self.vq = VectorQuantizer(
            num_embeddings = num_embeddings,
            embedding_dim = embedding_dim,
            beta = beta 
            )
        
    def forward(self, x):
        # Encoder output
        z_e = self.encoder(x)

        # Getting quantized and vq loss from VectorQuantizer
        z_q, vq_loss = self.vq(z_e)

        # Decoder output
        reconstructed_x = self.decoder(z_q)

        # Reconstruction loss
        reconstruction_loss = F.mse_loss(reconstructed_x, x)

        total_loss = reconstruction_loss + vq_loss

        return reconstructed_x, total_loss, reconstruction_loss, vq_loss