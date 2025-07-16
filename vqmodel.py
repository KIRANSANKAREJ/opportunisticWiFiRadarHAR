import torch
import torch.nn as nn

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
