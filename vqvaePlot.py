import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from vqvae import VQVAE, PWRDataset

# Settings
model_path = "model pt files/vqvae_pwr.pt"
pkl_path = "pkl files/raw_data.pkl"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = VQVAE()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()

# Load data
dataset = PWRDataset(pkl_path)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
batch = next(iter(dataloader))
inputs = batch[0].to(device)

# Inference
with torch.no_grad():
    recons, _, _ = model(inputs)

# Convert to numpy
inputs_np = inputs.cpu().numpy()
recons_np = recons.cpu().numpy()

# Plot
import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 4, figsize=(16, 6), sharex=True, sharey=True)

for i in range(4):
    for ch in range(3):
        axs[ch, i].plot(inputs_np[i, ch], label='Original', alpha=0.7)
        axs[ch, i].plot(recons_np[i, ch], label='Reconstructed', linestyle='--', alpha=0.7)
        if i == 0:
            axs[ch, i].set_ylabel(f'Ch{ch+1}')
        if ch == 0:
            axs[ch, i].set_title(f'Sample {i+1}')
        if ch == 2:
            axs[ch, i].legend(loc='upper right')

plt.tight_layout()
plt.suptitle("Original vs Reconstructed PWR Signals", fontsize=16, y=1.02)
plt.show()
