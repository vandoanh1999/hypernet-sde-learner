import torch
import numpy as np
import sys
sys.path.append('./src')
from main import HypernetSDEContinualLearner

model = HypernetSDEContinualLearner(
    input_dim=16, hidden_dim=32, latent_dim=8,
    task_emb_dim=8, num_tasks_expected=3
)
model.eval()

x = torch.randn(4, 16)
with torch.no_grad():
    recon, u, ldj, z_sde, z0 = model(x, task_id=0)

print("Input:", x.numpy())
print("Reconstruction:", recon.numpy())
