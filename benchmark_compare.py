import torch
import numpy as np
import time
from src.main import HypernetSDEContinualLearner

# Baseline: simple MLP
class SimpleMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# Tạo data test mẫu (có thể thay bằng dữ liệu thực)
x = torch.randn(64, 16)  # batch size 64, input dim 16

model = HypernetSDEContinualLearner(16, 32, 8, 8, 3)
baseline = SimpleMLP(16, 32, 16)

# Chạy thử model của bạn
t0 = time.time()
with torch.no_grad():
    recon, _, _, _, _ = model(x, task_id=0)
loss_yours = torch.nn.functional.mse_loss(recon, x)
t1 = time.time()
print(f"HypernetSDE: Recon Loss = {loss_yours:.4f}, Time = {t1-t0:.4f}s")

# Chạy thử baseline
t0 = time.time()
with torch.no_grad():
    recon_base = baseline(x)
loss_base = torch.nn.functional.mse_loss(recon_base, x)
t1 = time.time()
print(f"Baseline MLP: Recon Loss = {loss_base:.4f}, Time = {t1-t0:.4f}s")
