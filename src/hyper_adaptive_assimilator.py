# MVP API for Hyper-Adaptive Continual Learner
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Security, BackgroundTasks
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Tuple, Optional
import os
import re

# --- API Security ---
# Load API Key from environment variable for better security
API_KEY = os.getenv("API_KEY", "your_secret_api_key_dev")
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

# --- Model Configuration ---
INPUT_DIM = 3
HIDDEN_DIM = 64
LATENT_DIM = 24
TASK_EMB_DIM = 16
NUM_TASKS_EXPECTED = 10
EPOCHS_PER_TASK = 100 # Reduced for API speed
LR = 7e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure torchsde is imported; it should be in requirements.txt
import torchsde

# ======================================================================
# MODEL IMPLEMENTATION (from benchmark_compare.py)
# ======================================================================

class NeuralSDE(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.drift_net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
        self.diffusion_net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, dim), nn.Sigmoid()
        )

    def f(self, t: float, z: torch.Tensor) -> torch.Tensor:
        t_vec = torch.ones(z.size(0), 1, device=z.device) * t
        tz = torch.cat([t_vec, z], dim=1)
        return self.drift_net(tz)

    def g(self, t: float, z: torch.Tensor) -> torch.Tensor:
        t_vec = torch.ones(z.size(0), 1, device=z.device) * t
        tz = torch.cat([t_vec, z], dim=1)
        return self.diffusion_net(tz)

    def forward(self, z0: torch.Tensor, t: float = 1.0) -> torch.Tensor:
        t_span = torch.tensor([0.0, t], device=z0.device)
        solution = torchsde.sdeint(self, z0, t_span, method='srk', dt=t/10.0)
        return solution[1]

class DynamicLinear(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F.linear(x, w, b)

class HyperNetwork(nn.Module):
    def __init__(self, task_emb_dim: int, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        hidden_dim = max(32, (task_emb_dim + in_features * out_features) // 8)
        self.weight_generator = nn.Sequential(
            nn.Linear(task_emb_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, in_features * out_features)
        )
        self.bias_generator = nn.Sequential(
            nn.Linear(task_emb_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_features)
        )
    def forward(self, z_task: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = self.weight_generator(z_task).view(self.out_features, self.in_features)
        bias = self.bias_generator(z_task).view(self.out_features)
        return weights, bias

class AffineCoupling(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2
        self.net = nn.Sequential(
            nn.Linear(self.half_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, (dim - self.half_dim) * 2)
        )
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z1, z2 = z.chunk(2, dim=1)
        log_s_t = self.net(z1)
        log_s, t = log_s_t.chunk(2, dim=1)
        s = torch.exp(log_s.tanh())
        u2 = (z2 + t) * s
        u = torch.cat([z1, u2], dim=-1)
        log_det_jac = s.log().sum(dim=-1)
        return u, log_det_jac

class Permutation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("perm", torch.randperm(dim))
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, self.perm]

class ManifoldNormalizingFlow(nn.Module):
    def __init__(self, latent_dim: int, num_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(AffineCoupling(latent_dim))
            self.layers.append(Permutation(latent_dim))
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        total_log_det_jac = torch.zeros(z.size(0), device=z.device)
        for layer in self.layers:
            if isinstance(layer, AffineCoupling):
                z, log_det_jac = layer(z)
                total_log_det_jac += log_det_jac
            else:
                z = layer(z)
        return z, total_log_det_jac

class HypernetSDEContinualLearner(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int,
                 task_emb_dim: int, num_tasks_expected: int):
        super().__init__()
        self.task_embeddings = nn.Parameter(torch.randn(num_tasks_expected, task_emb_dim) * 0.02)
        self.encoder_1 = DynamicLinear()
        self.hypernet_enc1 = HyperNetwork(task_emb_dim, input_dim, hidden_dim)
        self.encoder_2 = DynamicLinear()
        self.hypernet_enc2 = HyperNetwork(task_emb_dim, hidden_dim, hidden_dim)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.flow = ManifoldNormalizingFlow(latent_dim, num_layers=4)
        self.neural_sde = NeuralSDE(latent_dim)
        self.decoder_sde = NeuralSDE(latent_dim)
        self.decoder_1 = DynamicLinear()
        self.hypernet_dec1 = HyperNetwork(task_emb_dim, latent_dim, hidden_dim)
        self.decoder_2 = nn.Linear(hidden_dim, input_dim)
        self.base_dist = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))

    def to(self, device):
        super().to(device)
        self.base_dist = torch.distributions.Normal(
            torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
        )
        return self

    def encode(self, x: torch.Tensor, task_id_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        task_vec = self.task_embeddings[task_id_tensor]
        enc1_w, enc1_b = self.hypernet_enc1(task_vec)
        enc2_w, enc2_b = self.hypernet_enc2(task_vec)
        h1 = F.gelu(torch.einsum('bi,oi->bo', x, enc1_w) + enc1_b.unsqueeze(0))
        h2 = F.gelu(torch.einsum('bi,oi->bo', h1, enc2_w) + enc2_b.unsqueeze(0))
        z0 = self.to_latent(h2)
        u, log_det_jac = self.flow(z0)
        return z0, u, log_det_jac

    def decode(self, z: torch.Tensor, task_id_tensor: torch.Tensor) -> torch.Tensor:
        task_vec = self.task_embeddings[task_id_tensor]
        dec1_w, dec1_b = self.hypernet_dec1(task_vec)
        z_evolved = self.decoder_sde(z, t=0.5)
        h1 = F.gelu(torch.einsum('bi,oi->bo', z_evolved, dec1_w) + dec1_b.unsqueeze(0))
        reconstruction = self.decoder_2(h1)
        return reconstruction

    def compute_loss(self, x: torch.Tensor, recon: torch.Tensor, u: torch.Tensor, log_det_jac: torch.Tensor, task_id: int):
        recon_loss = F.mse_loss(recon, x)
        log_p_u = self.base_dist.log_prob(u).sum(dim=1)
        nll_loss = -torch.mean(log_p_u + log_det_jac)
        task_sep_loss = torch.tensor(0.0, device=x.device)
        if task_id > 0:
            current_proto = self.task_embeddings[task_id]
            prev_protos = self.task_embeddings[:task_id]
            distances = torch.norm(current_proto - prev_protos, dim=1)
            task_sep_loss = torch.mean(torch.exp(-distances * 2))
        total_loss = recon_loss + 0.1 * nll_loss + 0.3 * task_sep_loss
        return total_loss

# ======================================================================
# API IMPLEMENTATION
# ======================================================================
app = FastAPI(title="Hyper-Adaptive Assimilator MVP")

# --- State Persistence ---
MODEL_DATA_DIR = "data/models"
os.makedirs(MODEL_DATA_DIR, exist_ok=True)


class AssimilateRequest(BaseModel):
    task_id: int = Field(..., example=0, ge=0, lt=NUM_TASKS_EXPECTED)
    data: List[List[float]] = Field(..., example=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

class PredictRequest(BaseModel):
    task_id: int = Field(..., example=0, ge=0, lt=NUM_TASKS_EXPECTED)
    data: List[List[float]] = Field(..., example=[[0.7, 0.8, 0.9]])

def get_model_path(user_id: str) -> str:
    # Sanitize user_id to prevent path traversal attacks
    safe_user_id = re.sub(r'[^a-zA-Z0-9_-]', '', user_id)
    if not safe_user_id:
        raise HTTPException(status_code=400, detail="Invalid user_id format.")
    return os.path.join(MODEL_DATA_DIR, f"{safe_user_id}.pt")

def load_or_create_model(user_id: str) -> HypernetSDEContinualLearner:
    model_path = get_model_path(user_id) # The path is now sanitized
    # Create a new model instance. This is the recommended practice.
    model = HypernetSDEContinualLearner(
        input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM,
        task_emb_dim=TASK_EMB_DIM, num_tasks_expected=NUM_TASKS_EXPECTED
    )
    if os.path.exists(model_path):
        print(f"Loading model state for user: {user_id}")
        # Load the state dictionary, which is safer than loading the whole object.
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print(f"Creating a new model for user: {user_id}")
        # Model is already created, no further action needed.
    return model.to(DEVICE)

def save_model(user_id: str, model: HypernetSDEContinualLearner):
    model_path = get_model_path(user_id)
    print(f"Saving model state for user: {user_id}")
    # Save the state dictionary, not the entire model object.
    torch.save(model.state_dict(), model_path)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    else:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
def train_model_background(user_id: str, task_id: int, data: List[List[float]]):
    """Function to handle the model training in the background."""
    model = load_or_create_model(user_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    try:
        x_train = torch.tensor(data, dtype=torch.float32).to(DEVICE)
        train_loader = DataLoader(TensorDataset(x_train), batch_size=64, shuffle=True)
    except Exception as e:
        print(f"Error processing training data for user {user_id}: {e}")
        return

    model.train()
    for epoch in range(EPOCHS_PER_TASK):
        for (x_batch,) in train_loader:
            optimizer.zero_grad()
            task_id_tensor = torch.tensor(task_id, device=DEVICE)
            z0, u, log_det_jac = model.encode(x_batch, task_id_tensor)
            z_sde = model.neural_sde(z0, t=1.0)
            recon = model.decode(z_sde, task_id_tensor)
            loss = model.compute_loss(x_batch, recon, u, log_det_jac, task_id)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"BG-TRAIN | User: {user_id}, Task: {task_id}, Epoch: {epoch+1}/{EPOCHS_PER_TASK}, Loss: {loss.item():.6f}")

    save_model(user_id, model)
    print(f"BG-TRAIN | Assimilation complete for user: {user_id}, task: {task_id}")

@app.post("/assimilate/{user_id}")
async def assimilate(user_id: str, request: AssimilateRequest, background_tasks: BackgroundTasks, api_key: str = Depends(get_api_key)):
    # Validate data dimensions before starting background task
    if not request.data or len(request.data[0]) != INPUT_DIM:
        raise HTTPException(status_code=400, detail=f"Input data must have {INPUT_DIM} features.")

    background_tasks.add_task(train_model_background, user_id, request.task_id, request.data)

    return {"status": "assimilation_queued", "user_id": user_id, "task_id": request.task_id}

@app.post("/predict/{user_id}", response_model=List[List[float]])
async def predict(user_id: str, request: PredictRequest, api_key: str = Depends(get_api_key)):
    model = load_or_create_model(user_id)
    model.eval()

    try:
        x_test = torch.tensor(request.data, dtype=torch.float32).to(DEVICE)
        if x_test.shape[1] != INPUT_DIM:
            raise ValueError(f"Input data must have {INPUT_DIM} features.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid data format: {e}")

    with torch.no_grad():
        task_id_tensor = torch.tensor(request.task_id, device=DEVICE)
        z0, _, _ = model.encode(x_test, task_id_tensor)
        z_sde = model.neural_sde(z0, t=1.0)
        reconstruction = model.decode(z_sde, task_id_tensor)

    return reconstruction.cpu().numpy().tolist()

if __name__ == "__main__":
    print(f"Starting Hyper-Adaptive Assimilator MVP on {DEVICE}...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
