import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

# ======================================================================
# Neural SDE, Dynamic Linear, HyperNetwork, AffineCoupling, Flow, Main Model
# ======================================================================
class NeuralSDE(nn.Module):
    def __init__(self, dim: int, num_layers: int = 2, hidden_dim: int = 64):
        super().__init__()
        def build_net(in_d, out_d):
            layers = []
            for _ in range(num_layers):
                layers.extend([nn.Linear(in_d, hidden_dim), nn.Tanh()])
                in_d = hidden_dim
            layers.append(nn.Linear(hidden_dim, out_d))
            return nn.Sequential(*layers)
        self.drift = build_net(dim, dim)
        self.diffusion = nn.Sequential(build_net(dim, dim), nn.Sigmoid())

    def forward(self, z0: torch.Tensor, t: float = 1.0, steps: int = 10) -> torch.Tensor:
        dt = t / steps
        current_z = z0
        for _ in range(steps):
            dW = torch.randn_like(current_z) * np.sqrt(dt)
            drift_term = self.drift(current_z) * dt
            diffusion_term = self.diffusion(current_z) * dW
            current_z = current_z + drift_term + diffusion_term
        return current_z

class DynamicLinear(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, weights: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if weights.dim() == 3:
            # Reshape input for batched matrix multiplication
            x_reshaped = x.unsqueeze(1)
            if bias is not None:
                # Perform batched add-matrix-multiply
                res = torch.baddbmm(bias.unsqueeze(1), x_reshaped, weights.transpose(1, 2))
            else:
                # Perform batched matrix-multiply
                res = torch.bmm(x_reshaped, weights.transpose(1, 2))
            # Squeeze the output to remove the temporary dimension
            return res.squeeze(1)
        else:
            return F.linear(x, weights, bias)

class HyperNetwork(nn.Module):
    def __init__(self, task_emb_dim: int, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        hidden_dim = (task_emb_dim + in_features * out_features) // 4
        self.weight_generator = nn.Sequential(
            nn.Linear(task_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_features * out_features)
        )
        self.bias_generator = nn.Sequential(
            nn.Linear(task_emb_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_features)
        )

    def forward(self, z_task: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = self.weight_generator(z_task)
        bias = self.bias_generator(z_task)
        if z_task.dim() == 2:
            batch_size = z_task.size(0)
            return (weights.view(batch_size, self.out_features, self.in_features),
                    bias.view(batch_size, self.out_features))
        else:
            return (weights.view(self.out_features, self.in_features),
                    bias.view(self.out_features))

class AffineCoupling(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2
        self.net = nn.Sequential(
            nn.Linear(self.half_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.dim - self.half_dim),
            nn.Tanh()
        )
        self.scale = nn.Parameter(torch.randn(self.dim - self.half_dim) * 0.01)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z1 = z[:, :self.half_dim]
        z2 = z[:, self.half_dim:]
        st = self.net(z1)
        log_s = self.scale * st
        t = st
        s = torch.exp(log_s)
        u2 = (z2 + t) * s
        u = torch.cat([z1, u2], dim=-1)
        log_det_jac = s.log().sum(dim=-1)
        return u, log_det_jac

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        u1 = u[:, :self.half_dim]
        u2 = u[:, self.half_dim:]
        st = self.net(u1)
        log_s = self.scale * st
        t = st
        s = torch.exp(log_s)
        z2 = (u2 / (s + 1e-8)) - t
        z = torch.cat([u1, z2], dim=-1)
        return z

class ManifoldNormalizingFlow(nn.Module):
    def __init__(self, latent_dim: int, num_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(AffineCoupling(latent_dim))
            self.layers.append(nn.Identity())

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        total_log_det_jac = torch.zeros(z.size(0), device=z.device)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Identity):
                z = torch.cat([z[:, self.layers[i-1].half_dim:], z[:, :self.layers[i-1].half_dim]], dim=1)
                continue
            z, log_det_jac = layer(z)
            total_log_det_jac += log_det_jac
        return z, total_log_det_jac

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if isinstance(layer, nn.Identity):
                u = torch.cat([u[:, -self.layers[i-1].half_dim:], u[:, :-self.layers[i-1].half_dim]], dim=1)
                continue
            u = layer.inverse(u)
        return u

class HypernetSDEContinualLearner(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 48,
                 task_emb_dim: int = 32, num_tasks_expected: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.task_emb_dim = task_emb_dim
        self.task_embeddings = nn.Parameter(torch.randn(num_tasks_expected, task_emb_dim) * 0.02)
        self.encoder_1 = DynamicLinear()
        self.hypernet_enc1 = HyperNetwork(task_emb_dim, input_dim, hidden_dim)
        self.encoder_2 = DynamicLinear()
        self.hypernet_enc2 = HyperNetwork(task_emb_dim, hidden_dim, hidden_dim)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.flow = ManifoldNormalizingFlow(latent_dim, num_layers=4)
        self.neural_sde = NeuralSDE(latent_dim)
        self.decoder_sde = NeuralSDE(latent_dim, num_layers=2)
        self.decoder_1 = DynamicLinear()
        self.hypernet_dec1 = HyperNetwork(task_emb_dim, latent_dim, hidden_dim)
        self.decoder_2 = nn.Linear(hidden_dim, input_dim)
        self.episodic_memory: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
        self.task_statistics: Dict[int, Dict] = {}

    def encode(self, x: torch.Tensor, task_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        task_vec = self.task_embeddings[task_id].unsqueeze(0).expand(x.size(0), -1)
        enc1_w, enc1_b = self.hypernet_enc1(task_vec)
        enc2_w, enc2_b = self.hypernet_enc2(task_vec)
        h1 = F.gelu(self.encoder_1(x, enc1_w, enc1_b))
        h2 = F.gelu(self.encoder_2(h1, enc2_w, enc2_b))
        z0 = self.to_latent(h2)
        u, log_det_jac = self.flow(z0)
        return z0, u, log_det_jac

    def decode(self, z: torch.Tensor, task_id: int) -> torch.Tensor:
        task_vec = self.task_embeddings[task_id].unsqueeze(0).expand(z.size(0), -1)
        dec1_w, dec1_b = self.hypernet_dec1(task_vec)
        z_evolved = self.decoder_sde(z, t=0.5)
        h1 = F.gelu(self.decoder_1(z_evolved, dec1_w, dec1_b))
        reconstruction = self.decoder_2(h1)
        return reconstruction

    def forward(self, x: torch.Tensor, task_id: int) -> Tuple:
        z0, u, log_det_jac = self.encode(x, task_id)
        z_ode_sde = self.neural_sde(z0, t=1.0)
        reconstruction = self.decode(z_ode_sde, task_id)
        return reconstruction, u, log_det_jac, z_ode_sde, z0

    def compute_loss(self, x: torch.Tensor, reconstruction: torch.Tensor, 
                    u: torch.Tensor, log_det_jac: torch.Tensor, z0: torch.Tensor,
                    task_id: int, epoch: int) -> Tuple[torch.Tensor, Dict]:
        recon_loss = F.mse_loss(reconstruction, x, reduction='mean')
        log_p_u = -0.5 * u.pow(2).sum(dim=1) - 0.5 * self.latent_dim * np.log(2 * np.pi)
        log_p_x = log_p_u + log_det_jac
        nll_loss = -torch.mean(log_p_x)
        replay_loss = torch.tensor(0.0, device=x.device)
        if len(self.episodic_memory) > 0 and epoch % 5 == 0:
            num_replay = min(32, len(self.episodic_memory))
            replay_indices = np.random.choice(len(self.episodic_memory), num_replay, replace=False)
            for idx in replay_indices:
                x_replay, z_replay, tid_replay = self.episodic_memory[idx]
                recon_replay = self.decode(z_replay.unsqueeze(0), tid_replay)
                replay_loss += F.mse_loss(recon_replay, x_replay.unsqueeze(0))
            replay_loss = replay_loss / num_replay
        task_separation_loss = torch.tensor(0.0, device=x.device)
        if task_id > 0:
            current_proto = self.task_embeddings[task_id]
            for tid in range(task_id):
                prev_proto = self.task_embeddings[tid]
                distance = torch.norm(current_proto - prev_proto)
                task_separation_loss += torch.exp(-distance)
        beta_nll = min(1.0, 0.01 + 0.99 * epoch / 300)
        # Tăng cường đáng kể trọng số của các thành phần chống quên lãng
        total_loss = (recon_loss + beta_nll * nll_loss + 2.0 * replay_loss + 1.5 * task_separation_loss)
        loss_components = {
            'reconstruction': recon_loss.item(),
            'nll_loss': nll_loss.item(),
            'replay': replay_loss.item(),
            'task_separation': task_separation_loss.item(),
            'total': total_loss.item()
        }
        return total_loss, loss_components

    def update_task_statistics(self, x: torch.Tensor, task_id: int):
        with torch.no_grad():
            z0, _, _ = self.encode(x, task_id)
            z_evolved = self.neural_sde(z0, t=1.0)
            num_samples_to_store = min(50, x.size(0))
            indices = torch.randperm(x.size(0))[:num_samples_to_store]
            for idx in indices:
                self.episodic_memory.append((x[idx].detach().clone(), z_evolved[idx].detach().clone(), task_id))
            if len(self.episodic_memory) > 500:
                self.episodic_memory = self.episodic_memory[-500:]

def train_hypernet_continual(model: HypernetSDEContinualLearner, tasks: List[Tuple], epochs: int = 400, lr: float = 5e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    total_steps = len(tasks) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
    )
    all_task_losses = {}
    for task_id, (x_train, _) in enumerate(tasks):
        print(f"\n{'='*70}")
        print(f"Training on Task {task_id}")
        print(f"{'='*70}")
        task_losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            reconstruction, u, log_det_jac, z_sde, z0 = model(x_train, task_id)
            loss, loss_components = model.compute_loss(
                x_train, reconstruction, u, log_det_jac, z0, task_id, epoch
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()
            task_losses.append(loss_components)
            if epoch % 50 == 0:
                print(f"Epoch {epoch:3d} | Total: {loss_components['total']:.6f} | "
                      f"Recon: {loss_components['reconstruction']:.6f} | "
                      f"NLL: {loss_components['nll_loss']:.6f} | "
                      f"Replay: {loss_components['replay']:.6f}")
        all_task_losses[task_id] = task_losses
        model.update_task_statistics(x_train, task_id)
        print(f"\n{'─'*70}")
        print(f"Evaluating retention across all tasks:")
        print(f"{'─'*70}")
        retention_scores = []
        for eval_task_id in range(task_id + 1):
            x_eval, _ = tasks[eval_task_id]
            with torch.no_grad():
                recon_eval, u_eval, ldj_eval, z_sde_eval, z0_eval = model(x_eval, eval_task_id)
                eval_loss, eval_components = model.compute_loss(
                    x_eval, recon_eval, u_eval, ldj_eval, z0_eval, eval_task_id, epochs-1
                )
                retention_scores.append(eval_components['reconstruction'])
                print(f"  Task {eval_task_id}: Reconstruction Loss = {eval_components['reconstruction']:.6f}")
        avg_retention = np.mean(retention_scores)
        print(f"\n  Average Retention Score: {avg_retention:.6f}")
    return all_task_losses

if __name__ == "__main__":
    print("Module loaded. Import and run training from external script.")
