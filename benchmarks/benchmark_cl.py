import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, List, Tuple, Optional

# ======================================================================
# ĐỊNH NGHĨA CÁC MÔ HÌNH
# ======================================================================

# --- MÔ HÌNH 1: Baseline MLP (Xe đạp) ---
# Đây là một Autoencoder MLP đơn giản
class BaselineMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 48):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

# --- MÔ HÌNH 2: Hypernet-SDE-Flow (F-35) ---
# Toàn bộ code đột phá từ lần trước
# (Bao gồm SDE, HyperNetwork, Flow, và mô hình chính)

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
    def __init__(self): super().__init__()
    def forward(self, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F.linear(x, w, b)

class HyperNetwork(nn.Module):
    def __init__(self, task_emb_dim: int, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        hidden_dim = (task_emb_dim + in_features * out_features) // 4
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
            nn.Linear(hidden_dim, self.dim - self.half_dim), nn.Tanh()
        )
        self.scale = nn.Parameter(torch.randn(self.dim - self.half_dim) * 0.01)
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z1, z2 = z[:, :self.half_dim], z[:, self.half_dim:]
        st = self.net(z1)
        log_s, t = self.scale * st, st
        s = torch.exp(log_s)
        u2 = (z2 + t) * s
        u = torch.cat([z1, u2], dim=-1)
        log_det_jac = s.log().sum(dim=-1)
        return u, log_det_jac
    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        u1, u2 = u[:, :self.half_dim], u[:, self.half_dim:]
        st = self.net(u1)
        log_s, t = self.scale * st, st
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
            self.layers.append(nn.Identity()) # Placeholder for permutation
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
        
        # Dynamic Encoder
        self.encoder_1 = DynamicLinear()
        self.hypernet_enc1 = HyperNetwork(task_emb_dim, input_dim, hidden_dim)
        self.encoder_2 = DynamicLinear()
        self.hypernet_enc2 = HyperNetwork(task_emb_dim, hidden_dim, hidden_dim)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        
        # Manifold & Dynamics
        self.flow = ManifoldNormalizingFlow(latent_dim, num_layers=4)
        self.neural_sde = NeuralSDE(latent_dim)
        
        # Dynamic Decoder
        self.decoder_sde = NeuralSDE(latent_dim, num_layers=2)
        self.decoder_1 = DynamicLinear()
        self.hypernet_dec1 = HyperNetwork(task_emb_dim, latent_dim, hidden_dim)
        self.decoder_2 = nn.Linear(hidden_dim, input_dim) # Output layer (static)
        
        self.episodic_memory: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
        self.task_statistics: Dict[int, Dict] = {}

    def encode(self, x: torch.Tensor, task_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        task_vec = self.task_embeddings[task_id]
        enc1_w, enc1_b = self.hypernet_enc1(task_vec)
        enc2_w, enc2_b = self.hypernet_enc2(task_vec)
        
        h1 = F.gelu(self.encoder_1(x, enc1_w, enc1_b))
        h2 = F.gelu(self.encoder_2(h1, enc2_w, enc2_b))
        z0 = self.to_latent(h2)
        u, log_det_jac = self.flow(z0)
        return z0, u, log_det_jac

    def decode(self, z: torch.Tensor, task_id: int) -> torch.Tensor:
        task_vec = self.task_embeddings[task_id]
        dec1_w, dec1_b = self.hypernet_dec1(task_vec)
        z_evolved = self.decoder_sde(z, t=0.5)
        h1 = F.gelu(self.decoder_1(z_evolved, dec1_w, dec1_b))
        reconstruction = self.decoder_2(h1)
        return reconstruction
    
    def forward(self, x: torch.Tensor, task_id: int) -> Tuple:
        z0, u, log_det_jac = self.encode(x, task_id)
        z_ode_sde = self.neural_sde(z0, t=1.0)
        reconstruction = self.decode(z_ode_sde, task_id)
        return reconstruction, u, log_det_jac, z0
    
    def compute_loss(self, x: torch.Tensor, reconstruction: torch.Tensor, 
                    u: torch.Tensor, log_det_jac: torch.Tensor,
                    task_id: int, epoch: int) -> torch.Tensor:
        
        recon_loss = F.mse_loss(reconstruction, x, reduction='mean')
        
        log_p_u = -0.5 * u.pow(2).sum(dim=1) - 0.5 * self.latent_dim * np.log(2 * np.pi)
        log_p_x = log_p_u + log_det_jac
        nll_loss = -torch.mean(log_p_x)
        
        replay_loss = torch.tensor(0.0, device=x.device)
        if len(self.episodic_memory) > 0 and epoch % 5 == 0:
            num_replay = min(16, len(self.episodic_memory))
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
                task_separation_loss += torch.exp(-distance * 2) # Tăng lực đẩy
        
        beta_nll = min(1.0, 0.01 + 0.99 * epoch / 150) # Annealing nhanh hơn
        total_loss = (recon_loss + 
                     beta_nll * nll_loss * 0.1 + # Giảm trọng số NLL
                     0.5 * replay_loss +
                     0.3 * task_separation_loss)
        return total_loss, recon_loss

    def update_task_statistics(self, x: torch.Tensor, task_id: int):
        with torch.no_grad():
            z0, _, _ = self.encode(x, task_id)
            z_evolved = self.neural_sde(z0, t=1.0)
            num_samples_to_store = min(50, x.size(0))
            indices = torch.randperm(x.size(0))[:num_samples_to_store]
            for idx in indices:
                self.episodic_memory.append((x[idx].detach().clone(), 
                                            z_evolved[idx].detach().clone(), 
                                            task_id))
            if len(self.episodic_memory) > 200: # Giới hạn bộ nhớ 
                self.episodic_memory = self.episodic_memory[-200:]

# ======================================================================
# PHẦN BENCHMARK (CỐT LÕI CỦA BÀI TEST)
# ======================================================================

def create_tasks(n_samples=300, device='cpu'):
    """Tạo 5 bộ dữ liệu task"""
    x_base = torch.linspace(-4, 4, n_samples).unsqueeze(1).to(device)
    tasks = [
        (torch.cat([torch.sin(x_base), torch.cos(x_base), x_base**2 / 16], dim=1), None),
        (torch.cat([torch.exp(-x_base**2), torch.tanh(x_base * 2), torch.sin(2*x_base)], dim=1), None),
        (torch.cat([x_base**3 / 64, torch.sigmoid(x_base), torch.cos(3*x_base)], dim=1), None),
        (torch.cat([torch.sin(x_base) * torch.cos(x_base), torch.abs(x_base) / 4, torch.exp(-torch.abs(x_base))], dim=1), None),
        (torch.cat([torch.sin(x_base**2), torch.sign(x_base) * torch.sqrt(torch.abs(x_base)) / 2, torch.tanh(x_base)], dim=1), None),
    ]
    return tasks

def main_benchmark():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Khởi tạo ---
    INPUT_DIM = 3
    HIDDEN_DIM = 64  # Giảm độ phức tạp để train nhanh hơn
    LATENT_DIM = 24  # Giảm độ phức tạp
    TASK_EMB_DIM = 16
    NUM_TASKS = 5
    EPOCHS_PER_TASK = 300 # Giảm epochs để test nhanh
    LR = 5e-4
    
    tasks = create_tasks(device=device)
    
    # Khởi tạo 2 mô hình
    model_mlp = BaselineMLP(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(device)
    model_hypernet = HypernetSDEContinualLearner(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        task_emb_dim=TASK_EMB_DIM,
        num_tasks_expected=NUM_TASKS
    ).to(device)
    
    optimizer_mlp = torch.optim.AdamW(model_mlp.parameters(), lr=LR)
    optimizer_hypernet = torch.optim.AdamW(model_hypernet.parameters(), lr=LR)
    
    # Bảng kết quả (Ma trận Lỗi)
    # results[model_name][task_trained][task_eval]
    results = {
        "MLP": np.zeros((NUM_TASKS, NUM_TASKS)),
        "Hypernet": np.zeros((NUM_TASKS, NUM_TASKS))
    }
    
    print(f"\n{'='*70}")
    print(f"BẮT ĐẦU BENCHMARK HỌC LIÊN TỤC (5 TÁC VỤ)")
    print(f"MLP Params: {sum(p.numel() for p in model_mlp.parameters()):,}")
    print(f"Hypernet Params: {sum(p.numel() for p in model_hypernet.parameters()):,}")
    print(f"{'='*70}\n")
    
    # --- Vòng lặp Học Liên Tục ---
    for task_id in range(NUM_TASKS):
        x_train, _ = tasks[task_id]
        
        print(f"\n--- Training on TASK {task_id} ---")
        
        # --- Train MLP ---
        # Thằng này không biết task_id là gì, nó chỉ bị ghi đè (overwrite)
        model_mlp.train()
        for epoch in range(EPOCHS_PER_TASK):
            optimizer_mlp.zero_grad()
            recon = model_mlp(x_train)
            loss = F.mse_loss(recon, x_train)
            loss.backward()
            optimizer_mlp.step()
            if epoch % (EPOCHS_PER_TASK - 1) == 0 or epoch == 0:
                print(f"  [MLP] Epoch {epoch:3d}, Recon Loss: {loss.item():.6f}")

        # --- Train Hypernet ---
        # Thằng này biết nó đang học task_id
        model_hypernet.train()
        for epoch in range(EPOCHS_PER_TASK):
            optimizer_hypernet.zero_grad()
            recon, u, ldj, z0 = model_hypernet(x_train, task_id)
            loss, recon_loss = model_hypernet.compute_loss(
                x_train, recon, u, ldj, task_id, epoch
            )
            loss.backward()
            optimizer_hypernet.step()
            if epoch % (EPOCHS_PER_TASK - 1) == 0 or epoch == 0:
                print(f"  [Hypernet] Epoch {epoch:3d}, Total Loss: {loss.item():.6f}, Recon Loss: {recon_loss.item():.6f}")

        # Cập nhật bộ nhớ cho Hypernet
        model_hypernet.update_task_statistics(x_train, task_id)
        
        # --- GIAI ĐOẠN ĐÁNH GIÁ (MẤU CHỐT) ---
        print(f"\n--- Evaluating memory on ALL tasks (0 to {task_id}) ---")
        model_mlp.eval()
        model_hypernet.eval()
        
        with torch.no_grad():
            for eval_task_id in range(task_id + 1):
                x_test, _ = tasks[eval_task_id]
                
                # Test MLP
                recon_mlp = model_mlp(x_test)
                loss_mlp = F.mse_loss(recon_mlp, x_test).item()
                
                # Test Hypernet (phải cung cấp đúng eval_task_id)
                recon_hyper, _, _, _ = model_hypernet(x_test, eval_task_id)
                loss_hyper = F.mse_loss(recon_hyper, x_test).item()
                
                results["MLP"][task_id, eval_task_id] = loss_mlp
                results["Hypernet"][task_id, eval_task_id] = loss_hyper
                
                print(f"  Task {eval_task_id} MSE | MLP: {loss_mlp:.6f} | Hypernet: {loss_hyper:.6f}")

    # --- TỔNG KẾT ---
    print(f"\n{'='*70}")
    print(f"KẾT QUẢ CUỐI CÙNG (MA TRẬN RECONSTRUCTION MSE)")
    print(f"{'='*70}")
    
    print("\nBaseline MLP (Cột: Task Test, Hàng: Task Vừa Học Xong)")
    print("-----------------------------------------------------")
    print("       ", end="")
    for i in range(NUM_TASKS): print(f"  T{i} Test ", end="")
    print("\n")
    for i in range(NUM_TASKS):
        print(f"Học T{i}: ", end="")
        for j in range(i + 1):
            print(f"  {results['MLP'][i, j]:.4f}  ", end="")
        print("")

    print("\nHypernet-SDE (Cột: Task Test, Hàng: Task Vừa Học Xong)")
    print("-------------------------------------------------------")
    print("       ", end="")
    for i in range(NUM_TASKS): print(f"  T{i} Test ", end="")
    print("\n")
    for i in range(NUM_TASKS):
        print(f"Học T{i}: ", end="")
        for j in range(i + 1):
            print(f"  {results['Hypernet'][i, j]:.4f}  ", end="")
        print("")

    print(f"\n{'='*70}")
    print(f"PHÂN TÍCH SỰ QUÊN LÃNG (FORGETTING)")
    print(f"{'='*70}")
    
    # Lấy hàng cuối cùng của ma trận kết quả (sau khi đã học T4)
    final_mlp_losses = results["MLP"][-1, :]
    final_hypernet_losses = results["Hypernet"][-1, :]
    
    print(f"MLP Final Avg Loss (all tasks):   {np.mean(final_mlp_losses):.6f}")
    print(f"Hypernet Final Avg Loss (all tasks): {np.mean(final_hypernet_losses):.6f}")
    
    print("\n--- Chi tiết về Trí Nhớ Task 0 ---")
    loss_t0_mlp_start = results["MLP"][0, 0]
    loss_t0_mlp_end = results["MLP"][NUM_TASKS - 1, 0]
    forgetting_mlp = loss_t0_mlp_end - loss_t0_mlp_start
    
    loss_t0_hyper_start = results["Hypernet"][0, 0]
    loss_t0_hyper_end = results["Hypernet"][NUM_TASKS - 1, 0]
    forgetting_hyper = loss_t0_hyper_end - loss_t0_hyper_start
    
    print(f"MLP Loss T0 (Sau khi học T0):    {loss_t0_mlp_start:.6f}")
    print(f"MLP Loss T0 (Sau khi học T4):    {loss_t0_mlp_end:.6f}")
    print(f"==> MLP Forgetting:              {forgetting_mlp:.6f}")
    
    print(f"\nHypernet Loss T0 (Sau khi học T0): {loss_t0_hyper_start:.6f}")
    print(f"Hypernet Loss T0 (Sau khi học T4): {loss_t0_hyper_end:.6f}")
    print(f"==> Hypernet Forgetting:         {forgetting_hyper:.6f}")
    
    if forgetting_mlp > forgetting_hyper * 5: # Nếu MLP quên gấp 5 lần
        print("\n[KẾT LUẬN]: Đột phá thành công. Baseline MLP đã bị 'Quên Lãng Thảm Khốc'.")
    else:
        print("\n[KẾT LUẬN]: Thất bại. Hypernet không giữ được trí nhớ tốt hơn MLP.")


if __name__ == "__main__":
    main_benchmark()
      
