import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional

# ======================================================================
# CÀI ĐẶT BẮT BUỘC (Cho Colab)
# !pip install torchsde matplotlib
# ======================================================================
try:
    import torchsde
except ImportError:
    print("torchsde chưa được cài. Đang cài đặt...")
    import os
    os.system('pip install torchsde matplotlib')
    import torchsde

# ======================================================================
# NÂNG CẤP #1: BỘ ĐỆM REPLAY CHUẨN (RESERVOIR BUFFER)
# Thay thế cho List + np.random.choice
# ======================================================================
class ReservoirReplayBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.buffer_x: Optional[torch.Tensor] = None
        self.buffer_z: Optional[torch.Tensor] = None
        self.buffer_tid: Optional[torch.Tensor] = None
        self.counter = 0

    def add(self, x: torch.Tensor, z: torch.Tensor, tid: int):
        batch_size = x.size(0)
        
        if self.buffer_x is None:
            self.buffer_x = torch.empty((self.capacity, *x.shape[1:]), dtype=x.dtype, device=self.device)
            self.buffer_z = torch.empty((self.capacity, *z.shape[1:]), dtype=z.dtype, device=self.device)
            self.buffer_tid = torch.empty((self.capacity,), dtype=torch.int64, device=self.device)

        for i in range(batch_size):
            index = self.counter % self.capacity
            self.buffer_x[index] = x[i]
            self.buffer_z[index] = z[i]
            self.buffer_tid[index] = tid
            self.counter += 1

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.is_empty():
            raise ValueError("Không thể sample từ buffer rỗng.")
            
        max_idx = min(self.counter, self.capacity)
        indices = torch.randint(0, max_idx, (batch_size,), device=self.device)
        
        return self.buffer_x[indices], self.buffer_z[indices], self.buffer_tid[indices]

    def __len__(self) -> int:
        return min(self.counter, self.capacity)

    def is_empty(self) -> bool:
        return self.counter == 0

# ======================================================================
# MÔ HÌNH 1: Baseline MLP (Compile-ready)
# ======================================================================
class BaselineMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 48):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

# ======================================================================
# MÔ HÌNH 2: Hypernet-SDE-Flow (Nâng Cấp Toàn Diện)
# ======================================================================

# --- NÂNG CẤP #2: Tích hợp `torchsde` ---
class NeuralSDE(nn.Module):
    """
    Định nghĩa SDE, tách riêng drift/diffusion để `torchsde` gọi.
    Đây là SDE chuẩn Ito: dZ = f(t, Z)dt + g(t, Z)dW_t
    """
    sde_type = "ito"
    noise_type = "general"

    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        
        # f(t, Z) - Drift
        self.drift_net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
        # g(t, Z) - Diffusion
        self.diffusion_net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, dim), nn.Sigmoid()
        )

    def f(self, t: float, z: torch.Tensor) -> torch.Tensor:
        # Thêm 't' vào input
        t_vec = torch.ones(z.size(0), 1, device=z.device) * t
        tz = torch.cat([t_vec, z], dim=1)
        return self.drift_net(tz)

    def g(self, t: float, z: torch.Tensor) -> torch.Tensor:
        t_vec = torch.ones(z.size(0), 1, device=z.device) * t
        tz = torch.cat([t_vec, z], dim=1)
        return self.diffusion_net(tz)
    
    def forward(self, z0: torch.Tensor, t: float = 1.0) -> torch.Tensor:
        t_span = torch.tensor([0.0, t], device=z0.device)
        # Dùng solver SRK (Stochastic Runge-Kutta) hiệu suất cao
        solution = torchsde.sdeint(self, z0, t_span, method='srk', dt=t/10.0)
        return solution[1] # Lấy điểm cuối (t=1.0)


class DynamicLinear(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F.linear(x, w, b)

class HyperNetwork(nn.Module):
    def __init__(self, task_emb_dim: int, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        hidden_dim = max(32, (task_emb_dim + in_features * out_features) // 8) # Tối ưu
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
            nn.Linear(hidden_dim, (dim - self.half_dim) * 2) # Output (log_s, t)
        )
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z1, z2 = z.chunk(2, dim=1)
        log_s_t = self.net(z1)
        log_s, t = log_s_t.chunk(2, dim=1)
        s = torch.exp(log_s.tanh()) # Ổn định s
        u2 = (z2 + t) * s
        u = torch.cat([z1, u2], dim=-1)
        log_det_jac = s.log().sum(dim=-1)
        return u, log_det_jac

class Permutation(nn.Module):
    """Permutation chuẩn, thay cho nn.Identity() hack"""
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("perm", torch.randperm(dim))
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, self.perm]
    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        inv_perm = torch.argsort(self.perm)
        return u[:, inv_perm]

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
        
        self.base_dist = torch.distributions.Normal(
            torch.tensor(0.0, device='cpu'), torch.tensor(1.0, device='cpu')
        )
        
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
        
        # Hypernet chỉ sinh 1 W, b cho 1 task_id. Cần `einsum` cho mini-batch.
        # x: [B, Din], w: [Dout, Din] -> [B, Dout]
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
    
    def forward(self, x: torch.Tensor, task_id: int) -> Tuple:
        task_id_tensor = torch.tensor(task_id, device=x.device) # Tensor cho Hypernet
        z0, u, log_det_jac = self.encode(x, task_id_tensor)
        z_sde = self.neural_sde(z0, t=1.0)
        reconstruction = self.decode(z_sde, task_id_tensor)
        return reconstruction, u, log_det_jac, z0
    
    def compute_loss(self, x: torch.Tensor, recon: torch.Tensor, 
                    u: torch.Tensor, log_det_jac: torch.Tensor,
                    replay_batch: Optional[Tuple], task_id: int, epoch: int
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        recon_loss = F.mse_loss(recon, x)
        
        log_p_u = self.base_dist.log_prob(u).sum(dim=1)
        nll_loss = -torch.mean(log_p_u + log_det_jac)
        
        replay_loss = torch.tensor(0.0, device=x.device)
        if replay_batch is not None:
            x_rep, z_rep, tid_rep = replay_batch
            # Phải lặp vì tid_rep có thể chứa nhiều task_id khác nhau
            recon_replay_list = []
            for tid_u in torch.unique(tid_rep):
                mask = (tid_rep == tid_u)
                recon_rep = self.decode(z_rep[mask], tid_u)
                recon_replay_list.append(F.mse_loss(recon_rep, x_rep[mask], reduction='sum'))
            replay_loss = torch.sum(torch.stack(recon_replay_list)) / x_rep.size(0)

        task_sep_loss = torch.tensor(0.0, device=x.device)
        if task_id > 0:
            current_proto = self.task_embeddings[task_id]
            prev_protos = self.task_embeddings[:task_id]
            distances = torch.norm(current_proto - prev_protos, dim=1)
            task_sep_loss = torch.mean(torch.exp(-distances * 2))
        
        beta_nll = min(1.0, 0.01 + 0.99 * epoch / 150)
        total_loss = (recon_loss + 
                     beta_nll * nll_loss * 0.1 + 
                     0.5 * replay_loss +
                     0.3 * task_sep_loss)
        return total_loss, recon_loss

# ======================================================================
# PHẦN BENCHMARK (Đã nâng cấp)
# ======================================================================

def create_tasks(n_samples=300, device='cpu'):
    x_base = torch.linspace(-4, 4, n_samples).unsqueeze(1).to(device)
    tasks_data = [
        (torch.cat([torch.sin(x_base), torch.cos(x_base), x_base**2 / 16], dim=1), None),
        (torch.cat([torch.exp(-x_base**2), torch.tanh(x_base * 2), torch.sin(2*x_base)], dim=1), None),
        (torch.cat([x_base**3 / 64, torch.sigmoid(x_base), torch.cos(3*x_base)], dim=1), None),
        (torch.cat([torch.sin(x_base) * torch.cos(x_base), torch.abs(x_base) / 4, torch.exp(-torch.abs(x_base))], dim=1), None),
        (torch.cat([torch.sin(x_base**2), torch.sign(x_base) * torch.sqrt(torch.abs(x_base)) / 2, torch.tanh(x_base)], dim=1), None),
    ]
    # NÂNG CẤP #5: Tạo DataLoader
    task_loaders = [DataLoader(TensorDataset(x), batch_size=64, shuffle=True) for x, _ in tasks_data]
    task_test_data = [x for x, _ in tasks_data] # Dữ liệu test (full-batch)
    return task_loaders, task_test_data

def main_benchmark():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    INPUT_DIM = 3
    HIDDEN_DIM = 64
    LATENT_DIM = 24
    TASK_EMB_DIM = 16
    NUM_TASKS = 5
    EPOCHS_PER_TASK = 200 # Giảm epochs vì mini-batch hội tụ nhanh hơn
    LR = 7e-4
    
    train_loaders, test_data = create_tasks(device=device)
    
    model_mlp = BaselineMLP(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(device)
    model_hypernet = HypernetSDEContinualLearner(
        input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM,
        task_emb_dim=TASK_EMB_DIM, num_tasks_expected=NUM_TASKS
    ).to(device)
    
    # NÂNG CẤP #1 (khởi tạo):
    replay_buffer = ReservoirReplayBuffer(capacity=200, device=device)
    
    # NÂNG CẤP #2: Áp dụng `torch.compile`
    print("\nĐang compile mô hình (lần đầu có thể mất 1-2 phút)...")
    try:
        # Thử compile ở chế độ tối đa
        model_mlp_compiled = torch.compile(model_mlp, mode="max-autotune")
        model_hypernet_compiled = torch.compile(model_hypernet, mode="max-autotune")
        print("Compile thành công (max-autotune)!")
    except Exception as e:
        print(f"Compile max-autotune thất bại ({e}), dùng chế độ 'default'.")
        model_mlp_compiled = torch.compile(model_mlp)
        model_hypernet_compiled = torch.compile(model_hypernet)
        
    
    optimizer_mlp = torch.optim.AdamW(model_mlp_compiled.parameters(), lr=LR)
    optimizer_hypernet = torch.optim.AdamW(model_hypernet_compiled.parameters(), lr=LR)
    
    results = {
        "MLP": np.full((NUM_TASKS, NUM_TASKS), np.nan),
        "Hypernet": np.full((NUM_TASKS, NUM_TASKS), np.nan)
    }
    
    print(f"{'='*70}\nBẮT ĐẦU BENCHMARK (PRO-LEVEL)\n{'='*70}")
    
    start_time_total = time.time()
    
    for task_id in range(NUM_TASKS):
        x_train_loader = train_loaders[task_id]
        
        print(f"\n--- Training on TASK {task_id} ---")
        
        # --- Train MLP ---
        model_mlp_compiled.train()
        for epoch in range(EPOCHS_PER_TASK):
            for (x_batch,) in x_train_loader:
                optimizer_mlp.zero_grad(set_to_none=True)
                recon = model_mlp_compiled(x_batch)
                loss = F.mse_loss(recon, x_batch)
                loss.backward()
                optimizer_mlp.step()
        
        # --- Train Hypernet ---
        model_hypernet_compiled.train()
        for epoch in range(EPOCHS_PER_TASK):
            for (x_batch,) in x_train_loader:
                optimizer_hypernet.zero_grad(set_to_none=True)
                
                recon, u, ldj, z0 = model_hypernet_compiled(x_batch, task_id)
                
                replay_batch = None
                if not replay_buffer.is_empty():
                    replay_batch = replay_buffer.sample(x_batch.size(0) // 2)
                    
                loss, recon_loss = model_hypernet_compiled.compute_loss(
                    x_batch, recon, u, ldj, replay_batch, task_id, epoch
                )
                loss.backward()
                optimizer_hypernet.step()
        
        # Cập nhật buffer (chạy 1 lần sau khi train)
        model_hypernet_compiled.eval()
        with torch.no_grad():
            x_full = test_data[task_id]
            _, _, _, z0 = model_hypernet_compiled(x_full, task_id)
            z_sde = model_hypernet_compiled.neural_sde(z0, t=1.0)
            replay_buffer.add(x_full, z_sde, task_id)

        # --- GIAI ĐOẠN ĐÁNH GIÁ (MẤU CHỐT) ---
        print(f"--- Evaluating memory on ALL tasks (0 to {task_id}) ---")
        model_mlp_compiled.eval()
        model_hypernet_compiled.eval()
        
        with torch.no_grad():
            for eval_task_id in range(task_id + 1):
                x_test = test_data[eval_task_id]
                
                loss_mlp = F.mse_loss(model_mlp_compiled(x_test), x_test).item()
                recon_hyper, _, _, _ = model_hypernet_compiled(x_test, eval_task_id)
                loss_hyper = F.mse_loss(recon_hyper, x_test).item()
                
                results["MLP"][task_id, eval_task_id] = loss_mlp
                results["Hypernet"][task_id, eval_task_id] = loss_hyper
                
                print(f"  Task {eval_task_id} MSE | MLP: {loss_mlp:8.5f} | Hypernet: {loss_hyper:8.5f}")

    end_time_total = time.time()
    print(f"\nTổng thời gian benchmark: {end_time_total - start_time_total:.2f} giây")

    # --- TỔNG KẾT & NÂNG CẤP #3: VẼ ĐỒ THỊ ---
    print(f"\n{'='*70}\nKẾT QUẢ CUỐI CÙNG & TRỰC QUAN HÓA\n{'='*70}")
    
    # Lấy dữ liệu cho đồ thị
    task_axis = np.arange(NUM_TASKS)
    # Lấy cột T0 (sự quên lãng của task 0)
    forgetting_mlp = results["MLP"][:, 0]
    forgetting_hypernet = results["Hypernet"][:, 0]
    
    # Tính toán chỉ số CL chuẩn
    final_avg_loss_mlp = np.nanmean(results["MLP"][-1, :])
    final_avg_loss_hyper = np.nanmean(results["Hypernet"][-1, :])
    
    avg_forgetting_mlp = np.nanmean([results["MLP"][-1, j] - results["MLP"][j, j] for j in range(NUM_TASKS)])
    avg_forgetting_hyper = np.nanmean([results["Hypernet"][-1, j] - results["Hypernet"][j, j] for j in range(NUM_TASKS)])

    print(f"--- Chỉ số tổng kết (sau khi học hết T{NUM_TASKS-1}) ---")
    print(f"Average Loss (MLP):      {final_avg_loss_mlp:.6f}")
    print(f"Average Loss (Hypernet): {final_avg_loss_hyper:.6f}")
    print(f"Average Forgetting (MLP):      {avg_forgetting_mlp:.6f} (Càng cao càng cùi)")
    print(f"Average Forgetting (Hypernet): {avg_forgetting_hyper:.6f} (Càng gần 0 càng tốt)")

    # Vẽ đồ thị
    plt.figure(figsize=(10, 6))
    plt.plot(task_axis, forgetting_mlp, 'r-o', label=f'Baseline MLP (Avg Loss: {final_avg_loss_mlp:.3f})')
    plt.plot(task_axis, forgetting_hypernet, 'g-s', label=f'Hypernet-SDE (Avg Loss: {final_avg_loss_hyper:.3f})')
    
    plt.title(f'Phân tích "Quên Lãng Thảm Khốc" (Catastrophic Forgetting)', fontsize=16)
    plt.ylabel('MSE Loss on Task 0', fontsize=12)
    plt.xlabel('Task $T_i$ vừa học xong', fontsize=12)
    plt.xticks(task_axis, [f'Học T{i}' for i in task_axis])
        plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.yscale('log') # Dùng thang log để thấy rõ sự chênh lệch
    
    # Chú thích kết luận
    conclusion = (f"Hypernet Avg Forgetting: {avg_forgetting_hyper:.4f}\n"
                  f"MLP Avg Forgetting: {avg_forgetting_mlp:.4f}")
    plt.text(0.05, 0.5, conclusion, transform=plt.gca().transAxes, 
             fontsize=12, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    print("\nĐang hiển thị đồ thị kết quả...")
    # Lưu file ảnh thay vì show (tốt hơn cho môi trường server)
    plt.savefig("benchmark_forgetting_graph.png")
    print("Đã lưu đồ thị vào file 'benchmark_forgetting_graph.png'")
    # plt.show() # Bật dòng này nếu mày chạy local và muốn nó tự mở
    

if __name__ == "__main__":
    main_benchmark()
