import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import sys
from typing import Dict, List, Tuple, Optional

# Thêm đường dẫn src để import mô hình gốc
sys.path.append('./src')
from main import HypernetSDEContinualLearner

# ======================================================================
# ĐỊNH NGHĨA CÁC MÔ HÌNH
# ======================================================================

# --- MÔ HÌNH 1: Baseline MLP (Xe đạp) ---
# Đây là một Autoencoder MLP đơn giản, giữ nguyên
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
# XÓA BỎ TOÀN BỘ CODE CŨ, VÌ NÓ BỊ LỖI VÀ KHÔNG ĐÚNG
# THAY VÀO ĐÓ, CHÚNG TA SẼ IMPORT TRỰC TIẾP TỪ `src/main.py`
# Điều này đảm bảo chúng ta đang benchmark đúng mô hình, mạnh mẽ và chính xác

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
            # The forward pass now returns a tuple of tensors
            reconstruction, u, log_det_jac, z_ode_sde, z0 = model_hypernet(x_train, task_id)
            # The loss function is called with all the necessary components
            loss, loss_components = model_hypernet.compute_loss(
                x_train, reconstruction, u, log_det_jac, z0, task_id, epoch
            )
            loss.backward()
            optimizer_hypernet.step()
            if epoch % (EPOCHS_PER_TASK - 1) == 0 or epoch == 0:
                print(f"  [Hypernet] Epoch {epoch:3d}, Total Loss: {loss.item():.6f}, Recon Loss: {loss_components['reconstruction']:.6f}")

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
                # Correctly call the forward pass and calculate reconstruction loss
                recon_hyper, _, _, _, _ = model_hypernet(x_test, eval_task_id)
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
      
