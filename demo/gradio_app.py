import torch
import numpy as np
import gradio as gr
import sys
sys.path.append('./src')
from main import HypernetSDEContinualLearner

# Demo model với kích thước nhỏ để chạy nhanh
model = HypernetSDEContinualLearner(
    input_dim=16, hidden_dim=32, latent_dim=8,
    task_emb_dim=8, num_tasks_expected=3
)
model.eval()

def predict(x_input):
    x = torch.tensor([x_input], dtype=torch.float32)
    with torch.no_grad():
        recon, _, _, _, _ = model(x, task_id=0)
    return recon.numpy().flatten().tolist()

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.inputs.Dataframe(
            type="numpy", headers=False, row_count=1, col_count=16,
            label="Nhập vector đầu vào (16 số)"
        )
    ],
    outputs=[
        gr.outputs.Dataframe(
            type="numpy", headers=False, row_count=1, col_count=16,
            label="Kết quả tái tạo"
        )
    ],
    examples=[(np.random.randn(16).tolist(),)],
    title="Hypernet SDE Learner Demo"
)
iface.launch(share=False)
