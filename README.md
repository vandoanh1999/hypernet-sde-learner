# hypernet-sde-learner

**Thuật toán continual learning đột phá: Hypernet + SDE + Flow**

Repo này dành cho bán thương mại, chứa demo Gradio/Colab để khách hàng thử nghiệm nhanh.

## Nội dung chính
- `src/main.py`: mã nguồn lõi model
- `requirements.txt`: môi trường cần thiết (Torch/Numpy/Gradio)
- `LICENSE-MERCHANT.txt`: bản quyền thương mại, bán độc quyền
- `demo/gradio_app.py`: demo Gradio (web thử ngay input)
- `demo/colab_demo.py`: demo nhanh trên Colab
- `.gitignore`, `.github/workflows/python-ci.yml`: hỗ trợ dev

## Hướng dẫn dùng nhanh:
1. Tạo môi trường:  
   `pip install -r requirements.txt`
2. Chạy Gradio demo:  
   `python demo/gradio_app.py`
3. Chạy thử Colab demo:  
   Tải lên Colab và chạy:  
   `python demo/colab_demo.py`

## Bảo mật & bản quyền:
- Không chia sẻ repo này ra công khai nếu bán độc quyền
- Khách chỉ được sử dụng theo LICENSE-MERCHANT.txt

---
