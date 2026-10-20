import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")   # ต้องได้ True
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")        # RTX 4060

    print(torch.cuda.get_device_properties(0).total_memory)

    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM: {vram:.1f} GB")