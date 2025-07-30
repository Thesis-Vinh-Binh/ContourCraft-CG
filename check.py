import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version (from torch):", torch.version.cuda)
print("Device name:", torch.cuda.get_device_name(0))