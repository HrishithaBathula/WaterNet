
import torch
from models.waternet import WaterNet
from thop import profile, clever_format

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dummy inputs
x = torch.randn(1, 3, 256, 256).to(device)
model = WaterNet().to(device)

# Run FLOPs and Params
flops, params = profile(model, inputs=(x, x, x, x))
flops, params = clever_format([flops, params], "%.3f")

# Save to flops.txt
with open("flops.txt", "w") as f:
    f.write(f"FLOPs: {flops}\n")
    f.write(f"Parameters: {params}\n")

print("✅ FLOPs saved to flops.txt")
