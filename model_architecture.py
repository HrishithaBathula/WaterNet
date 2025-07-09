
import torch
from models.waternet import WaterNet
from torchviz import make_dot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WaterNet().to(device)
x = torch.randn(1, 3, 256, 256).to(device)

output = model(x, x, x, x)

graph = make_dot(output, params=dict(model.named_parameters()))

graph.format = "png"
graph.directory = "."
graph.render("waternet_architecture")

print("✅ Model flowchart saved as 'waternet_architecture.png'")
