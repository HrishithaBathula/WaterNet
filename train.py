
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.waternet import WaterNet
from utils import UIEBDataset, save_checkpoint
from metrics import ssim_loss

EPOCHS = 100
BATCH_SIZE = 2
LR = 1e-4
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])
train_dataset = UIEBDataset('data/input', 'data/ground_truth', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

model = WaterNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.L1Loss()

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for raw, wb, he, gc, gt in train_loader:
        raw, wb, he, gc, gt = raw.to(DEVICE), wb.to(DEVICE), he.to(DEVICE), gc.to(DEVICE), gt.to(DEVICE)

        out = model(raw, wb, he, gc)
        loss = criterion(out, gt) + ssim_loss(out, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss / len(train_loader):.4f}")
    save_checkpoint(model, f"checkpoints/epoch_{epoch+1}.pth")

print("Training complete.")
