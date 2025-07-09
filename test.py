
import os
import torch
from torchvision import transforms
from PIL import Image
from models.waternet import WaterNet
from utils import save_image
from metrics import psnr, ssim, mse, mae
from tqdm import tqdm

raw_dir = 'data/input'
ref_dir = 'data/ground_truth'
output_dir = 'output'
checkpoint_path = 'checkpoints/epoch_100.pth'  # Update if needed

os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WaterNet().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

score_lines = []
avg_scores = {'psnr': 0, 'ssim': 0, 'mse': 0, 'mae': 0}
count = 0

for fname in tqdm(sorted(os.listdir(raw_dir))):
    raw_path = os.path.join(raw_dir, fname)
    ref_path = os.path.join(ref_dir, fname)

    raw = Image.open(raw_path).convert('RGB')
    ref = Image.open(ref_path).convert('RGB')

    raw_t = transform(raw).unsqueeze(0).to(device)
    wb_t = raw_t  
    he_t = raw_t
    gc_t = raw_t
    ref_t = transform(ref).to(device)

    with torch.no_grad():
        pred_t = model(raw_t, wb_t, he_t, gc_t).squeeze(0).clamp(0, 1)

    save_image(pred_t, os.path.join(output_dir, fname))

    psnr_val = psnr(pred_t, ref_t)
    ssim_val = ssim(pred_t, ref_t)
    mse_val = mse(pred_t, ref_t)
    mae_val = mae(pred_t, ref_t)

    avg_scores['psnr'] += psnr_val
    avg_scores['ssim'] += ssim_val
    avg_scores['mse'] += mse_val
    avg_scores['mae'] += mae_val
    count += 1

    score_lines.append(f"{fname:<20} PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}, MSE: {mse_val:.5f}, MAE: {mae_val:.5f}")

avg_scores = {k: v / count for k, v in avg_scores.items()}
score_lines.append("-" * 60)
score_lines.append(f"Average           PSNR: {avg_scores['psnr']:.2f}, SSIM: {avg_scores['ssim']:.4f}, MSE: {avg_scores['mse']:.5f}, MAE: {avg_scores['mae']:.5f}")

with open("scores.txt", "w") as f:
    f.write("\n".join(score_lines))

print("✅ Testing complete. Results saved to `output/` and `scores.txt`.")
