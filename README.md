# WaterNet
WaterNet is a deep learning-based method for underwater image enhancement, introduced in the paper:

"Water-Net: An Underwater Image Enhancement Model Based on Water Type Classification"
by Chongyi Li, Jichang Guo, Fatih Porikli (TIP, 2019)
🔗 Paper Link
---

WaterNet method for enhancing underwater images uses a fusion of color correction techniques and confidence-based weighting.

This implementation supports training and testing in both Google Colab and terminal (VS Code) environments using the UIEB dataset.
---
##  Average Evaluation metrics

| Metric | Value |
|--------|-------|
| PSNR   | 19.21 |
| SSIM   | 0.8443 |
| MSE    | 0.01559 |
| MAE    | 0.09382 |
 
---

## 🖥️ For VS Code / Terminal

> ✅ Recommended for GPU-enabled desktops or laptops.

### 🔧 Setup
```bash
pip install torch torchvision thop torchinfo pillow tqdm graphviz (or) pip install requirements.txt
choco install graphviz (or) conda install -c conda-forge python-graphviz (optional for model architecture)
python train.py
python test.py
type scores.txt
python flops.py
type flops.txt
python model_architecture.py
type waternet_architecture.txt
```
---

## 🚀 For Google Colab 

> ✅ Recommended for systems without GPUs.

### 🔧 Setup
```bash
!pip install torch torchvision thop torchinfo pillow tqdm
!apt install graphviz
!python train.py
!python test.py
!cat scores.txt
!python flops.py
!cat flops.txt
!python model_architecture.py
!cat waternet_architecture.txt
```
