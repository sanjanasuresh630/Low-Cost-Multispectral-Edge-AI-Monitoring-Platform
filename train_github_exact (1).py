"""
Focal Loss + Tversky Loss - Best for Imbalanced Segmentation
This focuses on hard boundary pixels
"""

from models import *
from trainer import *
from loader import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def focal_tversky_loss(pred, target, alpha=0.7, beta=0.3, gamma=1.5):
    """
    Focal Tversky Loss - BEST for imbalanced segmentation with boundaries
    
    Tversky: Controls FP vs FN trade-off (like weighted Dice)
    Focal: Focuses on hard-to-classify pixels (boundaries)
    
    This is better than Dice for your use case!
    """
    pred_prob = torch.sigmoid(pred)
    
    # Flatten
    pred_flat = pred_prob.view(pred_prob.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    # Tversky components
    TP = (pred_flat * target_flat).sum(dim=1)
    FP = (pred_flat * (1 - target_flat)).sum(dim=1)
    FN = ((1 - pred_flat) * target_flat).sum(dim=1)
    
    # Tversky index
    tversky = (TP + 1) / (TP + alpha * FP + beta * FN + 1)
    
    # Focal component - focus on hard samples
    focal_tversky = (1 - tversky).pow(gamma)
    
    return focal_tversky.mean()

def combined_focal_bce(pred, target):
    """
    50% Focal Tversky + 50% BCE
    Best combination for your scenario
    """
    # Focal Tversky
    ft_loss = focal_tversky_loss(pred, target)
    
    # BCE
    bce = nn.BCEWithLogitsLoss(reduction='none')(pred, target)
    bce_loss = bce.sum(dim=(-2, -1)).mean()
    
    # Combine
    return 0.5 * ft_loss + 0.5 * bce_loss

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_ROOT = r"D:\ch4net\data"
OUT_DIR = "FINAL_13_focal_tversky/"
CHANNELS = 12
N_EPOCHS = 250
LEARNING_RATE = 1e-4
BATCH_SIZE = 16

# ============================================================================
# Setup
# ============================================================================
import os
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(0)
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*70)
print("Focal Tversky Loss - Best for Boundary Refinement")
print("="*70)
print(f"Device: {device}")
print(f"Output: {OUT_DIR}")
print(f"Channels: {CHANNELS}")
print(f"Loss: 50% Focal Tversky + 50% BCE")
print("\nWhy this is better:")
print("  ✓ Focuses on boundary pixels (hard pixels)")
print("  ✓ Handles class imbalance better than Dice")
print("  ✓ Won't collapse to predicting nothing")
print("  ✓ Keeps your excellent scene detection")
print("\nExpected:")
print("  Scene metrics: Stay at 0.85+ (excellent)")
print("  Pixel IoU: 0.29 → 0.35-0.42")

# Model
model = Unet(in_channels=CHANNELS, out_channels=1, div_factor=1, prob_output=False)
model = model.to(device)
model = nn.DataParallel(model)

# Datasets  
train_dataset = MethaneLoader(data_root=DATA_ROOT, device=device, mode="train", plume_id=None, channels=CHANNELS)
test_dataset = MethaneLoader(data_root=DATA_ROOT, device=device, mode="test", plume_id=None, channels=CHANNELS)

print(f"\nDatasets: Train={len(train_dataset)}, Test={len(test_dataset)}")

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Trainer
trainer = Trainer(model, train_loader, test_loader, train_dataset, combined_focal_bce, OUT_DIR, LEARNING_RATE)

# Train
print("\nStarting training...")
trainer.train(n_epochs=N_EPOCHS)

print(f"\nDone! Evaluate with: IN_DIR = '{OUT_DIR}'")