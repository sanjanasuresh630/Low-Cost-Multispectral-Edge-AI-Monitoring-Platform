import sys
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from glob import glob
import numpy as np
import torch
from random import shuffle
import random
from numpy import random as npr
import pandas as pd
import imageio
import os


class MethaneLoader(Dataset):
    """
    Adapted for HuggingFace dataset structure while keeping original logic
    
    Original expects:
        turkmenistan_plumes_raw/plume_X/DATE-raw.npy
        final_annotations/plume_X/{train,val,test}/{pos,neg}/*.npy
    
    HuggingFace has:
        data/{train,val,test}/s2/*.npy
        data/{train,val,test}/label/{pos,neg}/*.npy
        data/{train,val,test}/mbmp/*.npy (optional)
    """
    
    def __init__(self, data_root, device, mode, plume_id=None, red=False, alli=False, channels=12):
        self.data_root = data_root
        self.device = device
        self.mode = mode
        self.reduce = red
        self.channels = channels

        if mode == "train":
            persist = False
        else:
            persist = True

        # Adapt to HuggingFace structure
        label_base = os.path.join(data_root, mode, 'label')
        self.pos_labels = sorted(glob(os.path.join(label_base, 'pos', '*.npy')))
        self.neg_labels = sorted(glob(os.path.join(label_base, 'neg', '*.npy')))
        
        self.labels = self.pos_labels + self.neg_labels
        if not alli:
            self.sample_labels_and_combine(persist=persist)
        
    def sample_labels_and_combine(self, persist=False):
        """
        Sample a subset of negative labels for each epoch
        """
        if self.mode in ["test", "val"]:
            self.labels = self.pos_labels + self.neg_labels
        else:
            if persist:
                random.seed(555)
            
            shuffle(self.neg_labels)
            self.labels = self.pos_labels + self.neg_labels[:len(self.pos_labels)]

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        f = self.labels[index]
        
        # Get filename and construct s2 path
        filename = os.path.basename(f)
        s2_path = os.path.join(self.data_root, self.mode, 's2', filename)
        
        # Load target and context
        target = np.load(f)
        context = np.load(s2_path)
        
        # Channel selection (matching original code exactly)
        if self.channels == 2:
            context = context[..., 10:12]  # Bands 11-12
        elif self.channels == 5:
            context = np.concatenate([context[..., 1:4], context[..., 10:12]], axis=-1)  # RGB + SWIR
        elif self.channels == 12:
            context = context[..., :12]  # All except band 10
        # else: use all 13 bands
        
        # Crop to center (matching original code exactly)
        s = 50  # Half of 100x100 patch
        
        if self.mode == "train":
            rng = npr.RandomState()
            mid_loc_x = rng.randint(s, target.shape[0] - s)
            mid_loc_y = rng.randint(s, target.shape[1] - s)
        else:
            mid_loc_x = target.shape[0] // 2
            mid_loc_y = target.shape[1] // 2
        
        # Crop target and context
        target = target[mid_loc_x - s:mid_loc_x + s,
                       mid_loc_y - s:mid_loc_y + s]
        
        context = context[mid_loc_x - s:mid_loc_x + s,
                         mid_loc_y - s:mid_loc_y + s, :]
        
        # Create RGB image for visualization (from bands 4,3,2 - R,G,B)
        # Original context before channel selection for RGB
        full_context = np.load(s2_path)
        full_context_crop = full_context[mid_loc_x - s:mid_loc_x + s,
                                        mid_loc_y - s:mid_loc_y + s, :]
        
        if full_context_crop.shape[2] >= 4:
            # Sentinel-2: Band 4 (Red), Band 3 (Green), Band 2 (Blue)
            rgb_img = full_context_crop[..., [3, 2, 1]]  # R, G, B indices
        else:
            rgb_img = np.zeros((100, 100, 3))
        
        # Check if MBMP exists (for diff_img)
        mbmp_path = os.path.join(self.data_root, self.mode, 'mbmp', filename)
        if os.path.exists(mbmp_path):
            diff_img_full = np.load(mbmp_path)
            diff_img = diff_img_full[mid_loc_x - s:mid_loc_x + s,
                                    mid_loc_y - s:mid_loc_y + s, ...]
            if len(diff_img.shape) == 2:
                diff_img_g = diff_img
                diff_img = np.stack([diff_img] * 3, axis=-1)
            else:
                diff_img_g = diff_img[..., 0] if diff_img.shape[2] > 0 else diff_img
        else:
            # Create dummy diff images if not available
            diff_img = np.zeros((100, 100, 3))
            diff_img_g = np.zeros((100, 100))
        
        if self.reduce:
            target = np.array([np.int(target.any())])
        
        # Return dictionary matching original format exactly
        # IMPORTANT: Keep /255 normalization as in original
        d = {
            "pred": torch.from_numpy(context).float().to(self.device).permute(2, 0, 1) / 255,
            "target": torch.from_numpy(target).float().to(self.device),
            "rgb_img": torch.from_numpy(rgb_img).float(),  # Don't move to device yet
            "diff_img": torch.from_numpy(diff_img).float(),
            "diff_img_g": torch.from_numpy(diff_img_g).float()
        }
        
        return d


# For backward compatibility, keep the original DataLoader name
# But use Dataset as base class (original mistakenly used DataLoader)
if __name__ == "__main__":
    # Test the loader with your data
    data_root = r"D:\ch4net\data"
    
    print(f"Testing loader with data from: {data_root}")
    
    loader = MethaneLoader(
        data_root=data_root,
        device="cuda" if torch.cuda.is_available() else "cpu",
        mode="train",
        plume_id=None,
        channels=12
    )
    
    print(f"Dataset size: {len(loader)}")
    
    # Test loading one sample
    sample = loader[0]
    print(f"pred shape: {sample['pred'].shape}")
    print(f"target shape: {sample['target'].shape}")
    print(f"pred range: [{sample['pred'].min():.3f}, {sample['pred'].max():.3f}]")
