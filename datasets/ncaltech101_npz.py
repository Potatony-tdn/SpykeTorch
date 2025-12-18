'''
N-Caltech101 Dataset processing utilities.

This module contains tools to convert .npz files into PyTorch tensors for training. To download N-Caltech101 and convert it to .npz files, use scripts/download_ncaltech101.py.

Usage:
    #TODO

Attributes:
    #TODO
'''

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

def npz_to_spike_tensor(x, y, p, t, timesteps, H, W):
    '''
    Transforms arrays loaded from .npz into tensors.

    Args:
        x, y (np.uint16): x and y coordinates of an event
        p (np.uint8): polarity of the event (0 for OFF, 1 for ON)
        t (np.uint32): timestamp (in microseconds)
    
    Returns:
        spikes (torch.Tensor): shape (timesteps, 2, H, W)
    '''
    spikes = torch.zeros(timesteps, 2, H, W, dtype = torch.float32)
    x_idx = torch.from_numpy(x).long()
    y_idx = torch.from_numpy(y).long()
    t_raw = torch.from_numpy(t).float()
    p_idx = torch.from_numpy(p).long()

    t_bin = ((t_raw-t_raw.min())/(t_raw.max()-t_raw.min())*(timesteps-1)).floor().long()
    spikes[t_bin, p_idx, y_idx, x_idx] = 1.0
    
    return spikes

class NCaltechDataset(Dataset):
    def __init__(self, root_dir, T=120, H=320, W=320):
        self.root = root_dir / "data" / "decoded_npz" / "ncaltech101" / "events"
        self.T = T
        self.H = H
        self.W = W
        self.classes = sorted([d for d in self.root.iterdir() if d.is_dir()])
        self.classes_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            for sample in cls.rglob("*.npz"):
                self.samples.append((sample, self.classes_to_idx[cls]))

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with np.load(path) as data:
            x = data["x"]
            y = data["y"]
            p = data["p"]
            t = data["t"]
        
        spikes = npz_to_spike_tensor(x, y, p, t, self.T, self.H, self.W)
        
        return spikes, torch.tensor(label, dtype=torch.long) 
