import torch
import numpy as np
from datasets.ncaltech101_npz import npz_to_spike_tensor

def test_npz_to_spike_tensor_logic():
    x = np.array([10, 20, 30], dtype=np.uint16)
    y = np.array([5, 5, 5], dtype=np.uint16)
    p = np.array([0, 1, 0], dtype=np.uint8) # OFF, ON, OFF
    t = np.array([0, 50, 100], dtype=np.uint32)

    T= 5
    H = 32
    W = 32

    spikes = npz_to_spike_tensor (x,y,p,t,T,H,W)
    print("Shape: ", spikes.shape)
    assert spikes.shape == (5,2,32,32), "Shape"
    assert spikes[0, 0, 5, 10] == 1.0, "First event missing!"
    assert spikes[4, 0, 5, 30] == 1.0, "Last event missing!"
