import matplotlib.pyplot as plt
import torch
from datasets.ncaltech101_npz import NCaltechDataset
from pathlib import Path

dataset = NCaltechDataset(root_dir = Path(__file__).parents[1],H=240,W=240,T=120)
spikes, label = dataset[0]
image = torch.sum(spikes[:,:,:,:], dim=[0,1])
plt.imshow(image.numpy())
plt.show()
