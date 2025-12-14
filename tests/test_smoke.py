import torch

def test_torch():
    print(torch.__version__)
    print(torch.backends.mps.is_built())

def test_device():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    x = torch.randn(2,3,device=device)
    assert x.device.type == device

def test_device_allocation():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    x = torch.randn(2,3)
    x = x.to(device)
    assert x.device.type == device
