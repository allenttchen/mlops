import torch


def set_device(mps: bool = False, cuda: bool = False):
    """
    Set the device to cuda and default tensor types to FloatTensor on the device
    """
    device = "cpu"
    if torch.cuda.is_available() and cuda:
        device = "cuda"
    elif torch.backends.mps.is_available() and mps:
        device = "mps"

    torch_device = torch.device(device)
    return torch_device
