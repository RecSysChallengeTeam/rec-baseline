import torch


def get_device() -> str:
    """
    Returns the device that available for use. 
    Priority: M1 (MPS) > GPU (CUDA) > CPU
    
    Returns:
        string of device name
    """
    
    if hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
