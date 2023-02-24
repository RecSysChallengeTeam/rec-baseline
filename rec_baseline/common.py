import torch


def get_device() -> str:
    """
    Returns the device that available for use. 
    Priority:  GPU (CUDA) > M1 (MPS) > CPU
    
    Returns:
        string of device name
    """
    
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    return "cpu"

