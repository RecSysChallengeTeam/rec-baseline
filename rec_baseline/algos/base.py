import torch

import torch.nn as nn


class BaseModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def predict(self, *args, **kwargs):
        raise NotImplementedError
    
    def loss_fn(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:        
        raise NotImplementedError
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)
