import torch
import torch.nn as nn


class VanillaMF(nn.Module):
    """
    An implementation of vanilla matrix factorization model.
    """
    
    def __init__(self, n_users: int, n_items: int, latent_dim: int = 10):
        super().__init__()
        
        self.user_embeddings = nn.Embedding(n_users, latent_dim)
        self.item_embeddings = nn.Embedding(n_items, latent_dim)
        
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embeddings(user_ids).squeeze(dim=1)  # (batch_size, latent_dim)
        item_emb = self.item_embeddings(item_ids).squeeze(dim=1)  # (batch_size, latent_dim)
        
        return (user_emb * item_emb).sum(dim=1)  # (batch_size, )
    
    @classmethod
    def loss_fn(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Simple MSE loss as the loss function for the model.
        
        Args:
            preds: predicted scores
            targets: ground truth scores
            
        Returns:
            MSE of the predictions and the ground truth
        """
        
        # return nn.MSELoss()(preds, targets)  # use package to calculate loss
        return ((preds - targets) ** 2).mean()
