import torch
import torch.nn as nn
import torch.nn.functional as F

from rec_baseline.algos.base import BaseModel


class GRU4Rec(BaseModel):
    """
    An implementation of GRU4Rec.
    """
    
    def __init__(self, 
                 n_items: int, 
                 latent_dim: int = 64, 
                 gru_latent_dim: int = 32, 
                 n_layers: int = 2, 
                 dropout: float = 0.2):
        super().__init__()
        
        self.item_embeddings = nn.Embedding(n_items + 1, latent_dim, padding_idx=n_items)
        self.gru = nn.GRU(latent_dim, gru_latent_dim, n_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.last_layers = nn.ModuleList([
            nn.Linear(gru_latent_dim, latent_dim * 2),
            nn.Linear(latent_dim * 2, latent_dim)
        ])
        
    def session_embedding(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Calculate the session embedding.
        
        Args:
            item_ids: the item ids of the session
            
        Returns:
            the session embedding
        """
        
        item_emb = self.item_embeddings(item_ids)  # (batch_size, seq_len, latent_dim)
        _, hs = self.gru(item_emb)  # (n_layers, batch_size, gru_latent_dim)
        hs = hs.transpose(0, 1)  # (batch_size, n_layers, gru_latent_dim)
        
        s_emb = self.dropout(hs).mean(dim=1)  # (batch_size, gru_latent_dim)
        
        for layer in self.last_layers:
            s_emb = layer(s_emb)
            
        return s_emb
        
    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        s_emb = self.session_embedding(item_ids)  # (batch_size, latent_dim)
        logits = (s_emb @ F.normalize(self.item_embeddings.weight[:-1, :]).T)  # (batch_size, n_items)
        return logits

    def loss_fn(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Cross entropy loss as the loss function for the model
        
        Args:
            preds: the predicted scores
            targets: the ground truth scores
            
        Returns:
            the loss
        """
        
        return F.cross_entropy(preds, targets)
    
    def predict(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Predict the ordered item ids.
        
        Args:
            item_ids: the item ids of the session
            
        Returns:
            the ordered item ids
        """
        
        logits = self.forward(item_ids)  # (batch_size, n_items)
        
        ordered_item_ids = torch.argsort(logits, dim=1, descending=True)  # (batch_size, n_items)
        
        return ordered_item_ids
