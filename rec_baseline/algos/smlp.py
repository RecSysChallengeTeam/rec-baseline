import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict
from rec_baseline.algos.base import BaseModel


class SMLP(BaseModel):
    """
    Sequence-Aware Multi-Layer Perceptron without scalar features.
    
    Ref: https://dl.acm.org/doi/abs/10.1145/3556702.3556851
    """
    
    def __init__(self, 
                 item2vec: Dict,
                 n_items: int,
                 k: int = 5, 
                 latent_dim: int = 64, 
                 dropout: float = 0.2):
        super().__init__()
        
        self.k = k
        item_vec_dim = len(item2vec[list(item2vec.keys())[0]])
        self.item2vec = torch.FloatTensor([item2vec[i] for i in range(n_items)] + [torch.zeros(item_vec_dim)])
        self.item_embeddings = nn.Embedding(n_items + 1, latent_dim, padding_idx=n_items)
        self.dropout = nn.Dropout(dropout)
        self.last_layers = nn.ModuleList([
            nn.Linear(latent_dim + item_vec_dim, latent_dim * 2),
            nn.Linear(latent_dim * 2, latent_dim)
        ])
        
    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_emb = self.item_embeddings(item_ids)  # (batch_size, seq_len, latent_dim)
        feature_emb = F.embedding(item_ids, self.item2vec)  # (batch_size, seq_len, item_vec_dim)
        
        last_k_item_emb = item_emb[:, -self.k:, :]  # (batch_size, k, latent_dim)
        last_k_feature_emb = feature_emb[:, -self.k:, :]  # (batch_size, k, item_vec_dim)
        
        item_emb = last_k_item_emb.sum(dim=1)  # (batch_size, latent_dim)
        feature_emb = last_k_feature_emb.sum(dim=1)  # (batch_size, item_vec_dim)
        
        concat = torch.cat([item_emb, feature_emb], dim=1)  # (batch_size, latent_dim + item_vec_dim)
        
        for layer in self.last_layers:
            concat = layer(concat)
            
        logits = (concat @ F.normalize(self.item_embeddings.weight[:-1, :]).T)  # (batch_size, n_items)
        
        return logits

    def loss_fn(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Cross entropy loss as the loss function for the model
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
