import torch
import torch.nn as nn
import torch.nn.functional as F

from rec_baseline.algos.base import BaseModel


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer.
    """
    
    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.n_heads, self.d_k)  # (bs, seq_len, n_heads, d_k)
        q = self.q_linear(q).view(bs, -1, self.n_heads, self.d_k)  # (bs, seq_len, n_heads, d_k)
        v = self.v_linear(v).view(bs, -1, self.n_heads, self.d_k)  # (bs, seq_len, n_heads, d_k)

        k = k.transpose(1, 2)  # (bs, n_heads, seq_len, d_k)
        q = q.transpose(1, 2)  # (bs, n_heads, seq_len, d_k)
        v = v.transpose(1, 2)  # (bs, n_heads, seq_len, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)  # (bs, n_heads, seq_len, seq_len)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)  # (bs, n_heads, seq_len, d_k)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)  # (bs, seq_len, d_model)
        output = self.out(output)  # (bs, seq_len, d_model)
        return output
    
    
class PositionwiseFeedForward(nn.Module):
    """
    Position-wise feed-forward layer.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x) -> torch.Tensor:
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))  # (bs, seq_len, d_model)
    

class TransformerLayer(nn.Module):
    """
    A single transformer layer.
    """
    
    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_heads, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm_1(x + self.dropout_1(self.self_attn(x, x, x)))
        x = self.layer_norm_2(x + self.dropout_2(self.feed_forward(x)))
        return x


class BERT4Rec(BaseModel):
    """
    An implementation of BERT4Rec.
    
    Ref: https://arxiv.org/abs/1904.06690
    """

    def __init__(self, 
                 n_items: int, 
                 latent_dim: int = 64, 
                 n_heads: int = 2, 
                 n_layers: int = 2, 
                 dropout: float = 0.2):
        super().__init__()
        
        self.n_layers = n_layers
        
        self.item_embeddings = nn.Embedding(n_items + 1, latent_dim, padding_idx=n_items)
        self.position_embeddings = nn.Embedding(100, latent_dim)
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(latent_dim, latent_dim * 4, n_heads, dropout) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, seq_len = x.size()
        positions = torch.arange(seq_len, device=x.device).repeat(bs, 1)
        
        x = self.item_embeddings(x) + self.position_embeddings(positions)
        
        for i in range(self.n_layers):
            x = self.transformer_layers[i](x)
            
        # use only the last hidden state
        x = x[:, -1, :]  # (bs, latent_dim)
        
        # apply linear projection
        logits = x @ self.item_embeddings.weight.t()  # (bs, n_items)
        
        return logits
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.argsort(logits, dim=-1, descending=True)
    
    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels)
