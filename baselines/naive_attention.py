import torch
import torch.nn as nn
import math

class NaiveAttention(nn.Module):
    """
    A standard PyTorch implementation of Dot-Product Attention.
    Used as a baseline for correctness and performance comparison on ROCm.
    """
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: (Batch, Heads, Seq_Q, Dim)
            k: (Batch, Heads, Seq_K, Dim)
            v: (Batch, Heads, Seq_V, Dim)
            mask: Optional mask (Batch, 1, 1, Seq_K)
        """
        d_k = q.size(-1)
        
        # Scaling dot product
        # (Batch, Heads, Seq_Q, Seq_K)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax: This is the memory bottleneck (O(N^2))
        p_attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        
        # Final output
        # (Batch, Heads, Seq_Q, Dim)
        return torch.matmul(p_attn, v)

# TODO: Implement a version that uses torch.nn.functional.scaled_dot_product_attention 
# as a secondary baseline for ROCm internal optimizations.
