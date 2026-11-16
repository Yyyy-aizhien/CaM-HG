import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Stochastic Depth (DropPath) ---
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# --- Layers ---

class CausalDynamicAdjacency(nn.Module):
    def __init__(self, dim, heads=8, alpha=4.0, window_size=3):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.alpha = alpha
        self.window_size = window_size
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(self, vertices, hyperedges):
        B, N, D = vertices.shape
        K = hyperedges.shape[1]
        
        k_proj = self.W_k(vertices).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        q_proj = self.W_q(hyperedges).view(B, K, self.heads, self.head_dim).transpose(1, 2)
        
        logits = (q_proj @ k_proj.transpose(-2, -1)) * self.scale
        
        device = vertices.device
        q_indices = torch.arange(K, device=device)
        k_indices = torch.arange(N, device=device)
        q_blocks = q_indices.div(3, rounding_mode='floor').unsqueeze(1)
        k_blocks = k_indices.div(3, rounding_mode='floor').unsqueeze(0)
        
        causal_mask = (q_blocks >= k_blocks)
        window_mask = (q_blocks - k_blocks) <= self.window_size
        final_mask = causal_mask & window_mask
        final_mask = final_mask.float()
        
        logits = logits + (1.0 - final_mask.unsqueeze(0).unsqueeze(0)) * -1e9
        A = torch.sigmoid(self.alpha * logits)
        return A 

class CausalHypergraphAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, adjacency_mask):
        B, T_q, _ = query.shape
        B, T_k, _ = key.shape
        q = self.W_q(query).view(B, T_q, self.heads, self.head_dim).transpose(1, 2)
        k = self.W_k(key).view(B, T_k, self.heads, self.head_dim).transpose(1, 2)
        v = self.W_v(value).view(B, T_k, self.heads, self.head_dim).transpose(1, 2)
        
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        if adjacency_mask is not None:
            gate = adjacency_mask.clamp(min=1e-6)
            attn_scores = attn_scores + torch.log(gate)
        
        attn_probs = attn_scores.softmax(dim=-1)
        attn_probs = self.dropout(attn_probs)
        out = (attn_probs @ v).transpose(1, 2).contiguous().view(B, T_q, self.dim)
        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)