import torch
import torch.nn as nn
import math 
from typing import Tuple

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, proj_bias: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, f"dim ({d_model}) must be divisible by heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k= d_model // n_heads
        self.scale = self.d_k ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias = proj_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias = proj_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias = proj_bias)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """
        x_q: (B, Lq, D) query side tokens
        x_kv: (B, Lk, D)
        """
        B, Lq, _ = x_q.shape
        Lk = x_kv.shape[1]

        # split the outputs into the heads - the one linear layer represents all heads at once and is split after the forward pass
        q = self.q_proj(x_q).view(B, Lq, self.n_heads, self.d_k).transpose(1,2)
        k = self.k_proj(x_kv).view(B, Lk, self.n_heads, self.d_k).transpose(1,2)
        v = self.v_proj(x_kv).view(B, Lk, self.n_heads, self.d_k).transpose(1,2)

        logits = (q @ k.transpose(-2,-1)) * self.scale
        if bias is not None:
            logits = logits + bias.unsqueeze(1)
        
        attn = logits.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1,2).reshape(B, Lq, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """
    Basic FFN thta has a two linear layers with a GELU inbetween

    d_model (float): controls the relative scale of the hidden layer
    """
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout)
        )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IntraShapeBlock(nn.Module):
    """
    MHA block with FFN and norm used to updated features within a shape. 
    This uses a prenorm formulation with LN
    """
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, mlp_ratio, dropout)

    
    def forward(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm_attn(x)
        x = x + self.attn(h, h, bias=bias)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class InterShapeBlock(nn.Module):
    """
    MHA block with FFN and norm for cross attending shape features
    This uses prenorm formulation with LN
    """
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, mlp_ratio, dropout)

    
    def forward(self, x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        hx, hy = self.norm(x), self.norm(y)                 # same norm -> symmetric
        bias_t = bias.transpose(-1, -2) if bias is not None else None
 
        # cross-attention; both updates read the pre-update normed Tensors
        x = x + self.attn(hx, hy, bias=bias)  # X <- Y
        y = y + self.attn(hy, hx, bias=bias_t)  # Y <- X
 
        x = x + self.mlp(self.norm_mlp(x))
        y = y + self.mlp(self.norm_mlp(y))
        return x, y


class ShapeMatchingEncoder(nn.Module):
    """
    The overall shape matching block. This combines inter and intra shape attention blocks interleave

    Args:
        in_dim (int): input feature dimensins
        dim (int): the dimension of the latent transformer space
        heads (int): the nunber of attention heads (must divide dim)
        depth (int): the number of layer clones
        mlp_ratio (float): relative size of the hidden layer of the FFN components
        dropout (float): dropout probability
    """
    def __init__(self, in_dim: int, dim: int, heads: int, depth: int = 2,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.embed = nn.Linear(in_dim, dim)
        # ONE intra block per layer, reused for both shapes => weight sharing.
        self.intra = nn.ModuleList(
            IntraShapeBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth))
        self.inter = nn.ModuleList(
            InterShapeBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth))
        self.out_norm = nn.LayerNorm(dim)
 
    def forward(self, feat_x: torch.Tensor, feat_y: torch.Tensor,
                intra_bias_x: torch.Tensor | None = None,
                intra_bias_y: torch.Tensor | None = None,
                inter_bias: torch.Tensor | None = None):
        """
        feat_x: (B, Nx, in_dim), feat_y: (B, Ny, in_dim)
        intra_bias_*: (B, N, N) per-shape relation (kNN / geodesic) or None
        inter_bias:   (B, Nx, Ny) cross relation (noisy assignment) or None
        returns refined embeddings (emb_x, emb_y), each (B, N, dim)
        """
        x, y = self.embed(feat_x), self.embed(feat_y)
        for intra, inter in zip(self.intra, self.inter):
            x = intra(x, bias=intra_bias_x)
            y = intra(y, bias=intra_bias_y)   # same instance
            x, y = inter(x, y, bias=inter_bias)
        return self.out_norm(x), self.out_norm(y)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, in_dim, dim, heads = 2, 16, 8, 32, 4
    enc = ShapeMatchingEncoder(in_dim, dim, heads, depth=2).eval()
 
    fx = torch.randn(B, N, in_dim)
    fy = torch.randn(B, N, in_dim)
 
    ex, ey = enc(fx, fy)
    # swap inputs; with inter_bias=None there is nothing to transpose
    ex_s, ey_s = enc(fy, fx)
 
    err_x = (ex - ey_s).abs().max().item()
    err_y = (ey - ex_s).abs().max().item()
    print(f"output shapes: {tuple(ex.shape)}, {tuple(ey.shape)}")
    print(f"symmetry residuals (want ~0): {err_x:.2e}, {err_y:.2e}")
 
    # symmetry with a non-trivial inter bias: swap inputs AND transpose the bias
    bias = torch.randn(B, N, N)
    ax, ay = enc(fx, fy, inter_bias=bias)
    bx, by = enc(fy, fx, inter_bias=bias.transpose(-1, -2))
    print("symmetry residuals w/ bias: "
          f"{(ax - by).abs().max().item():.2e}, {(ay - bx).abs().max().item():.2e}")