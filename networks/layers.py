import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, proj_bias: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, f"dim ({d_model}) must be divisible by heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = self.d_k ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=proj_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=proj_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=proj_bias)
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
        q = self.q_proj(x_q).view(B, Lq, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x_kv).view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x_kv).view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)

        logits = (q @ k.transpose(-2, -1)) * self.scale
        if bias is not None:
            logits = logits + bias.unsqueeze(1)

        attn = logits.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, Lq, self.d_model)
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
