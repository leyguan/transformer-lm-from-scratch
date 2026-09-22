import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from einops import rearrange

from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.rope import RoPE
from cs336_basics.linear import Linear


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = Linear(d_model, d_model)
        self.W_K = Linear(d_model, d_model)
        self.W_V = Linear(d_model, d_model)
        self.W_O = Linear(d_model, d_model)

        if max_seq_len is not None and theta is not None:
            self.rope = RoPE(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)
        else:
            self.rope = None

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
    ) -> Float[Tensor, "... seq_len d_model"]:
        seq_len = x.shape[-2]

        # 不改变形状的哈，三个 W 都是方阵
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # ... 在 einops.rearrange 是支持的
        Q = rearrange(Q, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        K = rearrange(K, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        V = rearrange(V, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)

        # RoPE
        if self.rope:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)

            
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # 下三角矩阵
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )

        attn_res = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        attn_res = rearrange(attn_res, "... h seq_len d_k -> ... seq_len (h d_k)")

        return self.W_O(attn_res)
