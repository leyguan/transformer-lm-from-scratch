import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Float, Int
from torch import Tensor


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        exp = torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k
        # [0, 2, 4, ..., d_k - 2]
        freqs = 1.0 / (theta ** exp)
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        # [0, 1, 2, ..., max_seq_len - 1]
        angles = einsum(positions, freqs, "max_seq_len, d_k_half -> max_seq_len d_k_half")
        self.register_buffer("sin_buf", torch.sin(angles), persistent=False)
        self.register_buffer("cos_buf", torch.cos(angles), persistent=False)

    def forward(self, x: Float[Tensor, " ... sequence_length d_k"], token_positions: Int[Tensor, " ... sequence_length"]) -> torch.Tensor:
        sin_vals = self.sin_buf[token_positions]
        cos_vals = self.cos_buf[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = einsum(x_even, cos_vals, "... seq_len d_k_half, ... seq_len d_k_half -> ... seq_len d_k_half") - \
                       einsum(x_odd, sin_vals, "... seq_len d_k_half, ... seq_len d_k_half -> ... seq_len d_k_half")
        rotated_odd = einsum(x_even, sin_vals, "... seq_len d_k_half, ... seq_len d_k_half -> ... seq_len d_k_half") + \
                       einsum(x_odd, cos_vals, "... seq_len d_k_half, ... seq_len d_k_half -> ... seq_len d_k_half")
        # rotated_even = x_even * cos_vals - x_odd * sin_vals
        # rotated_odd = x_even * sin_vals + x_odd * cos_vals

        result = rearrange([rotated_even, rotated_odd], "two ... d_k_half -> ... (d_k_half two)")
        return result
