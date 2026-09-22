from torch import Tensor
from einops import einsum
from jaxtyping import Bool, Float

from cs336_basics.softmax import softmax

def scaled_dot_product_attention(
    query: Float[Tensor, "batch_size ... seq_len d_k"],
    key: Float[Tensor, "batch_size ... seq_len d_k"],
    value: Float[Tensor, "batch_size ... seq_len d_v"],
    mask: Bool[Tensor, "seq_len seq_len"] | None = None,
) -> Float[Tensor, "batch_size ... seq_len d_v"]:
    d_k = query.shape[-1]

    scores = einsum(query, key, "... seq_len_i d_k, ... seq_len_j d_k -> ... seq_len_i seq_len_j")
    scores = scores / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # weights = torch.softmax(scores, dim=-1)
    weights = softmax(scores, dim=-1)

    output = einsum(weights, value, "... seq_len_i seq_len_j, ... seq_len_j d_v -> ... seq_len_i d_v")
    return output
