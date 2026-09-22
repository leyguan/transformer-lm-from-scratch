from jaxtyping import Float, Int
from torch import Tensor
from cs336_basics.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.swiglu import swiglu
from cs336_basics.rmsnorm import RMSNorm


def transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    mha = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta)
    mha.load_state_dict({
        "W_Q.weight": weights["attn.q_proj.weight"],
        "W_K.weight": weights["attn.k_proj.weight"],
        "W_V.weight": weights["attn.v_proj.weight"],
        "W_O.weight": weights["attn.output_proj.weight"],
    })

    rmsnorm1 = RMSNorm(d_model)
    rmsnorm1.load_state_dict({"weight": weights["ln1.weight"]})

    rmsnorm2 = RMSNorm(d_model)
    rmsnorm2.load_state_dict({"weight": weights["ln2.weight"]})

    x1 = in_features + mha(rmsnorm1(in_features))
    x2 = x1 + swiglu(d_model, d_ff, weights["ffn.w1.weight"],
                     weights["ffn.w2.weight"], weights["ffn.w3.weight"], rmsnorm2(x1))
    
    return x2
