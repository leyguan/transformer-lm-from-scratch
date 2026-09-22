from jaxtyping import Float, Int
from torch import Tensor
from cs336_basics.linear import Linear
from cs336_basics.embedding import Embedding
from cs336_basics.transformer_block import transformer_block
from cs336_basics.rmsnorm import RMSNorm


def transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    device = weights["token_embeddings.weight"].device
    dtype = weights["token_embeddings.weight"].dtype

    # 1. Token embedding
    emb = Embedding(vocab_size, d_model, device=device, dtype=dtype)
    emb.load_state_dict({"weight": weights["token_embeddings.weight"]})
    x = emb(in_indices)

    # 2. N transformer blocks
    for i in range(num_layers):
        prefix = f"layers.{i}."
        layer_weights = {
            key[len(prefix):]: value
            for key, value in weights.items()
            if key.startswith(prefix)
        }
        x = transformer_block(
            d_model, num_heads, d_ff, context_length, rope_theta,
            layer_weights, x,
        )

    # 3. Final RMSNorm
    norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
    norm.load_state_dict({"weight": weights["ln_final.weight"]})
    x = norm(x)

    # 4. LM head projection
    lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    lm_head.load_state_dict({"weight": weights["lm_head.weight"]})
    logits = lm_head(x)

    return logits

