from einops import einsum
import torch

def swiglu(d_model, d_ff, w1_weight, w2_weight, w3_weight, in_features):
    w1x = einsum(w1_weight, in_features, "d_ff d_model, ... d_model -> ... d_ff")
    silu_w1x = w1x * torch.sigmoid(w1x)
    w3x = einsum(w3_weight, in_features, "d_ff d_model, ... d_model -> ... d_ff")
    return einsum(w2_weight, silu_w1x * w3x, "d_model d_ff, ... d_ff -> ... d_model")
