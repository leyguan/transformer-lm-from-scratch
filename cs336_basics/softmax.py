import torch

def softmax(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    # 沿指定维度取最大值，keepdim 保持形状可广播
    max_vals = tensor.max(dim=dim, keepdim=True).values
    # 减去最大值，防止 exp 溢出
    shifted = tensor - max_vals
    exp_vals = torch.exp(shifted)
    # 沿同一维度求和，作为归一化分母
    sum_exp = exp_vals.sum(dim=dim, keepdim=True)
    return exp_vals / sum_exp