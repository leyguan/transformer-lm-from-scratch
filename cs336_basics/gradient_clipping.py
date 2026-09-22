import torch
from torch import Tensor
from typing import List
from jaxtyping import Float

def clip_gradients(
    parameters: List[Float[Tensor, "..."]], 
    max_norm: float, 
    eps: float = 1e-6
) -> None:
    """
    对一组参数的梯度进行 L2 范数裁剪（原地修改）。
    
    Args:
        parameters: 包含 PyTorch Parameter 或 Tensor 的列表。
        max_norm: 允许的最大 L2 范数 M。
        eps: 用于数值稳定的极小值 epsilon（默认 1e-6）。
    """
    # 1. 过滤出所有含有梯度的参数
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    # 2. 计算所有梯度拼接后的全局 L2 范数
    # 为了节省显存，不进行实际的拼接，而是累加每个 Tensor 的平方和
    device = grads[0].device
    total_norm_sq = torch.tensor(0.0, device=device)
    
    for g in grads:
        # 使用 detach() 确保这一步计算不留存在 autograd 图中
        total_norm_sq += g.detach().pow(2).sum()
        
    total_norm = torch.sqrt(total_norm_sq)

    # 3. 如果总范数超过了最大阈值，则按比例缩放梯度（原地操作）
    if total_norm >= max_norm:
        clip_coef = max_norm / (total_norm + eps)
        for g in grads:
            # mul_ 是 PyTorch 中的原地乘法操作
            g.detach().mul_(clip_coef)
