import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        # 可学习的 gain 参数，每个维度一个，初始化为 1
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # RMS(a) = sqrt( mean(a_i^2) + eps )
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        # RMSNorm(a_i) = (a_i / RMS(a)) * g_i
        result = (x / rms) * self.weight.to(torch.float32)

        return result.to(in_dtype)



# -------------------------------------------------------------------------------- #
# 下面是默写的

# class RMSNorm(nn.Module):
#     def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         in_dtype = x.dtype
#         x = x.to(torch.float32)
#         rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
#         result = (x / rms) * self.weight.to(torch.float32)
#         return result.to(in_dtype)