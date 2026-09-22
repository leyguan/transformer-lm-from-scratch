"""
这个class究竟是干什么的？不就是矩阵乘法？那跟那个矩阵乘呢？
这个就取一个linear.py的名字放在cs336_basics文件夹下面吗？
我不太懂python的面向对象啊，input叫in_features，那我存这个的local variable应该叫什么呢？
in_features和out_features是干鸡毛的？
怎么subclass nn.Module？怎么 call the superclass constructor？怎么‘construct and store your parameter as 𝑊 (not 𝑊 ⊤), putting it in an nn.Parameter’？
话说𝑊在哪？
哪里需要initialization了，需要torch.nn.init.trunc_normal_？
我不用手写矩阵乘法对吧，是在太无聊，直接用einsum，as suggested？
我记得spec里提到了leading dimensions，我不需要手动支持这个吧？
"""

import torch
import torch.nn as nn
import math
from einops import einsum

class Linear(nn.Module):                          # ← 这就是subclass nn.Module
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()                         # ← 这就是call superclass constructor
        
        # in_features / out_features 就是矩阵的维度：
        # nn.Parameter告诉PyTorch"这是可训练参数"
        # W ∈ ℝ^(d_out × d_in)，就是 (out_features, in_features)
        self.weight = nn.Parameter(                # ← 这就是construct and store as nn.Parameter
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        
        # σ = sqrt(2 / (d_in + d_out))，截断在 [-3σ, 3σ]
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
