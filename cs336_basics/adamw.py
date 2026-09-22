import math
import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # 初始化状态
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)   # 一阶矩 (momentum)
                    state["v"] = torch.zeros_like(p)   # 二阶矩 (variance)

                m = state["m"]
                v = state["v"]

                # t 从 1 开始
                state["t"] += 1
                t = state["t"]

                # 第 7 行
                alpha_t = lr * math.sqrt(1.0 - beta2 ** t) / (1.0 - beta1 ** t)

                # 第 8 行
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                # 第 9 行
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # 第 10 行
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 第 11 行
                denom = v.sqrt().add_(eps)
                p.addcdiv_(m, denom, value=-alpha_t)

        return loss
