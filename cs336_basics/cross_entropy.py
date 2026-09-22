import torch
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy(
    logits: Float[Tensor, "... vocab_size"],
    targets: Int[Tensor, "..."],
) -> Float[Tensor, ""]:
    logits_max = logits.max(dim=-1, keepdim=True).values
    shifted_logits = logits - logits_max
    # "Subtract the largest element for numerical stability."
    # from now on, we work with 'shifted_logits'


    # 【第一版】
    # exp_shifted_logits = torch.exp(shifted_logits)
    # sum_exp_shifted_logits = einsum(exp_shifted_logits, "... vocab -> ...")
    # # 'sum' 之后最后一维也就是 vocab_size 会坍塌
    # log_sum_exp_shifted_logits = torch.log(sum_exp_shifted_logits)

    # 【第二版】
    log_sum_exp_shifted_logits = torch.logsumexp(shifted_logits, dim=-1, keepdim=True)


    # 【第一版 内存爆炸】
    # vocab_size = logits.shape[-1]
    # one_hot_masking = torch.nn.functional.one_hot(targets, num_classes=vocab_size).float()
    # # 把 "..." 展开成 "... vocab_size"
    # target_logits = einsum(shifted_logits, one_hot_masking, "... vocab, ... vocab -> ...")
    # # 掩码，本质上是做内积

    # 【第二版】
    target_logits = shifted_logits.gather(
        dim=-1,
        index=targets.unsqueeze(-1)
    )

    loss = -target_logits + log_sum_exp_shifted_logits

    return loss.mean()
    # scalar vector，"" 就是一个数，可以参与后面的运算
