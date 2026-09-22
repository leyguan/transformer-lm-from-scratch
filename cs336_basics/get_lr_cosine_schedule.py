import math

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    计算基于余弦退火和线性预热的学习率。
    """
    # 1. Warm-up 阶段: t < T_w
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
        
    # 2. Post-annealing 阶段: t > T_c
    # 注意：如果 t == cosine_cycle_iters，会走到下面的余弦阶段并返回 min_learning_rate，
    # 但为了防止余弦阶段中分母 (T_c - T_w) 为 0 的边界情况，将大于号放在这里更安全。
    if it >= cosine_cycle_iters:
        return min_learning_rate
        
    # 3. Cosine annealing 阶段: T_w <= t <= T_c
    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)
