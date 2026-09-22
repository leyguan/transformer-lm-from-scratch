import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 随机采 batch_size 个起始索引，范围 [0, len-context_length-1)
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)

    x = np.stack([dataset[i : i + context_length] for i in starts])
    y = np.stack([dataset[i + 1 : i + 1 + context_length] for i in starts])

    return (
        torch.tensor(x, dtype=torch.long, device=device),
        torch.tensor(y, dtype=torch.long, device=device),
    )
