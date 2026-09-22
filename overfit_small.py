"""
过拟合诊断：拿 4 条固定序列往死里训，看模型能不能把它们背下来。

和旧版最大的区别：
    模型不再在本文件里重新定义一遍，而是直接从 train.py import。
    旧版抄了一份模型定义，测的是「抄件」，跟真正在训的那个模型可能早就不是一回事了。
    现在这个脚本测的就是本尊——train.py 改了，这里立刻跟着改。

跑法（放项目根目录，和 train.py 同级）：
    python overfit_small.py
    python overfit_small.py --steps 3000 --lr 1e-3

通过标准：
    1. 梯度体检：每个参数都有非零梯度        ← 一票否决项
    2. 初始 loss ≈ ln(vocab_size)
    3. 最终 loss < 0.05，token 准确率 = 100%  ← 真正「背下来了」的样子
"""
from __future__ import annotations

import json
import argparse

import numpy as np
import torch

# ★ 关键：模型和常量全部来自 train.py，保证测的是真家伙
from train import TransformerLM, CONTEXT_LENGTH, get_device
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.adamw import AdamW

DATA_PATH = "data/TinyStories_train.npy"
VOCAB_PATH = "bpe/TinyStories/bpe_vocab.json"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4, help="固定住的序列条数")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--target", type=float, default=0.01, help="到这个 loss 就提前收工")
    return ap.parse_args()


def make_fixed_batch(batch_size, device):
    """取数据开头的 batch_size 条不重叠序列，全程固定不变。"""
    data = np.load(DATA_PATH, mmap_mode="r")
    xs, ys = [], []
    for i in range(batch_size):
        s = i * CONTEXT_LENGTH
        xs.append(data[s:s + CONTEXT_LENGTH])
        ys.append(data[s + 1:s + CONTEXT_LENGTH + 1])
    x = torch.tensor(np.stack(xs), dtype=torch.long, device=device)
    y = torch.tensor(np.stack(ys), dtype=torch.long, device=device)
    return x, y


def gradient_report(model):
    """
    逐参数梯度报告。这就是当初照出旧版原形的那面镜子，格式沿用你熟悉的那套。
    返回 (没梯度的, 梯度恒零的) 两个列表。
    """
    print("\n===== gradient check =====")
    dead, zero = [], []
    for name, p in model.named_parameters():
        if p.grad is None:
            print(f"NO GRAD   : {name}")
            dead.append(name)
        else:
            n = p.grad.norm().item()
            tag = "ZERO GRAD " if n == 0.0 else "GRAD      "
            print(f"{tag}: {name:48s} norm={n:.6e}")
            if n == 0.0:
                zero.append(name)
    total = sum(1 for _ in model.parameters())
    alive = total - len(dead) - len(zero)
    print(f"---- {alive}/{total} 组参数有梯度 "
          f"{'✓' if alive == total else '✗ 有参数训不到，下面的 loss 不用看了'}")
    return dead, zero


def lr_at(step, total, base_lr):
    """
    过拟合专用的土办法 lr：先大步冲，后面两次降到 1/10。
    不降的话 AdamW 会在 0.3~0.5 附近反复横跳下不去——旧版停在 0.45 有一半是这个原因。
    """
    if step < 0.60 * total:
        return base_lr
    if step < 0.85 * total:
        return base_lr * 0.1
    return base_lr * 0.01


def main():
    args = parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    device = get_device()
    print("device:", device)

    with open(VOCAB_PATH) as f:
        vocab_size = len(json.load(f))
    print("vocab_size:", vocab_size)

    x, y = make_fixed_batch(args.batch, device)
    print("x:", tuple(x.shape), " y:", tuple(y.shape),
          f" 一共要背 {x.numel()} 个预测")

    # 注意：这里**不开** bf16 混合精度。过拟合测试要的是能压到多低，
    # bf16 的精度地板本身就在 1e-2 量级，会污染结论。
    model = TransformerLM(vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量 {n_params/1e6:.1f}M")

    opt = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                eps=1e-8, weight_decay=0.0)   # 过拟合不要 weight decay
    model.train()

    # ---------- 第一次前向 + 梯度体检 ----------
    logits = model(x)
    loss = cross_entropy(logits.reshape(-1, vocab_size).float(), y.reshape(-1))
    print(f"\ninitial loss: {loss.item():.6f}   "
          f"（理论值 ln(V)={np.log(vocab_size):.4f}，偏差大说明初始化或前向有问题）")

    opt.zero_grad(set_to_none=True)
    loss.backward()
    dead, zero = gradient_report(model)
    if dead or zero:
        raise SystemExit("\n梯度体检没过，先修模型，别浪费时间跑 overfit。")

    # ---------- 开始背书 ----------
    print("\n===== overfit =====")
    best = float("inf")
    hit = {}                       # 记录首次跌破各档位的步数
    marks = [1.0, 0.1, 0.05, 0.01]

    for step in range(args.steps):
        lr = lr_at(step, args.steps, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr

        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, vocab_size).float(), y.reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        v = loss.item()
        best = min(best, v)
        for m in marks:
            if m not in hit and v < m:
                hit[m] = step

        if step % args.log_every == 0:
            with torch.no_grad():
                acc = (logits.argmax(-1) == y).float().mean().item()
            print(f"step {step:5d}  loss {v:.6f}  acc {acc*100:5.1f}%  lr {lr:.1e}",
                  flush=True)

        if v < args.target:
            print(f"step {step:5d}  loss {v:.6f}  ← 到达目标 {args.target}，提前收工")
            break

    # ---------- 结论 ----------
    with torch.no_grad():
        logits = model(x)
        final = cross_entropy(logits.reshape(-1, vocab_size).float(),
                              y.reshape(-1)).item()
        acc = (logits.argmax(-1) == y).float().mean().item()
        wrong = int((logits.argmax(-1) != y).sum().item())

    print("\n===== 结论 =====")
    print(f"final loss : {final:.6f}")
    print(f"best  loss : {best:.6f}")
    print(f"token 准确率: {acc*100:.2f}%   （错 {wrong} / {y.numel()} 个）")
    for m in marks:
        s = hit.get(m)
        print(f"  首次跌破 {m:<5}: {'step ' + str(s) if s is not None else '未达到'}")

    if final < 0.05 and acc > 0.999:
        print("\n✓ 通过。模型有能力把这批数据完整背下来，"
              "说明全部参数都在参与学习，可以放心去跑长训了。")
    elif final < 0.5:
        print("\n△ 半通过。能学但压不下去，多半是 lr 或步数不够，"
              f"试试 --steps {args.steps*2}；如果加了还是卡在同一个数上，回来找我。")
    else:
        print("\n✗ 没通过。梯度体检明明过了却背不下来，"
              "问题可能在 attention 的掩码、RoPE 的位置，或者 loss 的对齐方式上。")


if __name__ == "__main__":
    main()
