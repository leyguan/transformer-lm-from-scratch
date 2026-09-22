"""
用训好的 checkpoint 生成文本（注意：这是“续写”，不是对话）。

============================================================
相比旧版，只改了一件事（但这件事同样是致命的）：
------------------------------------------------------------
旧版：generate.py 自己又拼了一遍模型，用的还是被淘汰的老写法——
      nn.ParameterDict 塞参数 + 打包成 weights 字典丢给函数式
      transformer_lm()，而且注意力的键名是
      attn.q_proj / k_proj / v_proj / output_proj。

      但你新版 train.py 存档时，模型持有的是真·子模块，键名是
      attn.W_Q / W_K / W_V / W_O（层级也不同：没有 params. 前缀，
      不是下划线安全名）。两边的 state_dict 键根本对不上。
      于是 load_checkpoint 里的 load_state_dict 要么当场报
      Missing/Unexpected keys，要么在 strict=False 下静默地
      一个权重都没加载——模型停留在随机初始化，生成一堆乱码。

新版：直接从 train.py import 同一个 TransformerLM。
      训练端怎么定义模型、怎么存档，这里就怎么建、怎么读，
      键名和形状永远一字不差，train.py 以后再改结构也不会错位。
      好处还有一个：架构常量（D_MODEL / D_FF / NUM_LAYERS ...）
      都由模型自己持有，generate.py 不必再抄一份、也就不会抄错。

采样部分（温度 / top-p 核采样）旧版本身是对的，原样保留。
============================================================

单条 prompt：
    python generate.py --dataset tinystories --ckpt checkpoints/tinystories/lr6e-4_best.pt \
        --prompt "Once upon a time"

交互式（不带 --prompt，逐行输入开头；空行或 Ctrl-C 退出）：
    python generate.py --dataset tinystories --ckpt checkpoints/tinystories/lr6e-4_best.pt

无条件从头生成一段：命令行传空 prompt 即可，即 --prompt ""
（交互模式下空行是“退出”，不是无条件生成，别搞混。）

采样参数：--max-new-tokens 256  --temperature 0.8  --top-p 0.9  --seed 0

前置：先跑过 train.py，checkpoint 存在；bpe/<name>/ 里有 vocab 和 merges。
运行目录和 train.py 一样，都在项目根目录（下面全是相对路径）。
"""
from __future__ import annotations
import os
import json
import argparse

import torch
import torch.nn.functional as F

# ★ 关键：复用 train.py 里那份已经修好的模型定义与结构常量，
#   保证存档键名/形状与训练端完全一致，绝不再自己拼一遍模型。
#   （train.py 有 if __name__ == "__main__" 保护，import 它不会触发训练。）
from train import (
    TransformerLM,      # 真·子模块版模型，与存档键名一一对应
    PATHS,              # 复用同一份路径配置，bpe 目录不必再写一遍
    CONTEXT_LENGTH,     # 生成时滑动窗口要用
    get_device,
)
from cs336_basics.adamw import AdamW
from cs336_basics.checkpointing import load_checkpoint
from cs336_basics.tokenizer import Tokenizer

SPECIAL_TOKENS = ["<|endoftext|>"]


def load_tokenizer(dataset):
    """从 bpe/<name>/ 读回词表和 merges，重建 Tokenizer。"""
    bpe_dir = PATHS[dataset]["bpe_dir"]
    with open(os.path.join(bpe_dir, "bpe_vocab.json")) as f:
        vocab = {int(k): bytes.fromhex(v) for k, v in json.load(f).items()}
    merges = []
    with open(os.path.join(bpe_dir, "bpe_merges.txt")) as f:
        for line in f:
            a, b = line.split()
            merges.append((bytes.fromhex(a), bytes.fromhex(b)))
    return Tokenizer(vocab=vocab, merges=merges, special_tokens=SPECIAL_TOKENS), len(vocab)


def find_eot_id(tok, vocab_size):
    """
    找 <|endoftext|> 这个特殊 token 的 id。
    先在词表里按字节匹配；匹配不到就退回“直接编码它、取第一个 id”；
    再不行就用词表最后一个 id 兜底。
    """
    eot_bytes = SPECIAL_TOKENS[0].encode("utf-8")
    vocab = getattr(tok, "vocab", None)
    if vocab:
        for k, v in vocab.items():
            if v == eot_bytes:
                return k
    enc = tok.encode(SPECIAL_TOKENS[0])
    return enc[0] if enc else vocab_size - 1


@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens, temperature, top_p, device, eot_id):
    seeded_empty = not prompt
    # 空 prompt 用 <|endoftext|> 起头，等于从一篇新文档的开头开始采
    ids = [eot_id] if seeded_empty else tok.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    generated = []
    for _ in range(max_new_tokens):
        logits = model(x[:, -CONTEXT_LENGTH:])         # (1, T, V)，超窗口就只喂最后 256 个
        logits = logits[:, -1, :].float().squeeze(0)   # 只取最后一个位置的分布 -> (V,)

        if temperature <= 0:
            next_id = int(torch.argmax(logits))        # 贪心
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            if top_p < 1.0:
                # 核采样：按概率降序排，丢掉“前缀累计已超过 top_p”的尾部
                sp, si = torch.sort(probs, descending=True)
                cum = torch.cumsum(sp, dim=-1)
                sp[(cum - sp) > top_p] = 0.0           # 保留跨过阈值的那一个，只砍它后面的
                sp /= sp.sum()
                next_id = int(si[torch.multinomial(sp, 1)])
            else:
                next_id = int(torch.multinomial(probs, 1))

        if next_id == eot_id:
            break
        generated.append(next_id)
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)

    full = generated if seeded_empty else ids + generated
    return tok.decode(full)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tinystories", choices=list(PATHS))
    ap.add_argument("--ckpt", required=True,
                    help="train.py 存的 .pt，比如 checkpoints/tinystories/lr6e-4_best.pt")
    ap.add_argument("--prompt", default=None, help="不给就进交互模式；给空串 '' 则无条件生成")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device()

    tok, vocab_size = load_tokenizer(args.dataset)
    eot_id = find_eot_id(tok, vocab_size)

    # 建模型：只需 vocab_size，其余架构参数模型自己从 train.py 的常量里拿
    model = TransformerLM(vocab_size).to(device)

    # load_checkpoint 的签名要求带一个优化器，推理用不上，给个占位的即可。
    # （因为这里的模型与训练端结构完全一致，优化器 state 也能对上，只是我们不用它。）
    dummy_opt = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95),
                      eps=1e-8, weight_decay=0.1)
    step = load_checkpoint(args.ckpt, model, dummy_opt)
    model.eval()

    print(f"device={device}  dataset={args.dataset}  vocab={vocab_size}  "
          f"ckpt={args.ckpt}  (trained to step {step})")
    print(f"采样: temperature={args.temperature}  top_p={args.top_p}  "
          f"max_new_tokens={args.max_new_tokens}")
    print("注意：这是续写式基础模型，不是对话。给它一个开头，它往下编。\n")

    if args.prompt is not None:
        print(generate(model, tok, args.prompt, args.max_new_tokens,
                       args.temperature, args.top_p, device, eot_id))
        return

    print("交互模式：输入一个开头回车；空行或 Ctrl-C 退出。")
    while True:
        try:
            prompt = input("\n>>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.strip() == "":
            break
        print(generate(model, tok, prompt, args.max_new_tokens,
                       args.temperature, args.top_p, device, eot_id))


if __name__ == "__main__":
    main()
