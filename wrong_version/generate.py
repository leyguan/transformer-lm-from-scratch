"""
用训好的 checkpoint 生成文本（注意：这是"续写"，不是对话）。

单条 prompt：
    python generate.py --dataset tinystories --ckpt checkpoints/tinystories/lr6e-4_best.pt \
        --prompt "Once upon a time"

交互式（不带 --prompt，逐行输入开头，空行或 Ctrl-C 退出）：
    python generate.py --dataset tinystories --ckpt checkpoints/tinystories/lr6e-4_best.pt

采样参数：--max-new-tokens 256  --temperature 0.8  --top-p 0.9  --seed 0
不给 --prompt 且交互时直接回车 = 让它从头无条件生成一段。

前置：先跑过 train.py，checkpoint 存在；bpe/<name>/ 里有 vocab 和 merges。
"""
from __future__ import annotations
import os
import sys
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from cs336_basics.transformer_lm import transformer_lm
from cs336_basics.adamw import AdamW
from cs336_basics.checkpointing import load_checkpoint
from cs336_basics.tokenizer import Tokenizer

# ---- 必须和 train.py 完全一致，否则权重对不上 ----
CONTEXT_LENGTH = 256
D_MODEL = 512
D_FF = 1344
NUM_LAYERS = 4
NUM_HEADS = 16
ROPE_THETA = 10000.0
SPECIAL_TOKENS = ["<|endoftext|>"]

BPE_DIR = {"tinystories": "bpe/TinyStories", "owt": "bpe/owt"}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TransformerLM(nn.Module):
    """结构与 train.py 里的一字不差，这样 load_checkpoint 的 key/shape 才能对上。"""

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.params = nn.ParameterDict()
        self._dotted = {}

        def add(dotted, shape, kind):
            safe = dotted.replace(".", "_")
            if kind == "matrix":
                t = torch.empty(*shape)
                nn.init.trunc_normal_(t, mean=0.0, std=0.02, a=-0.06, b=0.06)
            else:
                t = torch.ones(*shape)
            self.params[safe] = nn.Parameter(t)
            self._dotted[safe] = dotted

        add("token_embeddings.weight", (vocab_size, D_MODEL), "matrix")
        for i in range(NUM_LAYERS):
            p = f"layers.{i}."
            add(p + "attn.q_proj.weight", (D_MODEL, D_MODEL), "matrix")
            add(p + "attn.k_proj.weight", (D_MODEL, D_MODEL), "matrix")
            add(p + "attn.v_proj.weight", (D_MODEL, D_MODEL), "matrix")
            add(p + "attn.output_proj.weight", (D_MODEL, D_MODEL), "matrix")
            add(p + "ln1.weight", (D_MODEL,), "norm")
            add(p + "ffn.w1.weight", (D_FF, D_MODEL), "matrix")
            add(p + "ffn.w2.weight", (D_MODEL, D_FF), "matrix")
            add(p + "ffn.w3.weight", (D_FF, D_MODEL), "matrix")
            add(p + "ln2.weight", (D_MODEL,), "norm")
        add("ln_final.weight", (D_MODEL,), "norm")
        add("lm_head.weight", (vocab_size, D_MODEL), "matrix")

    def forward(self, in_indices):
        weights = {self._dotted[k]: v for k, v in self.params.items()}
        return transformer_lm(
            self.vocab_size, CONTEXT_LENGTH, D_MODEL, NUM_LAYERS,
            NUM_HEADS, D_FF, ROPE_THETA, weights, in_indices,
        )


def load_tokenizer(dataset):
    bpe_dir = BPE_DIR[dataset]
    with open(os.path.join(bpe_dir, "bpe_vocab.json")) as f:
        vocab = {int(k): bytes.fromhex(v) for k, v in json.load(f).items()}
    merges = []
    with open(os.path.join(bpe_dir, "bpe_merges.txt")) as f:
        for line in f:
            a, b = line.split()
            merges.append((bytes.fromhex(a), bytes.fromhex(b)))
    return Tokenizer(vocab=vocab, merges=merges, special_tokens=SPECIAL_TOKENS), len(vocab)


@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens, temperature, top_p, device, eot_id):
    seeded_empty = not prompt
    ids = [eot_id] if seeded_empty else tok.encode(prompt)   # 空 prompt 用 <|endoftext|> 起头，等于从新文档开头采
    x = torch.tensor([ids], dtype=torch.long, device=device)

    generated = []
    for _ in range(max_new_tokens):
        logits = model(x[:, -CONTEXT_LENGTH:])        # (1, T, V)，超过窗口就只喂最后 256 个
        logits = logits[:, -1, :].float().squeeze(0)  # (V,)

        if temperature <= 0:
            next_id = int(torch.argmax(logits))
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            if top_p < 1.0:
                sp, si = torch.sort(probs, descending=True)
                cum = torch.cumsum(sp, dim=-1)
                sp[(cum - sp) > top_p] = 0.0          # 丢掉前缀累计已过 top_p 的（保留跨过阈值的那个）
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
    ap.add_argument("--dataset", default="tinystories", choices=list(BPE_DIR))
    ap.add_argument("--ckpt", required=True, help="train.py 存的 .pt，比如 checkpoints/tinystories/lr6e-4_best.pt")
    ap.add_argument("--prompt", default=None, help="不给就进交互模式")
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
    eot_id = None
    eot_bytes = SPECIAL_TOKENS[0].encode("utf-8")
    for k, v in tok.vocab.items() if hasattr(tok, "vocab") else []:
        if v == eot_bytes:
            eot_id = k
            break
    if eot_id is None:  # 退路：直接编码特殊 token 拿它的 id
        enc = tok.encode(SPECIAL_TOKENS[0])
        eot_id = enc[0] if enc else vocab_size - 1

    model = TransformerLM(vocab_size).to(device)
    opt = AdamW(model.parameters(), lr=1e-3)          # load_checkpoint 需要一个 opt，推理用不上
    step = load_checkpoint(args.ckpt, model, opt)
    model.eval()
    print(f"device={device}  dataset={args.dataset}  vocab={vocab_size}  ckpt={args.ckpt}  (trained to step {step})")
    print(f"采样: temperature={args.temperature}  top_p={args.top_p}  max_new_tokens={args.max_new_tokens}")
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
