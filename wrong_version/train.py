"""
训练 Transformer 语言模型。放项目根目录，直接跑。

最简单：
    python train.py                       # 默认 tinystories，用下面 DEFAULTS 里的超参

指定数据集 / 学习率 / 起个名字（sweep 就靠这个）：
    python train.py --dataset tinystories --lr 4e-4 --name lr4e-4

后台跑 + 看日志：
    nohup python train.py --dataset tinystories --lr 4e-4 --name lr4e-4 > train_lr4e-4.log 2>&1 &
    tail -f train_lr4e-4.log

一次扫多个学习率（sweep）：
    for lr in 1e-3 6e-4 3e-4 1e-4; do
        python train.py --dataset tinystories --lr $lr --name lr$lr
    done

前置：先跑过 run_train_bpe.py 和 encode_dataset.py，
data/<name>_train.npy 和 data/<name>_valid.npy 要存在。
"""
from __future__ import annotations
import os
import json
import time
import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn

from cs336_basics.transformer_lm import transformer_lm
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.adamw import AdamW
from cs336_basics.get_lr_cosine_schedule import get_lr_cosine_schedule
from cs336_basics.gradient_clipping import clip_gradients
from cs336_basics.get_batch import get_batch
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint

# ============ 两个数据源的地址，全写在这 ============
PATHS = {
    "tinystories": {
        "bpe_dir": "bpe/TinyStories",
        "train_tokens": "data/TinyStories_train.npy",
        "valid_tokens": "data/TinyStories_valid.npy",
        "ckpt_dir": "checkpoints/tinystories",
    },
    "owt": {
        "bpe_dir": "bpe/owt",
        "train_tokens": "data/owt_train.npy",
        "valid_tokens": "data/owt_valid.npy",
        "ckpt_dir": "checkpoints/owt",
    },
}

# ============ 模型结构（作业给定，别动） ============
CONTEXT_LENGTH = 256
D_MODEL = 512
D_FF = 1344
NUM_LAYERS = 4
NUM_HEADS = 16
ROPE_THETA = 10000.0

# ============ 训练超参默认值（命令行可覆盖） ============
DEFAULTS = dict(
    dataset="tinystories",
    batch=128,
    total_tokens=327_680_000,   # batch * steps * context ≈ 这个数
    lr=4e-4,                    # 要 sweep 的主要就是它
    min_lr=4e-5,
    warmup_frac=0.05,           # 前 5% 步 warmup
    weight_decay=0.1,
    grad_clip=1.0,
    eval_every=250,
    eval_batches=20,
    log_every=20,
    ckpt_every=1000,
    seed=0,
    name="run",
    wandb=False,
    compile=False,
    no_amp=False,
    resume="",
)
BETAS = (0.9, 0.95)
EPS = 1e-8


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TransformerLM(nn.Module):
    """
    包一层壳：把可训练参数按 adapters.py 里 run_transformer_lm 文档规定的 key/shape
    建好，前向时组成 weights 字典喂给你已有的 transformer_lm 函数。
    这样保证跟你测过的实现完全对得上，不用重写模型。
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.params = nn.ParameterDict()   # ParameterDict 的 key 不能带点，所以做个映射
        self._dotted = {}                  # safe_key -> 真正的带点 key

        def add(dotted, shape, kind):
            safe = dotted.replace(".", "_")
            if kind == "matrix":
                t = torch.empty(*shape)
                nn.init.trunc_normal_(t, mean=0.0, std=0.02, a=-0.06, b=0.06)
            else:  # norm 层，初始化为 1
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


@torch.no_grad()
def evaluate(model, data, args, device):
    model.eval()
    losses = []
    for _ in range(args.eval_batches):
        x, y = get_batch(data, args.batch, CONTEXT_LENGTH, device)
        logits = model(x)
        loss = cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            y.reshape(-1),
        )
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULTS["dataset"], choices=list(PATHS))
    ap.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    ap.add_argument("--total-tokens", type=int, default=DEFAULTS["total_tokens"])
    ap.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    ap.add_argument("--min-lr", type=float, default=DEFAULTS["min_lr"])
    ap.add_argument("--warmup-frac", type=float, default=DEFAULTS["warmup_frac"])
    ap.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    ap.add_argument("--grad-clip", type=float, default=DEFAULTS["grad_clip"])
    ap.add_argument("--eval-every", type=int, default=DEFAULTS["eval_every"])
    ap.add_argument("--eval-batches", type=int, default=DEFAULTS["eval_batches"])
    ap.add_argument("--log-every", type=int, default=DEFAULTS["log_every"])
    ap.add_argument("--ckpt-every", type=int, default=DEFAULTS["ckpt_every"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--name", default=DEFAULTS["name"])
    ap.add_argument("--wandb", action="store_true", default=DEFAULTS["wandb"])
    ap.add_argument("--compile", action="store_true", default=DEFAULTS["compile"])
    ap.add_argument("--no-amp", action="store_true", default=DEFAULTS["no_amp"])
    ap.add_argument("--resume", default=DEFAULTS["resume"], help="从某个 .pt 断点续训")
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    paths = PATHS[args.dataset]
    os.makedirs(paths["ckpt_dir"], exist_ok=True)

    # 模型 vocab_size 直接从分词器 vocab 数出来，绝不会跟数据对不上
    with open(os.path.join(paths["bpe_dir"], "bpe_vocab.json")) as f:
        vocab_size = len(json.load(f))

    # 内存映射加载，不会把几个 G 一次读进内存
    train_data = np.load(paths["train_tokens"], mmap_mode="r")
    valid_data = np.load(paths["valid_tokens"], mmap_mode="r")

    tokens_per_step = args.batch * CONTEXT_LENGTH
    num_steps = args.total_tokens // tokens_per_step
    warmup = int(num_steps * args.warmup_frac)

    print(f"device={device}  dataset={args.dataset}  vocab={vocab_size}")
    print(f"batch={args.batch}  ctx={CONTEXT_LENGTH}  steps={num_steps}  "
          f"warmup={warmup}  tokens/step={tokens_per_step}")

    model = TransformerLM(vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量 {n_params/1e6:.1f}M（含 embedding）")

    if args.compile:
        model = torch.compile(model)

    opt = AdamW(model.parameters(), lr=args.lr, betas=BETAS,
                eps=EPS, weight_decay=args.weight_decay)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, opt)
        print(f"从 {args.resume} 续训，起始 step={start_step}")

    use_wandb = args.wandb
    if use_wandb:
        import wandb
        wandb.init(project="cs336-a1", name=f"{args.dataset}-{args.name}",
                   config=dict(vars(args), context=CONTEXT_LENGTH, d_model=D_MODEL,
                               d_ff=D_FF, num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
                               num_steps=num_steps, vocab_size=vocab_size))

    use_amp = (not args.no_amp) and device == "cuda"
    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    if use_amp:
        print("已开启 bf16 混合精度（更快更省显存）")

    t0 = time.time()
    best_val = float("inf")
    model.train()
    for step in range(start_step, num_steps):
        lr = get_lr_cosine_schedule(step, args.lr, args.min_lr, warmup, num_steps)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_data, args.batch, CONTEXT_LENGTH, device)
        with amp_ctx:
            logits = model(x)
        loss = cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            y.reshape(-1),
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_gradients(model.parameters(), args.grad_clip, eps=1e-6)
        opt.step()

        if step % args.log_every == 0:
            dt = time.time() - t0
            print(f"step {step:6d}/{num_steps}  train_loss {loss.item():.4f}  "
                  f"lr {lr:.2e}  {dt:.0f}s", flush=True)
            if use_wandb:
                wandb.log({"train_loss": loss.item(), "lr": lr, "wall_s": dt}, step=step)

        if step > 0 and step % args.eval_every == 0:
            vloss = evaluate(model, valid_data, args, device)
            dt = time.time() - t0
            print(f"  [eval] step {step}  val_loss {vloss:.4f}  ({dt:.0f}s)", flush=True)
            if use_wandb:
                wandb.log({"val_loss": vloss}, step=step)
            if vloss < best_val:
                best_val = vloss
                save_checkpoint(model, opt, step, os.path.join(paths["ckpt_dir"], f"{args.name}_best.pt"))

        if step > 0 and step % args.ckpt_every == 0:
            save_checkpoint(model, opt, step, os.path.join(paths["ckpt_dir"], f"{args.name}_last.pt"))

    # 收尾：再评一次，存最后一版
    vloss = evaluate(model, valid_data, args, device)
    best_val = min(best_val, vloss)
    save_checkpoint(model, opt, num_steps, os.path.join(paths["ckpt_dir"], f"{args.name}_last.pt"))
    print(f"[done] final val_loss {vloss:.4f}  best val_loss {best_val:.4f}  "
          f"用时 {time.time()-t0:.0f}s")
    if use_wandb:
        wandb.log({"final_val_loss": vloss, "best_val_loss": best_val})
        wandb.finish()


if __name__ == "__main__":
    main()
