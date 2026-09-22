"""
训练 Transformer 语言模型（修正版）。放项目根目录，直接跑。

============================================================
相比旧版，只改了一件事（但这件事是致命的）：
------------------------------------------------------------
旧版：所有参数塞在 nn.ParameterDict 里，前向时打包成 weights 字典
      丢给函数式的 transformer_lm()。而 transformer_lm 内部对
      embedding / attention / rmsnorm / lm_head 走的是
      「临时新建 nn.Module + load_state_dict(权重)」——
      load_state_dict 是**拷贝值**，拷完之后参与计算的是临时模块
      自己的参数，你 ParameterDict 里的原始参数根本没进计算图。
      于是反向传播时它们 .grad 全是 None，优化器每步都跳过它们。
      只有 ffn 的 w1/w2/w3 因为是被当实参直接传进函数式 swiglu，
      才真正在图里、才有梯度。

新版：模型持有真正的子模块（Embedding / MultiHeadSelfAttention /
      RMSNorm / Linear），在 __init__ 里建一次、forward 里直接调用。
      参数天然属于模型，天然在计算图里，梯度天然回得来。

cs336_basics/ 下一个文件都没动，adapters.py 里小零件的签名全部原样。
（那些 load_state_dict 的写法给测试用完全没问题——测试只对前向数值，
  不需要梯度。它只是不能拿来当训练代码用。）
============================================================

用法（和旧版完全一致）：
    python train.py
    python train.py --dataset tinystories --lr 6e-4 --name lr6e-4
    nohup python train.py --dataset tinystories --lr 6e-4 --name lr6e-4 > train_lr6e-4.log 2>&1 &

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

# ↓ 全是你自己写过、测过的小零件。注意这里 import 的是「类」和「纯函数」，
#   不再 import transformer_lm —— 因为模型结构现在直接在本文件里拼。
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import swiglu
from cs336_basics.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.adamw import AdamW
from cs336_basics.get_lr_cosine_schedule import get_lr_cosine_schedule
from cs336_basics.gradient_clipping import clip_gradients
from cs336_basics.get_batch import get_batch
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint

# ============ 两个数据源的地址 ============
PATHS = {
    "tinystories": { #$ 统一大小写，都用大写 
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
RMS_EPS = 1e-5

# ============ 训练超参默认值（命令行可覆盖） ============
DEFAULTS = dict(
    dataset="tinystories",
    batch=128,
    total_tokens=327_680_000,
    lr=6e-4,
    min_lr=4e-5,
    warmup_frac=0.05,
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


# ==================================================================
#                            模型
# ==================================================================
# 核心原则（这次踩坑的教训）：
#   参数必须由模块**持有**，不能在 forward 里现建模块再把值拷进去。
#   持有 = 在 __init__ 里 self.xxx = SomeModule(...)，然后 forward 里 self.xxx(...)。
# ==================================================================

class FFN(nn.Module):
    """
    SwiGLU 前馈层。
    这里之所以用三个 Linear 当"参数容器"，是为了让 state_dict 的 key
    刚好是 ffn.w1.weight / ffn.w2.weight / ffn.w3.weight，跟作业文档一致。
    前向时把 .weight 当实参传给你写的纯函数 swiglu —— 纯函数拿到的就是
    参数张量本人，所以梯度直接回到 self.w1.weight 上。
    （旧版能训到的恰恰只有这三个，就是因为它们走的是这条路。）
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff)    # weight: (d_ff, d_model)
        self.w2 = Linear(d_ff, d_model)    # weight: (d_model, d_ff)
        self.w3 = Linear(d_model, d_ff)    # weight: (d_ff, d_model)

    def forward(self, x):
        return swiglu(
            self.d_model, self.d_ff,
            self.w1.weight, self.w2.weight, self.w3.weight,
            x,
        )


class Block(nn.Module):
    """
    pre-norm transformer block：
        x = x + attn(ln1(x))
        x = x + ffn(ln2(x))
    注意 attn 是在 __init__ 里建的，整个训练过程复用同一个实例，
    它内部的 W_Q/W_K/W_V/W_O 就是本模型的真参数。
    """

    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta):
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model, eps=RMS_EPS)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model, num_heads=num_heads,
            max_seq_len=max_seq_len, theta=theta,
        )
        self.ln2 = RMSNorm(d_model=d_model, eps=RMS_EPS)
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x):
        # 不传 token_positions，让 MHA 内部默认生成 arange（跟测试里的调用方式一致）。
        # 如果你的 MHA 要求必须显式传，把下面两行换成：
        #   pos = torch.arange(x.shape[-2], device=x.device)
        #   x = x + self.attn(self.ln1(x), token_positions=pos)
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """
    完整语言模型。state_dict 的 key 长这样：
        token_embeddings.weight
        layers.0.ln1.weight
        layers.0.attn.W_Q.weight ...     ← 唯一跟作业文档不同的地方
        layers.0.ffn.w1.weight ...
        layers.0.ln2.weight
        ln_final.weight
        lm_head.weight
    （文档里注意力叫 attn.q_proj.weight，你的类里叫 W_Q。自洽即可，
      存档和续训都在本文件内闭环，不影响任何测试。）
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embeddings = Embedding(vocab_size, D_MODEL)
        self.layers = nn.ModuleList([
            Block(D_MODEL, NUM_HEADS, D_FF, CONTEXT_LENGTH, ROPE_THETA)
            for _ in range(NUM_LAYERS)
        ])
        self.ln_final = RMSNorm(d_model=D_MODEL, eps=RMS_EPS)
        self.lm_head = Linear(D_MODEL, vocab_size)   # weight: (vocab_size, d_model)
        self._init_weights()

    def _init_weights(self):
        """
        沿用旧脚本的初始化口径：矩阵用 std=0.02 的截断正态，norm 的 gain 全 1。
        （每个小零件自己的 __init__ 也有一套作业规定的初始化，
          想用那套的话把这个函数调用删掉即可，两套都能训。）
        """
        for _, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.trunc_normal_(p, mean=0.0, std=0.02, a=-0.06, b=0.06)
            else:
                nn.init.ones_(p)

    def forward(self, in_indices):
        x = self.token_embeddings(in_indices)      # (B, T) -> (B, T, d_model)
        for blk in self.layers:
            x = blk(x)
        x = self.ln_final(x)
        return self.lm_head(x)                     # (B, T, vocab_size)


# ==================================================================
#                    开训前的体检（照妖镜）
# ==================================================================

def grad_check(model, vocab_size, device):
    """
    跑一次假的前向+反向，确认**每一个**参数都拿到了非零梯度。
    这就是旧版那个 bug 的检测器：旧版跑这个会当场列出一屏 NO GRAD。
    只花不到一秒，以后每次改模型结构都该跑一次。
    """
    model.train()
    x = torch.randint(0, vocab_size, (2, CONTEXT_LENGTH), device=device)
    y = torch.randint(0, vocab_size, (2, CONTEXT_LENGTH), device=device)

    logits = model(x)
    loss = cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        y.reshape(-1),
    )
    model.zero_grad(set_to_none=True)
    loss.backward()

    no_grad, zero_grad = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            no_grad.append(name)          # 压根没进计算图 —— 就是旧版的病
        elif p.grad.abs().sum().item() == 0.0:
            zero_grad.append(name)        # 进了图但梯度恒零，也不正常
    model.zero_grad(set_to_none=True)

    if no_grad or zero_grad:
        msg = ["梯度体检没过，这样训是白训："]
        for n in no_grad:
            msg.append(f"  NO GRAD   {n}")
        for n in zero_grad:
            msg.append(f"  ZERO GRAD {n}")
        raise RuntimeError("\n".join(msg))

    n_tensors = sum(1 for _ in model.parameters())
    # 初始 loss 应该 ≈ ln(vocab_size)，明显偏离说明前向或初始化有问题
    print(f"[体检] {n_tensors} 组参数全部拿到梯度 ✓   "
          f"初始 loss {loss.item():.3f}（理论值 ln(V)={np.log(vocab_size):.3f}）",
          flush=True)


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
    # ===== 第一段：准备 =====
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    paths = PATHS[args.dataset]
    os.makedirs(paths["ckpt_dir"], exist_ok=True)

    with open(os.path.join(paths["bpe_dir"], "bpe_vocab.json")) as f:
        vocab_size = len(json.load(f))

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

    # ★ 开训前体检：不过就直接抛异常，绝不让你再白跑三小时
    grad_check(model, vocab_size, device)

    raw_model = model            # compile 之后 state_dict 的 key 会多一层前缀，
    if args.compile:             # 存档统一用没包装过的 raw_model
        model = torch.compile(model)

    opt = AdamW(model.parameters(), lr=args.lr, betas=BETAS,
                eps=EPS, weight_decay=args.weight_decay)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, raw_model, opt)
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

    # ===== 第二段：训练主循环 =====
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
                save_checkpoint(raw_model, opt, step,
                                os.path.join(paths["ckpt_dir"], f"{args.name}_best.pt"))

        if step > 0 and step % args.ckpt_every == 0:
            save_checkpoint(raw_model, opt, step,
                            os.path.join(paths["ckpt_dir"], f"{args.name}_last.pt"))

    # ===== 第三段：收尾 =====
    vloss = evaluate(model, valid_data, args, device)
    best_val = min(best_val, vloss)
    save_checkpoint(raw_model, opt, num_steps,
                    os.path.join(paths["ckpt_dir"], f"{args.name}_last.pt"))
    print(f"[done] final val_loss {vloss:.4f}  best val_loss {best_val:.4f}  "
          f"用时 {time.time()-t0:.0f}s")
    if use_wandb:
        wandb.log({"final_val_loss": vloss, "best_val_loss": best_val})
        wandb.finish()


if __name__ == "__main__":
    main()
