"""
把语料编码成 uint16 的 token id 数组，训练时 memmap 直接用。
必须在 run_train_bpe.py 之后跑（要先有 bpe/<name>/ 里的 vocab 和 merges）。

用法（项目根目录）：
    python encode_dataset.py tinystories
    python encode_dataset.py owt

    # 语料大时建议后台跑（这一步可能要几分钟到更久，正常）
    nohup python encode_dataset.py tinystories > data/ts_encode.log 2>&1 &
    tail -f data/ts_encode.log

输出：
    data/<name>_train.npy
    data/<name>_valid.npy
"""
import os
import sys
import json
import time
import numpy as np
from cs336_basics.tokenizer import Tokenizer

CONFIGS = {
    "tinystories": {
        "bpe_dir": "bpe/TinyStories",
        "out_prefix": "data/TinyStories",
        "splits": {
            "train": "data/TinyStoriesV2-GPT4-train.txt",
            "valid": "data/TinyStoriesV2-GPT4-valid.txt",
        },
    },
    "owt": {
        "bpe_dir": "bpe/owt",
        "out_prefix": "data/owt",
        "splits": {
            "train": "data/owt_train.txt",
            "valid": "data/owt_valid.txt",
        },
    },
}
SPECIAL_TOKENS = ["<|endoftext|>"]


def load_tokenizer(bpe_dir):
    """从 run_train_bpe.py 存下来的 hex 格式还原出 Tokenizer。"""
    with open(os.path.join(bpe_dir, "bpe_vocab.json")) as f:
        raw = json.load(f)
    vocab = {int(k): bytes.fromhex(v) for k, v in raw.items()}
    merges = []
    with open(os.path.join(bpe_dir, "bpe_merges.txt")) as f:
        for line in f:
            a, b = line.split()
            merges.append((bytes.fromhex(a), bytes.fromhex(b)))
    return Tokenizer(vocab=vocab, merges=merges, special_tokens=SPECIAL_TOKENS)


def token_stream(tok, f, every=5_000_000):
    """逐行编码，边吐 token 边报进度。encode_iterable 没有就退回 encode。"""
    if hasattr(tok, "encode_iterable"):
        gen = tok.encode_iterable(f)
    else:
        def gen_():
            for line in f:
                yield from tok.encode(line)
        gen = gen_()
    n = 0
    for tid in gen:
        yield tid
        n += 1
        if n % every == 0:
            print(f"  ... 已编码 {n:,} tokens", flush=True)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "tinystories"
    if name not in CONFIGS:
        raise SystemExit(f"未知数据集 '{name}'，只能是 {list(CONFIGS)}")
    cfg = CONFIGS[name]
    tok = load_tokenizer(cfg["bpe_dir"])

    for split, path in cfg["splits"].items():
        print(f"[{name}/{split}] 编码 {path} ...", flush=True)
        t0 = time.time()
        with open(path, encoding="utf-8") as f:
            ids = np.fromiter(token_stream(tok, f), dtype=np.uint16)
        out = f"{cfg['out_prefix']}_{split}.npy"
        np.save(out, ids)
        print(f"[{name}/{split}] {len(ids):,} tokens -> {out} | 用时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
