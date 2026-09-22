# """
# 训练 BPE 分词器。放在项目根目录，从根目录跑。

# 用法：
#     # TinyStories（vocab 10000）
#     python run_train_bpe.py tinystories

#     # OWT（vocab 32000）
#     python run_train_bpe.py owt

#     # 后台跑 + 看日志
#     nohup python run_train_bpe.py tinystories > bpe/ts_bpe.log 2>&1 &
#     tail -f bpe/ts_bpe.log

# 输出到 bpe/<name>/ ：
#     bpe_vocab.json   {token_id: hex}
#     bpe_merges.txt   每行 "hex hex"
# """
# import os
# import sys
# import json
# import time
# import resource
# from cs336_basics.train_bpe import train_bpe

# CONFIGS = {
#     "tinystories": {
#         "input_path": "data/TinyStoriesV2-GPT4-train.txt",
#         "out_dir": "bpe/TinyStories",
#         "vocab_size": 10000,
#     },
#     "owt": {
#         "input_path": "data/owt_train.txt",
#         "out_dir": "bpe/owt",
#         "vocab_size": 32000,
#     },
# }
# SPECIAL_TOKENS = ["<|endoftext|>"]


# def main():
#     name = sys.argv[1] if len(sys.argv) > 1 else "tinystories"
#     if name not in CONFIGS:
#         raise SystemExit(f"未知数据集 '{name}'，只能是 {list(CONFIGS)}")
#     cfg = CONFIGS[name]
#     os.makedirs(cfg["out_dir"], exist_ok=True)

#     print(f"[{name}] BPE 开始  vocab_size={cfg['vocab_size']}  input={cfg['input_path']}")
#     t0 = time.time()
#     vocab, merges = train_bpe(cfg["input_path"], cfg["vocab_size"], SPECIAL_TOKENS)
#     dt = time.time() - t0
#     peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB -> MB
#     print(f"[{name}] 完成  用时 {dt:.1f}s | 峰值内存 ≈ {peak_mb:.0f} MB | merges={len(merges)}")

#     # 看一眼最长的 token（能感觉到 BPE 学到了什么）
#     longest_id = max(vocab, key=lambda k: len(vocab[k]))
#     lb = vocab[longest_id]
#     try:
#         ls = lb.decode("utf-8")
#     except UnicodeDecodeError:
#         ls = repr(lb)
#     print(f"[{name}] 最长 token (id={longest_id}): {ls!r} ({len(lb)} bytes)")

#     # 存 vocab（id -> hex）
#     vpath = os.path.join(cfg["out_dir"], "bpe_vocab.json")
#     with open(vpath, "w") as f:
#         json.dump({str(k): v.hex() for k, v in vocab.items()}, f, indent=2)

#     # 存 merges（每行 "hex hex"）
#     mpath = os.path.join(cfg["out_dir"], "bpe_merges.txt")
#     with open(mpath, "w") as f:
#         for a, b in merges:
#             f.write(f"{a.hex()} {b.hex()}\n")

#     print(f"[{name}] 已保存 {vpath} 和 {mpath}")


# if __name__ == "__main__":
#     main()







"""
Train a BPE tokenizer. Place in the project root; run from the project root.

Usage:
    # TinyStories (vocab 10000)
    python run_train_bpe.py tinystories

    # OWT (vocab 32000)
    python run_train_bpe.py owt

    # OWT smoke test: first 500 MB only, same vocab size, for timing.
    # Build the file once yourself, then run:
    #   head -c 500M data/owt_train.txt > data/owt_smoke.txt
    #   python run_train_bpe.py owt_smoke

    # Background run + follow the log
    nohup python run_train_bpe.py owt_smoke > bpe/owt_smoke.log 2>&1 &
    tail -f bpe/owt_smoke.log

Outputs go to bpe/<name>/ :
    bpe_vocab.json   {token_id: hex}
    bpe_merges.txt   one "hex hex" per line
"""
import os
import sys
import json
import time
import resource
from cs336_basics.train_bpe import train_bpe

CONFIGS = {
    "tinystories": {
        "input_path": "data/TinyStoriesV2-GPT4-train.txt",
        "out_dir": "bpe/TinyStories",
        "vocab_size": 10000,
    },
    "owt": {
        "input_path": "data/owt_train.txt",
        "out_dir": "bpe/owt",
        "vocab_size": 32000,
    },
    # Smoke test: first 500 MB of OWT. Build data/owt_smoke.txt yourself:
    #   head -c 500M data/owt_train.txt > data/owt_smoke.txt
    # Same vocab_size as owt so the merge count matches and timing extrapolates.
    "owt_smoke": {
        "input_path": "data/owt_smoke.txt",
        "out_dir": "bpe/owt_smoke",
        "vocab_size": 32000,
    },
}
SPECIAL_TOKENS = ["<|endoftext|>"]


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "tinystories"
    if name not in CONFIGS:
        raise SystemExit(f"未知数据集 '{name}'，只能是 {list(CONFIGS)}")
    cfg = CONFIGS[name]
    os.makedirs(cfg["out_dir"], exist_ok=True)

    input_path = cfg["input_path"]
    if not os.path.exists(input_path):
        raise SystemExit(
            f"输入文件不存在：{input_path}\n"
            f"（owt_smoke 需要你先手动建：head -c 500M data/owt_train.txt > {input_path}）"
        )

    print(f"[{name}] BPE 开始  vocab_size={cfg['vocab_size']}  input={input_path}")
    t0 = time.time()
    vocab, merges = train_bpe(input_path, cfg["vocab_size"], SPECIAL_TOKENS)
    dt = time.time() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB -> MB
    print(f"[{name}] 完成  用时 {dt:.1f}s | 峰值内存 ≈ {peak_mb:.0f} MB | merges={len(merges)}")

    # Peek at the longest token (a feel for what BPE learned)
    longest_id = max(vocab, key=lambda k: len(vocab[k]))
    lb = vocab[longest_id]
    try:
        ls = lb.decode("utf-8")
    except UnicodeDecodeError:
        ls = repr(lb)
    print(f"[{name}] 最长 token (id={longest_id}): {ls!r} ({len(lb)} bytes)")

    # Save vocab (id -> hex)
    vpath = os.path.join(cfg["out_dir"], "bpe_vocab.json")
    with open(vpath, "w") as f:
        json.dump({str(k): v.hex() for k, v in vocab.items()}, f, indent=2)

    # Save merges (one "hex hex" per line)
    mpath = os.path.join(cfg["out_dir"], "bpe_merges.txt")
    with open(mpath, "w") as f:
        for a, b in merges:
            f.write(f"{a.hex()} {b.hex()}\n")

    print(f"[{name}] 已保存 {vpath} 和 {mpath}")


if __name__ == "__main__":
    main()
