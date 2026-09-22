"""
用法:
    nohup python run_train_bpe.py > bpe_train.log 2>&1 &
    tail -f bpe_train.log

输出:
    bpe_vocab.json   — {token_id: hex_bytes}
    bpe_merges.txt   — 每行一个 merge pair (hex)
    训练耗时、内存峰值、最长 token 会打印到 stdout
"""

import json
import time
import resource
from cs336_basics.train_bpe import train_bpe

INPUT_PATH = "data/TinyStoriesV2-GPT4-train.txt"
VOCAB_SIZE = 10000
SPECIAL_TOKENS = ["<|endoftext|>"]

def main():
    print(f"Starting BPE training: vocab_size={VOCAB_SIZE}")
    t0 = time.time()

    vocab, merges = train_bpe(INPUT_PATH, VOCAB_SIZE, SPECIAL_TOKENS)

    elapsed = time.time() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB -> MB
    print(f"Done in {elapsed:.1f}s | Peak RSS ≈ {peak_mb:.0f} MB")

    # --- 最长 token ---
    longest_id = max(vocab, key=lambda k: len(vocab[k]))
    longest_bytes = vocab[longest_id]
    try:
        longest_str = longest_bytes.decode("utf-8")
    except UnicodeDecodeError:
        longest_str = repr(longest_bytes)
    print(f"Longest token (id={longest_id}): {repr(longest_str)} ({len(longest_bytes)} bytes)")

    # --- 序列化 vocab ---
    vocab_serializable = {str(k): v.hex() for k, v in vocab.items()}
    with open("bpe_vocab.json", "w") as f:
        json.dump(vocab_serializable, f, indent=2)
    print("Saved bpe_vocab.json")

    # --- 序列化 merges ---
    with open("bpe_merges.txt", "w") as f:
        for a, b in merges:
            f.write(f"{a.hex()} {b.hex()}\n")
    print(f"Saved bpe_merges.txt ({len(merges)} merges)")

if __name__ == "__main__":
    main()
