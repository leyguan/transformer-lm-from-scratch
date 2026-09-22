import json
from cs336_basics.tokenizer import Tokenizer
V = {int(k): bytes.fromhex(v) for k, v in json.load(open("bpe/TinyStories/bpe_vocab.json")).items()}
M = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in (l.split() for l in open("bpe/TinyStories/bpe_merges.txt"))]
tok = Tokenizer(V, M, ["<|endoftext|>"])
s = open("data/TinyStoriesV2-GPT4-valid.txt", encoding="utf-8").read(3000)
ids = tok.encode(s)
assert tok.decode(ids) == s, "round-trip 对不上，encode/decode 有问题"
assert max(ids) < len(V)
print("OK  tokens:", len(ids), " bytes/token:", len(s.encode()) / len(ids))