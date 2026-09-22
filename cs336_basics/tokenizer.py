import regex as re
from typing import Iterable, Iterator


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)  # id -> bytes
        self.merges = list(merges)
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)

        # 把 special tokens 加入 vocab（如果还没在里面）
        existing_bytes = set(self.vocab.values())
        for token in self.special_tokens:
            tb = token.encode("utf-8")
            if tb not in existing_bytes:
                self.vocab[len(self.vocab)] = tb
                existing_bytes.add(tb)

        # 反向映射: bytes -> id
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}

        # special token 的正则（长的优先匹配，避免前缀冲突）
        if self.special_tokens:
            self.special_pattern = (
                "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
            )
        else:
            self.special_pattern = None

        # merge pair -> rank（用于 encode 时快速找最高优先级 pair）
        self.merge_order = {pair: i for i, pair in enumerate(self.merges)}

        # GPT-2 style pre-tokenization 正则
        self.pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # ------------------------------------------------------------------
    # from_files: 从磁盘加载 vocab / merges
    # ------------------------------------------------------------------
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        import json

        # --- 读 vocab ---
        vocab: dict[int, bytes] = {}
        with open(vocab_filepath, "r") as f:
            raw = json.load(f)
        for k, v in raw.items():
            # v 可能是 list[int]（字节列表）或 hex string，视你 train 的序列化格式而定
            vocab[int(k)] = bytes(v)

        # --- 读 merges ---
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 每行格式: "b1_hex b2_hex"  (空格分隔的两段 hex)
                parts = line.split(" ")
                a = bytes.fromhex(parts[0])
                b = bytes.fromhex(parts[1])
                merges.append((a, b))

        return cls(vocab, merges, special_tokens)

    # ------------------------------------------------------------------
    # encode
    # ------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        ids: list[int] = []

        # 1) 按 special tokens 切分
        if self.special_pattern:
            segments = re.split(self.special_pattern, text)
        else:
            segments = [text]

        special_set = set(self.special_tokens)

        for seg in segments:
            if not seg:
                continue

            # special token 直接查表
            if seg in special_set:
                ids.append(self.bytes_to_id[seg.encode("utf-8")])
                continue

            # 2) pre-tokenize
            pre_tokens = re.findall(self.pat, seg)

            for pt in pre_tokens:
                # 初始：每个字节单独一个 token
                token_seq = [bytes([b]) for b in pt.encode("utf-8")]

                # 3) 每轮找当前序列里 rank 最小的 pair，合并它
                while len(token_seq) >= 2:
                    # 扫一遍当前所有相邻 pair，找 merge_order 里 rank 最小的
                    best_pair = None
                    best_rank = float("inf")
                    for i in range(len(token_seq) - 1):
                        p = (token_seq[i], token_seq[i + 1])
                        r = self.merge_order.get(p)
                        if r is not None and r < best_rank:
                            best_rank = r
                            best_pair = p

                    if best_pair is None:
                        break  # 没有可合并的了

                    a, b = best_pair
                    new_seq: list[bytes] = []
                    i = 0
                    while i < len(token_seq):
                        if (
                            i < len(token_seq) - 1
                            and token_seq[i] == a
                            and token_seq[i + 1] == b
                        ):
                            new_seq.append(a + b)
                            i += 2
                        else:
                            new_seq.append(token_seq[i])
                            i += 1
                    token_seq = new_seq

                # 4) 查 bytes -> id
                for tb in token_seq:
                    ids.append(self.bytes_to_id[tb])

        return ids

    # ------------------------------------------------------------------
    # encode_iterable: 惰性逐块编码，内存 O(1)
    # ------------------------------------------------------------------
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------
    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")