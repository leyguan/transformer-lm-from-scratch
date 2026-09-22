# 1.multiprocessing - pretokenize
# chunk = f.read(end - start).decode("utf-8", errors="ignore")
# chunk是string吗？
# 怎么用py自己的包变成词数统计的dict？
# 怎么合并呢？

# 2.第一遍把它拆成{['t', 'h', 'e'] : 3, ...}这样的吗？
# 这样做有什么用？或许不如直接改成{['th', 'he'] : 3}？
# 然后挨个扫描dict每个item的每个list，做成{'th' : x，'he'：y}
# 然后做成heap，为啥非要-x，-y呢？python没有大顶堆是吗？
# 使用lazy deletion+定期重构，那究竟怎么个定期重构法呢？
# 核心问题，合并之后怎么修改{['t', 'h', 'e'] : 3, ...} {['th', 'he'] : 3} {'th' : x，'he'：y}这三个dict呢？
# 除了

# 还有就是我们是byte-level的，怎么play with byte呢？比如把string变成byte string？
# import os
# import multiprocessing
# import regex as re
# from cs336_basics.pretokenization_example import find_chunk_boundaries
# from collections import Counter, defaultdict


# def process_chunk(args):
#     res = Counter()

#     start, end, input_path, special_pattern = args
#     with open(input_path, "rb") as f:
#         f.seek(start)
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")

#     pat_str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

#     last_index = 0
#     for match in re.finditer(special_pattern, chunk):
#         prefix = chunk[last_index:match.start()]
#         res += Counter(re.findall(pat_str, prefix))
#         last_index = match.end()

#     prefix = chunk[last_index:]
#     res += Counter(re.findall(pat_str, prefix))
#     return res


# def train_bpe(
#     input_path: str | os.PathLike,
#     vocab_size: int,
#     special_tokens: list[str],
# ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
#     # Results to return
#     vocab = {}
#     merges = []

#     # Initial 256 & special tokens
#     vocab = {i: bytes([i]) for i in range(256)}
#     for token in special_tokens:
#         vocab[len(vocab)] = token.encode("utf-8")

#     # Reg of special tokens
#     special_pattern = ("(" + "|".join(re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True)) + ")")

#     # Multi-processing chunks
#     num_processes = 6
#     with open(input_path, "rb") as f:
#         boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
#     tasks = [(start, end, input_path, special_pattern)
#              for start, end in zip(boundaries[:-1], boundaries[1:])]
#     with multiprocessing.Pool(num_processes) as pool:
#         results = pool.map(process_chunk, tasks)

#     # Combine chunks to initial Counter
#     pre_tok_res = Counter()
#     for result in results:
#         pre_tok_res += result

#     # Convert word to byte string in Counter
#     ByteStringCounter = {tuple(bytes([b]) for b in word.encode("utf-8")): count for word, count in pre_tok_res.items()}
#     ## 1. ByteStringCounter
#     ## Ground truth of byte string to count

#     # Get pair -> count & pair -> byte strings
#     PairCounter = Counter()
#     ## 2. PairCounter
#     ## Ground truth of byte pair -> count
#     PairToByteStrings = defaultdict(list)
#     ## 3. PairToByteStrings
#     ## Ground truth of byte pair to byte strings contraining it
#     for byte_word, count in ByteStringCounter.items():
#         for i in range(len(byte_word) - 1):
#             pair = (byte_word[i], byte_word[i + 1])
#             PairCounter[pair] += count
#             PairToByteStrings[pair].append(byte_word)

#     # Start merging
#     for new_id in range(len(vocab), vocab_size):
#         if not PairCounter:
#             break
#         pair = max(PairCounter, key=lambda p: (PairCounter[p], p))
#         count = PairCounter[pair]
#         del PairCounter[pair] ## maintenance of 2. PairCounter is done
#         ## 了吗？并没有，还有左右的 pair 

#         merges.append(pair)
#         merged_pair = pair[0] + pair[1]
#         vocab[new_id] = merged_pair        

#         # 那就修改 1. ByteStringCounter 3. PairToByteStrings
#         for byte_string in set(PairToByteStrings[pair]):
#             byte_string_count = ByteStringCounter[byte_string]
#             del ByteStringCounter[byte_string]
#             # 维护的是 1. ByteStringCounter
#             # 旧的 byte string 删了，新的还没加上


#             new_byte_string = []
#             i = 0
#             while i < len(byte_string):
#                 if i == len(byte_string) - 1:
#                     new_byte_string.append(byte_string[i])
#                     break
#                 if byte_string[i] == pair[0] and byte_string[i + 1] == pair[1]:
#                     new_byte_string.append(merged_pair)
#                     i += 2
#                 else:
#                     new_byte_string.append(byte_string[i])
#                     i += 1
#             new_byte_string_tuple = tuple(new_byte_string)
#             ByteStringCounter[new_byte_string_tuple] = byte_string_count
#             # 维护的是 1. ByteStringCounter
#             # 找到新的，并把新的加上了，1. ByteStringCounter 维护完成 


#             # --- 用 diff 更新 PairCounter 和 PairToByteStrings ---
#             old_pairs = Counter()
#             for j in range(len(byte_string) - 1):
#                 old_pairs[(byte_string[j], byte_string[j + 1])] += 1

#             new_pairs = Counter()
#             for j in range(len(new_byte_string_tuple) - 1):
#                 new_pairs[(new_byte_string_tuple[j], new_byte_string_tuple[j + 1])] += 1

#             for p, cnt in old_pairs.items():
#                 if p == pair:
#                     continue  # 已经在循环外 del 了
#                 PairCounter[p] -= cnt * byte_string_count
#                 if PairCounter[p] <= 0:
#                     del PairCounter[p]
#                 for _ in range(cnt):
#                     PairToByteStrings[p].remove(byte_string)

#             for p, cnt in new_pairs.items():
#                 PairCounter[p] += cnt * byte_string_count
#                 for _ in range(cnt):
#                     PairToByteStrings[p].append(new_byte_string_tuple)
            
#         del PairToByteStrings[pair]
            
#     return (vocab, merges)

# // 第二版
# import os
# import multiprocessing
# import regex as re
# from cs336_basics.pretokenization_example import find_chunk_boundaries
# from collections import Counter, defaultdict


# def process_chunk(args):
#     res = Counter()

#     start, end, input_path, special_pattern = args
#     with open(input_path, "rb") as f:
#         f.seek(start)
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")

#     pat_str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

#     last_index = 0
#     for match in re.finditer(special_pattern, chunk):
#         prefix = chunk[last_index:match.start()]
#         res += Counter(re.findall(pat_str, prefix))
#         last_index = match.end()

#     prefix = chunk[last_index:]
#     res += Counter(re.findall(pat_str, prefix))
#     return res


# def train_bpe(
#     input_path: str | os.PathLike,
#     vocab_size: int,
#     special_tokens: list[str],
# ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
#     vocab = {}
#     merges = []

#     # Base 256 bytes + special tokens
#     vocab = {i: bytes([i]) for i in range(256)}
#     for token in special_tokens:
#         vocab[len(vocab)] = token.encode("utf-8")

#     # Regex that splits on special tokens
#     special_pattern = ("(" + "|".join(re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True)) + ")")

#     # Parallel pre-tokenization over file chunks
#     num_processes = 14
#     with open(input_path, "rb") as f:
#         boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
#     tasks = [(start, end, input_path, special_pattern)
#              for start, end in zip(boundaries[:-1], boundaries[1:])]
#     with multiprocessing.Pool(num_processes) as pool:
#         results = pool.map(process_chunk, tasks)

#     pre_tok_res = Counter()
#     for result in results:
#         pre_tok_res += result

#     # Table 1: ByteStringCounter -- byte-string word -> count (ground truth)
#     ByteStringCounter = {tuple(bytes([b]) for b in word.encode("utf-8")): count for word, count in pre_tok_res.items()}

#     # Table 2: PairCounter -- byte pair -> total count (ground truth)
#     PairCounter = Counter()
#     # Table 3: PairToByteStrings -- byte pair -> set of words containing it
#     PairToByteStrings = defaultdict(set)
#     for byte_word, count in ByteStringCounter.items():
#         for i in range(len(byte_word) - 1):
#             pair = (byte_word[i], byte_word[i + 1])
#             PairCounter[pair] += count
#             PairToByteStrings[pair].add(byte_word)

#     # Merge loop
#     for new_id in range(len(vocab), vocab_size):
#         if not PairCounter:
#             break
#         pair = max(PairCounter, key=lambda p: (PairCounter[p], p))
#         count = PairCounter[pair]
#         del PairCounter[pair]

#         merges.append(pair)
#         merged_pair = pair[0] + pair[1]
#         vocab[new_id] = merged_pair

#         for byte_string in set(PairToByteStrings[pair]):
#             byte_string_count = ByteStringCounter[byte_string]
#             del ByteStringCounter[byte_string]

#             new_byte_string = []
#             i = 0
#             while i < len(byte_string):
#                 if i == len(byte_string) - 1:
#                     new_byte_string.append(byte_string[i])
#                     break
#                 if byte_string[i] == pair[0] and byte_string[i + 1] == pair[1]:
#                     new_byte_string.append(merged_pair)
#                     i += 2
#                 else:
#                     new_byte_string.append(byte_string[i])
#                     i += 1
#             new_byte_string_tuple = tuple(new_byte_string)
#             ByteStringCounter[new_byte_string_tuple] = byte_string_count

#             # Diff-update PairCounter and PairToByteStrings
#             old_pairs = Counter()
#             for j in range(len(byte_string) - 1):
#                 old_pairs[(byte_string[j], byte_string[j + 1])] += 1

#             new_pairs = Counter()
#             for j in range(len(new_byte_string_tuple) - 1):
#                 new_pairs[(new_byte_string_tuple[j], new_byte_string_tuple[j + 1])] += 1

#             for p, cnt in old_pairs.items():
#                 if p == pair:
#                     continue
#                 PairCounter[p] -= cnt * byte_string_count
#                 if PairCounter[p] <= 0:
#                     del PairCounter[p]
#                 PairToByteStrings[p].discard(byte_string)

#             for p, cnt in new_pairs.items():
#                 PairCounter[p] += cnt * byte_string_count
#                 PairToByteStrings[p].add(new_byte_string_tuple)

#         del PairToByteStrings[pair]

#     return (vocab, merges)







import os
import time
import heapq
import multiprocessing
import regex as re
from cs336_basics.pretokenization_example import find_chunk_boundaries
from collections import Counter, defaultdict


def process_chunk(args):
    res = Counter()

    start, end, input_path, special_pattern = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    pat_str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    last_index = 0
    for match in re.finditer(special_pattern, chunk):
        prefix = chunk[last_index:match.start()]
        res += Counter(re.findall(pat_str, prefix))
        last_index = match.end()

    prefix = chunk[last_index:]
    res += Counter(re.findall(pat_str, prefix))
    return res


class _RevPair:
    """Wraps a byte pair so heapq (a min-heap) breaks count ties toward the
    LARGER pair, matching max(PairCounter, key=(count, pair)). Bytes can't be
    negated the way counts can, so the ordering is reversed here instead."""
    __slots__ = ("p",)

    def __init__(self, p):
        self.p = p

    def __lt__(self, other):
        return self.p > other.p


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    verbose: bool = True,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {}
    merges = []

    # Base 256 bytes + special tokens
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    # Regex that splits on special tokens
    special_pattern = ("(" + "|".join(re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True)) + ")")

    # Parallel pre-tokenization over file chunks
    t0 = time.time()
    num_processes = 14
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    tasks = [(start, end, input_path, special_pattern)
             for start, end in zip(boundaries[:-1], boundaries[1:])]
    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(process_chunk, tasks)

    pre_tok_res = Counter()
    for result in results:
        pre_tok_res += result
    if verbose:
        print(f"  [pretok] done in {time.time() - t0:.1f}s  distinct words={len(pre_tok_res)}", flush=True)

    # Table 1: ByteStringCounter -- byte-string word -> count (ground truth)
    ByteStringCounter = {tuple(bytes([b]) for b in word.encode("utf-8")): count for word, count in pre_tok_res.items()}

    # Table 2: PairCounter -- byte pair -> total count (ground truth)
    PairCounter = Counter()
    # Table 3: PairToByteStrings -- byte pair -> set of words containing it
    PairToByteStrings = defaultdict(set)
    for byte_word, count in ByteStringCounter.items():
        for i in range(len(byte_word) - 1):
            pair = (byte_word[i], byte_word[i + 1])
            PairCounter[pair] += count
            PairToByteStrings[pair].add(byte_word)

    # Max-pair heap (lazy deletion): entries are (-count, _RevPair(pair), pair).
    # A stale entry (its -count no longer matches PairCounter) is skipped on pop.
    # Within one merge, each pair whose count changes is pushed once at the end
    # rather than on every intermediate mutation, so a hot pair touching many
    # words no longer floods the heap with millions of throwaway entries.
    # The heap is rebuilt from PairCounter once it bloats past ~2x the live pairs.
    heap = [(-c, _RevPair(p), p) for p, c in PairCounter.items()]
    heapq.heapify(heap)

    if verbose:
        print(f"  [init] tables built in {time.time() - t0:.1f}s  distinct pairs={len(PairCounter)}", flush=True)

    # Merge loop
    start_id = len(vocab)
    total_merges = vocab_size - start_id
    t_merge0 = time.time()

    for new_id in range(start_id, vocab_size):
        # Pop the best still-valid pair (skip stale heap entries)
        pair = None
        while heap:
            neg, _, cand = heapq.heappop(heap)
            if PairCounter.get(cand) == -neg:
                pair = cand
                break
        if pair is None:
            break
        count = PairCounter[pair]
        del PairCounter[pair]

        merges.append(pair)
        merged_pair = pair[0] + pair[1]
        vocab[new_id] = merged_pair

        dirty = set()  # pairs whose count changed this merge -> push once at the end
        for byte_string in set(PairToByteStrings[pair]):
            byte_string_count = ByteStringCounter[byte_string]
            del ByteStringCounter[byte_string]

            new_byte_string = []
            i = 0
            while i < len(byte_string):
                if i == len(byte_string) - 1:
                    new_byte_string.append(byte_string[i])
                    break
                if byte_string[i] == pair[0] and byte_string[i + 1] == pair[1]:
                    new_byte_string.append(merged_pair)
                    i += 2
                else:
                    new_byte_string.append(byte_string[i])
                    i += 1
            new_byte_string_tuple = tuple(new_byte_string)
            ByteStringCounter[new_byte_string_tuple] = byte_string_count

            # Diff-update PairCounter and PairToByteStrings
            old_pairs = Counter()
            for j in range(len(byte_string) - 1):
                old_pairs[(byte_string[j], byte_string[j + 1])] += 1

            new_pairs = Counter()
            for j in range(len(new_byte_string_tuple) - 1):
                new_pairs[(new_byte_string_tuple[j], new_byte_string_tuple[j + 1])] += 1

            for p, cnt in old_pairs.items():
                if p == pair:
                    continue
                PairCounter[p] -= cnt * byte_string_count
                if PairCounter[p] <= 0:
                    del PairCounter[p]
                dirty.add(p)
                PairToByteStrings[p].discard(byte_string)

            for p, cnt in new_pairs.items():
                PairCounter[p] += cnt * byte_string_count
                dirty.add(p)
                PairToByteStrings[p].add(new_byte_string_tuple)

        del PairToByteStrings[pair]

        # Push each changed pair once, with its final count
        for p in dirty:
            if p in PairCounter:
                heapq.heappush(heap, (-PairCounter[p], _RevPair(p), p))

        # Rebuild the heap when stale entries pile up, to keep pops cheap
        if len(heap) > 2 * len(PairCounter) + 1024:
            heap = [(-c, _RevPair(p), p) for p, c in PairCounter.items()]
            heapq.heapify(heap)

        if verbose:
            done = new_id - start_id + 1
            if done <= 10 or done % 200 == 0 or done == total_merges:
                print(f"  [merge {done}/{total_merges}] {time.time() - t_merge0:.0f}s  "
                      f"top_count={count}  live_pairs={len(PairCounter)}  heap={len(heap)}", flush=True)

    return (vocab, merges)