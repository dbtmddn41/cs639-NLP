"""
CS639 HW: Byte-Pair Encoding (BPE) Tokenizer

You will implement the core pieces of a GPT-2-style byte-level BPE tokenizer:
1) Pre-tokenization using the provided PAT regex
2) Special token preservation (e.g., <|endoftext|>)
3) Pair counting and efficient merge updates during training

"""

from __future__ import annotations

import os
import time
import regex as re
from collections import defaultdict, Counter
from multiprocessing import Pool
import heapq
import pickle

from pretokenization_example import find_chunk_boundaries

tiny_stories_val_path = "data/TinyStoriesV2-GPT4-valid.txt"
tiny_stories_train_path = "data/TinyStoriesV2-GPT4-train.txt"

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# ============================================================
# Part 1: Pre-tokenization
# ============================================================

def build_split_expr(special_tokens: list[str]) -> str:
    """
    Build a regex that lets us split text into normal text and special tokens,
    while preserving special tokens in the split output.

    Hint: `re.split` only preserves delimiters if the delimiter regex is CAPTURED.

    Example:
        text = "a <|endoftext|> b"
        parts = re.split(split_expr, text)
        parts should include "<|endoftext|>" as an element.
    """
    # TODO (student): implement
    raise NotImplementedError


def pretokenize_text(text: str, special_tokens: list[str]) -> Counter[tuple[int, ...]]:
    """
    Convert a text string into a Counter over byte-tuples, using:
      - special token splitting (preserve special tokens)
      - PAT regex to produce pre-tokens from normal text
      - UTF-8 bytes converted to tuple[int, ...] for hashing

    Output keys are tuples of ints, e.g. b"hello" -> (104, 101, 108, 108, 111)

    Notes:
    - Special tokens should be treated as atomic tokens.
    - For normal text pieces, run PAT and count each match.
    """
    # TODO: implement
    raise NotImplementedError


def process_chunk(
    start: int,
    end: int,
    file_path: str | os.PathLike,
    split_expr: str,
    special_tokens: list[str],
) -> Counter[tuple[int, ...]]:
    """
    Process a file chunk [start, end) and return pre-token counts.

    Notes:
    - This should call `pretokenize_text` on the decoded chunk.
    - Use errors="ignore" for decoding to avoid crashing on boundary artifacts.
    """
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    # TODO: you may directly call pretokenize_text(chunk, special_tokens)
    # and return it.
    raise NotImplementedError


def pre_tokenize(
    file_path: str | os.PathLike, special_tokens: list[str], num_processes: int = 1
) -> Counter[tuple[int, ...]]:
    """
    Pre-tokenize the whole file using multiple processes.

    For grading, we mainly care about correctness of the returned Counter.
    Multiprocessing is kept for realism but may be disabled in tests.
    """
    start_time = time.time()
    if not file_path:
        raise ValueError("No file path passed to pre_tokenize.")

    print("Starting pre-tokenization...")

    split_expr = build_split_expr(special_tokens)

    # Use first special token as a delimiter for boundary finding (course-provided helper)
    delimiter = special_tokens[0].encode("utf-8") if special_tokens else b"\n"

    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, delimiter)

    # boundaries is a list of positions; we have len(boundaries)-1 chunks
    n_chunks = len(boundaries) - 1
    process_args = list(
        zip(
            boundaries,
            boundaries[1:],
            [file_path] * n_chunks,
            [split_expr] * n_chunks,
            [special_tokens] * n_chunks,
        )
    )

    pre_tokens_count: Counter[tuple[int, ...]] = Counter()

    if num_processes == 1:
        # single-process path (useful for debugging + deterministic tests)
        for args in process_args:
            pre_tokens_count.update(process_chunk(*args))
    else:
        with Pool(processes=num_processes) as pool:
            results = pool.starmap(process_chunk, process_args)
        for r in results:
            pre_tokens_count.update(r)

    print(f"Pre-tokenization time: {time.time() - start_time:.2f}s")
    return pre_tokens_count


# ============================================================
# Part 2: BPE training
# ============================================================

def count_pairs(
    inp: dict[tuple[int, ...], int],
) -> tuple[
    dict[tuple[int, int], int],
    dict[tuple[int, int], set[tuple[int, ...]]],
    dict[tuple[int, ...], set[tuple[int, int]]],
]:
    """
    Count adjacent byte-pairs across all sequences in `inp`.

    Returns:
      pairs_count: (a,b) -> total count
      pairs_to_sequences: (a,b) -> set of sequences containing that pair
      sequences_to_pairs: seq -> set of pairs present in that sequence

    Note:
    - sequences are tuples of token IDs (ints), e.g., (104,101,108,108,111)
    - Do not count pairs in sequences of length < 2.
    """
    # TODO: implement
    raise NotImplementedError


def merge_pair(
    byte_tokens_count: dict[tuple[int, ...], int],
    pairs_to_sequences: dict[tuple[int, int], set[tuple[int, ...]]],
    old: tuple[int, int],
    new: int,
) -> tuple[list[tuple[int, ...]], list[tuple[list[int], int]]]:
    """
    Apply one merge (old=(a,b) -> new_token_id) to every sequence that contains old.

    Returns:
      keys_to_remove: old sequences that are replaced
      keys_to_add: list of (new_sequence_as_list, count) to add

    Important:
    - Replace all occurrences of (a,b) in a sequence left-to-right.
    - Only sequences that change should appear in keys_to_remove/keys_to_add.
    """
    # TODO: implement
    raise NotImplementedError


def select_best_pair(
    pairs: dict[tuple[int, int], int],
    vocab: dict[int, bytes],
) -> tuple[int, int]:
    """
    Deterministically select the best pair to merge.

    Staff policy (simple, deterministic):
      1) highest count
      2) break ties by lexicographic order of (vocab[a], vocab[b])  (bytes compare)

    This avoids non-determinism across platforms.
    """
    # You may keep this staff-provided or let students implement (up to you).
    # Keeping staff-provided makes grading less fragile.
    top = heapq.nlargest(100, pairs.items(), key=lambda kv: kv[1])
    max_count = top[0][1]
    ties: list[tuple[bytes, bytes, tuple[int, int]]] = []
    for (a, b), c in top:
        if c != max_count:
            break
        ties.append((vocab[a], vocab[b], (a, b)))
    ties.sort()  # lexicographic on bytes then pair
    return ties[-1][2]  # pick lexicographically largest among max-count ties


def train_byte_pair_encoder(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 2,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.

    Returns:
      vocab: token_id -> bytes
      merges: list of merges in order, each as (bytes_a, bytes_b)
    """
    if not input_path or not vocab_size:
        raise ValueError("Missing input_path or vocab_size.")
    if special_tokens is None:
        special_tokens = []

    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")

    num_merges = vocab_size - len(vocab)
    if num_merges < 0:
        raise ValueError("vocab_size too small for base vocab + special tokens.")

    byte_tokens_count = pre_tokenize(input_path, special_tokens, num_processes=num_processes)

    t0 = time.time()
    pairs, pairs_to_sequences, seqs_to_pairs = count_pairs(byte_tokens_count)
    print(f"Initial pair counting time: {time.time() - t0:.2f}s")

    t1 = time.time()
    for _ in range(num_merges):
        if not pairs:
            break

        best_pair = select_best_pair(pairs, vocab)
        a, b = best_pair
        merges.append((vocab[a], vocab[b]))

        new_tok_id = len(vocab)
        vocab[new_tok_id] = vocab[a] + vocab[b]

        keys_to_remove, keys_to_add = merge_pair(
            byte_tokens_count, pairs_to_sequences, best_pair, new_tok_id
        )

        # incremental update logic
        for s in keys_to_remove:
            # decrement old pair counts from the removed sequence
            if len(s) >= 2:
                for p in zip(s, s[1:]):
                    pairs[p] -= byte_tokens_count[s]
                    if pairs[p] == 0:
                        pairs.pop(p, None)

            # update reverse maps
            for p in seqs_to_pairs.get(s, set()):
                if p in pairs_to_sequences:
                    pairs_to_sequences[p].discard(s)
                    if not pairs_to_sequences[p]:
                        pairs_to_sequences.pop(p, None)

            byte_tokens_count.pop(s, None)
            seqs_to_pairs.pop(s, None)

        # add in new sequences with their counts
        for w_list, c in keys_to_add:
            w = tuple(w_list)
            byte_tokens_count[w] = byte_tokens_count.get(w, 0) + c

        # recount pairs only for newly added sequences
        for w_list, c in keys_to_add:
            w = tuple(w_list)
            if len(w) < 2:
                continue
            for j in range(len(w) - 1):
                p = (w[j], w[j + 1])
                pairs[p] = pairs.get(p, 0) + c
                pairs_to_sequences[p].add(w)
                seqs_to_pairs[w].add(p)

    print(f"Merge loop time ({len(merges)} merges): {time.time() - t1:.2f}s")
    return vocab, merges


# ============================================================
# Part 3: Sanity checks
# ============================================================

def save_artifacts(out_dir: str, filename_base: str, vocab, merges) -> None:
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, f"{filename_base}_vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    with open(os.path.join(out_dir, f"{filename_base}_merges.pkl"), "wb") as f:
        pickle.dump(merges, f)


if __name__ == "__main__":
    start = time.time()
    data_path = tiny_stories_train_path
    special_tokens = ["<|endoftext|>"]

    vocab, merges = train_byte_pair_encoder(data_path, 32000, special_tokens, num_processes=2)

    end = time.time()
    print(f"Total time: {end - start:.2f}s")

    import os.path
    filename_base = os.path.splitext(os.path.basename(data_path))[0]
    save_artifacts("tokenizer_results", filename_base, vocab, merges)
