"""Build a PG19-based associative-reconstruction dataset for GradMemGPT.

Each example is derived from one PG19 chunk (produced by
`prepare_pg19_chunks.py`, schema `{context, word_count, n_sentences,
sentence_boundaries, ...}`). The chunk's text is split into three parts:

    context : text[:context_end_char]   -- written to memory at test time
    query   : span_prefix               -- READ-phase prompt (held-out substring prefix)
    target  : span_suffix               -- the completion GradMemGPT must reconstruct

where the held-out substring `span = text[context_end_char:
context_end_char + span_len]` is divided at `query_fraction` into
`query` + `target`. The target appears nowhere in `context`, so it can
only be reconstructed from memory of the surrounding passage (the
context IS the passage up to the held-out point).

Token-length budgets are estimated with the pretrained tokenizer purely
for length control (mirrors `babilong_utils.SentenceSampler`); final
re-tokenization happens at train time in `collate_fn`.

The output is an HF `DatasetDict` saved with `save_to_disk` containing
the standard GradMemGPT schema `{context, query, target}` (raw strings)
and the splits `train`, `valid`. A `valid_no_context` split (identical
rows but `context = ""`) is also emitted so the "without memory"
baseline can be evaluated by feeding all-pad context (which skips the
WRITE phase in `grad_memgpt.py`).
"""

import argparse
import os
import random

import datasets
from tqdm import tqdm
from transformers import AutoTokenizer


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _build_example(text, rng, tokenizer,
                   context_min_tokens, context_max_tokens,
                   span_min_tokens, span_max_tokens, query_fraction):
    """Sample one (context, query, target) triple from a chunk's text.

    Returns None if the text is too short to satisfy the requested budgets.
    """
    text = text.strip()
    if not text:
        return None

    # Estimate characters-per-token from a small prefix so we can pick
    # boundaries that respect the token budgets. 1.3 is a conservative
    # chars/token for English BPE (GPT-2 averages ~4 chars/token, but we want
    # an upper bound on chars to avoid overshooting the token budget).
    probe = text[: min(512, len(text))]
    n_probe_tok = len(tokenizer.encode(probe, add_special_tokens=False))
    if n_probe_tok == 0:
        return None
    chars_per_tok = max(1.0, len(probe) / n_probe_tok)

    # Convert token budgets to char budgets.
    context_min_chars = int(context_min_tokens * chars_per_tok)
    context_max_chars = int(context_max_tokens * chars_per_tok)
    span_min_chars = int(span_min_tokens * chars_per_tok)
    span_max_chars = int(span_max_tokens * chars_per_tok)

    if len(text) < context_min_chars + span_min_chars:
        return None

    # Sample the held-out span first (it must fit at the end of context).
    span_max_chars = min(span_max_chars, len(text) - context_min_chars)
    if span_max_chars < span_min_chars:
        return None
    span_len = rng.randint(span_min_chars, span_max_chars)

    # Context boundary must leave room for the span.
    context_min_chars = min(context_min_chars, len(text) - span_len)
    context_max_chars = min(context_max_chars, len(text) - span_len)
    if context_max_chars < context_min_chars:
        return None
    context_end_char = rng.randint(context_min_chars, context_max_chars)

    span_start = context_end_char
    span_end = context_end_char + span_len
    span = text[span_start:span_end]

    # Split the span into query (prefix) + target (completion) at a token
    # boundary. We split on chars using query_fraction, then snap the query to
    # end on a whitespace boundary so the tokenizer offset-mapping in
    # collate_fn lands cleanly between query and target.
    query_char_len = max(1, int(round(len(span) * query_fraction)))
    # snap forward to the next whitespace (so query ends mid-word only if no
    # whitespace exists). This keeps query/target cleanly separable.
    snap_to = span.find(' ', query_char_len)
    if snap_to == -1 or snap_to == 0:
        query_char_len = min(query_char_len, len(span) - 1)
    else:
        query_char_len = snap_to + 1  # include the space in the query
    query_char_len = max(1, min(query_char_len, len(span) - 1))

    query = span[:query_char_len]
    target = span[query_char_len:]
    if not target.strip():
        return None

    context = text[:context_end_char]

    return {'context': context, 'query': query, 'target': target}


def build_split(chunk_dataset, n_samples, rng, tokenizer, **sample_kwargs):
    """Draw `n_samples` reconstruction examples by sampling random chunks.

    Multiple examples may be drawn from the same long chunk (each with an
    independent random boundary), and short/empty chunks are retried.
    """
    n_chunks = len(chunk_dataset)
    if n_chunks == 0:
        raise ValueError("Chunk dataset is empty.")

    rows = []
    pbar = tqdm(total=n_samples, desc="sampling examples", unit="ex")
    attempts = 0
    max_attempts = n_samples * 50
    while len(rows) < n_samples and attempts < max_attempts:
        attempts += 1
        chunk_idx = rng.randrange(n_chunks)
        text = chunk_dataset[int(chunk_idx)]['context']
        ex = _build_example(text, rng, tokenizer, **sample_kwargs)
        if ex is None:
            continue
        rows.append(ex)
        pbar.update(1)
    pbar.close()

    if len(rows) < n_samples:
        raise RuntimeError(
            f"Only built {len(rows)} examples after {attempts} attempts "
            f"(requested {n_samples}). Loosen length budgets or use more chunks.")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Build a PG19 associative-reconstruction dataset for GradMemGPT.")
    parser.add_argument("--chunks_path", type=str, required=True,
                        help="Path to the chunk DatasetDict from prepare_pg19_chunks.py.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Where to save the output DatasetDict.")
    parser.add_argument("--chunk_train_split", type=str, default="train",
                        help="Split of the chunk dataset to draw training examples from.")
    parser.add_argument("--chunk_valid_split", type=str, default="validation",
                        help="Split of the chunk dataset to draw validation examples from.")
    parser.add_argument("--context_min_tokens", type=int, default=512)
    parser.add_argument("--context_max_tokens", type=int, default=1024)
    parser.add_argument("--span_min_tokens", type=int, default=32)
    parser.add_argument("--span_max_tokens", type=int, default=128)
    parser.add_argument("--query_fraction", type=float, default=0.5,
                        help="Fraction of the held-out span given to the query (rest is target).")
    parser.add_argument("--n_train", type=int, default=100000)
    parser.add_argument("--n_valid", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--pretrained_model", type=str, default="gpt2",
                        help="Tokenizer used for length budgeting (re-tokenized at train time).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.output_path):
        if not args.overwrite:
            raise FileExistsError(
                f"Output path exists: {args.output_path}. Use --overwrite to replace.")
        import shutil
        shutil.rmtree(args.output_path)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    chunks = datasets.load_from_disk(args.chunks_path)
    if not isinstance(chunks, datasets.DatasetDict):
        raise ValueError("chunks_path must point to a DatasetDict saved with save_to_disk.")

    def _resolve_split(ds, preferred):
        if preferred in ds:
            return ds[preferred]
        # fall back to the first available split
        return ds[list(ds.keys())[0]]

    train_chunks = _resolve_split(chunks, args.chunk_train_split)
    valid_chunks = _resolve_split(chunks, args.chunk_valid_split)

    rng = random.Random(args.seed)

    sample_kwargs = dict(
        tokenizer=tokenizer,
        context_min_tokens=args.context_min_tokens,
        context_max_tokens=args.context_max_tokens,
        span_min_tokens=args.span_min_tokens,
        span_max_tokens=args.span_max_tokens,
        query_fraction=args.query_fraction,
    )

    train_rows = build_split(train_chunks, args.n_train, rng, **sample_kwargs)
    valid_rows = build_split(valid_chunks, args.n_valid, rng, **sample_kwargs)

    # valid_no_context: same examples with empty context (WRITE phase skipped).
    valid_no_context_rows = [{**r, 'context': ''} for r in valid_rows]

    out = datasets.DatasetDict({
        'train': datasets.Dataset.from_list(train_rows),
        'valid': datasets.Dataset.from_list(valid_rows),
        'valid_no_context': datasets.Dataset.from_list(valid_no_context_rows),
    })

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or '.', exist_ok=True)
    out.save_to_disk(args.output_path)
    print(f"Dataset saved to {args.output_path}")
    print(f"  train: {len(out['train'])}  valid: {len(out['valid'])}  "
          f"valid_no_context: {len(out['valid_no_context'])}")
    print("Example row:")
    ex = out['train'][0]
    print(f"  context[-120:]: ...{ex['context'][-120:]!r}")
    print(f"  query:  {ex['query']!r}")
    print(f"  target: {ex['target']!r}")


if __name__ == "__main__":
    main()
