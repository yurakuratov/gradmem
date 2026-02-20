import argparse
import os
import random
import shutil
from functools import partial

import datasets
from tqdm import tqdm

import nltk


def _ensure_punkt():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _pack_sentences(text, spans, min_words, drop_short_last=True):
    chunks = []
    cur_spans = []
    cur_words = 0

    def _flush_current():
        nonlocal cur_spans, cur_words
        if not cur_spans:
            return
        chunk_start = cur_spans[0][0]
        chunk_end = cur_spans[-1][1]
        context = text[chunk_start:chunk_end]
        word_count = len(context.split())
        sentence_boundaries = [[start - chunk_start, end - chunk_start] for start, end in cur_spans]
        chunks.append(
            {
                "context": context,
                "word_count": word_count,
                "n_sentences": len(cur_spans),
                "char_start": int(chunk_start),
                "char_end": int(chunk_end),
                "sentence_boundaries": sentence_boundaries,
            }
        )
        cur_spans = []
        cur_words = 0

    for start, end in spans:
        sent = text[start:end].strip()
        if not sent:
            continue
        sent_words = len(sent.split())
        if sent_words == 0:
            continue

        # Add sentence to current chunk
        cur_spans.append((start, end))
        cur_words += sent_words

        # Check if we've reached min_words - if so, flush immediately
        if cur_words >= min_words:
            _flush_current()

    # Handle remaining sentences
    if cur_spans:
        if cur_words >= min_words:
            _flush_current()
        elif not drop_short_last:
            _flush_current()

    return chunks


def iter_validation_sample(args):
    # Set random seed for reproducible sampling
    if args.seed is not None:
        random.seed(args.seed)

    ds = datasets.load_dataset(args.dataset_name, split="validation")
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

    books = []
    book_pbar = tqdm(desc="Loading validation books", unit="book")
    for idx, item in enumerate(ds):
        text = item.get(args.text_field, "")
        text = _normalize_text(text)
        if not text:
            book_pbar.update(1)
            continue
        spans = list(tokenizer.span_tokenize(text))
        if not spans:
            book_pbar.update(1)
            continue
        book_id = item.get("book_id", idx)
        books.append({"text": text, "spans": spans, "book_id": str(book_id)})
        book_pbar.update(1)
    book_pbar.close()

    if not books:
        raise ValueError("No valid validation books found.")

    chunk_pbar = tqdm(total=args.sample_n_valid_chunks, desc="Sampling validation chunks", unit="chunk")
    chunk_id = 0
    while chunk_id < args.sample_n_valid_chunks:
        made_progress = False
        for book in books:
            if chunk_id >= args.sample_n_valid_chunks:
                break
            spans = book["spans"]
            if not spans:
                continue
            start_idx = random.randrange(len(spans))
            chunk_spans = []
            cur_words = 0
            for start, end in spans[start_idx:]:
                sent = book["text"][start:end].strip()
                if not sent:
                    continue
                sent_words = len(sent.split())
                if sent_words == 0:
                    continue
                chunk_spans.append((start, end))
                cur_words += sent_words
                if cur_words >= args.min_words:
                    break
            if cur_words < args.min_words or not chunk_spans:
                continue

            chunk_start = chunk_spans[0][0]
            chunk_end = chunk_spans[-1][1]
            context = book["text"][chunk_start:chunk_end]
            word_count = len(context.split())
            sentence_boundaries = [[start - chunk_start, end - chunk_start] for start, end in chunk_spans]

            yield {
                "context": context,
                "word_count": int(word_count),
                "n_sentences": int(len(chunk_spans)),
                "char_start": int(chunk_start),
                "char_end": int(chunk_end),
                "sentence_boundaries": sentence_boundaries,
                "book_id": book["book_id"],
                "chunk_id": int(chunk_id),
                "split": "validation",
            }
            chunk_id += 1
            chunk_pbar.update(1)
            made_progress = True

        if not made_progress:
            chunk_pbar.close()
            raise ValueError("Unable to sample any validation chunks meeting min_words.")

    chunk_pbar.close()


def iter_chunks(split, args):
    ds = datasets.load_dataset(args.dataset_name, split=split)
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

    book_pbar = tqdm(desc=f"Processing {split} books", unit="book")

    total_chunks = 0
    for idx, item in enumerate(ds):
        text = item.get(args.text_field, "")
        text = _normalize_text(text)
        if not text:
            book_pbar.update(1)
            continue
        spans = list(tokenizer.span_tokenize(text))
        chunks = _pack_sentences(
            text,
            spans,
            min_words=args.min_words,
            drop_short_last=args.drop_short_last,
        )
        book_id = item.get("book_id", idx)
        for chunk_id, chunk in enumerate(chunks):
            total_chunks += 1
            yield {
                "context": chunk["context"],
                "word_count": int(chunk["word_count"]),
                "n_sentences": int(chunk["n_sentences"]),
                "char_start": int(chunk["char_start"]),
                "char_end": int(chunk["char_end"]),
                "sentence_boundaries": chunk["sentence_boundaries"],
                "book_id": str(book_id),
                "chunk_id": int(chunk_id),
                "split": split,
            }

        book_pbar.set_postfix({"chunks": total_chunks})
        book_pbar.update(1)

    book_pbar.close()


def main():
    parser = argparse.ArgumentParser(description="Prepare PG19 sentence-aligned chunks by word count.")
    parser.add_argument("--dataset_name", type=str, default="deepmind/pg19")
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--splits", type=str, default="train,validation,test")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--min_words", type=int, required=True)
    parser.add_argument("--keep_short_last", action="store_true")
    parser.add_argument("--sample_n_valid_chunks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.min_words <= 0:
        raise ValueError("min_words must be a positive integer.")

    if args.sample_n_valid_chunks is not None and args.sample_n_valid_chunks <= 0:
        raise ValueError("sample_n_valid_chunks must be a positive integer.")

    if os.path.exists(args.output_path):
        if not args.overwrite:
            raise FileExistsError(f"Output path exists: {args.output_path}. Use --overwrite to replace.")
        shutil.rmtree(args.output_path)

    _ensure_punkt()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    drop_short_last = not args.keep_short_last
    features = datasets.Features(
        {
            "context": datasets.Value("string"),
            "word_count": datasets.Value("int32"),
            "n_sentences": datasets.Value("int32"),
            "book_id": datasets.Value("string"),
            "chunk_id": datasets.Value("int32"),
            "split": datasets.Value("string"),
            "char_start": datasets.Value("int32"),
            "char_end": datasets.Value("int32"),
            "sentence_boundaries": datasets.Sequence(datasets.Sequence(datasets.Value("int32"))),
        }
    )

    split_datasets = {}

    # Progress bar for splits
    split_pbar = tqdm(splits, desc="Processing splits", unit="split")

    for split in split_pbar:
        split_pbar.set_description(f"Processing {split} split")
        args.drop_short_last = drop_short_last
        if split == "validation" and args.sample_n_valid_chunks is not None:
            gen = partial(iter_validation_sample, args)
        else:
            gen = partial(iter_chunks, split, args)
        split_datasets[split] = datasets.Dataset.from_generator(gen, features=features)

    split_pbar.close()

    print("Saving dataset to disk...")
    out = datasets.DatasetDict(split_datasets)
    out.save_to_disk(args.output_path)
    print(f"Dataset saved to {args.output_path}")


if __name__ == "__main__":
    main()
