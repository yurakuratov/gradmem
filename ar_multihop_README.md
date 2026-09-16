---
pretty_name: AR Multihop Associative Retrieval
task_categories:
- question-answering
language:
- zxx
---

# AR Multihop Associative Retrieval

`irodkin/ar_multihop` is a deterministic synthetic associative-retrieval
dataset with multiple queries per context. Hub rows are raw structured integer
arrays; model-specific structural tokens and sequence formatting are added by a
training collator.

## Configurations

Configurations are named `N{N}-H{H}-V16` for every combination of:

- `N`: 8, 16, 32, 64, or 128 total key-value edges.
- `H`: 1, 2, 4, or 8 edges per chain; `H` must divide `N`.
- `V16`: one shared alphabet of exactly 16 ordinary entity symbols, numbered
  0 through 15. Structural model tokens are not part of this alphabet.

Each sample contains `N/H` chains. The number of distinct nodes is
`N + N/H`. Every node is represented by a fixed-width base-16 sequence. The
width is the smallest positive integer `L` satisfying
`16**L >= N + N/H`. The generator finds `L` with integer multiplication, so
exact powers of 16 are handled without floating-point rounding.

## Raw schema

For a configuration with entity width `L`, every row has:

| Field | Arrow shape | Meaning |
|---|---:|---|
| `sample_id` | scalar `int64` | Zero-based row ID within the split |
| `context_keys` | `[N, L]` `int8` | Keys in shuffled context-pair order |
| `context_values` | `[N, L]` `int8` | Values aligned with `context_keys` |
| `query_keys` | `[N, L]` `int8` | Every nonterminal node, independently shuffled |
| `targets` | `[N, L]` `int8` | Terminal entity aligned with each query |
| `hop_distances` | `[N]` `int16` | Remaining edges from query to terminal |

All entries in the four entity fields are raw symbols in `[0, 16)`. Rows do
not contain BOS, EOS, PAD, MASK, separators, boundaries, KV markers, query
markers, model token IDs, segmentation data, or other structural symbols.

## Chains and queries

The generator samples `N + N/H` unique integer node IDs without replacement
from `[0, 16**L)` and only then converts them to fixed-width base 16, retaining
leading zeros. It partitions these nodes into `N/H` disjoint acyclic chains of
`H+1` nodes. Consecutive nodes form the `H` directed edges in each chain.

Consequently, keys are unique, values are unique, internal values exactly equal
the key of the following edge, chains do not share nodes, and there are no
cycles, branches, or shortcut edges. All context pairs are shuffled while
preserving key/value alignment.

Every one of the `N` nonterminal nodes is queried exactly once. Its target is
the terminal node of its chain, and its `hop_distance` is the correct remaining
number of edges. Each distance from 1 through `H` occurs exactly `N/H` times.
Query order is shuffled with a deterministic random stream independent from the
context-pair shuffle, while query keys, targets, and distances stay aligned.

## Splits and determinism

| Split | Default rows | Default seed |
|---|---:|---:|
| train | 1,000,000 | 42 |
| validation | 5,000 | 43 |
| test | 10,000 | 44 |

Split sizes and seeds are configurable. Split name, seed, configuration, and
sample index enter a versioned SHA-256 seed derivation, giving deterministic
but distinct random streams.

## Model-side formatting

The dataset intentionally leaves sequence construction to the collator. The
provided EnergyGradMem runner maps each raw symbol into a disjoint model-entity
range and inserts distinct PAD, BOS, EOS, context, KV, query, and MASK tokens.
It formats KV records, segments only between complete records, and creates one
MASK prediction position per target digit. Ground-truth target digits are labels
only and are never inserted into query inputs.

## Generation and upload

`push_ar_multihop_dataset.py` validates every row before Arrow writing and uses
`Dataset.from_generator`, avoiding a million-row Python list. Every invocation
first generates small local preflight versions of the selected configurations
and prints decoded examples.

```bash
# Validate all configurations without full generation.
python push_ar_multihop_dataset.py --dry-run

# Materialize full datasets locally.
python push_ar_multihop_dataset.py --local-only

# Upload directly from completed generation caches.
python push_ar_multihop_dataset.py --push

# Rebuild the complete Hub configuration index from uploaded Parquet files.
python push_ar_multihop_dataset.py --sync-hub-metadata
```

Use `--n-values`, `--h-values`, split-size/seed options, `--output-dir`, and
`--cache-dir` to select or customize generation. Uploads use the user's existing
Hugging Face authentication. Existing cards are not overwritten, completed
caches are removed only after a successful save/upload, and failed persistence
keeps the only completed copy available for resume. Metadata synchronization
indexes all complete configurations found on the Hub, including those uploaded
by earlier partial invocations.
