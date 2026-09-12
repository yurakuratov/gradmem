---
pretty_name: Zoology Multihop Associative Retrieval
task_categories:
- question-answering
language:
- zxx
---

# Zoology Multihop Associative Retrieval

`irodkin/zoology_multihop` is a deterministic synthetic associative-retrieval
dataset with multiple queries per context. It uses integer token arrays rather
than natural-language text.

## Configurations

Every configuration is named `N{N}-H{H}-V4096` and uses one of:

- `N`: 8, 16, 32, 64, or 128 total key-value edges.
- `H`: 1, 2, 4, or 8 edges per chain.
- `V`: exactly 4,096 total tokens.

The complete release contains all 20 combinations. A sample has `N/H`
node-disjoint chains. `H` is both the chain length and the maximum required
query hop distance.

## Vocabulary allocation

The complete vocabulary has exactly 4,096 IDs. IDs 0–9 are structural:

| ID | Name | Meaning |
|---:|---|---|
| 0 | `PAD` | Batch padding |
| 1 | `BOS` | Sequence beginning |
| 2 | `EOS` | Sequence end |
| 3 | `CONTEXT_START` | Context boundary |
| 4 | `CONTEXT_END` | Context boundary |
| 5 | `KV_START` | KV-record beginning |
| 6 | `KV_SEPARATOR` | Separates a key from its value |
| 7 | `KV_END` | KV-record end |
| 8 | `QUERY_START` | Query-record beginning |
| 9 | `QUERY_END` | Query-record end |

IDs 10–4095 form one shared, unpartitioned pool of 4,086 entity tokens. Keys,
values, and targets are sampled from this same pool; there are no key-only or
value-only sub-vocabularies.

## Integer encoding and schema

Each KV record has fixed width five:

```text
[KV_START, key, KV_SEPARATOR, value, KV_END]
```

`context_input_ids` is one flat sequence:

```text
[BOS, CONTEXT_START, KV record 1, ..., KV record N, CONTEXT_END, EOS]
```

Each query record has fixed width three and contains no target:

```text
[QUERY_START, key, QUERY_END]
```

`query_input_ids` is a flat query collection:

```text
[BOS, query record 1, ..., query record N, EOS]
```

The Arrow schema is:

| Field | Type | Length | Meaning |
|---|---|---:|---|
| `sample_id` | `int64` | 1 | Zero-based row ID within the split |
| `context_input_ids` | `list<int32>` | `5N + 4` | Flat shuffled KV context |
| `query_input_ids` | `list<int32>` | `3N + 2` | Flat independently shuffled query records |
| `targets` | `list<int32>` | `N` | Terminal entity aligned with each query |
| `hop_distances` | `list<int16>` | `N` | Required number of mapping traversals |

A collator can split `query_input_ids[1:-1]` into width-three records and pair
their middle tokens with `targets` and `hop_distances`. Targets are never part
of `query_input_ids`, preventing teacher-forced leakage between queries.

The dataset deliberately contains no segmentation fields or metadata. Model
code may segment the flat context later, provided a model boundary never splits
a width-five KV record.

## Chain and query generation

For each sample, the generator samples `N + N/H` entity IDs without
replacement and divides them into `N/H` chains of `H+1` nodes. Consecutive
nodes produce the `H` directed KV edges in each chain. Consequently:

- every key and every value is unique;
- internal nodes occur once as a value and once as the next key;
- chains are acyclic and node-disjoint;
- there are no branches, shortcuts, duplicate mappings, or cycles.

All `N` KV records are shuffled before context encoding. Every nonterminal node
is queried exactly once, including chain starts and intermediate nodes. Query
order uses a random stream independent from context order. Each query targets
its chain's terminal node. Every effective depth from 1 through `H` occurs
exactly `N/H` times.

## Splits and determinism

The pregenerated split defaults match the original KV-retrieval uploader:

| Split | Default rows | Default seed |
|---|---:|---:|
| train | 1,000,000 | 42 |
| validation | 5,000 | 43 |
| test | 10,000 | 44 |

Split sizes and seeds are CLI-configurable. Split name, split seed,
configuration, and sample index are combined with a versioned SHA-256 seed
derivation, giving each split a distinct deterministic random stream.

## Generation

The standalone pipeline is `push_zoology_multihop_dataset.py`. It first creates
and validates small local Arrow datasets for every selected configuration and
prints decoded examples. `--dry-run` stops after this preflight,
`--local-only` materializes the requested full local datasets, and `--push`
uploads completed configurations directly from the generation cache using the
user's existing Hugging Face authentication. Local-only mode removes each
configuration's generation cache only after `save_to_disk()` succeeds. Push
mode does not create a second full copy under `--output-dir`; it removes the
generation cache only after the upload succeeds. A failed save or upload keeps
the completed cache for a resumable retry.

```bash
# Validate and inspect small Arrow versions of all 20 configurations.
python push_zoology_multihop_dataset.py --dry-run

# Materialize the complete release locally without network writes.
python push_zoology_multihop_dataset.py --local-only

# Generate and upload directly, without retaining a second full local copy.
python push_zoology_multihop_dataset.py --push

# Rebuild the Hub configuration index from shards that are already uploaded.
# This does not regenerate or re-upload any dataset data.
python push_zoology_multihop_dataset.py --sync-hub-metadata
```

Use `--n-values` and `--h-values` for a subset, and use the split-size,
split-seed, `--output-dir`, and `--cache-dir` options to override generation
defaults. Separate partial `--push` invocations preserve and extend the existing
configuration index. No access token is accepted on the command line; uploads
use the existing Hugging Face login.
