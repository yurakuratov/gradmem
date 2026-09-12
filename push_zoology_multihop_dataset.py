#!/usr/bin/env python3
"""Materialize and optionally upload the zoology multihop dataset."""

from __future__ import annotations

import argparse
import gc
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Sequence

from datasets import Dataset, DatasetDict, Features, Sequence as HFSequence, Value

from zoology_multihop_dataset import (
    DEFAULT_H_VALUES,
    DEFAULT_N_VALUES,
    DEFAULT_SPLIT_SEEDS,
    DEFAULT_SPLIT_SIZES,
    VOCAB_SIZE,
    configuration_name,
    decode_sample,
    generate_split_records,
    validate_configuration,
    validate_sample,
)


FEATURES = Features(
    {
        "sample_id": Value("int64"),
        "context_input_ids": HFSequence(Value("int32")),
        "query_input_ids": HFSequence(Value("int32")),
        "targets": HFSequence(Value("int32")),
        "hop_distances": HFSequence(Value("int16")),
    }
)
DATASET_CARD_PATH = Path(__file__).with_name("zoology_multihop_README.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="irodkin/zoology_multihop")
    parser.add_argument(
        "--n-values",
        nargs="+",
        type=int,
        choices=DEFAULT_N_VALUES,
        default=list(DEFAULT_N_VALUES),
    )
    parser.add_argument(
        "--h-values",
        nargs="+",
        type=int,
        choices=DEFAULT_H_VALUES,
        default=list(DEFAULT_H_VALUES),
    )
    parser.add_argument("--train-size", type=int, default=DEFAULT_SPLIT_SIZES["train"])
    parser.add_argument(
        "--validation-size",
        type=int,
        default=DEFAULT_SPLIT_SIZES["validation"],
    )
    parser.add_argument("--test-size", type=int, default=DEFAULT_SPLIT_SIZES["test"])
    parser.add_argument("--train-seed", type=int, default=DEFAULT_SPLIT_SEEDS["train"])
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=DEFAULT_SPLIT_SEEDS["validation"],
    )
    parser.add_argument("--test-seed", type=int, default=DEFAULT_SPLIT_SEEDS["test"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/zoology_multihop"),
        help=(
            "Directory for preflight samples and, in local-only mode, saved "
            "full Arrow datasets."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/zoology_multihop"),
        help="Hugging Face generation cache directory.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--local-only",
        action="store_true",
        help="Materialize locally without uploading (the default mode).",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate only the small local preflight datasets.",
    )
    mode.add_argument(
        "--push",
        action="store_true",
        help=(
            "Upload each configuration directly from its completed generation cache "
            "using existing HF authentication."
        ),
    )
    mode.add_argument(
        "--sync-hub-metadata",
        action="store_true",
        help=(
            "Repair the Hub dataset-card configuration index from the Parquet "
            "files already present in the repository, without generating or "
            "uploading dataset shards."
        ),
    )
    parser.add_argument(
        "--preflight-samples",
        type=int,
        default=2,
        help="Samples per split and configuration in the mandatory preflight.",
    )
    parser.add_argument(
        "--examples-per-config",
        type=int,
        default=1,
        help="Number of decoded preflight training examples to print per configuration.",
    )
    parser.add_argument("--max-decoded-items", type=int, default=12)
    parser.add_argument(
        "--overwrite-local",
        action="store_true",
        help="Replace exact configuration directories that already exist locally.",
    )
    parser.add_argument("--max-shard-size", default="500MB")
    return parser.parse_args()


def _validated_unique(values: Iterable[int], name: str) -> Sequence[int]:
    result = tuple(dict.fromkeys(int(value) for value in values))
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def validate_cli_args(args: argparse.Namespace) -> None:
    args.n_values = _validated_unique(args.n_values, "n_values")
    args.h_values = _validated_unique(args.h_values, "h_values")
    for n_pairs in args.n_values:
        for hop_length in args.h_values:
            validate_configuration(n_pairs, hop_length)

    sizes = (args.train_size, args.validation_size, args.test_size)
    if any(size < 1 for size in sizes):
        raise ValueError("split sizes must be positive")
    if args.preflight_samples < 1:
        raise ValueError("preflight_samples must be at least 1")
    if args.examples_per_config < 0:
        raise ValueError("examples_per_config must be non-negative")
    if args.max_decoded_items < 1:
        raise ValueError("max_decoded_items must be at least 1")

    seeds = (args.train_seed, args.validation_seed, args.test_seed)
    if len(set(seeds)) != len(seeds):
        raise ValueError("train, validation, and test seeds must be distinct")
    if VOCAB_SIZE != 4096:
        raise RuntimeError("zoology_multihop requires V=4096")


def split_sizes_from_args(args: argparse.Namespace) -> Dict[str, int]:
    return {
        "train": args.train_size,
        "validation": args.validation_size,
        "test": args.test_size,
    }


def split_seeds_from_args(args: argparse.Namespace) -> Dict[str, int]:
    return {
        "train": args.train_seed,
        "validation": args.validation_seed,
        "test": args.test_seed,
    }


def materialize_configuration(
    n_pairs: int,
    hop_length: int,
    *,
    split_sizes: Dict[str, int],
    split_seeds: Dict[str, int],
    cache_dir: Path,
) -> DatasetDict:
    """Build memory-mapped Arrow splits incrementally from Python generators."""

    config_name = configuration_name(n_pairs, hop_length)
    splits = {}
    for split_name in ("train", "validation", "test"):
        splits[split_name] = Dataset.from_generator(
            generate_split_records,
            features=FEATURES,
            cache_dir=str(cache_dir / config_name / split_name),
            keep_in_memory=False,
            split=split_name,
            gen_kwargs={
                "n_pairs": n_pairs,
                "hop_length": hop_length,
                "num_samples": split_sizes[split_name],
                "split_seed": split_seeds[split_name],
                "split_name": split_name,
                "validate": True,
            },
        )
    return DatasetDict(splits)


def save_configuration(
    dataset: DatasetDict,
    destination: Path,
    *,
    overwrite: bool,
) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"local dataset already exists: {destination}; "
                "pass --overwrite-local to replace this exact configuration"
            )
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(destination))


def validate_materialized_dataset(
    dataset: DatasetDict,
    n_pairs: int,
    hop_length: int,
    split_sizes: Dict[str, int],
    *,
    validate_rows: bool = True,
) -> None:
    if set(dataset) != {"train", "validation", "test"}:
        raise ValueError(f"unexpected split names: {sorted(dataset)}")
    for split_name, expected_size in split_sizes.items():
        split = dataset[split_name]
        if len(split) != expected_size:
            raise ValueError(
                f"{split_name} contains {len(split)} samples; expected {expected_size}"
            )
        if validate_rows:
            for sample in split:
                validate_sample(sample, n_pairs, hop_length)


def remove_generation_cache(cache_path: Path) -> None:
    """Remove one exact configuration cache after another durable copy exists."""

    if cache_path.exists():
        shutil.rmtree(cache_path)


def process_full_configuration(
    n_pairs: int,
    hop_length: int,
    *,
    split_sizes: Dict[str, int],
    split_seeds: Dict[str, int],
    cache_dir: Path,
    output_dir: Path,
    overwrite_local: bool,
    push: bool,
    repo_id: str,
    max_shard_size: str,
) -> None:
    """Generate and persist one full configuration without keeping two copies.

    In push mode, the Hugging Face generation cache is uploaded directly and no
    full ``save_to_disk`` copy is made. In local mode, the saved dataset is the
    durable copy. The generation cache is removed only after the selected
    persistence operation has returned successfully, so a failed save or upload
    leaves the completed cache available for retry.
    """

    config_name = configuration_name(n_pairs, hop_length)
    configuration_cache = cache_dir / config_name
    print(f"\nMaterializing {config_name}")
    dataset = materialize_configuration(
        n_pairs,
        hop_length,
        split_sizes=split_sizes,
        split_seeds=split_seeds,
        cache_dir=cache_dir,
    )
    # Every row was validated inside generate_split_records before the Arrow
    # writer received it; avoid a second million-row pass here.
    validate_materialized_dataset(
        dataset,
        n_pairs,
        hop_length,
        split_sizes,
        validate_rows=False,
    )

    if push:
        dataset.push_to_hub(
            repo_id,
            config_name=config_name,
            max_shard_size=max_shard_size,
        )
        print(f"Uploaded {repo_id}/{config_name}")
    else:
        destination = output_dir / config_name
        save_configuration(dataset, destination, overwrite=overwrite_local)
        print(f"Saved completed configuration to {destination}")

    # Close memory-mapped Arrow handles before removing the now-redundant cache.
    del dataset
    gc.collect()
    remove_generation_cache(configuration_cache)


def run_preflight(args: argparse.Namespace) -> None:
    preflight_sizes = {
        split_name: args.preflight_samples
        for split_name in ("train", "validation", "test")
    }
    split_seeds = split_seeds_from_args(args)
    print(
        f"Preflight: {len(args.n_values) * len(args.h_values)} configurations, "
        f"{args.preflight_samples} samples per split"
    )
    for n_pairs in args.n_values:
        for hop_length in args.h_values:
            config_name = configuration_name(n_pairs, hop_length)
            dataset = materialize_configuration(
                n_pairs,
                hop_length,
                split_sizes=preflight_sizes,
                split_seeds=split_seeds,
                cache_dir=args.cache_dir / "preflight",
            )
            validate_materialized_dataset(
                dataset,
                n_pairs,
                hop_length,
                preflight_sizes,
            )
            destination = args.output_dir / "_preflight" / config_name
            if destination.exists() and not args.overwrite_local:
                print(
                    f"\n[{config_name}] validated; existing local preflight preserved at "
                    f"{destination}"
                )
            else:
                save_configuration(dataset, destination, overwrite=args.overwrite_local)
                print(f"\n[{config_name}] validated and saved to {destination}")
            for index in range(min(args.examples_per_config, len(dataset["train"]))):
                print(decode_sample(dataset["train"][index], args.max_decoded_items))
            del dataset
            gc.collect()
            remove_generation_cache(
                args.cache_dir / "preflight" / config_name
            )


def initialize_hub_repository(repo_id: str, *, api=None) -> None:
    """Create the repository and install the static card only when it is absent.

    ``DatasetDict.push_to_hub`` merges a newly uploaded configuration into the
    existing card metadata. Re-uploading the static local card on every process
    invocation would erase that accumulated metadata first.
    """

    from huggingface_hub import HfApi

    api = HfApi() if api is None else api
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    if api.file_exists(repo_id, "README.md", repo_type="dataset"):
        print(f"Preserving existing dataset card for {repo_id}")
        return
    api.upload_file(
        path_or_fileobj=str(DATASET_CARD_PATH),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add zoology multihop dataset card",
    )


def metadata_configs_from_repo_files(repo_files: Iterable[str]) -> Dict[str, object]:
    """Build Hub ``configs`` entries for complete uploaded configurations."""

    ordered_config_names = [
        configuration_name(n_pairs, hop_length)
        for n_pairs in DEFAULT_N_VALUES
        for hop_length in DEFAULT_H_VALUES
    ]
    valid_config_names = set(ordered_config_names)
    splits_by_config = defaultdict(set)
    for repo_file in repo_files:
        parts = str(repo_file).split("/")
        if len(parts) != 2 or parts[0] not in valid_config_names:
            continue
        config_name, filename = parts
        for split_name in ("train", "validation", "test"):
            if filename.startswith(f"{split_name}-") and filename.endswith(".parquet"):
                splits_by_config[config_name].add(split_name)

    required_splits = {"train", "validation", "test"}
    complete_configs = {
        config_name: {
            "data_files": [
                {"split": split_name, "path": f"{config_name}/{split_name}-*"}
                for split_name in ("train", "validation", "test")
            ]
        }
        for config_name, split_names in splits_by_config.items()
        if required_splits.issubset(split_names)
    }
    if not complete_configs:
        raise ValueError("no complete Zoology multihop configurations found on the Hub")
    return {
        config_name: complete_configs[config_name]
        for config_name in ordered_config_names
        if config_name in complete_configs
    }


def sync_hub_metadata(repo_id: str, *, api=None) -> Sequence[str]:
    """Synchronize the card's configuration index with uploaded Parquet files."""

    from datasets.utils.metadata import MetadataConfigs
    from huggingface_hub import DatasetCard, HfApi

    api = HfApi() if api is None else api
    repo_files = api.list_repo_files(repo_id, repo_type="dataset")
    metadata_configs = metadata_configs_from_repo_files(repo_files)
    card_path = api.hf_hub_download(
        repo_id,
        "README.md",
        repo_type="dataset",
    )
    dataset_card = DatasetCard.load(card_path)

    # Replace only the configuration index. Preserve descriptive card fields,
    # body text, and any dataset_info entries written by DatasetDict.push_to_hub.
    dataset_card.data.pop(MetadataConfigs.FIELD_NAME, None)
    MetadataConfigs(metadata_configs).to_dataset_card_data(dataset_card.data)
    api.upload_file(
        path_or_fileobj=str(dataset_card).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Synchronize zoology multihop configuration metadata",
    )
    config_names = tuple(metadata_configs)
    print(f"Indexed {len(config_names)} complete configurations in {repo_id}")
    return config_names


def main() -> None:
    args = parse_args()
    validate_cli_args(args)
    args.output_dir = args.output_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()

    if args.sync_hub_metadata:
        initialize_hub_repository(args.repo_id)
        sync_hub_metadata(args.repo_id)
        return

    run_preflight(args)
    if args.dry_run:
        print("\nDry run complete; full datasets were not generated or uploaded.")
        return

    if args.push:
        initialize_hub_repository(args.repo_id)

    split_sizes = split_sizes_from_args(args)
    split_seeds = split_seeds_from_args(args)
    for n_pairs in args.n_values:
        for hop_length in args.h_values:
            process_full_configuration(
                n_pairs,
                hop_length,
                split_sizes=split_sizes,
                split_seeds=split_seeds,
                cache_dir=args.cache_dir / "full",
                output_dir=args.output_dir,
                overwrite_local=args.overwrite_local,
                push=args.push,
                repo_id=args.repo_id,
                max_shard_size=args.max_shard_size,
            )

    if args.push:
        sync_hub_metadata(args.repo_id)


if __name__ == "__main__":
    main()
