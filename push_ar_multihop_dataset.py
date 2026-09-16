#!/usr/bin/env python3
"""Materialize and optionally upload the raw AR-multihop dataset."""

from __future__ import annotations

import argparse
import gc
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Sequence

from datasets import Dataset, DatasetDict, Features, Sequence as HFSequence, Value

from ar_multihop_dataset import (
    DEFAULT_H_VALUES,
    DEFAULT_N_VALUES,
    DEFAULT_SPLIT_SEEDS,
    DEFAULT_SPLIT_SIZES,
    ENTITY_ALPHABET_SIZE,
    configuration_name,
    decode_sample,
    entity_length_for_configuration,
    generate_split_records,
    validate_configuration,
    validate_sample,
)


DATASET_CARD_PATH = Path(__file__).with_name("ar_multihop_README.md")


def features_for_configuration(n_pairs: int, hop_length: int) -> Features:
    """Return fixed-shape Arrow features for one N/H configuration."""

    entity_length = entity_length_for_configuration(n_pairs, hop_length)
    entity = HFSequence(Value("int8"), length=entity_length)
    entities = HFSequence(entity, length=int(n_pairs))
    return Features(
        {
            "sample_id": Value("int64"),
            "context_keys": entities,
            "context_values": entities,
            "query_keys": entities,
            "targets": entities,
            "hop_distances": HFSequence(Value("int16"), length=int(n_pairs)),
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="irodkin/ar_multihop")
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
        default=Path("data/ar_multihop"),
        help="Preflight output and local-only full Arrow datasets.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/ar_multihop"),
        help="Incremental Hugging Face generation cache.",
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
        help="Generate only small local preflight datasets.",
    )
    mode.add_argument(
        "--push",
        action="store_true",
        help="Upload completed configurations directly from generation caches.",
    )
    mode.add_argument(
        "--sync-hub-metadata",
        action="store_true",
        help="Reconstruct the complete Hub configuration index from Parquet files.",
    )
    parser.add_argument("--preflight-samples", type=int, default=2)
    parser.add_argument("--examples-per-config", type=int, default=1)
    parser.add_argument("--max-decoded-items", type=int, default=12)
    parser.add_argument(
        "--overwrite-local",
        action="store_true",
        help="Replace exact local configuration directories that already exist.",
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

    if any(
        size < 1 for size in (args.train_size, args.validation_size, args.test_size)
    ):
        raise ValueError("split sizes must be positive")
    if args.preflight_samples < 1:
        raise ValueError("preflight_samples must be at least 1")
    if args.examples_per_config < 0:
        raise ValueError("examples_per_config must be non-negative")
    if args.max_decoded_items < 1:
        raise ValueError("max_decoded_items must be at least 1")
    if len({args.train_seed, args.validation_seed, args.test_seed}) != 3:
        raise ValueError("train, validation, and test seeds must be distinct")
    if ENTITY_ALPHABET_SIZE != 16:
        raise RuntimeError("ar_multihop configuration names require V=16")


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
    """Build memory-mapped Arrow splits incrementally from validated generators."""

    config_name = configuration_name(n_pairs, hop_length)
    features = features_for_configuration(n_pairs, hop_length)
    splits = {}
    for split_name in ("train", "validation", "test"):
        splits[split_name] = Dataset.from_generator(
            generate_split_records,
            features=features,
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
    expected_features = features_for_configuration(n_pairs, hop_length)
    for split_name, expected_size in split_sizes.items():
        split = dataset[split_name]
        if len(split) != expected_size:
            raise ValueError(
                f"{split_name} contains {len(split)} samples; expected {expected_size}"
            )
        if split.features != expected_features:
            raise ValueError(f"{split_name} has unexpected Arrow features")
        if validate_rows:
            for sample in split:
                validate_sample(sample, n_pairs, hop_length)


def remove_generation_cache(cache_path: Path) -> None:
    """Remove one cache only after another durable copy or upload exists."""

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
    """Generate then persist one config without retaining duplicate full copies."""

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
    # Rows were validated before reaching the Arrow writer.
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
                    f"\n[{config_name}] validated; existing preflight preserved at "
                    f"{destination}"
                )
            else:
                save_configuration(dataset, destination, overwrite=args.overwrite_local)
                print(f"\n[{config_name}] validated and saved to {destination}")
            for index in range(min(args.examples_per_config, len(dataset["train"]))):
                print(decode_sample(dataset["train"][index], args.max_decoded_items))
            del dataset
            gc.collect()
            remove_generation_cache(args.cache_dir / "preflight" / config_name)


def initialize_hub_repository(repo_id: str, *, api=None) -> None:
    """Create the repository without overwriting an existing metadata index."""

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
        commit_message="Add AR-multihop dataset card",
    )


def metadata_configs_from_repo_files(repo_files: Iterable[str]) -> Dict[str, object]:
    """Build config entries for every complete uploaded Parquet configuration."""

    ordered_names = [
        configuration_name(n_pairs, hop_length)
        for n_pairs in DEFAULT_N_VALUES
        for hop_length in DEFAULT_H_VALUES
        if n_pairs % hop_length == 0
    ]
    valid_names = set(ordered_names)
    splits_by_config = defaultdict(set)
    for repo_file in repo_files:
        parts = str(repo_file).split("/")
        if len(parts) != 2 or parts[0] not in valid_names:
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
        raise ValueError("no complete AR-multihop configurations found on the Hub")
    return {
        config_name: complete_configs[config_name]
        for config_name in ordered_names
        if config_name in complete_configs
    }


def sync_hub_metadata(repo_id: str, *, api=None) -> Sequence[str]:
    """Synchronize the card config index with all uploaded Parquet files."""

    from datasets.utils.metadata import MetadataConfigs
    from huggingface_hub import DatasetCard, HfApi

    api = HfApi() if api is None else api
    repo_files = api.list_repo_files(repo_id, repo_type="dataset")
    metadata_configs = metadata_configs_from_repo_files(repo_files)
    card_path = api.hf_hub_download(repo_id, "README.md", repo_type="dataset")
    dataset_card = DatasetCard.load(card_path)
    dataset_card.data.pop(MetadataConfigs.FIELD_NAME, None)
    MetadataConfigs(metadata_configs).to_dataset_card_data(dataset_card.data)
    api.upload_file(
        path_or_fileobj=str(dataset_card).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Synchronize AR-multihop configuration metadata",
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
