import argparse
import json
import shutil
from pathlib import Path

import datasets

from zoology_mqar_data import build_mqar_datasets


DEFAULT_PAIR_COUNTS = (8, 16, 32, 64)
DEFAULT_NOISE_LEVELS = (0.0,)


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare deterministic dense MQAR datasets.')
    parser.add_argument('--output_dir', default='./data')
    parser.add_argument('--pair_counts', nargs='+', type=int, default=DEFAULT_PAIR_COUNTS)
    parser.add_argument('--noise_levels', nargs='+', type=float, default=DEFAULT_NOISE_LEVELS)
    parser.add_argument('--vocab_size', type=int, default=8192)
    parser.add_argument('--train_num_examples', type=int, default=1_000_000)
    parser.add_argument('--valid_num_examples', type=int, default=5_000)
    parser.add_argument('--data_seed', type=int, default=123)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def save_dataset(*, output_path, train_dataset, valid_dataset, metadata, overwrite):
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f'{output_path} already exists; pass --overwrite to replace it.'
            )
        shutil.rmtree(output_path)

    dataset_dict = datasets.DatasetDict({
        'train': datasets.Dataset.from_dict({
            'input_ids': train_dataset.inputs.numpy(),
            'labels': train_dataset.labels.numpy(),
        }),
        'valid': datasets.Dataset.from_dict({
            'input_ids': valid_dataset.inputs.numpy(),
            'labels': valid_dataset.labels.numpy(),
        }),
    })
    dataset_dict.save_to_disk(str(output_path))
    with (output_path / 'mqar_metadata.json').open('w') as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


def main():
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for num_kv_pairs in args.pair_counts:
        input_seq_len = 5 * num_kv_pairs
        for noise_level in args.noise_levels:
            train_dataset, valid_dataset, train_seed, valid_seed = build_mqar_datasets(
                vocab_size=args.vocab_size,
                input_seq_len=input_seq_len,
                num_kv_pairs=num_kv_pairs,
                train_num_examples=args.train_num_examples,
                valid_num_examples=args.valid_num_examples,
                power_a=0.01,
                random_non_queries=False,
                data_seed=args.data_seed,
                dense_queries=True,
                query_sampling='uniform',
                mqar_noise_lvl=noise_level,
            )
            noise_suffix = '' if noise_level == 0.0 else f'_noise{noise_level:g}'
            output_path = output_root / (
                f'mqar_N{num_kv_pairs}_V{args.vocab_size}_L{input_seq_len}{noise_suffix}'
            )
            metadata = {
                'schema_version': 2,
                'task': 'dense uniform MQAR',
                'vocab_size': args.vocab_size,
                'num_kv_pairs': num_kv_pairs,
                'input_seq_len': int(train_dataset.inputs.shape[1]),
                'train_num_examples': args.train_num_examples,
                'valid_num_examples': args.valid_num_examples,
                'data_seed': args.data_seed,
                'train_data_seed': train_seed,
                'valid_data_seed': valid_seed,
                'dense_queries': True,
                'query_sampling': 'uniform',
                'query_distribution': train_dataset.slices.get('query_distribution'),
                'data_generator': train_dataset.slices.get('generator'),
                'power_a': 0.01,
                'random_non_queries': False,
                'mqar_noise_lvl': noise_level,
                'pair_open_token': 0,
                'pair_close_token': 1,
                'context_size': train_dataset.slices['context_size'],
                'query_size': train_dataset.inputs.shape[1] - train_dataset.slices['context_size'],
            }
            print(f'Saving {output_path}', flush=True)
            save_dataset(
                output_path=output_path,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                metadata=metadata,
                overwrite=args.overwrite,
            )


if __name__ == '__main__':
    main()
