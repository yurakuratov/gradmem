from datasets import Dataset, DatasetDict
from tqdm import tqdm
import gc 

from kv_dataset_utils import (
    BASE_KV_ALPHABET,
    ComplexityValueMapper,
    generate_sequence,
    get_extra_chars,
)


def create_complexity_function(complexity, v_length, kv_alphabet=BASE_KV_ALPHABET, seed=0):
    """Create a stateless, collision-free latent-to-visible value mapping."""
    return ComplexityValueMapper(
        complexity=complexity,
        v_length=v_length,
        alphabet=kv_alphabet,
        seed=seed,
    )


def _create_split(
    split_name,
    num_samples,
    num_kv_pairs,
    k_length,
    v_length,
    n_segments,
    min_segment_len,
    max_segment_len,
    kv_alphabet,
    complexity,
    complexity_function,
):
    data = []
    for _ in tqdm(range(num_samples), total=num_samples, desc=f"Creating {split_name} data"):
        sample = generate_sequence(
            num_kv_pairs,
            k_length,
            v_length,
            n_segments,
            min_segment_len,
            max_segment_len,
            kv_alphabet,
            complexity=complexity,
            complexity_function=complexity_function,
        )
        data.append(
            {
                "context": sample["context"],
                "query": sample["query"],
                "target": sample["target"],
            }
        )
    return Dataset.from_list(data)


def create_dataset(
    num_kv_pairs: int,
    k_length: int = 2,
    v_length: int = 2,
    n_segments: int = 1,
    min_segment_len: int = 0,
    max_segment_len: int = 0,
    kv_vocab_size: int = 62,
    train_samples: int = 1_000_000,
    valid_samples: int = 5_000,
    test_samples: int = 10_000,
    complexity: int = None,
    complexity_seed: int = 0,
) -> DatasetDict:
    """
    Create a dataset with train, valid, and test splits.
    
    Args:
        num_kv_pairs: Number of key-value pairs
        k_length: Length of each key
        v_length: Length of each value
        n_segments: Number of segments/messages in the sequence
        min_segment_len: Minimum length of each segment
        max_segment_len: Maximum length of each segment
        kv_vocab_size: Vocabulary size for KV alphabet
        train_samples: Number of training samples
        valid_samples: Number of validation samples
        test_samples: Number of test samples
        complexity: Number of latent alphabet characters represented by each value
        complexity_seed: Seed defining the latent-to-visible value permutation
    
    Returns:
        DatasetDict with 'train', 'valid', and 'test' splits
    """
    kv_alphabet = BASE_KV_ALPHABET + get_extra_chars(kv_vocab_size)
    
    complexity_function = None
    if complexity is not None and complexity < v_length:
        complexity_function = create_complexity_function(
            complexity,
            v_length,
            kv_alphabet=kv_alphabet,
            seed=complexity_seed,
        )

    split_sizes = {
        "train": train_samples,
        "valid": valid_samples,
        "test": test_samples,
    }
    return DatasetDict(
        {
            split_name: _create_split(
                split_name,
                num_samples,
                num_kv_pairs,
                k_length,
                v_length,
                n_segments,
                min_segment_len,
                max_segment_len,
                kv_alphabet,
                complexity,
                complexity_function,
            )
            for split_name, num_samples in split_sizes.items()
        }
    )


def push_dataset_to_hub(
    dataset: DatasetDict,
    repo_id: str,
    num_kv_pairs: int,
    k_length: int = 2,
    v_length: int = 2,
    kv_vocab_size: int = 62,
    complexity: int = None,
) -> None:
    """
    Push a dataset to HuggingFace Hub.
    
    Args:
        dataset: DatasetDict to push
        repo_id: Repository ID on HuggingFace Hub (e.g., 'irodkin/kv_retrieval')
        num_kv_pairs: Number of key-value pairs
        k_length: Length of each key
        v_length: Length of each value
        kv_vocab_size: Vocabulary size for KV alphabet
        complexity: Latent value length, when complexity control is enabled
    """
    complexity_suffix = "" if complexity is None else f"C{complexity}"
    config_name = f"N{num_kv_pairs}-K{k_length}V{v_length}{complexity_suffix}-V{kv_vocab_size}"
    dataset.push_to_hub(repo_id, config_name=config_name)
    print(f"Successfully pushed dataset with config: {config_name}")


if __name__ == "__main__":
    # Hyperparameters
    num_kv_pairs_list = [8, 16, 32, 64, 128]
    k_length = 2
    v_length = 6

    complexities = [1, 2, 3, 6]
    complexity_seed = 0
    n_segments = 1
    min_segment_len = 0
    max_segment_len = 0
    kv_vocab_size = 62
    repo_id = 'irodkin/kv_retrieval'
    
    # Process each num_kv_pairs value
    for num_kv_pairs in num_kv_pairs_list:
        for complexity in complexities:
            print(f"\n{'='*60}")
            print(f"Processing num_kv_pairs={num_kv_pairs}")
            print(f"{'='*60}\n")
            
            # Create dataset
            dataset = create_dataset(
                num_kv_pairs=num_kv_pairs,
                k_length=k_length,
                v_length=v_length,
                n_segments=n_segments,
                min_segment_len=min_segment_len,
                max_segment_len=max_segment_len,
                kv_vocab_size=kv_vocab_size,
                complexity=complexity,
                complexity_seed=complexity_seed,
            )
            # Push to hub
            push_dataset_to_hub(
                dataset=dataset,
                repo_id=repo_id,
                num_kv_pairs=num_kv_pairs,
                k_length=k_length,
                v_length=v_length,
                kv_vocab_size=kv_vocab_size,
                complexity=complexity,
            )
            del dataset
            gc.collect()
            
            print(f"Completed num_kv_pairs={num_kv_pairs}\n")
