import os
from datasets import Dataset, DatasetDict
from tqdm import tqdm

from kv_dataset_utils import generate_sequence, get_extra_chars, BASE_KV_ALPHABET


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
    
    Returns:
        DatasetDict with 'train', 'valid', and 'test' splits
    """
    kv_alphabet = BASE_KV_ALPHABET + get_extra_chars(kv_vocab_size)
    
    # Create training data
    train_data = []
    for _ in tqdm(range(train_samples), total=train_samples, desc="Creating train data"):
        sample = generate_sequence(num_kv_pairs, k_length, v_length, n_segments,
                                   min_segment_len, max_segment_len, kv_alphabet)
        train_data.append({
            'context': sample['context'],
            'query': sample['query'],
            'target': sample['target'],
        })
    train_dataset = Dataset.from_list(train_data)
    
    # Create validation data
    valid_data = []
    for _ in tqdm(range(valid_samples), total=valid_samples, desc="Creating valid data"):
        sample = generate_sequence(num_kv_pairs, k_length, v_length, n_segments,
                                   min_segment_len, max_segment_len, kv_alphabet)
        valid_data.append({
            'context': sample['context'],
            'query': sample['query'],
            'target': sample['target'],
        })
    valid_dataset = Dataset.from_list(valid_data)
    
    # Create test data
    test_data = []
    for _ in tqdm(range(test_samples), total=test_samples, desc="Creating test data"):
        sample = generate_sequence(num_kv_pairs, k_length, v_length, n_segments,
                                   min_segment_len, max_segment_len, kv_alphabet)
        test_data.append({
            'context': sample['context'],
            'query': sample['query'],
            'target': sample['target'],
        })
    test_dataset = Dataset.from_list(test_data)
    
    dataset = DatasetDict({
        'train': train_dataset,
        'valid': valid_dataset,
        'test': test_dataset
    })
    
    return dataset


def push_dataset_to_hub(
    dataset: DatasetDict,
    repo_id: str,
    num_kv_pairs: int,
    k_length: int = 2,
    v_length: int = 2,
    kv_vocab_size: int = 62,
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
    """
    config_name = f"N{num_kv_pairs}-K{k_length}V{v_length}-V{kv_vocab_size}"
    dataset.push_to_hub(repo_id, config_name=config_name)
    print(f"Successfully pushed dataset with config: {config_name}")


if __name__ == "__main__":
    # Hyperparameters
    num_kv_pairs_list = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    k_length = 2
    v_length = 2
    n_segments = 1
    min_segment_len = 0
    max_segment_len = 0
    kv_vocab_size = 62
    repo_id = 'irodkin/kv_retrieval'
    
    # Process each num_kv_pairs value
    for num_kv_pairs in num_kv_pairs_list:
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
        )
        
        # Push to hub
        push_dataset_to_hub(
            dataset=dataset,
            repo_id=repo_id,
            num_kv_pairs=num_kv_pairs,
            k_length=k_length,
            v_length=v_length,
            kv_vocab_size=kv_vocab_size,
        )
        
        print(f"Completed num_kv_pairs={num_kv_pairs}\n")

