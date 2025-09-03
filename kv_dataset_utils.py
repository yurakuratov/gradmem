import random
import string
import unicodedata as ud

from torch.utils.data import Dataset

from tokenizers import Tokenizer, Regex
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers.decoders import BPEDecoder
from transformers import PreTrainedTokenizerFast

# Define alphabets for character generation
STRUCT = "!?:|"  # keep these reserved for syntax
BASE_KV_ALPHABET = string.ascii_letters + string.digits

# Add pool of extra chars to choose from (if vocab size > 62 is needed)
RANGES = [
    (0x2460, 0x2473+1),    # Enclosed Alphanumerics
    (0x24B6, 0x24F4+1),    # Enclosed Alphanumerics
    (0x1F600, 0x1F64F+1),  # Emoticons
    (0x0391, 0x03A9+1),    # Greek Uppercase
    (0x03B1, 0x03C9+1),    # Greek Lowercase
    (0x0410, 0x044F+1),    # Cyrillic basic (Ѐ..џ)
    (0x00A1, 0x0100),      # Latin-1 Supplement (¡..ÿ)
    (0x25A0, 0x2600),      # Geometric shapes
    (0x2500, 0x2580),      # Box drawing
]


def generate_sequence(num_kv_pairs=3, k_length=4, v_length=4, n_segments=4,
                      min_segment_len=32, max_segment_len=64, kv_alphabet=BASE_KV_ALPHABET):
    """
    Generate a sequence with random text, key-value pairs, and a query.

    This sequence emulates chat with each message length between min_segment_len and max_segment_len.
    Each message ends with '|'.
    Each message can contain random sequence of characters along with key-value pairs in format !K:V!.
    Total number of key-value pairs in the full sequence (all messages/segments) is set by num_kv_pairs.
    The last segment requests one of the previous values by key: ?!K:

    ...!K:V!...|?!K_i:V_i!|
    ..context..|query:target|

    All keys are unique. Values can be repeated.

    Example representation of sequence that consists of 3 content segments and 4th segment with question/request:
    1: random_seq!K1:V1!random_seq!K2:V2!random_seq|
    2: random_seq|
    3: !K3:V3!random_seq!K4:V4!random_seq|
    4: ?!K2:

    Args:
        num_kv_pairs: Number of key-value pairs to include
        k_length: Length of each key
        v_length: Length of each value
        n_segments: Number of segments/messages in the sequence
        min_segment_len: Minimum length of each segment
        max_segment_len: Maximum length of each segment

    Returns:
        Dictionary containing:
        - kv_pairs: List of key-value pairs in format !K:V!
        - segment_ids_to_kv_ids: Mapping from segment IDs to key-value pair indices
        - context: Complete context string (all segments concatenated)
        - query: Query string in format ?!K:
        - input_sequence: Complete sequence string (context + query)
        - target: Target value for the query in format V!|
    """
    # generate unique keys and values
    keys = []
    values = []
    for _ in range(num_kv_pairs):
        while True:
            key = ''.join(random.choice(kv_alphabet) for _ in range(k_length))
            if key not in keys:
                break
        value = ''.join(random.choice(kv_alphabet) for _ in range(v_length))
        keys.append(key)
        values.append(value)
    kv_pairs_dict = dict(zip(keys, values))
    kv_pairs = [f'!{k}:{v}!' for k, v in kv_pairs_dict.items()]

    # distribute facts by segments
    segments_ids = list(range(n_segments))
    segments_ids_with_kv_pairs = random.choices(segments_ids, k=len(kv_pairs))
    # create a mapping from segment_id to list of fact indices
    # [1, 1, 0] -> {0: [2], 1: [0, 1], 2: [], ...}
    # [0, 3, 0] -> {0: [0, 2], 1: [], 2: [], 3: [1], ...}
    segment_ids_to_kv_ids = {seg_id: [] for seg_id in segments_ids}
    for kv_idx, seg_id in enumerate(segments_ids_with_kv_pairs):
        segment_ids_to_kv_ids[seg_id].append(kv_idx)

    segments = []
    for i in range(n_segments):
        kv_pairs_n_chars = sum([len(kv_pairs[kv_pair_idx]) for kv_pair_idx in segment_ids_to_kv_ids[i]])
        min_n_random_chars = max(min_segment_len - kv_pairs_n_chars, 0)
        max_n_random_chars = max(max_segment_len - kv_pairs_n_chars, 0)
        random_chars = ''.join(random.choice(kv_alphabet) for _ in range(random.randint(min_n_random_chars,
                                                                                        max_n_random_chars)))
        # insert kv pairs into random places in the segment
        # first, determine all insertion positions
        kv_pairs_idxs = segment_ids_to_kv_ids[i]
        insertion_positions = []
        for _ in range(len(kv_pairs_idxs)):
            # generate a random position in the current random_chars string
            pos = random.randint(0, len(random_chars))
            insertion_positions += [pos]

        # sort positions in descending order to avoid shifting issues
        insertion_positions.sort(reverse=True)

        # insert kv_pairs at the predetermined positions
        for pos, kv_idx in zip(insertion_positions, kv_pairs_idxs):
            random_chars = random_chars[:pos] + kv_pairs[kv_idx] + random_chars[pos:]
        segments += [random_chars + '|']
    context = ''.join(segments)
    if num_kv_pairs > 0:
        # sample random k for query:
        k_for_query = random.choice(keys)
        query = f'?!{k_for_query}:'
        target = f'{kv_pairs_dict[k_for_query]}!|'
    else:
        query = '?!:'
        target = ''
    input_sequence = context + query
    return {'kv_pairs': kv_pairs, 'segment_ids_to_kv_ids': segment_ids_to_kv_ids,
            'context': context, 'query': query, 'input_sequence': input_sequence, 'target': target}


class KVDataset(Dataset):
    def __init__(self, num_samples, **gen_params):
        self.samples = [
            generate_sequence(**gen_params)
            for _ in range(num_samples)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_seq = sample['input_sequence']
        target_seq = sample['target']

        return {
            'input_seq': input_seq,
            'target_seq': target_seq,
        }


def is_extra_char(ch: str) -> bool:
    if ch in STRUCT:
        # keep reserved out of the general pool
        return False
    if ch in BASE_KV_ALPHABET:
        return False
    if not ch.isprintable():
        return False
    cat = ud.category(ch)
    # exclude controls/marks/spacing
    if cat[0] in {"C", "M", "Z"}:
        return False
    return True


def get_extra_chars(vocab_size: int):
    if len(BASE_KV_ALPHABET) < vocab_size:
        extra = ''.join(ch for a, b in RANGES for ch in map(chr, range(a, b)) if is_extra_char(ch))
        return extra[:vocab_size-len(BASE_KV_ALPHABET)]
    return ''


def verify_char_vocab_and_tokenizer_compatibility(tokenizer, vocab):
    # just to make sure that all chars in vocab are supported by the tokenizer
    for ch in vocab:
        ids = tokenizer(ch, add_special_tokens=False).input_ids
        assert len(ids) == 1, f"{repr(ch)} split into {ids}"
        back = tokenizer.decode(ids)
        assert back == ch, f"tokenizer check mismatch: {repr(ch)} -> {back}"
    return True


def create_tokenizer(vocab_size: int = 62):
    # Create character tokenizer
    extra_kv_chars = get_extra_chars(vocab_size=vocab_size)
    chars = BASE_KV_ALPHABET + STRUCT + extra_kv_chars
    special = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2, '[UNK]': 3}
    vocab = {ch: i + len(special) for i, ch in enumerate(chars)}
    vocab.update(special)

    tokenizer = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Split(Regex(r'.'), behavior="isolated", invert=True)
    # to remove spaces between tokens
    tokenizer.decoder = BPEDecoder(' ')

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                        pad_token='[PAD]', eos_token='[EOS]',
                                        bos_token='[BOS]', unk_token='[UNK]'
                                        )

    if verify_char_vocab_and_tokenizer_compatibility(tokenizer, chars):
        return tokenizer
