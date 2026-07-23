import pandas as pd
import numpy as np
import re
import nltk
from torch.utils.data import Dataset


def compare_answers(target, output):
    target = target.lower()
    output = output.lower()
    # take only the first sentence from output
    output = output.split('.')[0]
    # filter responses when model tries to generate examples
    output = output.split('<context>')[0]
    output = output.split('<example>')[0]

    # we consider prediction correct if target is in output
    if target in output:
        return True

    return False


# preprocess babi text files
def get_dataset_df(dataset_path, max_n_facts=None):
    with open(dataset_path, 'r') as f:
        texts = f.read().strip()
        texts = texts.split('\n')
        df = pd.DataFrame(texts, columns=['text'])

    # parse samples
    df['phrase_num'] = df.text.apply(lambda x: int(x.split(' ')[0]))
    df.text = df.text.apply(lambda x: x[x.index(' ') + 1:])
    df['answer'] = df.text.apply(lambda x: x[x.index('\t') + 1:] if '\t' in x else None)
    df['reference_num'] = df.answer.apply(lambda x: x if pd.isna(x) else [int(n) for n in re.split('\t| ', x)[1:]])
    df.answer = df.answer.apply(lambda x: x if pd.isna(x) else x.split('\t')[0])
    df.text = df.text.apply(lambda x: x.split('\t')[0] if '\t' in x else x)

    # mark each sample
    sample_start_inds = list(np.where(df.phrase_num == 1)[0]) + [df.shape[0]]
    for i, (start, end) in enumerate(zip(sample_start_inds, sample_start_inds[1:])):
        df.loc[start:end, 'initial_sample_num'] = i

    df.initial_sample_num = df.initial_sample_num.astype(int)

    # multiple questions in sample -> samples with single question
    initial_samples = [df[df.initial_sample_num == sn] for sn in df.initial_sample_num.unique()]

    single_question_slices = []
    for sample in initial_samples:
        answer_positions = sample[~sample.answer.isna()].index
        slices = [sample.loc[:ans_pos].copy() for ans_pos in answer_positions]
        for i, slc in enumerate(slices):
            slices[i] = slc[(slc.answer.isna()) | (slc.index == slc.index[-1])]
        if max_n_facts is not None:             # drop samples with too many facts
            slices = [slc for slc in slices if slc.shape[0] <= max_n_facts]
        single_question_slices += slices

    df = pd.concat(single_question_slices).reset_index(drop=True)

    # mark each sample again
    sample_start_inds = list(np.where(df.phrase_num == 1)[0]) + [df.shape[0]]
    for i, (start, end) in enumerate(zip(sample_start_inds, sample_start_inds[1:])):
        df.loc[start:end, 'sample_num'] = i

    df.sample_num = df.sample_num.astype(int)

    return df


# babi task loader dataset
class TaskDataset(Dataset):
    def __init__(self, dataset_path, max_n_facts=None):
        self.fact_dataset = get_dataset_df(dataset_path, max_n_facts=max_n_facts)

    def __getitem__(self, ind):
        slc = self.fact_dataset[self.fact_dataset.sample_num == ind]
        references = slc[slc.phrase_num.isin(slc.reference_num.values[-1])].text.values
        sample = {'facts': slc.text.values[:-1],
                  'question': slc.text.values[-1],
                  'answer': slc.answer.values[-1],
                  'references': references}
        return sample

    def __len__(self):
        return self.fact_dataset.sample_num.max()


def sum_lengths(sentences):
    return sum([len(s) for s in sentences])


# sampler of background text
class SentenceSampler:
    def __init__(self, dataset, tokenizer, min_sentence_len=10, max_sentence_len=None, shuffle=False, random_seed=42):
        self.sample_ind = 0
        self.dataset = dataset
        self.sentences = []
        self.tokenizer = tokenizer
        self.min_sentence_len = min_sentence_len
        self.max_sentence_len = max_sentence_len
        self.sentence_tokenizer = nltk.PunktSentenceTokenizer()
        self.shuffle = shuffle
        self.gen = np.random.default_rng(seed=random_seed)

    def get_sample(self, sample_size):
        sample = []
        total_len = 0
        while True:
            sentences = list(self.sentences)
            for i, sent in enumerate(sentences):  # add new sentence until sample_size is reached
                tokenized = self.tokenizer.encode(' ' + sent, add_special_tokens=False)
                if not self.length_is_ok(tokenized):
                    continue
                total_len += len(tokenized)
                sample.append(tokenized)
                if total_len >= sample_size:
                    self.sentences = self.sentences[i+1:]
                    cutoff = total_len - sample_size
                    if cutoff > 0:
                        sample[-1] = sample[-1][:-cutoff]
                    return sample

            self.sentences = []
            self.sample_sentences_(sample_size)  # appends new sentences, can be updated to just return new sentences

    def sample_sentences_(self, sample_size):
        sentences = []
        while len(sentences) == 0:
            text = self.next_sample_()
            if self.shuffle:
                if len(text) == 0:
                    continue
                text = text[self.gen.choice(len(text)):]  # start from random position in text
                text = text[:max(sample_size * 10, 4000)]  # cut too long texts to speed up tokenization
                                                          # floor 4000 chars so small sample_size
                                                          # (few facts / low noise_ratio) still yields
                                                          # >=3 sentences and the loop terminates
            sentences += self.sentence_tokenizer.tokenize(text)
            if self.shuffle:
                sentences = sentences[1:-1]
        self.sentences += sentences

    def next_sample_(self):
        if self.shuffle:
            self.total_tokens = 0
            sample_ind = self.gen.choice(len(self.dataset))
            sample = self.dataset[int(sample_ind)]['text']
        else:
            sample = self.dataset[int(self.sample_ind)]['text']
            self.sample_ind += 1
            self.sample_ind = self.sample_ind % len(self.dataset)
        return sample

    def length_is_ok(self, tokenized):
        if self.max_sentence_len is not None and len(tokenized) > self.max_sentence_len:
            return False
        if self.min_sentence_len is not None and len(tokenized) < self.min_sentence_len:
            return False
        return True


# sampler of random filler text (control for pg19-noise ablations)
# mode="random": uniform random printable-ASCII tokens (high-entropy, non-English)
# mode="repeat": a single token repeated (near-zero reconstruction difficulty)
# Same get_sample() interface as SentenceSampler, so NoiseInjectionDataset interleaves
# facts identically regardless of the noise source.
class RandomStringSampler:
    def __init__(self, tokenizer, mode="random", min_sentence_len=10, max_sentence_len=50,
                 fill_token_str=" the", random_seed=None):
        self.tokenizer = tokenizer
        self.mode = mode
        self.min_sentence_len = min_sentence_len
        self.max_sentence_len = max_sentence_len
        self.gen = np.random.default_rng(seed=random_seed)
        self.fill_token = tokenizer.encode(fill_token_str, add_special_tokens=False)[0]
        # printable-ASCII tokens that decode to clean, round-trippable text
        self.safe_tokens = [tid for tid in range(tokenizer.vocab_size)
                            if self._is_printable_ascii(tokenizer.decode([tid]))]

    @staticmethod
    def _is_printable_ascii(s):
        return len(s) > 0 and all(32 <= ord(c) < 127 for c in s)

    def _make_chunk(self, length):
        if self.mode == "repeat":
            return [self.fill_token] * length
        return self.gen.choice(self.safe_tokens, size=length).tolist()

    def get_sample(self, sample_size):
        # return a list of token chunks whose total length equals sample_size
        if sample_size <= 0:
            return []
        sample = []
        total_len = 0
        while total_len < sample_size:
            remaining = sample_size - total_len
            lo = min(self.min_sentence_len, remaining)
            hi = min(self.max_sentence_len, remaining)
            target = int(self.gen.integers(lo, hi + 1))
            chunk = self._make_chunk(target)
            sample.append(chunk)
            total_len += len(chunk)
        return sample


# combined dataset for noisy babi QA
# it's recommended to use sample_size >= 1024
# and task_end_pct - task_start_pct >= 0.2 in order to
class NoiseInjectionDataset(Dataset):
    def __init__(self, task_dataset, noise_sampler, tokenizer,
                 task_start_pct=None,                   # left border of facts in sample, between 0 and 1
                 task_end_pct=None,                     # right border of facts in sample, between task_start_pct and 1
                 sample_size=1024,
                 mixed_length_ratio=0.0,                # used for mixed length curriculum, prob for shorter samples
                 noise_ratio=None,                      # if set (and sample_size is None), total length =
                                                        # facts_len * (1 + noise_ratio); total length tracks the
                                                        # natural fact count -> 0k-like variable distribution, with noise
                 random_seed=42):
        self.task_dataset = task_dataset
        self.noise_sampler = noise_sampler
        self.sample_size = sample_size
        self.mixed_length_ratio = mixed_length_ratio
        self.noise_ratio = noise_ratio
        self.tokenizer = tokenizer
        self.task_start_pct = task_start_pct
        self.task_end_pct = task_end_pct
        if random_seed:
            self.gen = np.random.default_rng(seed=random_seed)

    def __getitem__(self, ind):
        sample = self.task_dataset[ind]
        facts = list(sample['facts'])
        facts_tok = self.tokenizer([' ' + f for f in facts], add_special_tokens=False)['input_ids']
        question_tok = self.tokenizer(sample['question'], add_special_tokens=False)['input_ids']
        answer_tok = self.tokenizer(sample['answer'], add_special_tokens=False)['input_ids']

        if self.noise_ratio == 0:
            flat = [i for s in facts_tok for i in s]
            sample['input_tokens'] = flat
            sample['question_tokens'] = question_tok
            sample['target_tokens'] = answer_tok
            return sample

        sample_size = self.get_sample_size()
        task_len = sum_lengths(facts_tok)
        if self.noise_ratio is not None:
            # ratio mode: noise budget scales with the fact count of THIS sample,
            # so total length = task_len * (1 + noise_ratio) and stays variable
            # across samples (0k-like distribution, but with noise added on top).
            # floor the noise budget so the noise sampler always gets a sane request:
            # with few facts and/or a small noise_ratio, task_len * noise_ratio can be
            # a handful of tokens, which starves SentenceSampler of prose and spins its
            # sentence-filling loop indefinitely. 64 tokens is well below any usable
            # noise budget but large enough to always yield a few sentences.
            background_text_len = max(int(round(task_len * self.noise_ratio)), 64)
            sample_size = task_len + background_text_len   # effective total, for pct-bounds logic
        else:
            background_text_len = sample_size - task_len
        background_text = self.noise_sampler.get_sample(background_text_len)
        sample['background_text'] = background_text

        if self.task_start_pct is None and self.task_end_pct is None:     # if fact position unspecified
            possible_positions = range(len(background_text) + 1) 
        else:
            task_start_ind = int(sample_size * self.task_start_pct)
            task_end_ind = int(sample_size * self.task_end_pct)
            total_facts_len = sum_lengths(facts_tok)

            possible_positions = []                                       # where can we insert facts?
            current_length = 0
            for i, text in enumerate(background_text):
                if (current_length >= task_start_ind) and (current_length < task_end_ind - total_facts_len):
                    possible_positions.append(i)
                current_length += len(text)

            if len(possible_positions) == 0:
                raise IndexError(f"Unable to insert facts in specified place: {self.task_start_pct, self.task_end_pct}."
                                 f"Total fact length: {total_facts_len}, "
                                 f"sentences length: {[len(t) for t in background_text]}. "
                                 f"Make the range wider or increase the sample size.")

        fact_positions = self.gen.choice(possible_positions, len(facts_tok))
        fact_positions.sort()
        sample['fact_positions'] = fact_positions                  # positions of facts between noise sentences

        updated_sample = [[] for _ in range(len(background_text) + 1)]
        for fact, pos in zip(facts_tok, fact_positions):
            updated_sample[pos].append(fact)

        for i, s in enumerate(background_text):
            updated_sample[i].append(s)

        flat = [i for s in updated_sample for i in s]
        tokens = [i for s in flat for i in s]

        sample['input_tokens'] = tokens
        sample['question_tokens'] = question_tok
        sample['target_tokens'] = answer_tok

        return sample

    def __len__(self):
        return len(self.task_dataset)

    def get_sample_size(self):
        if isinstance(self.sample_size, list):
            if self.gen.random() > self.mixed_length_ratio:
                return self.gen.choice(self.sample_size)
            return max(self.sample_size)
        else:
            return self.sample_size
