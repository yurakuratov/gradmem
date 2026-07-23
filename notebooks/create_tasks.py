import argparse
import numpy as np
import json
import torch
import os
import datasets

import sys
sys.path.append('..')
from babilong.babilong_utils import TaskDataset, SentenceSampler, RandomStringSampler, NoiseInjectionDataset
from transformers import AutoTokenizer

# qa1_single-supporting-fact qa2_two-supporting-facts qa3_three-supporting-facts qa4_two-arg-relations qa5_three-arg-relations qa6_yes-no-questions qa7_counting qa8_lists-sets qa9_simple-negation qa10_indefinite-knowledge
# qa11_basic-coreference qa12_conjunction qa13_compound-coreference qa14_time-reasoning qa15_basic-deduction qa16_basic-induction qa17_positional-reasoning qa18_size-reasoning qa19_path-finding qa20_agents-motivations

out_folder = "./generated_tasks"
task_folder = "./tasks_1-20_v1-2/en-valid-10k/"
splits = ['train', 'valid', 'test']
map_to_pg = {'train': 'train', 'valid': 'test', 'test': 'test'}
n_samples = [9000, 1000, 1000]
# different noise sample sizes up to message length
curriculum = True
os.makedirs(out_folder, exist_ok=True)


def build_curriculum_schedule(base_names, cmin, cmax, n_stages, schedule):
    """Build a list of stage names + per-stage fact-count caps for a fact-count-window curriculum.

    Each stage filters bAbI samples to those with <= max_facts facts; the cap expands from
    `cmin` (easiest/shortest) to `cmax` (hardest/longest), so dataset mass shifts from short
    sequences to long across stages. Names are '<base>_c<cap>' (or '<base>_call' for an
    uncapped final stage), which become the JSON basenames and downstream curriculum.levels
    entries.

    `cmax == 0` means: space the first n_stages-1 caps in [cmin, 2*cmin] and append a final
    UNCAPPED stage (full fact distribution).

    Returns (names, max_facts_per_name) where names has one entry per stage and
    max_facts_per_name maps each name -> its cap (None for uncapped).
    """
    if n_stages < 1:
        raise ValueError("curriculum_n must be >= 1")
    if cmax != 0 and cmax <= cmin:
        raise ValueError("curriculum_max must be > curriculum_min, or 0 for an uncapped final stage")

    uncapped_last = (cmax == 0)
    n_spaced = n_stages - 1 if uncapped_last else n_stages
    upper = cmin * 2 if uncapped_last else cmax

    if n_spaced >= 2:
        if schedule == 'geometric':
            spaced = np.geomspace(cmin, upper, n_spaced).astype(int)
        else:  # linear
            spaced = np.linspace(cmin, upper, n_spaced).astype(int)
        caps = sorted(set(int(c) for c in spaced))
    else:
        caps = [cmin]
    if uncapped_last:
        caps.append(None)  # final stage: no cap (full fact distribution)

    names = []
    max_facts_per_name = {}
    for cap in caps:
        name = f"{base_names[0]}_c{'all' if cap is None else cap}"
        names.append(name)
        max_facts_per_name[name] = cap
    return names, max_facts_per_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate babilong tasks with configurable noise source.")
    parser.add_argument('tasks', type=str, help="space-separated bAbI task names, e.g. 'qa3' or 'qa1 qa3'")
    parser.add_argument('--noise', type=str, default='pg19', choices=['pg19', 'random', 'repeat'],
                        help="noise source: 'pg19' (English prose), 'random' (random ASCII tokens), "
                             "'repeat' (single token repeated). Default: pg19")
    parser.add_argument('--noise_tag', type=str, default=None,
                        help="tag appended to the output folder/len_name to distinguish noise variants "
                             "(e.g. 'rand'). Defaults to the --noise value.")
    parser.add_argument('--length_mode', type=str, default='fixed', choices=['fixed', 'ratio'],
                        help="'fixed': each sample padded to message_length (current behavior; total length "
                             "is constant across samples). 'ratio': total length = facts_len * (1 + "
                             "--noise_ratio), so length tracks the natural fact count (0k-like variable "
                             "distribution) with noise added on top. Default: fixed")
    parser.add_argument('--noise_ratio', type=float, default=1.0,
                        help="noise tokens per fact token; only used with --length_mode ratio. "
                             "E.g. 1.0 -> as much noise as facts, 3.0 -> 3x. Default: 1.0")
    parser.add_argument('--curriculum_generate', action='store_true',
                        help="Generate a CURRICULUM of datasets (one per stage) whose fact-count window "
                             "expands from --curriculum_min to --curriculum_max. Each stage is written as a "
                             "separate JSON (name '<len_name>_c<cap>') to be wired into training "
                             "curriculum.levels. Overrides the hardcoded message_lengths/names.")
    parser.add_argument('--curriculum_min', type=int, default=10,
                        help="max_facts cap of the FIRST curriculum stage (smallest/easiest). Default: 10")
    parser.add_argument('--curriculum_max', type=int, default=150,
                        help="max_facts cap of the LAST curriculum stage. Use 0 (or a value larger than the "
                             "data) to make the last stage uncapped (full fact distribution). Default: 150")
    parser.add_argument('--curriculum_n', type=int, default=5,
                        help="number of curriculum stages. Default: 5")
    parser.add_argument('--curriculum_schedule', type=str, default='geometric',
                        choices=['geometric', 'linear'],
                        help="how to space the max_facts caps between min and max. 'geometric' spaces them "
                             "evenly on a log scale (matches the log-distributed fact counts). Default: geometric")
    args = parser.parse_args()

    tasks = args.tasks.split(' ')
    print(tasks)

    # message_lengths = [0, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1_000_000]
    message_lengths = [1000,]

    # names = ['0k', '1k', '2k', '4k', '8k', '16k', '32k', '64k', '128k', '256k', '512k', '1M']
    names = ['1k',]

    message_lengths = message_lengths

    message_lengths = [ml - 300 for ml in message_lengths] # take prompt length into account

    # Curriculum mode: replace the single (name, message_length) target with a list of
    # stages, each a (name, max_facts_cap) pair whose fact-count window expands from min to
    # max. Each stage is written to its own JSON and becomes one curriculum.levels entry.
    # max_facts_per_name maps each len_name -> the max_n_facts cap to pass to TaskDataset.
    max_facts_per_name = None
    if args.curriculum_generate:
        names, max_facts_per_name = build_curriculum_schedule(
            base_names=names,
            cmin=args.curriculum_min, cmax=args.curriculum_max,
            n_stages=args.curriculum_n, schedule=args.curriculum_schedule,
        )
        # in curriculum mode every stage shares the same (already -300-adjusted) noise budget;
        # message_lengths is broadcast 1:1 with names below, so expand it.
        message_lengths = [message_lengths[0]] * len(names)

    # tag that identifies the noise variant in output paths
    noise_tag = args.noise_tag if args.noise_tag is not None else args.noise
    print('noise source:', args.noise, '| tag:', noise_tag)
    print('length mode:', args.length_mode,
          '| noise_ratio:', args.noise_ratio if args.length_mode == 'ratio' else 'n/a')

    if max_facts_per_name is not None:
        # Preview the curriculum schedule and how much of the raw data each stage covers.
        # Coverage is measured on the train split (largest); shows how mass shifts short->long.
        preview_path = os.path.join(task_folder, tasks[0].split('_')[0] + f'_train.txt')
        if os.path.exists(preview_path):
            _prev = TaskDataset(preview_path)
            _nf = np.array([len(_prev[i]['facts']) for i in range(len(_prev))])
            print(f'curriculum schedule ({args.curriculum_schedule}, '
                  f'{args.curriculum_n} stages from max_facts {args.curriculum_min}->{args.curriculum_max}):')
            for nm in names:
                cap = max_facts_per_name[nm]
                cov = (_nf <= cap).mean() * 100 if cap is not None else 100.0
                print(f'  stage {nm:18s} max_facts={str(cap):>4s}  covers {cov:5.1f}% of raw train samples')
        print('curriculum_levels hint: ' + ','.join(
            f'babilong_{tasks[0].split("_")[0]}_{nm}' for nm in names))

    os.makedirs('tasks', exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    for task in tasks:
        print('processing', task)
        # a placeholder for all samples for the current task
        llm_tasks = dict()
        subfolder = os.path.join(out_folder, task.split('_')[0])
        os.makedirs(subfolder, exist_ok=True)

        for split, number_of_samples in zip(splits, n_samples):
            task_path = os.path.join(task_folder, task + f'_{split}.txt')

            for len_name, message_length in zip(names, message_lengths):
                print('message length', len_name, message_length)

                # Determine the fact-count cap for this stage:
                #  - curriculum mode: explicit per-stage cap from max_facts_per_name (expanding window).
                #    None means uncapped (full fact distribution, e.g. the last stage).
                #  - ratio mode (non-curriculum): no cap; length grows with the natural fact count.
                #  - fixed mode: cap facts so they fit the message_length budget.
                if max_facts_per_name is not None:
                    stage_cap = max_facts_per_name[len_name]
                    task_dataset_test = TaskDataset(task_path) if stage_cap is None \
                        else TaskDataset(task_path, max_n_facts=stage_cap)
                elif args.length_mode == 'ratio':
                    task_dataset_test = TaskDataset(task_path)
                elif message_length > 0:
                    max_n_facts = message_length // 8
                    task_dataset_test = TaskDataset(task_path, max_n_facts=max_n_facts)
                else:
                    task_dataset_test = TaskDataset(task_path)

                if args.noise == 'pg19':
                    noise_dataset = datasets.load_dataset("pg19")[map_to_pg[split]]
                    noise_sampler_test = SentenceSampler(noise_dataset, tokenizer=tokenizer, shuffle=True, random_seed=None)
                else:
                    # 'random' or 'repeat': no external dataset needed
                    noise_sampler_test = RandomStringSampler(tokenizer=tokenizer, mode=args.noise, random_seed=None)

                if args.length_mode == 'ratio':
                    # total length tracks each sample's natural fact count; noise added on top.
                    # sample_size is unused in this mode (NoiseInjectionDataset derives the budget
                    # from facts_len * noise_ratio per sample).
                    dataset_test = NoiseInjectionDataset(task_dataset=task_dataset_test,
                                                            noise_sampler=noise_sampler_test,
                                                            tokenizer=tokenizer,
                                                            sample_size=None,
                                                            noise_ratio=args.noise_ratio,
                                                            mixed_length_ratio=0.0,
                                                         )
                else:
                    dataset_test = NoiseInjectionDataset(task_dataset=task_dataset_test,
                                                            noise_sampler=noise_sampler_test,
                                                            tokenizer=tokenizer,
                                                            sample_size=message_length,
                                                            mixed_length_ratio=1.0 if curriculum else 0.0
                                                         )

                # get number_of_samples random indices
                inds = list(range(len(dataset_test)))
                np.random.shuffle(inds)
                inds = inds[:number_of_samples]

                # prepare samples for LLM evaluation
                samples = [dataset_test[i] for i in inds]

                questions = [sample['question'] for sample in samples]
                input_tokens = [torch.tensor(sample['input_tokens']) for sample in samples]
                target_tokens = [torch.tensor(sample['target_tokens']) for sample in samples]

                inputs = tokenizer.batch_decode(input_tokens, add_special_tokens=False)
                targets = tokenizer.batch_decode(target_tokens, add_special_tokens=False)

                llm_tasks[len_name] = [{'input': i.strip(), 'question': q, 'target': t} for (i, q, t) in zip(inputs, questions, targets)]

                # build an output name that distinguishes noise source and length mode so
                # variants don't overwrite each other
                out_len_name = len_name
                if args.length_mode == 'ratio':
                    out_len_name = f"{out_len_name}_ratio{args.noise_ratio:g}"
                if noise_tag != 'pg19':
                    out_len_name = f"{out_len_name}_{noise_tag}"
                json_path = os.path.join(subfolder, f"{out_len_name}_{split}.json")
                print(f"Writing", json_path)
                with open(json_path, 'w') as f:
                    json.dump(llm_tasks[len_name], f)
