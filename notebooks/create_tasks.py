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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate babilong tasks with configurable noise source.")
    parser.add_argument('tasks', type=str, help="space-separated bAbI task names, e.g. 'qa3' or 'qa1 qa3'")
    parser.add_argument('--noise', type=str, default='pg19', choices=['pg19', 'random', 'repeat'],
                        help="noise source: 'pg19' (English prose), 'random' (random ASCII tokens), "
                             "'repeat' (single token repeated). Default: pg19")
    parser.add_argument('--noise_tag', type=str, default=None,
                        help="tag appended to the output folder/len_name to distinguish noise variants "
                             "(e.g. 'rand'). Defaults to the --noise value.")
    args = parser.parse_args()

    tasks = args.tasks.split(' ')
    print(tasks)

    # message_lengths = [0, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1_000_000]
    message_lengths = [600,]

    # names = ['0k', '1k', '2k', '4k', '8k', '16k', '32k', '64k', '128k', '256k', '512k', '1M']
    names = ['600_curriculum']

    message_lengths = message_lengths

    message_lengths = [ml - 300 for ml in message_lengths] # take prompt length into account

    # tag that identifies the noise variant in output paths
    noise_tag = args.noise_tag if args.noise_tag is not None else args.noise
    print('noise source:', args.noise, '| tag:', noise_tag)

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

                if message_length > 0:
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

                # include the noise tag in the filename so variants don't overwrite each other
                out_len_name = len_name if noise_tag == 'pg19' else f"{len_name}_{noise_tag}"
                json_path = os.path.join(subfolder, f"{out_len_name}_{split}.json")
                print(f"Writing", json_path)
                with open(json_path, 'w') as f:
                    json.dump(llm_tasks[len_name], f)
