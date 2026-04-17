import json
import logging
import os
import random
from pathlib import Path

import torch
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass, field
from torch.utils.data import Sampler
import datasets

import accelerate
from safetensors.torch import load_file
import transformers
from transformers import (
    AutoConfig, AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback, TrainerCallback,
    HfArgumentParser
)
import yaml

from grad_memgpt import GradMemGPT, GradMemGPTConfig
from kv_dataset_utils import BASE_KV_ALPHABET
from run_gradmemgpt_on_kv_retrieval import (
    collate_fn,
    compute_metrics_fn,
    preprocess_logits_for_metrics,
    CustomTrainer,
    StopOnMetricValue,
)


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_lvl = logging.INFO
logging.basicConfig(format=logger_fmt, level=log_lvl)
logger = logging.getLogger('')

logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")


def generate_unique_kv_pool(n_pairs, k_length, v_length, kv_alphabet=BASE_KV_ALPHABET):
    keys = set()
    pool = []
    while len(pool) < n_pairs:
        key = ''.join(random.choice(kv_alphabet) for _ in range(k_length))
        if key in keys:
            continue
        keys.add(key)
        value = ''.join(random.choice(kv_alphabet) for _ in range(v_length))
        pool.append((key, value))
    return pool


def build_store_samples_from_pool(kv_pool, n_samples, pairs_per_sample):
    samples = []
    all_kv_dicts = []
    idx = 0
    for _ in range(n_samples):
        chunk = kv_pool[idx:idx + pairs_per_sample]
        idx += pairs_per_sample
        kv_pairs_str = ''.join(f'!{k}:{v}!' for k, v in chunk)
        context = kv_pairs_str + '|'
        pairs_dict = {k: v for k, v in chunk}
        query_key = random.choice(list(pairs_dict.keys()))
        query = f'?!{query_key}:'
        target = f'{pairs_dict[query_key]}!|'
        samples.append({
            'context': context,
            'query': query,
            'target': target,
        })
        all_kv_dicts.append(pairs_dict)
    return samples, all_kv_dicts


def generate_retrieve_samples(all_kv_dicts):
    samples = []
    for pairs in all_kv_dicts:
        for k, v in pairs.items():
            samples.append({
                'context': '',
                'query': f'?!{k}:',
                'target': f'{v}!|',
            })
    return samples


def retrieve_collate_fn(batch, tokenizer):
    pad_id = tokenizer.pad_token_id
    context_input_ids = torch.full(
        (len(batch), 8), pad_id, dtype=torch.long
    )

    query = [item['query'] + item['target'] for item in batch]
    query_encoded = tokenizer(
        query, return_tensors="pt", add_special_tokens=True,
        padding=True, pad_to_multiple_of=8, return_offsets_mapping=True,
    )
    query_input_ids = query_encoded['input_ids']
    offsets_mapping = query_encoded['offset_mapping']

    labels_mask = torch.zeros_like(query_input_ids)
    for i, item in enumerate(batch):
        query_seq_len = len(item['query'])
        target_seq_len = len(item['target'])
        target_st, target_end = query_seq_len, query_seq_len + target_seq_len
        in_target = False
        for j in range(len(offsets_mapping[i]) - 1, -1, -1):
            st, end = offsets_mapping[i][j]
            if st < target_end and end > target_st:
                labels_mask[i, j] = 1
                in_target = True
            elif in_target:
                break

    labels = query_input_ids * labels_mask + (1 - labels_mask) * (-100)
    return {
        'input_ids': {
            'context_input_ids': context_input_ids,
            'query_input_ids': query_input_ids,
        },
        'labels': labels,
    }


def compute_retrieve_metrics(predictions, labels, ignore_token_ids):
    preds = predictions.argmax(dim=-1).cpu().numpy()
    labels = labels.cpu().numpy()
    preds = preds[:, :-1]
    labels = labels[:, :]

    mask = (labels != -100)
    for t_id in ignore_token_ids:
        mask &= (labels != t_id)

    masked_predictions = preds[mask]
    masked_labels = labels[mask]

    if masked_predictions.size == 0:
        return {"token_accuracy": 0.0, "exact_match": 0.0}

    accuracy = (masked_predictions == masked_labels).mean()

    exact_match = np.mean([
        np.all(pred[mask[i]] == lab[mask[i]])
        for i, (pred, lab) in enumerate(zip(preds, labels))
        if np.any(mask[i])
    ])

    return {
        "token_accuracy": float(accuracy),
        "exact_match": float(exact_match),
    }


def generate_chunked_dataset(n_chunks, chunk_size, pairs_per_sample,
                              k_length, v_length, seed=0,
                              n_valid_chunks=1,
                              kv_alphabet=BASE_KV_ALPHABET):
    rng = random.Random(seed)

    def _gen_chunks(n):
        all_data = {'chunk_id': [], 'context': [], 'query': [], 'target': []}
        for chunk_idx in range(n):
            n_pairs = chunk_size * pairs_per_sample
            pool = []
            keys_set = set()
            while len(pool) < n_pairs:
                key = ''.join(rng.choice(kv_alphabet) for _ in range(k_length))
                if key in keys_set:
                    continue
                keys_set.add(key)
                value = ''.join(rng.choice(kv_alphabet) for _ in range(v_length))
                pool.append((key, value))
            samples, _ = build_store_samples_from_pool(pool, chunk_size, pairs_per_sample)
            for sample in samples:
                all_data['chunk_id'].append(chunk_idx)
                all_data['context'].append(sample['context'])
                all_data['query'].append(sample['query'])
                all_data['target'].append(sample['target'])
        return datasets.Dataset.from_dict(all_data)

    train_dataset = _gen_chunks(n_chunks)
    valid_dataset = _gen_chunks(n_valid_chunks)
    return datasets.DatasetDict({'train': train_dataset, 'valid': valid_dataset})


class ChunkPreservingSampler(Sampler):
    def __init__(self, chunk_ids, shuffle=True, seed=0, epoch=0):
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = epoch

        self.chunks = {}
        for idx, cid in enumerate(chunk_ids):
            self.chunks.setdefault(cid, []).append(idx)
        self.chunk_order = sorted(self.chunks.keys())

    def __iter__(self):
        chunk_order = list(self.chunk_order)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(chunk_order), generator=g).tolist()
            chunk_order = [chunk_order[i] for i in perm]

        for cid in chunk_order:
            indices = list(self.chunks[cid])
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch * 10000 + cid)
                perm = torch.randperm(len(indices), generator=g).tolist()
                indices = [indices[p] for p in perm]
            yield from indices

    def __len__(self):
        return sum(len(v) for v in self.chunks.values())

    def set_epoch(self, epoch):
        self.epoch = epoch


class HopfieldResetCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs.get('model')
        if model is not None and hasattr(model, 'W_hopfield') and model.W_hopfield is not None:
            model.W_hopfield.zero_()
            model._hopfield_has_patterns.fill_(False)
            model._hopfield_step_counter.zero_()


class HopfieldTrainer(CustomTrainer):
    def _get_train_sampler(self, dataset=None):
        if dataset is None:
            dataset = self.train_dataset
        chunk_ids = dataset['chunk_id']
        epoch = int(self.state.epoch) if self.state.epoch is not None else 0
        return ChunkPreservingSampler(
            chunk_ids, shuffle=True, seed=self.args.seed, epoch=epoch,
        )


def run_hopfield_eval(model, tokenizer, n_store_batches, eval_batch_size,
                      num_kv_pairs, k_length, v_length, seed, device):
    model.eval()
    ignore_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in ['!', '|']]
    all_results = []

    max_total_kv = max(n_batches for n_batches in n_store_batches) * eval_batch_size * num_kv_pairs

    random.seed(seed)
    torch.manual_seed(seed)
    kv_pool = generate_unique_kv_pool(max_total_kv, k_length, v_length)

    for n_batches in n_store_batches:
        n_total_pairs = n_batches * eval_batch_size * num_kv_pairs
        pool_subset = kv_pool[:n_total_pairs]

        model.W_hopfield.zero_()
        model._hopfield_has_patterns.fill_(False)

        store_batches = []
        all_kv_dicts = []
        for batch_idx in range(n_batches):
            chunk_start = batch_idx * eval_batch_size * num_kv_pairs
            chunk_end = chunk_start + eval_batch_size * num_kv_pairs
            chunk = pool_subset[chunk_start:chunk_end]
            samples, kv_dicts = build_store_samples_from_pool(
                chunk, eval_batch_size, num_kv_pairs,
            )
            store_batches.append(samples)
            all_kv_dicts.extend(kv_dicts)

        with torch.no_grad():
            for batch_samples in store_batches:
                batch = collate_fn(batch_samples, tokenizer)
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else
                         {k2: v2.to(device) for k2, v2 in v.items()}
                         if isinstance(v, dict) else v
                         for k, v in batch.items()}
                model(**batch)

        logger.info(f'Hopfield store done: {n_batches} batches ({n_total_pairs} unique KV pairs), '
                    f'W_hopfield norm = {model.W_hopfield.norm().item():.4f}')

        retrieve_samples = generate_retrieve_samples(all_kv_dicts)

        all_preds = []
        all_labels = []
        with torch.no_grad():
            for i in range(0, len(retrieve_samples), eval_batch_size):
                batch_slice = retrieve_samples[i:i + eval_batch_size]
                batch = retrieve_collate_fn(batch_slice, tokenizer)
                context_input_ids = batch['input_ids']['context_input_ids'].to(device)
                query_input_ids = batch['input_ids']['query_input_ids'].to(device)
                labels = batch['labels'].to(device)
                output = model(
                    input_ids={
                        'context_input_ids': context_input_ids,
                        'query_input_ids': query_input_ids,
                    },
                    labels=labels,
                )
                all_preds.append(output['predictions'].cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = compute_retrieve_metrics(all_preds, all_labels, ignore_token_ids)

        n_stored_samples = len(all_kv_dicts)
        result = {
            'n_store_batches': n_batches,
            'n_stored_samples': n_stored_samples,
            'n_total_kv_pairs': n_total_pairs,
            'n_retrieve_queries': len(retrieve_samples),
            'hopfield_W_norm': model.W_hopfield.norm().item(),
            **metrics,
        }
        all_results.append(result)
        logger.info(f'n_store_batches={n_batches}, n_total_kv={n_total_pairs}, '
                    f'n_queries={len(retrieve_samples)}, '
                    f'token_acc={metrics["token_accuracy"]:.4f}, '
                    f'exact_match={metrics["exact_match"]:.4f}')

    return all_results


@dataclass
class ExperimentArgs:
    config: Optional[str] = field(default=None)
    exp_path: Optional[str] = field(default=None)
    per_device_batch_size: int = field(default=64)
    data_path: str = field(default='./data/N2-K4V4-S4(32-64)_1M')
    tokenizer_path: str = field(default='./tokenizers/kv_alphabet_62/')
    gradient_accumulation_steps: Optional[int] = field(default=1)
    total_batch_size: Optional[int] = field(default=None)
    metric_for_best_model: Optional[str] = field(default='token_accuracy')
    warmup_steps: Optional[int] = field(default=1000)
    max_steps: Optional[int] = field(default=50000)
    logging_steps: Optional[int] = field(default=100)
    eval_steps: Optional[int] = field(default=100)
    weight_decay: Optional[float] = field(default=0.0)
    learning_rate: Optional[float] = field(default=1e-04)
    lr_scheduler_type: Optional[str] = field(default='constant_with_warmup')
    early_stopping_patience: Optional[int] = field(default=50)
    seed: Optional[int] = field(default=142)
    base_model: Optional[str] = field(default=None)
    pretrained_model: Optional[str] = field(default=None)
    init_checkpoint: Optional[str] = field(default=None)
    n_layer: Optional[int] = field(default=4)
    n_head: Optional[int] = field(default=4)
    n_embd: Optional[int] = field(default=128)
    max_context_length: Optional[int] = field(default=None)
    n_mem_tokens: Optional[int] = field(default=8)
    K: Optional[int] = field(default=3)
    last_K_second_order: Optional[int] = field(default=None)
    inner_lr: Optional[float] = field(default=0.01)
    use_adam: Optional[bool] = field(default=True)
    grad_mode: Optional[str] = field(default="none")
    n_ctrl_tokens: Optional[int] = field(default=0)
    inner_clip_value: Optional[float] = field(default=None)
    inner_clip_norm: Optional[float] = field(default=None)
    use_mem_proj: Optional[bool] = field(default=False)
    mem_proj_mode: Optional[str] = field(default="none")
    use_write_head: Optional[bool] = field(default=False)
    use_write_lora: Optional[bool] = field(default=False)
    write_lora_r: Optional[int] = field(default=8)
    write_lora_alpha: Optional[int] = field(default=16)
    write_lora_dropout: Optional[float] = field(default=0.0)
    write_lora_target_modules: Optional[str] = field(default=None)
    freeze_backbone: Optional[bool] = field(default=False)
    use_gradient_checkpointing: Optional[bool] = field(default=False)
    attn_implementation: Optional[str] = field(default="eager")
    add_inner_loss_to_outer: Optional[bool] = field(default=False)
    inner_loss_weight: Optional[float] = field(default=None)
    use_hopfield_memory: Optional[bool] = field(default=True)
    hopfield_dim: Optional[int] = field(default=0)
    hopfield_proj_freeze: Optional[bool] = field(default=True)
    hopfield_reset_interval: Optional[int] = field(default=None)
    concat_hopfield_memory: Optional[bool] = field(default=False)
    hopfield_chunk_size: Optional[int] = field(default=448)
    hopfield_n_chunks: Optional[int] = field(default=2000)
    hopfield_pairs_per_sample: Optional[int] = field(default=8)
    hopfield_k_length: Optional[int] = field(default=2)
    hopfield_v_length: Optional[int] = field(default=2)
    hopfield_n_valid_chunks: Optional[int] = field(default=1)
    skip_training: Optional[bool] = field(default=False)
    hopfield_eval_n_store_batches: Optional[str] = field(default="1,2,5,10,20")
    hopfield_eval_batch_size: Optional[int] = field(default=64)
    hopfield_eval_n_kv_pairs: Optional[int] = field(default=8)
    hopfield_eval_k_length: Optional[int] = field(default=2)
    hopfield_eval_v_length: Optional[int] = field(default=2)
    hopfield_eval_seed: Optional[int] = field(default=999)


def main(config_path: Optional[str] = None):
    parser = HfArgumentParser(ExperimentArgs)
    args = parser.parse_args_into_dataclasses()[0]

    if config_path is not None:
        args.config = config_path

    if args.config is not None:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        for section in ['model', 'training', 'dataset', 'gradmem', 'hopfield']:
            if section in cfg:
                for key, value in cfg[section].items():
                    if hasattr(args, key):
                        setattr(args, key, value)

        if 'hopfield_eval' in cfg:
            for key, value in cfg['hopfield_eval'].items():
                arg_name = f'hopfield_eval_{key}'
                if hasattr(args, arg_name):
                    setattr(args, arg_name, value)

        if 'exp_path' not in vars(args) or args.exp_path is None:
            from generate_run_name import generate_run_name, get_exp_path, get_data_path
            run_name = generate_run_name(cfg)
            exp_path = get_exp_path(cfg)
            args.exp_path = str(exp_path)

            dataset = cfg.get('dataset', {})
            if 'data_path' in dataset:
                args.data_path = dataset['data_path']
            elif 'data_name' in dataset:
                args.data_path = get_data_path(cfg)
            if 'tokenizer_path' in dataset:
                args.tokenizer_path = dataset['tokenizer_path']

    accel = accelerate.Accelerator()
    from accelerate.logging import get_logger
    logger = get_logger('')
    transformers.utils.logging.set_verbosity(log_lvl)

    logger.info(f'num processes: {accel.num_processes}')
    logger.info(f'mixed precision: {accel.mixed_precision}')
    logger.info(f'accelerator state: {accel.state}')

    assert not (args.pretrained_model is not None and args.base_model is not None), \
        "only one of these args must be set"

    if args.pretrained_model is None:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        if args.base_model == 'gpt2':
            config = AutoConfig.from_pretrained('gpt2')
            config.n_layer = args.n_layer
            config.n_head = args.n_head
            config.n_embd = args.n_embd
        elif args.base_model == 'pythia':
            config = AutoConfig.from_pretrained('EleutherAI/pythia-160m')
            config.num_hidden_layers = args.n_layer
            config.num_attention_heads = args.n_head
            config.hidden_size = args.n_embd
            config.intermediate_size = config.hidden_size * 4
        elif args.base_model == 'llama':
            config = AutoConfig.from_pretrained('meta-llama/Llama-3.2-1B')
            config.num_hidden_layers = args.n_layer
            config.num_attention_heads = args.n_head
            config.num_key_value_heads = args.n_head
            config.hidden_size = args.n_embd
            config.head_dim = config.hidden_size // config.num_attention_heads
            config.intermediate_size = config.hidden_size * 4
        else:
            raise ValueError(f'Unsupported base model: {args.base_model}')

        config.torch_dtype = "float32"
        config.vocab_size = tokenizer.vocab_size
        config.pad_token_id = tokenizer.pad_token_id
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.use_cache = False
    else:
        config = None
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    gradmem_config = GradMemGPTConfig(
        pretrained_model=args.pretrained_model, base_config=config,
        n_mem_tokens=args.n_mem_tokens, K=args.K,
        last_K_second_order=args.last_K_second_order,
        lr=args.inner_lr, use_adam=args.use_adam, grad_mode=args.grad_mode,
        n_ctrl_tokens=args.n_ctrl_tokens,
        inner_clip_value=args.inner_clip_value, inner_clip_norm=args.inner_clip_norm,
        use_mem_proj=args.use_mem_proj, mem_proj_mode=args.mem_proj_mode,
        use_write_head=args.use_write_head,
        use_write_lora=args.use_write_lora,
        write_lora_r=args.write_lora_r,
        write_lora_alpha=args.write_lora_alpha,
        write_lora_dropout=args.write_lora_dropout,
        write_lora_target_modules=args.write_lora_target_modules,
        freeze_backbone=args.freeze_backbone,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        attn_implementation=args.attn_implementation,
        add_inner_loss_to_outer=args.add_inner_loss_to_outer,
        inner_loss_weight=args.inner_loss_weight,
        use_hopfield_memory=args.use_hopfield_memory,
        hopfield_dim=args.hopfield_dim,
        hopfield_proj_freeze=args.hopfield_proj_freeze,
        hopfield_reset_interval=args.hopfield_reset_interval,
        concat_hopfield_memory=args.concat_hopfield_memory,
    )

    model = GradMemGPT(gradmem_config)

    if args.init_checkpoint is not None:
        missing_k, unexpected_k = model.load_state_dict(
            load_file(args.init_checkpoint), strict=False
        )
        if len(missing_k) != 0:
            logger.info(f'{missing_k} were not loaded from checkpoint!')
        if len(unexpected_k) != 0:
            logger.info(f'{unexpected_k} found in checkpoint but not expected!')

    if accel.mixed_precision == 'bf16':
        model.to(torch.bfloat16)

    output_dir = Path(args.exp_path) if args.exp_path else Path('./runs/hopfield_eval')

    if not args.skip_training:
        if accel.is_main_process:
            config_dict = {'cli_args': dict(vars(args))}
            logger.info(f'saving experiment configuration to {output_dir}')
            output_dir.mkdir(parents=True, exist_ok=True)
            json.dump(config_dict, open(output_dir / 'config.json', 'w'), indent=4)

        logger.info(f'model config: {model.config}')
        logger.info(f'model.dtype: {model.dtype}')

        if args.hopfield_reset_interval is None:
            assert args.hopfield_chunk_size % args.per_device_batch_size == 0, \
                f"hopfield_chunk_size ({args.hopfield_chunk_size}) must be a multiple of per_device_batch_size ({args.per_device_batch_size})"
            args.hopfield_reset_interval = args.hopfield_chunk_size // args.per_device_batch_size
            logger.info(f'auto-computed hopfield_reset_interval = {args.hopfield_reset_interval} '
                        f'(chunk_size={args.hopfield_chunk_size} / batch_size={args.per_device_batch_size})')

        dataset = generate_chunked_dataset(
            n_chunks=args.hopfield_n_chunks,
            chunk_size=args.hopfield_chunk_size,
            pairs_per_sample=args.hopfield_pairs_per_sample,
            k_length=args.hopfield_k_length,
            v_length=args.hopfield_v_length,
            seed=args.seed,
            n_valid_chunks=args.hopfield_n_valid_chunks,
        )
        logger.info(f'generated chunked dataset: train={len(dataset["train"])} samples '
                    f'({args.hopfield_n_chunks} chunks x {args.hopfield_chunk_size}), '
                    f'valid={len(dataset["valid"])} samples ({args.hopfield_n_valid_chunks} chunks)')

        def data_collator(batch):
            return collate_fn(batch, tokenizer, max_context_length=args.max_context_length)

        ignore_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in ['!', '|']]

        def compute_metrics(eval_pred):
            return compute_metrics_fn(eval_pred, ignore_token_ids, tokenizer)

        if args.total_batch_size is None:
            args.total_batch_size = (args.per_device_batch_size * accel.num_processes
                                     * args.gradient_accumulation_steps)
        else:
            args_total_bs = (args.per_device_batch_size * accel.num_processes
                             * args.gradient_accumulation_steps)
            assert args.total_batch_size == args_total_bs

        training_args = TrainingArguments(
            output_dir=output_dir,
            logging_dir=output_dir,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            gradient_checkpointing=args.use_gradient_checkpointing,
            eval_strategy='steps',
            save_strategy='steps',
            save_steps=args.eval_steps,
            eval_steps=args.eval_steps,
            logging_steps=args.logging_steps,
            report_to='comet_ml',
            metric_for_best_model=args.metric_for_best_model,
            load_best_model_at_end=True,
            eval_on_start=True,
            greater_is_better=True,
            remove_unused_columns=False,
            include_num_input_tokens_seen=False,
            include_for_metrics=['inputs'],
            save_total_limit=1,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            seed=args.seed,
        )

        trainer = HopfieldTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['valid'],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
                StopOnMetricValue(metric_name='exact_match', value=1.0, higher_is_better=True),
                HopfieldResetCallback(),
            ],
        )

        trainer.train()
        logger.info('training done. running final evaluation...')
        metrics = trainer.evaluate(dataset['valid'])
        logger.info(f'{metrics}')
        trainer.save_metrics(split='all', metrics=metrics)
        trainer.state.save_to_json(output_dir / 'trainer_state.json')

    n_store_batches_list = [int(x) for x in args.hopfield_eval_n_store_batches.split(',')]
    device = next(model.parameters()).device

    logger.info('Starting Hopfield store/retrieve evaluation...')
    logger.info(f'n_store_batches sweep: {n_store_batches_list}')
    logger.info(f'eval_batch_size: {args.hopfield_eval_batch_size}')
    logger.info(f'num_kv_pairs per sample: {args.hopfield_eval_n_kv_pairs}')
    logger.info(f'k_length: {args.hopfield_eval_k_length}, v_length: {args.hopfield_eval_v_length}')

    all_results = run_hopfield_eval(
        model=model,
        tokenizer=tokenizer,
        n_store_batches=n_store_batches_list,
        eval_batch_size=args.hopfield_eval_batch_size,
        num_kv_pairs=args.hopfield_eval_n_kv_pairs,
        k_length=args.hopfield_eval_k_length,
        v_length=args.hopfield_eval_v_length,
        seed=args.hopfield_eval_seed,
        device=device,
    )

    results_path = output_dir / 'hopfield_eval_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f'Hopfield eval results saved to {results_path}')

    for r in all_results:
        logger.info(
            f"n_store_batches={r['n_store_batches']}, "
            f"n_total_kv={r['n_total_kv_pairs']}, "
            f"n_queries={r['n_retrieve_queries']}, "
            f"token_acc={r['token_accuracy']:.4f}, "
            f"exact_match={r['exact_match']:.4f}, "
            f"W_norm={r['hopfield_W_norm']:.4f}"
        )


if __name__ == '__main__':
    main()
