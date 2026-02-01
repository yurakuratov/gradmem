import json
import logging
import os
from pathlib import Path
from itertools import chain

import torch
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass, field
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
from torch.nn.utils.rnn import pad_sequence

from rmt import RMT2Segm, RMT2SegmConfig

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_lvl = logging.INFO
logging.basicConfig(format=logger_fmt, level=log_lvl)
logger = logging.getLogger('')

logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")


def collate_fn(batch, tokenizer):
    context_ids_list = []
    query_ids_list = []
    
    for sample in batch:
        input_ids = torch.tensor(sample['input_ids'], dtype=torch.long)
        context_ids_list.append(input_ids[args.context_size:])
        query_ids_list.append(input_ids[:args.context_size])
    
    context_input_ids = pad_sequence(context_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    query_input_ids = pad_sequence(query_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = query_input_ids.clone()
    
    return {
        'input_ids': {
            'context_input_ids': context_input_ids,
            'query_input_ids': query_input_ids,
        },
        'labels': labels,
    }


class CustomTrainer(Trainer):
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        num_training_steps = int(num_training_steps / 0.9)  # to make final lr not zero, for linear it is lr/10.
        return super().create_scheduler(num_training_steps, optimizer)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        # log early stopping patience
        for cb in self.callback_handler.callbacks:
            if isinstance(cb, EarlyStoppingCallback):
                logs['patience'] = cb.early_stopping_patience_counter
                break
        return super().log(logs, start_time=start_time)


@dataclass
class ExperimentArgs:
    exp_path: str = field()
    per_device_batch_size: int = field()
    dataset_name: str = field(default='wikitext')
    tokenizer_path: str = field(
        default='./tokenizers/kv_alphabet_62/',
    )
    gradient_accumulation_steps: Optional[int] = field(default=1)
    total_batch_size: Optional[int] = field(default=None)
    metric_for_best_model: Optional[str] = field(default='loss')
    warmup_steps: Optional[int] = field(default=1000)
    max_steps: Optional[int] = field(default=50000)
    logging_steps: Optional[int] = field(default=100)
    eval_steps: Optional[int] = field(default=100)
    weight_decay: Optional[float] = field(default=0.0)
    learning_rate: Optional[float] = field(default=1e-04)
    adam_beta1: Optional[float] = field(default=0.9)
    adam_beta2: Optional[float] = field(default=0.999)
    adam_epsilon: Optional[float] = field(default=1e-8)
    lr_scheduler_type: Optional[str] = field(default='constant_with_warmup')
    early_stopping_patience: Optional[int] = field(default=50)
    seed: Optional[int] = field(default=142)
    base_model: Optional[str] = field(default=None)
    tokenizer: Optional[str] = field(default=None)
    pretrained_model: Optional[str] = field(default=None)
    init_base_checkpoint: Optional[str] = field(default=None, metadata={'help': 'checkpoint to initialize base model'})
    init_checkpoint: Optional[str] = field(default=None, metadata={'help': 'checkpoint to initialize gradmemgpt model'})
    n_layer: Optional[int] = field(default=4)
    n_head: Optional[int] = field(default=4)
    n_embd: Optional[int] = field(default=128)
    # RMT parameters
    n_mem_tokens: Optional[int] = field(default=8)
    K: Optional[int] = field(default=1)
    n_ctrl_tokens: Optional[int] = field(default=0)
    use_mem_proj: Optional[bool] = field(default=False)
    mem_proj_mode: Optional[str] = field(default="none")
    use_reconstruction_loss: Optional[bool] = field(default=False)
    reconstruction_loss_weight: Optional[float] = field(default=1.0)
    use_write_head: Optional[bool] = field(default=False)
    attn_implementation: Optional[str] = field(default='eager')
    # LM-specific parameters
    segment_size: int = field(default=1024)
    context_size: Optional[int] = field(default=None)


if __name__ == '__main__':
    parser = HfArgumentParser(ExperimentArgs)
    args = parser.parse_args_into_dataclasses()[0]

    accel = accelerate.Accelerator()
    from accelerate.logging import get_logger
    logger = get_logger('')
    # datasets.utils.logging.set_verbosity(logger.log_level)
    transformers.utils.logging.set_verbosity(log_lvl)

    logger.info(f'num processes: {accel.num_processes}')
    logger.info(f'mixed precision: {accel.mixed_precision}')
    logger.info(f'accelerator state: {accel.state}')

    assert not (args.pretrained_model is not None and args.base_model is not None), "only one of these args must be set"

    if accel.is_main_process:
        config = {
            'cli_args': dict(vars(args)),
        }
        logger.info(f'saving experiment configuration to {args.exp_path}')
        Path(args.exp_path).mkdir(parents=True)
        json.dump(config, open(os.path.join(args.exp_path, 'config.json'), 'w'), indent=4)

    if args.pretrained_model is None:
        # create tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        # create base model config
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

        config.torch_dtype = "float32"  # weights in float32, at training precision is controlled by accelerate
        config.vocab_size = tokenizer.vocab_size
        config.pad_token_id = tokenizer.pad_token_id
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.use_cache = False
    else:
        config = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
            print(f'using pretrained model tokenizer: {args.pretrained_model}')
        except:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            print(f'using base model tokenizer: {args.tokenizer}')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    rmt_config = RMT2SegmConfig(pretrained_model=args.pretrained_model, base_config=config,
                                n_mem_tokens=args.n_mem_tokens, K=args.K,
                                n_ctrl_tokens=args.n_ctrl_tokens,
                                use_mem_proj=args.use_mem_proj, mem_proj_mode=args.mem_proj_mode,
                                use_reconstruction_loss=args.use_reconstruction_loss,
                                reconstruction_loss_weight=args.reconstruction_loss_weight,
                                use_write_head=args.use_write_head, attn_implementation=args.attn_implementation)

    # Create rmt model
    model = RMT2Segm(rmt_config)

    if args.init_checkpoint is not None:
        missing_k, unexpected_k = model.load_state_dict(load_file(args.init_checkpoint), strict=False)
        if len(missing_k) != 0:
            logger.info(f'{missing_k} were not loaded from checkpoint! These parameters were randomly initialized.')
        if len(unexpected_k) != 0:
            logger.info(f'{unexpected_k} were found in checkpoint, but model is not expecting them!')

    if accel.mixed_precision == 'bf16':
        model.to(torch.bfloat16)

    logger.info(f'model config: {model.config}')
    logger.info(f'model: {model}')
    logger.info(f'model.dtype: {model.dtype}')

    segment_size = args.segment_size
    context_size = args.context_size
    
    logger.info(f"[dataset] segment_size: {segment_size}")
    logger.info(f"[dataset] context_size: {context_size}")
    
    def group_texts(examples, seg_size, context_size=None):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_len = len(concatenated[list(examples.keys())[0]])
        if context_size is None:
            result = {k: [t[i:i+seg_size] for i in range(0, total_len, seg_size)] for k, t in concatenated.items()}
        else:
            result = {k: [t[max(0, i-context_size):i+seg_size] for i in range(context_size, total_len, seg_size)] for k, t in concatenated.items()}
        return result
    
    with accel.main_process_first():
        if 'wikitext' in args.dataset_name:
            raw_dataset = datasets.load_dataset("wikitext", name="wikitext-103-raw-v1", streaming=False)
            train_dataset = raw_dataset["train"]#.select(range(1100, len(raw_dataset["train"])))
            valid_dataset = raw_dataset["validation"]#.select(range(100))
        else:
            raise NotImplementedError(f"LM dataset {args.dataset_name} not implemented")
        
        def tok(batch):
            return tokenizer(batch["text"], add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
        
        logger.info(f"[dataset] tokenizing train dataset...")
        train_dataset = train_dataset.map(tok, batched=True, batch_size=10_000, num_proc=32)
        logger.info(f"[dataset] tokenizing valid dataset...")
        valid_dataset = valid_dataset.map(tok, batched=True, batch_size=10_000, num_proc=32)
        
        logger.info(f"[dataset] grouping train dataset into segments...")
        train_dataset = train_dataset.select_columns(['input_ids']).map(lambda x: group_texts(x, segment_size, context_size), batched=True)
        logger.info(f"[dataset] grouping valid dataset into segments...")
        valid_dataset = valid_dataset.select_columns(['input_ids']).map(lambda x: group_texts(x, segment_size, context_size), batched=True)
        
    dataset = {'train': train_dataset, 'valid': valid_dataset}
    
    def data_collator(batch):
        return collate_fn(batch, tokenizer)

    output_dir = Path(args.exp_path)

    if args.total_batch_size is None:
        args.total_batch_size = args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps
    else:
        args_total_bs = args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps
        assert args.total_batch_size == args_total_bs

    # Training arguments
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
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        lr_scheduler_type=args.lr_scheduler_type,

        eval_strategy='steps',
        save_strategy='steps',
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to='tensorboard',
        metric_for_best_model=args.metric_for_best_model,
        load_best_model_at_end=True,
        eval_on_start=True,
        greater_is_better=False,
        remove_unused_columns=False,
        include_num_input_tokens_seen=False,  # input_ids is a dict, so HF Trainer cant get number of tokens
        include_for_metrics=['inputs'],
        save_total_limit=1,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=args.seed,
    )

    # Initialize Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        data_collator=data_collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    # Train the model
    trainer.train()
    logger.info('training done. running final evaluation...')
    metrics = trainer.evaluate(dataset['valid'])
    logger.info(f'{metrics}')
    trainer.save_metrics(split='all', metrics=metrics)
    trainer.state.save_to_json(output_dir / 'trainer_state.json')