import json
import logging
import os
from pathlib import Path
import math
from itertools import chain

import torch
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass, field
import datasets

import accelerate
import transformers
from transformers import (
    AutoConfig, AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback, TrainerCallback,
    HfArgumentParser
)
from torch.nn.utils.rnn import pad_sequence


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_lvl = logging.INFO
logging.basicConfig(format=logger_fmt, level=log_lvl)
logger = logging.getLogger('')

logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")


def collate_fn(batch, tokenizer):
    input_ids_list = []
    
    for sample in batch:
        input_ids = torch.tensor(sample['input_ids'], dtype=torch.long)
        input_ids_list.append(input_ids)
    
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (input_ids != tokenizer.pad_token_id).to(dtype=torch.long)
    labels = input_ids.clone()
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


class StopOnMetricValue(TrainerCallback):
    def __init__(self, metric_name: str, value: float, higher_is_better: bool = True):
        self.metric_name = metric_name
        self.value = value
        self.higher_is_better = higher_is_better

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if not self.metric_name.startswith("eval_"):
            metric_to_check = f"eval_{self.metric_name}"
        metric_value = metrics.get(metric_to_check)
        if metric_value is None:
            return
        operator = np.greater_equal if self.higher_is_better else np.less_equal
        if operator(metric_value, self.value):
            control.should_training_stop = True
            logger.info(f'metric {self.metric_name}={metric_value:.4f} >= {self.value:.4f}, stopping training..')


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
    pretrained_model: Optional[str] = field(default=None)
    n_layer: Optional[int] = field(default=4)
    n_head: Optional[int] = field(default=4)
    n_embd: Optional[int] = field(default=128)
    max_position_embeddings: Optional[int] = field(default=None)
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
        logger.info('saving experiment configuration..')
        Path(args.exp_path).mkdir(parents=True)
        json.dump(config, open(os.path.join(args.exp_path, 'config.json'), 'w'), indent=4)

    if args.pretrained_model is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        model = AutoModelForCausalLM.from_pretrained(args.pretrained_model)
    else:
        # create tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        # create model config
        if args.base_model == 'gpt2':
            config = AutoConfig.from_pretrained('gpt2')
            config.n_layer = args.n_layer
            config.n_head = args.n_head
            config.n_embd = args.n_embd
            if args.max_position_embeddings is not None:
                config.n_positions = args.max_position_embeddings
        elif args.base_model == 'pythia':
            config = AutoConfig.from_pretrained('EleutherAI/pythia-160m')
            config.num_hidden_layers = args.n_layer
            config.num_attention_heads = args.n_head
            config.hidden_size = args.n_embd
            config.intermediate_size = config.hidden_size * 4
            if args.max_position_embeddings is not None:
                config.max_position_embeddings = args.max_position_embeddings
        elif args.base_model == 'llama':
            config = AutoConfig.from_pretrained('meta-llama/Llama-3.2-1B')
            config.num_hidden_layers = args.n_layer
            config.num_attention_heads = args.n_head
            config.num_key_value_heads = args.n_head
            config.hidden_size = args.n_embd
            config.head_dim = config.hidden_size // config.num_attention_heads
            config.intermediate_size = config.hidden_size * 4
            if args.max_position_embeddings is not None:
                config.rope_scaling = None
                config.rope_theta = 10000.0
                config.max_position_embeddings = args.max_position_embeddings
        elif args.base_model == 'mamba':
            config = AutoConfig.from_pretrained('state-spaces/mamba-130m-hf')
            config.num_hidden_layers = args.n_layer
            config.n_layer = args.n_layer
            config.hidden_size = args.n_embd
            config.d_model = args.n_embd
            config.expand = 4
            config.intermediate_size = config.expand * config.hidden_size
            config.d_inner = config.expand * config.hidden_size
            config.time_step_rank = math.ceil(config.hidden_size / 16)
        else:
            raise ValueError(f'Unsupported base model: {args.base_model}')

        config.torch_dtype = "float32"  # weights in float32, at training precision is controlled by accelerate
        config.vocab_size = tokenizer.vocab_size
        config.pad_token_id = tokenizer.pad_token_id
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        # create model
        model = AutoModelForCausalLM.from_config(config)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.config.use_cache = False
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    logger.info(f'model config: {model.config}')
    logger.info(f'model: {model}')

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
            train_dataset = raw_dataset["train"]
            valid_dataset = raw_dataset["validation"]
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
        include_num_input_tokens_seen=True,
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
