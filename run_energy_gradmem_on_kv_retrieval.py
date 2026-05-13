import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import accelerate
import datasets
import torch
import transformers
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer, EarlyStoppingCallback, HfArgumentParser, Trainer, TrainingArguments

from energy_gradmem import EnergyGradMem, EnergyGradMemConfig
from run_gradmemgpt_on_kv_retrieval import (
    CustomTrainer,
    StopOnMetricValue,
    collate_fn,
    compute_metrics_fn,
    preprocess_logits_for_metrics,
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("WANDB_PROJECT", "gradmem")

logger_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
log_lvl = logging.INFO
logging.basicConfig(format=logger_fmt, level=log_lvl)
logger = logging.getLogger("")
logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")


@dataclass
class ExperimentArgs:
    exp_path: str = field()
    per_device_batch_size: int = field()
    data_path: Optional[str] = field(default=None)
    hf_dataset: Optional[str] = field(default="irodkin/kv_retrieval")
    hf_subset: Optional[str] = field(default=None)
    n_pairs: Optional[int] = field(default=8)
    key_size: Optional[int] = field(default=2)
    value_size: Optional[int] = field(default=2)
    vocab_size: Optional[int] = field(default=62)
    tokenizer_path: str = field(default="./tokenizers/kv_alphabet_62/")
    gradient_accumulation_steps: Optional[int] = field(default=1)
    total_batch_size: Optional[int] = field(default=None)
    metric_for_best_model: Optional[str] = field(default="token_accuracy")
    warmup_steps: Optional[int] = field(default=1000)
    max_steps: Optional[int] = field(default=50000)
    logging_steps: Optional[int] = field(default=100)
    eval_steps: Optional[int] = field(default=100)
    weight_decay: Optional[float] = field(default=0.0)
    learning_rate: Optional[float] = field(default=1e-4)
    lr_scheduler_type: Optional[str] = field(default="constant_with_warmup")
    early_stopping_patience: Optional[int] = field(default=50)
    seed: Optional[int] = field(default=142)
    base_model: Optional[str] = field(default=None)
    pretrained_model: Optional[str] = field(default=None)
    init_checkpoint: Optional[str] = field(default=None)
    n_layer: Optional[int] = field(default=4)
    n_head: Optional[int] = field(default=4)
    n_embd: Optional[int] = field(default=128)
    max_context_length: Optional[int] = field(default=None)

    memory_backend: Optional[str] = field(default="prefix")
    n_mem_tokens: Optional[int] = field(default=8)
    K: Optional[int] = field(default=2)
    last_K_second_order: Optional[int] = field(default=None)
    inner_lr: Optional[float] = field(default=0.01)
    use_adam: Optional[bool] = field(default=False)
    grad_mode: Optional[str] = field(default="second")
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
    lora_mem_placement: Optional[str] = field(default="between_layers")
    lora_mem_r: Optional[int] = field(default=8)
    lora_mem_alpha: Optional[int] = field(default=16)
    lora_mem_dropout: Optional[float] = field(default=0.0)
    lora_mem_layers: Optional[str] = field(default="all")
    lora_mem_target_modules: Optional[str] = field(default=None)
    kv_mem_layers: Optional[str] = field(default="all")
    freeze_backbone: Optional[bool] = field(default=False)
    use_gradient_checkpointing: Optional[bool] = field(default=False)
    attn_implementation: Optional[str] = field(default="eager")
    add_inner_loss_to_outer: Optional[bool] = field(default=False)
    inner_loss_weight: Optional[float] = field(default=None)

    inner_objective: Optional[str] = field(default="lstm")
    energy_hidden_size: Optional[int] = field(default=None)
    energy_num_layers: Optional[int] = field(default=2)
    energy_dropout: Optional[float] = field(default=0.0)
    energy_future_mode: Optional[str] = field(default="next_token")
    energy_ce_guidance: Optional[bool] = field(default=False)
    energy_ce_guidance_alpha: Optional[float] = field(default=0.01)


def build_base_config(args, tokenizer):
    if args.pretrained_model is not None:
        return None
    if args.base_model == "gpt2":
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = args.n_layer
        config.n_head = args.n_head
        config.n_embd = args.n_embd
    elif args.base_model == "pythia":
        config = AutoConfig.from_pretrained("EleutherAI/pythia-160m")
        config.num_hidden_layers = args.n_layer
        config.num_attention_heads = args.n_head
        config.hidden_size = args.n_embd
        config.intermediate_size = config.hidden_size * 4
    elif args.base_model == "llama":
        config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        config.num_hidden_layers = args.n_layer
        config.num_attention_heads = args.n_head
        config.num_key_value_heads = args.n_head
        config.hidden_size = args.n_embd
        config.head_dim = config.hidden_size // config.num_attention_heads
        config.intermediate_size = config.hidden_size * 4
    else:
        raise ValueError(f"Unsupported base model: {args.base_model}")

    config.torch_dtype = "float32"
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.use_cache = False
    return config


def build_model_config(args, base_config):
    return EnergyGradMemConfig(
        pretrained_model=args.pretrained_model,
        base_config=base_config,
        memory_backend=args.memory_backend,
        n_mem_tokens=args.n_mem_tokens,
        K=args.K,
        last_K_second_order=args.last_K_second_order,
        lr=args.inner_lr,
        use_adam=args.use_adam,
        grad_mode=args.grad_mode,
        n_ctrl_tokens=args.n_ctrl_tokens,
        inner_clip_value=args.inner_clip_value,
        inner_clip_norm=args.inner_clip_norm,
        use_mem_proj=args.use_mem_proj,
        mem_proj_mode=args.mem_proj_mode,
        use_write_head=args.use_write_head,
        use_write_lora=args.use_write_lora,
        write_lora_r=args.write_lora_r,
        write_lora_alpha=args.write_lora_alpha,
        write_lora_dropout=args.write_lora_dropout,
        write_lora_target_modules=args.write_lora_target_modules,
        lora_mem_placement=args.lora_mem_placement,
        lora_mem_r=args.lora_mem_r,
        lora_mem_alpha=args.lora_mem_alpha,
        lora_mem_dropout=args.lora_mem_dropout,
        lora_mem_layers=args.lora_mem_layers,
        lora_mem_target_modules=args.lora_mem_target_modules,
        kv_mem_layers=args.kv_mem_layers,
        freeze_backbone=args.freeze_backbone,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        attn_implementation=args.attn_implementation,
        add_inner_loss_to_outer=args.add_inner_loss_to_outer,
        inner_loss_weight=args.inner_loss_weight,
        inner_objective=args.inner_objective,
        energy_hidden_size=args.energy_hidden_size,
        energy_num_layers=args.energy_num_layers,
        energy_dropout=args.energy_dropout,
        energy_future_mode=args.energy_future_mode,
        energy_ce_guidance=args.energy_ce_guidance,
        energy_ce_guidance_alpha=args.energy_ce_guidance_alpha,
    )


def load_kv_dataset(args):
    if args.data_path is not None:
        return datasets.load_from_disk(args.data_path)

    subset = args.hf_subset
    if subset is None:
        subset = f"N{args.n_pairs}-K{args.key_size}V{args.value_size}-V{args.vocab_size}"
    return datasets.load_dataset(args.hf_dataset, subset)


def split_dataset(dataset):
    train = dataset["train"]
    if "valid" in dataset:
        valid = dataset["valid"]
    elif "validation" in dataset:
        valid = dataset["validation"]
    elif "test" in dataset:
        valid = dataset["test"]
    else:
        raise ValueError(f"Dataset has no valid/validation/test split. Available splits: {list(dataset.keys())}")
    return train, valid


if __name__ == "__main__":
    args = HfArgumentParser(ExperimentArgs).parse_args_into_dataclasses()[0]
    accel = accelerate.Accelerator()
    from accelerate.logging import get_logger

    logger = get_logger("")
    transformers.utils.logging.set_verbosity(log_lvl)
    logger.info(f"num processes: {accel.num_processes}")
    logger.info(f"mixed precision: {accel.mixed_precision}")
    logger.info(f"accelerator state: {accel.state}")

    assert not (args.pretrained_model is not None and args.base_model is not None), "only one of these args must be set"

    if accel.is_main_process:
        Path(args.exp_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(args.exp_path, "config.json"), "w") as f:
            json.dump({"cli_args": dict(vars(args))}, f, indent=4)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model or args.tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base_config = build_base_config(args, tokenizer)

    model = EnergyGradMem(build_model_config(args, base_config))
    if args.init_checkpoint is not None:
        missing_k, unexpected_k = model.load_state_dict(load_file(args.init_checkpoint), strict=False)
        logger.info(f"missing keys from checkpoint: {missing_k}")
        logger.info(f"unexpected keys from checkpoint: {unexpected_k}")
    if accel.mixed_precision == "bf16":
        model.to(torch.bfloat16)

    logger.info(f"model config: {model.config}")
    logger.info(f"model.dtype: {model.dtype}")

    dataset = load_kv_dataset(args)
    train_dataset, valid_dataset = split_dataset(dataset)

    def data_collator(batch):
        return collate_fn(batch, tokenizer, max_context_length=args.max_context_length)

    ignore_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in ["!", "|"]]

    def compute_metrics(eval_pred):
        return compute_metrics_fn(eval_pred, ignore_token_ids, tokenizer)

    if args.total_batch_size is None:
        args.total_batch_size = args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps
    else:
        actual_total = args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps
        assert args.total_batch_size == actual_total

    output_dir = Path(args.exp_path)
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
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to="wandb",
        run_name=os.environ.get("WANDB_NAME", output_dir.name),
        metric_for_best_model=args.metric_for_best_model,
        load_best_model_at_end=True,
        eval_on_start=True,
        greater_is_better=True,
        remove_unused_columns=False,
        include_num_input_tokens_seen=False,
        include_for_metrics=["inputs"],
        save_total_limit=1,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=args.seed,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
            StopOnMetricValue(metric_name="exact_match", value=1.0, higher_is_better=True),
        ],
    )
    trainer.train()
    logger.info("training done. running final evaluation...")
    metrics = trainer.evaluate(valid_dataset)
    logger.info(f"{metrics}")
    trainer.save_metrics(split="all", metrics=metrics)
    trainer.state.save_to_json(output_dir / "trainer_state.json")
