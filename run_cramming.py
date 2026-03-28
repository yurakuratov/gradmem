"""
Cramming experiment: compress text into memory via gradient descent on a frozen LM.

Supports multiple memory architectures via ``--model_class``:
    gradmem  – prepend trainable memory tokens  (GradMemGPT)
    gradlora – inject low-rank residual x ← x + B A x at layer l  (GradLoRA)

Interface contract — model.forward() accepts:
    input_ids = {"context_input_ids": [B,S], "query_input_ids": [B,S]}
and returns a dict with at least:
    "predictions": [B, S, V]  logits at query positions

Metrics reported per length:
    acc           – token-level reconstruction accuracy (t_2…t_S)
    mem_loss      – CE with optimised memory (sum over tokens, per sample)
    baseline_loss – CE without memory (sum over tokens, per sample)
    info_gain     – baseline_loss − mem_loss  (same token set, fair comparison)
"""

import argparse
import json
import logging
import os
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from datasets import load_dataset

log = logging.getLogger(__name__)


# ── helpers ─────────────────────────────────────────────────────────────────

def baseline_loss(base_model, token_ids):
    """Standard next-token CE of the base LM without memory (sum per sample)."""
    with torch.no_grad():
        logits = base_model(input_ids=token_ids).logits
        V = logits.shape[-1]
        B, S = token_ids.shape
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, V),
            token_ids[:, 1:].reshape(-1),
            reduction="none",
        ).view(B, S - 1).sum(1)  # [B]


def evaluate(model, token_ids):
    """Run the model's compress-then-reconstruct forward pass.

    Returns per-sample (acc, mem_loss, inner_loop_stats), evaluated on the
    common token set t_2…t_S so that info_gain is a fair comparison.
    """
    B, S = token_ids.shape

    with torch.no_grad():
        output = model(
            input_ids={
                "context_input_ids": token_ids,
                "query_input_ids": token_ids,
            },
        )

    predictions = output["predictions"]  # [B, S, V]
    V = predictions.shape[-1]
    targets = token_ids[:, 1:]           # [B, S-1]
    pred_logits = predictions[:, :-1]    # [B, S-1, V]

    acc = (pred_logits.argmax(-1) == targets).float().mean(1)  # [B]
    mem_loss = F.cross_entropy(
        pred_logits.reshape(-1, V), targets.reshape(-1), reduction="none",
    ).view(B, S - 1).sum(1)  # [B]

    return acc, mem_loss, output.get("inner_loop_stats", {})


# ── entry point ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Cramming: compress PG19 text into memory embeddings")

    g = ap.add_argument_group("model")
    g.add_argument("--model_class", choices=["gradmem", "gradlora"],
                   default="gradmem")
    g.add_argument("--model", default="gpt2")
    g.add_argument("--dtype", default="float32",
                   choices=["float32", "float16", "bfloat16"])
    g.add_argument("--device", default="cuda")

    g = ap.add_argument_group("memory optimisation")
    g.add_argument("--n_mem_tokens", type=int, default=1,
                   help="(gradmem only) number of memory tokens")
    g.add_argument("--layer_idx", type=int, default=0,
                   help="(gradlora only) layer index for low-rank residual")
    g.add_argument("--rank", type=int, default=4,
                   help="(gradlora only) rank of A, B matrices")
    g.add_argument("--n_steps", type=int, default=1000)
    g.add_argument("--lr", type=float, default=0.01)
    g.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    g.add_argument("--clip_norm", type=float, default=None)
    g.add_argument("--early_stop_acc", type=float, default=0.99)
    g.add_argument("--early_stop_check_every", type=int, default=100)

    g = ap.add_argument_group("data")
    g.add_argument("--dataset", default="pg19")
    g.add_argument("--split", default="test")
    g.add_argument("--n_samples", type=int, default=3,
                   help="number of PG19 texts to average over per length")

    g = ap.add_argument_group("sweep")
    g.add_argument("--lengths", type=int, nargs="+", default=[32, 64, 128, 256, 512],
                   help="list of text lengths to evaluate")
    g.add_argument("--threshold", type=float, default=0.99,
                   help="stop when mean accuracy drops below this")

    g = ap.add_argument_group("misc")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--output_dir", default="cramming_results",
                   help="directory for results and logs")
    g.add_argument("--run_name", default=None,
                   help="base name for .json and .log files (auto-generated if omitted)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.run_name is None:
        if args.model_class == "gradmem":
            args.run_name = (f"{args.model.split('/')[-1]}_mem{args.n_mem_tokens}"
                             f"_{args.optimizer}_lr{args.lr}_steps{args.n_steps}")
        else:
            args.run_name = (f"{args.model.split('/')[-1]}_lora_l{args.layer_idx}"
                             f"_r{args.rank}_{args.optimizer}_lr{args.lr}"
                             f"_steps{args.n_steps}")

    json_path = os.path.join(args.output_dir, f"{args.run_name}.json")
    log_path  = os.path.join(args.output_dir, f"{args.run_name}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="w"),
        ],
    )
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    # ── model ──────────────────────────────────────────────────────────────
    log.info(f"Loading {args.model_class} / {args.model} ({args.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    common_kw = dict(
        pretrained_model=args.model,
        K=args.n_steps,
        lr=args.lr,
        use_adam=(args.optimizer == "adam"),
        grad_mode="none",
        inner_clip_norm=args.clip_norm,
        early_stop_acc=args.early_stop_acc,
        early_stop_check_every=args.early_stop_check_every,
    )

    if args.model_class == "gradmem":
        from grad_memgpt import GradMemGPT, GradMemGPTConfig
        config = GradMemGPTConfig(n_mem_tokens=args.n_mem_tokens, **common_kw)
        model = GradMemGPT(config).to(device=device, dtype=dtype)
        mem_prefix_len = args.n_mem_tokens
    else:
        from grad_lora import GradLoRA, GradLoRAConfig
        config = GradLoRAConfig(
            layer_idx=args.layer_idx, rank=args.rank, **common_kw)
        model = GradLoRA(config).to(device=device, dtype=dtype)
        mem_prefix_len = 0

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    base_model = model.model
    max_ctx = getattr(base_model.config, "max_position_embeddings",
                      getattr(base_model.config, "n_positions", 1024))
    max_text = max_ctx - mem_prefix_len
    log.info(f"Context window {max_ctx}, max text length {max_text}")

    # ── data ───────────────────────────────────────────────────────────────
    log.info(f"Loading {args.dataset} ({args.split})")
    ds = load_dataset(args.dataset, split=args.split, trust_remote_code=True)

    token_cache = []
    for sample in ds:
        toks = tokenizer.encode(sample["text"], add_special_tokens=False)
        if len(toks) >= max_text:
            token_cache.append(toks)
        if len(token_cache) >= args.n_samples:
            break
    if not token_cache:
        raise RuntimeError(
            f"No texts >= {max_text} tokens in {args.dataset}/{args.split}")
    log.info(f"Cached {len(token_cache)} texts (>= {max_text} tokens each)")

    # ── sweep ──────────────────────────────────────────────────────────────
    results = []
    for length in args.lengths:
        if length > max_text:
            log.info(f"Length {length} exceeds context ({max_text}), stopping.")
            break

        log.info("=" * 60)
        log.info(f"Length = {length}")

        batch = torch.stack(
            [torch.tensor(toks[:length]) for toks in token_cache]
        ).to(device)

        try:
            bl = baseline_loss(base_model, batch)
            acc, ml, stats = evaluate(model, batch)
        except torch.cuda.OutOfMemoryError:
            log.warning(f"  OOM at length {length}, stopping sweep.")
            torch.cuda.empty_cache()
            break

        info_gain = bl - ml

        entry = dict(
            length=length,
            mean_acc=acc.mean().item(),
            mean_mem_loss=ml.mean().item(),
            mean_baseline_loss=bl.mean().item(),
            mean_info_gain=info_gain.mean().item(),
            accs=acc.tolist(),
            mem_losses=ml.tolist(),
            baseline_losses=bl.tolist(),
            info_gains=info_gain.tolist(),
        )
        if "inner_loss" in stats:
            entry["inner_loss"] = stats["inner_loss"].item()
        if "inner_steps" in stats:
            entry["inner_steps"] = int(stats["inner_steps"])
        results.append(entry)

        steps_str = f"  steps={entry['inner_steps']}" if "inner_steps" in entry else ""
        log.info(f"  acc={entry['mean_acc']:.4f}  "
                 f"mem_loss={entry['mean_mem_loss']:.4f}  "
                 f"baseline={entry['mean_baseline_loss']:.4f}  "
                 f"info_gain={entry['mean_info_gain']:.4f}"
                 f"{steps_str}")

        if entry["mean_acc"] < args.threshold:
            prev = results[-2]["length"] if len(results) > 1 else 0
            log.info(f"Accuracy {entry['mean_acc']:.4f} < {args.threshold}. "
                     f"Max compressible length: {prev}")
            break

    # ── summary table ──────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info(f"{'Length':>8}  {'Accuracy':>10}  {'Mem Loss':>10}  "
             f"{'Baseline':>10}  {'Info Gain':>10}  {'Steps':>6}")
    log.info("-" * 64)
    for r in results:
        steps = r.get("inner_steps", "")
        log.info(f"{r['length']:>8}  {r['mean_acc']:>10.4f}  "
                 f"{r['mean_mem_loss']:>10.4f}  "
                 f"{r['mean_baseline_loss']:>10.4f}  "
                 f"{r['mean_info_gain']:>10.4f}  "
                 f"{steps:>6}")

    with open(json_path, "w") as f:
        json.dump(dict(config=vars(args), results=results), f, indent=2)
    log.info(f"Results saved to {json_path}")
    log.info(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
