import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer

from grad_memgpt import GradMemGPT, GradMemGPTConfig

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.formatted_text import HTML
except ImportError:
    PromptSession = None
    WordCompleter = None
    FileHistory = None
    HTML = None

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.align import Align
except ImportError:
    Console = None
    Panel = None
    Align = None


CONSOLE = Console() if Console is not None else None


def _find_subsequence(haystack, needle):
    if len(needle) == 0:
        return 0
    if len(needle) > len(haystack):
        return None
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return None


def infer_special_wrapper_ids(tokenizer):
    probe_raw = tokenizer("Hi", add_special_tokens=False)["input_ids"]
    probe_full = tokenizer("Hi", add_special_tokens=True)["input_ids"]
    start = _find_subsequence(probe_full, probe_raw)
    if start is None:
        raise ValueError(
            "Could not infer tokenizer special-token wrapper from add_special_tokens behavior. "
            f"probe_raw={probe_raw}, probe_full={probe_full}"
        )
    return probe_full[:start], probe_full[start + len(probe_raw):]


def resolve_model_path(run_path, checkpoint=None):
    run_path = Path(run_path)
    if checkpoint is not None:
        p = Path(checkpoint)
        if p.is_dir():
            single_path = p / "model.safetensors"
            sharded_index_path = p / "model.safetensors.index.json"
            if single_path.exists():
                model_path = single_path
            elif sharded_index_path.exists():
                model_path = sharded_index_path
            else:
                raise ValueError(
                    f"Checkpoint directory has neither model.safetensors nor model.safetensors.index.json: {p}"
                )
        else:
            model_path = p
        if not model_path.exists():
            raise ValueError(f"Checkpoint path does not exist: {model_path}")
        return model_path

    candidates = []
    for d in run_path.glob("checkpoint-*"):
        if d.is_dir() and d.name.startswith("checkpoint-"):
            step_str = d.name.split("checkpoint-", 1)[1]
            if step_str.isdigit():
                single_path = d / "model.safetensors"
                sharded_index_path = d / "model.safetensors.index.json"
                if single_path.exists():
                    candidates.append((int(step_str), single_path))
                elif sharded_index_path.exists():
                    candidates.append((int(step_str), sharded_index_path))

    if not candidates:
        raise ValueError(
            "No checkpoint-* directories with model.safetensors or model.safetensors.index.json "
            f"found under run path: {run_path}. "
            "Pass --checkpoint explicitly."
        )

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def load_checkpoint_state_dict(model_path):
    model_path = Path(model_path)
    if model_path.name == "model.safetensors":
        return load_file(str(model_path)), "single"

    if model_path.name == "model.safetensors.index.json":
        with open(model_path, "r") as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map")
        if not isinstance(weight_map, dict) or len(weight_map) == 0:
            raise ValueError(f"Invalid sharded checkpoint index file: {model_path}")

        shard_names = sorted(set(weight_map.values()))
        state_dict = {}
        for shard_name in shard_names:
            shard_path = model_path.parent / shard_name
            if not shard_path.exists():
                raise ValueError(f"Missing shard file referenced by index: {shard_path}")
            state_dict.update(load_file(str(shard_path)))
        return state_dict, "sharded"

    raise ValueError(
        f"Unsupported checkpoint path: {model_path}. "
        "Expected model.safetensors or model.safetensors.index.json"
    )


def build_kv_base_config(cli_args, tokenizer):
    base_model = cli_args.get("base_model")
    if base_model is None:
        raise ValueError("KV retrieval run config requires base_model when pretrained_model is not set")

    n_layer = int(cli_args.get("n_layer", 4))
    n_head = int(cli_args.get("n_head", 4))
    n_embd = int(cli_args.get("n_embd", 128))

    if base_model == "gpt2":
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = n_layer
        config.n_head = n_head
        config.n_embd = n_embd
    elif base_model == "pythia":
        config = AutoConfig.from_pretrained("EleutherAI/pythia-160m")
        config.num_hidden_layers = n_layer
        config.num_attention_heads = n_head
        config.hidden_size = n_embd
        config.intermediate_size = config.hidden_size * 4
    elif base_model == "llama":
        config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        config.num_hidden_layers = n_layer
        config.num_attention_heads = n_head
        config.num_key_value_heads = n_head
        config.hidden_size = n_embd
        config.head_dim = config.hidden_size // config.num_attention_heads
        config.intermediate_size = config.hidden_size * 4
    else:
        raise ValueError(f"Unsupported base_model for KV retrieval: {base_model}")

    config.torch_dtype = "float32"
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.use_cache = False
    return config


def build_model_config(cli_args, base_config=None):
    pretrained_model = cli_args.get("pretrained_model")

    return GradMemGPTConfig(
        pretrained_model=pretrained_model,
        base_config=base_config,
        memory_backend=cli_args.get("memory_backend", "prefix"),
        n_mem_tokens=cli_args["n_mem_tokens"],
        K=cli_args["K"],
        last_K_second_order=cli_args.get("last_K_second_order"),
        lr=cli_args["inner_lr"],
        use_adam=cli_args["use_adam"],
        grad_mode=cli_args["grad_mode"],
        n_ctrl_tokens=cli_args["n_ctrl_tokens"],
        inner_clip_value=cli_args.get("inner_clip_value"),
        inner_clip_norm=cli_args.get("inner_clip_norm"),
        use_mem_proj=cli_args["use_mem_proj"],
        mem_proj_mode=cli_args["mem_proj_mode"],
        use_write_head=cli_args["use_write_head"],
        use_write_lora=cli_args.get("use_write_lora", False),
        write_lora_r=cli_args.get("write_lora_r", 8),
        write_lora_alpha=cli_args.get("write_lora_alpha", 16),
        write_lora_dropout=cli_args.get("write_lora_dropout", 0.0),
        write_lora_target_modules=cli_args.get("write_lora_target_modules"),
        lora_mem_placement=cli_args.get("lora_mem_placement", "between_layers"),
        lora_mem_r=cli_args.get("lora_mem_r", 8),
        lora_mem_alpha=cli_args.get("lora_mem_alpha", 16),
        lora_mem_dropout=cli_args.get("lora_mem_dropout", 0.0),
        lora_mem_layers=cli_args.get("lora_mem_layers", None),
        lora_mem_target_modules=cli_args.get("lora_mem_target_modules"),
        kv_mem_layers=cli_args.get("kv_mem_layers", None),
        freeze_backbone=cli_args.get("freeze_backbone", False),
        use_gradient_checkpointing=cli_args["use_gradient_checkpointing"],
        attn_implementation=cli_args.get("attn_implementation", "eager"),
        add_inner_loss_to_outer=cli_args.get("add_inner_loss_to_outer", False),
        inner_loss_weight=cli_args.get("inner_loss_weight"),
    )


def extract_memory_state(output, backend):
    if backend == "prefix":
        memory_state = {"mem_batch": output["mem"]}
        if "W" in output:
            memory_state["W_batch"] = output["W"]
        if "b" in output:
            memory_state["b_batch"] = output["b"]
        return memory_state
    if backend == "lora":
        return {"lora_mem": output["lora_mem"]}
    if backend == "kv_cache":
        return {"kv_mem": output["kv_mem"]}
    raise ValueError(f"Unsupported backend={backend}")


def encode_write_ids(tokenizer, text, special_prefix_ids, special_suffix_ids, max_context_length):
    text_prefix_ids = tokenizer("text: ", add_special_tokens=False)["input_ids"]
    tokenize_kwargs = {"add_special_tokens": False}
    if max_context_length is not None:
        tokenize_kwargs["max_length"] = max_context_length
        tokenize_kwargs["truncation"] = True
    ctx_ids = tokenizer(text, **tokenize_kwargs)["input_ids"]
    return special_prefix_ids + text_prefix_ids + ctx_ids + special_suffix_ids


def encode_query_ids(tokenizer, text, special_prefix_ids, special_suffix_ids):
    q_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return special_prefix_ids + q_ids + special_suffix_ids


def encode_write_ids_kv(tokenizer, text, max_context_length):
    kwargs = {"add_special_tokens": True}
    if max_context_length is not None:
        kwargs["max_length"] = max_context_length
        kwargs["truncation"] = True
    return tokenizer(text, **kwargs)["input_ids"]


def encode_query_ids_kv(tokenizer, text):
    return tokenizer(text, add_special_tokens=True)["input_ids"]


@torch.no_grad()
def write_memory_once(model, write_ids, device):
    ids = torch.tensor([write_ids], dtype=torch.long, device=device)
    out = model(
        {
            "context_input_ids": ids,
            "query_input_ids": ids,
        },
        labels=None,
        return_mem=True,
    )
    return extract_memory_state(out, model.memory_backend), out.get("inner_loop_stats", {})


@torch.no_grad()
def read_logits_from_memory(model, memory_state, query_ids, device):
    query_ids = torch.tensor([query_ids], dtype=torch.long, device=device)
    backend = model.memory_backend_impl
    pad_id = model.model.config.pad_token_id
    if pad_id is None:
        raise ValueError("Model pad_token_id is None")

    dummy_context = torch.full((1, 1), pad_id, dtype=torch.long, device=device)
    batch_ctx = backend.prepare_batch(dummy_context, query_ids, pad_id)
    read_batch = backend.build_read_inputs(memory_state, batch_ctx)
    read_model_kwargs = read_batch.get("model_kwargs", {})

    with backend.activation_context(memory_state):
        with model._disable_write_lora():
            read_out = model.model(
                inputs_embeds=read_batch["inputs_embeds"],
                return_dict=True,
                **read_model_kwargs,
            )
            logits = read_out.logits

    logits = logits[:, read_batch["logits_start"]:read_batch["logits_start"] + read_batch["pred_len"], :]
    return logits


def sample_next_token(next_logits, do_sample, temperature, top_p):
    if not do_sample:
        return torch.argmax(next_logits, dim=-1, keepdim=True)

    if temperature <= 0:
        raise ValueError(f"temperature must be > 0 when do_sample=True, got {temperature}")

    logits = next_logits / temperature
    probs = torch.softmax(logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        next_sorted = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_indices.gather(-1, next_sorted)
        return next_token

    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_from_memory(
    model,
    memory_state,
    prompt_ids,
    device,
    max_new_tokens,
    do_sample,
    temperature,
    top_p,
    eos_token_id,
):
    if len(prompt_ids) == 0:
        raise ValueError("Query must have at least one token")

    current_ids = list(prompt_ids)
    new_tokens = []

    for _ in range(max_new_tokens):
        logits = read_logits_from_memory(model, memory_state, current_ids, device)
        if logits.size(1) == 0:
            raise ValueError(
                "Read produced zero logits length. For lora/kv_cache use a non-empty query seed/prompt."
            )

        next_logits = logits[:, -1, :]
        next_token = sample_next_token(next_logits, do_sample, temperature, top_p)
        token_id = int(next_token.item())
        current_ids.append(token_id)
        new_tokens.append(token_id)

        if eos_token_id is not None and token_id == eos_token_id:
            break

    return new_tokens


def print_help():
    print("Commands:")
    print("  /write <text>   Write text into memory (overwrites current memory)")
    print("  /ask <query>    Query current memory and generate an answer")
    print("  /ask_base <q>   Query original model (no memory)")
    print("  /reset          Drop current memory state")
    print("  /help           Show commands")
    print("  /exit           Exit")


def print_bubble(role, text):
    if CONSOLE is None or Panel is None or Align is None:
        print(f"{role}: {text}")
        return

    if role == "You":
        panel = Panel(text, title=role, border_style="cyan", expand=False)
        CONSOLE.print(Align.right(panel))
    elif role == "Base Model":
        panel = Panel(text, title=role, border_style="yellow", expand=False)
        CONSOLE.print(Align.left(panel))
    elif role == "Memory + Base Model":
        panel = Panel(text, title=role, border_style="green", expand=False)
        CONSOLE.print(Align.left(panel))
    else:
        panel = Panel(text, title=role, border_style="magenta", expand=False)
        CONSOLE.print(Align.left(panel))


def print_status(text):
    print_bubble("System", text)


def read_command(session):
    if session is None:
        if CONSOLE is not None:
            CONSOLE.print("[bold cyan]┃ Type command (/help) >[/bold cyan]")
        return input("\n> ").strip()
    if HTML is not None:
        prompt = HTML("<ansicyan><b>┃ Type command (/help) ></b></ansicyan> ")
    else:
        prompt = "\n> "
    return session.prompt(prompt).strip()


@torch.no_grad()
def generate_without_memory(
    model,
    prompt_ids,
    device,
    max_new_tokens,
    do_sample,
    temperature,
    top_p,
    eos_token_id,
):
    if len(prompt_ids) == 0:
        raise ValueError("Query must have at least one token")

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with model._disable_write_lora():
        out_ids = model.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            pad_token_id=model.model.config.pad_token_id,
        )
    return out_ids[0, input_ids.size(1):].tolist()


def main():
    parser = argparse.ArgumentParser(description="Interactive playground for GradMemGPT runs")
    parser.add_argument("--run_path", required=True, help="Path to run directory containing config.json")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint dir or model.safetensors path")
    parser.add_argument("--task", required=True, choices=["text_compression", "kv_retrieval"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"], help="Model dtype")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--show_token_ids", action="store_true", help="Show token id diagnostics in /write and /ask")
    parser.add_argument("--plain_repl", action="store_true", help="Use basic input() even if prompt_toolkit is in env")
    args = parser.parse_args()

    run_path = Path(args.run_path)
    config_path = run_path / "config.json"
    if not config_path.exists():
        raise ValueError(f"config.json not found at: {config_path}")

    with open(config_path, "r") as f:
        run_cfg = json.load(f)
    cli_args = run_cfg.get("cli_args")
    if not isinstance(cli_args, dict):
        raise ValueError(f"Invalid run config format in {config_path}: missing cli_args dict")
    task = args.task

    model_path = resolve_model_path(run_path, args.checkpoint)

    pretrained_model = cli_args.get("pretrained_model")
    if task == "text_compression":
        if pretrained_model is None:
            raise ValueError("text_compression chat expects pretrained_model in run config")
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        model_base_config = None
        model_source = f"pretrained:{pretrained_model}"
        tokenizer_source = pretrained_model
    else:
        if pretrained_model is not None:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
            model_base_config = None
            model_source = f"pretrained:{pretrained_model}"
            tokenizer_source = pretrained_model
        else:
            tokenizer_path = cli_args.get("tokenizer_path")
            if tokenizer_path is None:
                raise ValueError("kv_retrieval run config requires tokenizer_path when pretrained_model is not set")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            model_base_config = build_kv_base_config(cli_args, tokenizer)
            model_source = f"base_config:{cli_args.get('base_model')}"
            tokenizer_source = tokenizer_path

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_cfg = build_model_config(cli_args, base_config=model_base_config)
    model = GradMemGPT(model_cfg)
    state_dict, checkpoint_format = load_checkpoint_state_dict(model_path)
    missing_k, unexpected_k = model.load_state_dict(state_dict, strict=False)
    if missing_k:
        print(f"[warn] missing keys from checkpoint: count={len(missing_k)}")
        print(f"[warn] missing sample: {missing_k[:20]}")
    if unexpected_k:
        print(f"[warn] unexpected keys in checkpoint: count={len(unexpected_k)}")
        print(f"[warn] unexpected sample: {unexpected_k[:20]}")

    missing_lora_mem = [k for k in missing_k if k.startswith("lora_mem_A0.") or k.startswith("lora_mem_B0.")]
    unexpected_lora_mem = [k for k in unexpected_k if k.startswith("lora_mem_A0.") or k.startswith("lora_mem_B0.")]
    if missing_lora_mem or unexpected_lora_mem:
        raise ValueError(
            "Checkpoint/model mismatch in lora memory parameters. "
            f"missing_lora_mem={len(missing_lora_mem)}, unexpected_lora_mem={len(unexpected_lora_mem)}. "
            "Ensure run config and checkpoint were produced with the same lora_mem_placement/"
            "lora_mem_layers/lora_mem_target_modules settings."
        )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model.to(device)
    if args.dtype == "bf16" and device.type == "cuda":
        model.to(torch.bfloat16)
    model.eval()

    max_context_length = cli_args.get("max_context_length")
    if task == "text_compression":
        special_prefix_ids, special_suffix_ids = infer_special_wrapper_ids(tokenizer)

        def encode_write(text):
            return encode_write_ids(tokenizer, text, special_prefix_ids, special_suffix_ids, max_context_length)

        def encode_query(text):
            return encode_query_ids(tokenizer, text, special_prefix_ids, special_suffix_ids)
    else:
        special_prefix_ids, special_suffix_ids = [], []

        def encode_write(text):
            return encode_write_ids_kv(tokenizer, text, max_context_length)

        def encode_query(text):
            return encode_query_ids_kv(tokenizer, text)

    print_status(
        f"Run: {run_path}\n"
        f"Checkpoint: {model_path} (format={checkpoint_format})\n"
        f"Task: {task}\n"
        f"Model source: {model_source}\n"
        f"Tokenizer source: {tokenizer_source}\n"
        f"Memory backend: {model.memory_backend}"
    )
    if PromptSession is None:
        print_status("prompt_toolkit is not installed; using plain REPL. Install via: pip install prompt_toolkit")
    elif args.plain_repl:
        print_status("Using plain REPL due to --plain_repl")
    if CONSOLE is None:
        print("[warn] rich is not installed; output bubbles are disabled. Install via: pip install rich")
    print_help()

    session = None
    if PromptSession is not None and not args.plain_repl:
        history_path = os.path.expanduser("~/.gradmemgpt_text_compression_history")
        completer = WordCompleter(
            ["/write", "/ask", "/ask_base", "/reset", "/help", "/exit"],
            ignore_case=True,
            sentence=True,
        )
        session = PromptSession(history=FileHistory(history_path), completer=completer)

    memory_state = None
    eos_token_id = tokenizer.eos_token_id

    while True:
        try:
            line = read_command(session)
        except (EOFError, KeyboardInterrupt):
            print_status("Exiting.")
            break

        if not line:
            continue
        if line in {"/exit", "exit", "quit", "q"}:
            break
        if line in {"/help", "help", "h"}:
            print_help()
            continue
        if line == "/reset":
            memory_state = None
            print_status("Memory state cleared.")
            continue

        if line.startswith("/write "):
            text = line[len("/write "):].strip()
            if not text:
                print_status("Empty write text.")
                continue
            print_bubble("You", f"/write {text}")
            write_ids = encode_write(text)
            memory_state, inner_loop_stats = write_memory_once(model, write_ids, device)
            inner_loss = inner_loop_stats.get("inner_loss")
            written_text = tokenizer.decode(write_ids, skip_special_tokens=False)
            if inner_loss is None:
                msg = f"text in memory: {written_text}\ninner_loss=NA"
                if args.show_token_ids:
                    token_ids_str = ", ".join(str(t) for t in write_ids)
                    msg = f"{msg}\nn_tokens={len(write_ids)}\ntoken_ids=[{token_ids_str}]"
                print_status(msg)
            else:
                if torch.is_tensor(inner_loss):
                    inner_loss = float(inner_loss.detach().item())
                else:
                    inner_loss = float(inner_loss)
                msg = f"text in memory: {written_text}\ninner_loss={inner_loss:.4f}"
                if args.show_token_ids:
                    token_ids_str = ", ".join(str(t) for t in write_ids)
                    msg = f"{msg}\nn_tokens={len(write_ids)}\ntoken_ids=[{token_ids_str}]"
                print_status(msg)
            continue

        if line.startswith("/ask "):
            if memory_state is None:
                print_status("Memory is empty. Use /write <text> first.")
                continue
            query = line[len("/ask "):].strip()
            if not query:
                print_status("Empty query.")
                continue
            print_bubble("You", f"/ask {query}")
            query_ids = encode_query(query)
            new_tokens = generate_from_memory(
                model=model,
                memory_state=memory_state,
                prompt_ids=query_ids,
                device=device,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=eos_token_id,
            )
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
            print_bubble("Memory + Base Model", answer)
            if args.show_token_ids:
                query_ids_str = ", ".join(str(t) for t in query_ids)
                gen_ids_str = ", ".join(str(t) for t in new_tokens)
                print_status(
                    f"read_input_token_ids=[{query_ids_str}]\n"
                    f"read_input_n_tokens={len(query_ids)}\n"
                    f"generated_token_ids=[{gen_ids_str}]\n"
                    f"generated_n_tokens={len(new_tokens)}"
                )
            continue

        if line.startswith("/ask_base "):
            query = line[len("/ask_base "):].strip()
            if not query:
                print_status("Empty query.")
                continue
            print_bubble("You", f"/ask_base {query}")
            query_ids = encode_query(query)
            new_tokens = generate_without_memory(
                model=model,
                prompt_ids=query_ids,
                device=device,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=eos_token_id,
            )
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
            print_bubble("Base", answer)
            if args.show_token_ids:
                query_ids_str = ", ".join(str(t) for t in query_ids)
                gen_ids_str = ", ".join(str(t) for t in new_tokens)
                print_status(
                    f"read_input_token_ids=[{query_ids_str}]\n"
                    f"read_input_n_tokens={len(query_ids)}\n"
                    f"generated_token_ids=[{gen_ids_str}]\n"
                    f"generated_n_tokens={len(new_tokens)}"
                )
            continue

        print_status("Unknown command. Use /help")


# CUDA_VISIBLE_DEVICES=2 python chat_with_gradmem.py \
# --run_path ./runs/pg19_chunks_w8000/gradmem_Llama-3.2-3B_N16_lora_target_modules_r8a16_gate_proj_up_proj_down_proj_layers_all_K1_ilr0.04_frozen_grad_second_bs_64_lr_5e-05/run_1_20260228142413_bf16 \
# --max_new_tokens 32 --dtype bf16 --task text_compression
# todo: chat templates?
if __name__ == "__main__":
    main()
