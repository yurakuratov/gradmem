import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
from transformers.cache_utils import DynamicCache
from contextlib import contextmanager
import math
import logging
try:
    import attn_double_bwd  # noqa: F401  # side-effect: registers attention kernels
except ImportError:
    attn_double_bwd = None

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:  # pragma: no cover - handled via runtime checks
    LoraConfig = None
    TaskType = None
    get_peft_model = None


def get_backbone(m):
    if hasattr(m, "get_base_model"):
        base = m.get_base_model()
        if base is not None and base is not m:
            return get_backbone(base)
    if hasattr(m, "base_model"):
        base = getattr(m, "base_model")
        if base is not None and base is not m:
            return get_backbone(base)
    # most HF CausalLM classes define base_model_prefix, e.g. "transformer" (GPT-2), "model" (LLaMA)
    if hasattr(m, "base_model_prefix") and hasattr(m, m.base_model_prefix):
        return getattr(m, m.base_model_prefix)
    # robust fallback
    for attr in ("model", "transformer", "gpt_neox", "backbone", "decoder", "base_model"):
        if hasattr(m, attr):
            return getattr(m, attr)
    raise AttributeError("Could not locate backbone submodule")


logger = logging.getLogger(__name__)


def _is_main_process():
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


class GradMemGPTConfig(PretrainedConfig):
    """
    Configuration class for GradMemGPT.
    """
    model_type = "grad_memgpt"

    def __init__(self,
                 pretrained_model=None,
                 base_config=None,
                 memory_backend="prefix",
                 n_mem_tokens=8,
                 K=2,
                 lr=0.01,
                 use_adam=False,
                 grad_mode="second",
                 last_K_second_order=None,
                 n_ctrl_tokens=0,
                 inner_clip_value=None,
                 inner_clip_norm=None,
                 use_mem_proj=False,
                 mem_proj_mode="none",
                 use_write_head=False,
                 use_write_lora=False,
                 write_lora_r=8,
                 write_lora_alpha=16,
                 write_lora_dropout=0.0,
                 write_lora_target_modules=None,
                 lora_mem_placement="between_layers",
                 lora_mem_r=8,
                 lora_mem_alpha=16,
                 lora_mem_dropout=0.0,
                 lora_mem_layers=None,
                 kv_mem_layers=None,
                 freeze_backbone=False,
                 use_gradient_checkpointing=False,
                 attn_implementation="eager",
                 add_inner_loss_to_outer=False,
                 inner_loss_weight=None,
                 **kwargs):
        """
        Args:
            pretrained_model: str, name of pretrained model to load (e.g., 'gpt2')
            base_config: dict or PretrainedConfig, config for base model when creating from scratch
            memory_backend: str, memory implementation ("prefix", "lora", "kv_cache")
            n_mem_tokens: int, number of memory tokens
            K: int, number of inner loop steps
            lr: float, inner loop learning rate, it is a effective learning rate per sample
            use_adam: bool, whether to use Adam optimizer in inner loop
            grad_mode: str, gradient mode ("none", "first", "second")
            last_K_second_order: int, use second order update for last K inner gradient steps only
            n_ctrl_tokens: int, number of control tokens
            inner_clip_value: float, gradient clipping value
            inner_clip_norm: float, gradient clipping norm
            use_mem_proj: bool, whether to use memory projection
            mem_proj_mode: str, memory projection mode ("none", "proj", "per_sample")
            use_write_head: bool, whether to use write head
            use_write_lora: bool, enable LoRA adapters during WRITE phase only
            write_lora_r: int, LoRA rank for WRITE adapters
            write_lora_alpha: int, LoRA alpha for WRITE adapters
            write_lora_dropout: float, LoRA dropout for WRITE adapters
            write_lora_target_modules: list[str]|str|None, target module names (None/"auto" for defaults)
            lora_mem_placement: str, where LoRA memory is injected (currently: "between_layers")
            lora_mem_r: int, rank of LoRA memory adapters
            lora_mem_alpha: int, alpha scaling for LoRA memory adapters
            lora_mem_dropout: float, dropout on hidden states before LoRA memory projection
            lora_mem_layers: str|list[int]|None, transformer layers to inject LoRA memory
                examples: None or "all" (all layers), "last_4" (last 4 layers),
                "0,3,7" (explicit indices), [0, 3, 7] (explicit indices as list),
                "none"/"auto"/"" (treated as all layers)
            kv_mem_layers: str|list[int]|None, transformer layers to inject KV-cache memory
                examples: None or "all" (all layers), "last_4" (last 4 layers),
                "0,3,7" (explicit indices), [0, 3, 7] (explicit indices as list),
                "none"/"auto"/"" (treated as all layers)
            freeze_backbone: bool, freeze backbone weights (READ+WRITE), except LoRA/write head/mem proj
            use_gradient_checkpointing: bool, turn on gradient checkpointing supported by HF models
            add_inner_loss_to_outer: bool, outer loss = target_loss + inner_loss_weight * inner_loss_mean
            inner_loss_weight: float, weight of inner loss in combined loss
        """
        super().__init__(**kwargs)

        if pretrained_model is not None:
            self.pretrained_model = pretrained_model
            self.base_config = None
        else:
            self.pretrained_model = None
            self.base_config = base_config

        # GradMemGPT specific parameters
        self.memory_backend = memory_backend
        self.n_mem_tokens = n_mem_tokens
        self.K = K
        self.lr = lr
        self.use_adam = use_adam
        self.grad_mode = grad_mode
        self.n_ctrl_tokens = n_ctrl_tokens
        self.inner_clip_value = inner_clip_value
        self.inner_clip_norm = inner_clip_norm
        self.use_mem_proj = use_mem_proj
        self.mem_proj_mode = mem_proj_mode
        self.use_write_head = use_write_head
        self.use_write_lora = use_write_lora
        self.write_lora_r = write_lora_r
        self.write_lora_alpha = write_lora_alpha
        self.write_lora_dropout = write_lora_dropout
        self.write_lora_target_modules = write_lora_target_modules
        self.lora_mem_placement = lora_mem_placement
        self.lora_mem_r = lora_mem_r
        self.lora_mem_alpha = lora_mem_alpha
        self.lora_mem_dropout = lora_mem_dropout
        self.lora_mem_layers = lora_mem_layers
        self.kv_mem_layers = kv_mem_layers
        self.freeze_backbone = freeze_backbone
        self.last_K_second_order = K if last_K_second_order is None else last_K_second_order
        if grad_mode != "second":
            self.last_K_second_order = 0
        self.last_K_second_order = max(0, min(self.last_K_second_order, K))
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.attn_implementation = attn_implementation
        self.add_inner_loss_to_outer = add_inner_loss_to_outer
        self.inner_loss_weight = inner_loss_weight

        # Validate mem_proj_mode settings
        assert mem_proj_mode in ["none", "proj", "per_sample"]
        assert self.use_mem_proj == (mem_proj_mode != 'none'), "use_mem_proj must be True if mem_proj_mode is set"
        assert self.memory_backend in ["prefix", "lora", "kv_cache"], (
            "memory_backend must be one of: prefix, lora, kv_cache"
        )
        if self.memory_backend == "lora":
            assert self.lora_mem_placement in ["between_layers"], (
                "lora_mem_placement currently supports only 'between_layers'"
            )


class MemoryBackend:
    def __init__(self, owner):
        self.owner = owner

    @contextmanager
    def activation_context(self, memory_state):
        _ = memory_state
        yield


class InputPrefixMemoryBackend(MemoryBackend):
    def init_memory_state(self, batch_size):
        o = self.owner
        mem_batch = o.mem.unsqueeze(0).expand(batch_size, -1, -1).clone()
        if o.grad_mode == "none":
            mem_batch = mem_batch.detach().requires_grad_(True)
        else:
            mem_batch = mem_batch.requires_grad_(True)

        memory_state = {"mem_batch": mem_batch}
        memory_state_initial = {"mem_batch": mem_batch.detach().clone()}

        if o.mem_proj_mode == "per_sample":
            W_batch = o.mem_proj.weight.unsqueeze(0).expand(batch_size, -1, -1).clone()
            b_batch = o.mem_proj.bias.unsqueeze(0).expand(batch_size, -1).clone()
            if o.grad_mode == "none":
                W_batch = W_batch.detach().requires_grad_(True)
                b_batch = b_batch.detach().requires_grad_(True)
            else:
                W_batch = W_batch.requires_grad_(True)
                b_batch = b_batch.requires_grad_(True)
            memory_state["W_batch"] = W_batch
            memory_state["b_batch"] = b_batch

        return memory_state, memory_state_initial

    def prepare_batch(self, context_input_ids, query_input_ids, pad_id):
        o = self.owner
        B = context_input_ids.size(0)
        mem_offset = o.n_mem_tokens + o.n_ctrl_tokens * 2

        batch_ctx = {
            "context_input_ids": context_input_ids,
            "query_input_ids": query_input_ids,
            "ctx_emb": o.model.get_input_embeddings()(context_input_ids),
            "qry_emb": o.model.get_input_embeddings()(query_input_ids),
            "mem_offset": mem_offset,
        }

        lm_labels = context_input_ids.clone()
        lm_labels[lm_labels == pad_id] = -100
        mask = (lm_labels != -100)

        if o.attn_implementation in ('jvp_flash', 'hvp_semi_manual'):
            pad_list = [0, -(batch_ctx["ctx_emb"].size(1) + mem_offset) % 32, 0, 0]
            mask = F.pad(mask, pad_list, "constant", 0)
            lm_labels = F.pad(lm_labels, pad_list, "constant", -100)
            batch_ctx["ctx_emb"] = F.pad(batch_ctx["ctx_emb"], [0, 0] + pad_list, "constant", 0)

        batch_ctx["lm_labels"] = lm_labels
        batch_ctx["mask"] = mask

        if o.n_ctrl_tokens > 0:
            batch_ctx["write_st_batch"] = o.write_st.unsqueeze(0).expand(B, -1, -1)
            batch_ctx["write_end_batch"] = o.write_end.unsqueeze(0).expand(B, -1, -1)
            batch_ctx["read_st_batch"] = o.read_st.unsqueeze(0).expand(B, -1, -1)
            batch_ctx["read_end_batch"] = o.read_end.unsqueeze(0).expand(B, -1, -1)

        return batch_ctx

    def build_write_inputs(self, memory_state, batch_ctx):
        o = self.owner
        mem_batch = memory_state["mem_batch"]
        if o.mem_proj_mode == "none":
            mem_inp = mem_batch
        elif o.mem_proj_mode == "proj":
            mem_inp = o.mem_proj(mem_batch)
        else:
            mem_inp = o._apply_linear(mem_batch, memory_state["W_batch"], memory_state["b_batch"])

        if o.n_ctrl_tokens > 0:
            x_ctx = torch.cat(
                [batch_ctx["write_st_batch"], mem_inp, batch_ctx["write_end_batch"], batch_ctx["ctx_emb"]],
                dim=1,
            )
        else:
            x_ctx = torch.cat([mem_inp, batch_ctx["ctx_emb"]], dim=1)

        return {
            "inputs_embeds": x_ctx,
            "lm_labels": batch_ctx["lm_labels"],
            "mask": batch_ctx["mask"],
            "logits_start": batch_ctx["mem_offset"] - 1,
            "label_shift": 0,
        }

    def build_read_inputs(self, memory_state, batch_ctx):
        o = self.owner
        mem_batch = memory_state["mem_batch"]
        if o.mem_proj_mode == "none":
            mem_inp = mem_batch
        elif o.mem_proj_mode == "proj":
            mem_inp = o.mem_proj(mem_batch)
        else:
            mem_inp = o._apply_linear(mem_batch, memory_state["W_batch"], memory_state["b_batch"])

        if o.n_ctrl_tokens > 0:
            x_qry = torch.cat(
                [batch_ctx["read_st_batch"], mem_inp, batch_ctx["read_end_batch"], batch_ctx["qry_emb"]],
                dim=1,
            )
        else:
            x_qry = torch.cat([mem_inp, batch_ctx["qry_emb"]], dim=1)

        if o.attn_implementation in ('jvp_flash', 'hvp_semi_manual'):
            x_qry = F.pad(x_qry, [0, 0, 0, -x_qry.size(1) % 32], "constant", 0)

        return {
            "inputs_embeds": x_qry,
            "logits_start": batch_ctx["mem_offset"] - 1,
            "pred_len": batch_ctx["qry_emb"].size(1) + 1,
            "label_shift": 0,
        }

    def inner_params(self, memory_state):
        params = [memory_state["mem_batch"]]
        if self.owner.mem_proj_mode == "per_sample":
            params += [memory_state["W_batch"], memory_state["b_batch"]]
        return params

    def assign_inner_params(self, memory_state, new_params):
        memory_state["mem_batch"] = new_params[0]
        if self.owner.mem_proj_mode == "per_sample":
            memory_state["W_batch"] = new_params[1]
            memory_state["b_batch"] = new_params[2]

    def maybe_detach_after_step(self, memory_state):
        if self.owner.grad_mode != "none":
            return
        memory_state["mem_batch"] = memory_state["mem_batch"].detach().requires_grad_(True)
        if self.owner.mem_proj_mode == "per_sample":
            memory_state["W_batch"] = memory_state["W_batch"].detach().requires_grad_(True)
            memory_state["b_batch"] = memory_state["b_batch"].detach().requires_grad_(True)

    def compute_memory_stats(self, memory_state, memory_state_initial):
        mem_batch = memory_state["mem_batch"]
        mem_init = memory_state_initial["mem_batch"]
        mem_norm = mem_batch.norm(dim=(1, 2)).detach()
        delta_mem_norm = (mem_batch - mem_init).detach().norm(dim=(1, 2))
        return mem_norm, delta_mem_norm

    def attach_return_memory(self, output, memory_state):
        output["mem"] = memory_state["mem_batch"]
        if self.owner.mem_proj_mode == "per_sample":
            output["W"] = memory_state["W_batch"]
            output["b"] = memory_state["b_batch"]


class LoraMemoryBackend(MemoryBackend):
    def init_memory_state(self, batch_size):
        o = self.owner
        device = o.model.get_input_embeddings().weight.device
        memory_state = {"lora_mem": {}}
        memory_state_initial = {"lora_mem": {}}
        for layer_idx in o.lora_mem_layer_ids:
            A0 = o.lora_mem_A0[str(layer_idx)].to(device=device)
            B0 = o.lora_mem_B0[str(layer_idx)].to(device=device)
            A_batch = A0.unsqueeze(0).expand(batch_size, -1, -1).clone()
            B_batch = B0.unsqueeze(0).expand(batch_size, -1, -1).clone()
            if o.grad_mode == "none":
                A_batch = A_batch.detach().requires_grad_(True)
                B_batch = B_batch.detach().requires_grad_(True)
            else:
                A_batch = A_batch.requires_grad_(True)
                B_batch = B_batch.requires_grad_(True)
            memory_state["lora_mem"][layer_idx] = (A_batch, B_batch)
            memory_state_initial["lora_mem"][layer_idx] = (A_batch.detach().clone(), B_batch.detach().clone())
        return memory_state, memory_state_initial

    @contextmanager
    def activation_context(self, memory_state):
        with self.owner._enable_lora_memory(memory_state["lora_mem"]):
            yield

    def prepare_batch(self, context_input_ids, query_input_ids, pad_id):
        o = self.owner
        emb_layer = o.model.get_input_embeddings()
        ctx_emb = emb_layer(context_input_ids)
        qry_emb = emb_layer(query_input_ids)

        lm_labels = context_input_ids.clone()
        lm_labels[lm_labels == pad_id] = -100
        mask = (lm_labels != -100)

        x_ctx = ctx_emb
        if o.attn_implementation in ('jvp_flash', 'hvp_semi_manual'):
            pad_n = (-x_ctx.size(1)) % 32
            if pad_n:
                x_ctx = F.pad(x_ctx, [0, 0, 0, pad_n], "constant", 0)
                lm_labels = F.pad(lm_labels, [0, pad_n], "constant", -100)
                mask = F.pad(mask, [0, pad_n], "constant", 0)

        return {
            "x_ctx": x_ctx,
            "lm_labels": lm_labels,
            "mask": mask,
            "qry_emb": qry_emb,
        }

    def build_write_inputs(self, memory_state, batch_ctx):
        _ = memory_state
        return {
            "inputs_embeds": batch_ctx["x_ctx"],
            "lm_labels": batch_ctx["lm_labels"],
            "mask": batch_ctx["mask"],
            "logits_start": 0,
            "label_shift": 1,
        }

    def build_read_inputs(self, memory_state, batch_ctx):
        _ = memory_state
        o = self.owner
        x_qry = batch_ctx["qry_emb"]
        if o.attn_implementation in ('jvp_flash', 'hvp_semi_manual'):
            x_qry = F.pad(x_qry, [0, 0, 0, (-x_qry.size(1)) % 32], "constant", 0)
        return {
            "inputs_embeds": x_qry,
            "logits_start": 0,
            "pred_len": batch_ctx["qry_emb"].size(1),
            "label_shift": 1,
        }

    def inner_params(self, memory_state):
        params = []
        for layer_idx in self.owner.lora_mem_layer_ids:
            A_batch, B_batch = memory_state["lora_mem"][layer_idx]
            params.extend([A_batch, B_batch])
        return params

    def assign_inner_params(self, memory_state, new_params):
        p = 0
        for layer_idx in self.owner.lora_mem_layer_ids:
            memory_state["lora_mem"][layer_idx] = (new_params[p], new_params[p + 1])
            p += 2

    def maybe_detach_after_step(self, memory_state):
        if self.owner.grad_mode != "none":
            return
        for layer_idx in self.owner.lora_mem_layer_ids:
            A_batch, B_batch = memory_state["lora_mem"][layer_idx]
            memory_state["lora_mem"][layer_idx] = (
                A_batch.detach().requires_grad_(True),
                B_batch.detach().requires_grad_(True),
            )

    def compute_memory_stats(self, memory_state, memory_state_initial):
        mem_device = next(iter(memory_state["lora_mem"].values()))[0].device
        mem_norm_sq = torch.zeros(next(iter(memory_state["lora_mem"].values()))[0].size(0), device=mem_device)
        delta_mem_norm_sq = torch.zeros_like(mem_norm_sq)
        for layer_idx in self.owner.lora_mem_layer_ids:
            A_batch, B_batch = memory_state["lora_mem"][layer_idx]
            A_init, B_init = memory_state_initial["lora_mem"][layer_idx]
            mem_norm_sq = mem_norm_sq + A_batch.detach().pow(2).sum(dim=(1, 2))
            mem_norm_sq = mem_norm_sq + B_batch.detach().pow(2).sum(dim=(1, 2))
            delta_mem_norm_sq = delta_mem_norm_sq + (A_batch.detach() - A_init).pow(2).sum(dim=(1, 2))
            delta_mem_norm_sq = delta_mem_norm_sq + (B_batch.detach() - B_init).pow(2).sum(dim=(1, 2))
        return mem_norm_sq.sqrt(), delta_mem_norm_sq.sqrt()

    def attach_return_memory(self, output, memory_state):
        output["lora_mem"] = memory_state["lora_mem"]


class KVCacheMemoryBackend(MemoryBackend):
    def init_memory_state(self, batch_size):
        o = self.owner
        device = o.model.get_input_embeddings().weight.device
        memory_state = {"kv_mem": {}}
        memory_state_initial = {"kv_mem": {}}
        for layer_idx in o.kv_mem_layer_ids:
            K0 = o.kv_mem_K0[str(layer_idx)].to(device=device)
            V0 = o.kv_mem_V0[str(layer_idx)].to(device=device)
            K_batch = K0.unsqueeze(0).expand(batch_size, -1, -1, -1).clone()
            V_batch = V0.unsqueeze(0).expand(batch_size, -1, -1, -1).clone()
            if o.grad_mode == "none":
                K_batch = K_batch.detach().requires_grad_(True)
                V_batch = V_batch.detach().requires_grad_(True)
            else:
                K_batch = K_batch.requires_grad_(True)
                V_batch = V_batch.requires_grad_(True)
            memory_state["kv_mem"][layer_idx] = (K_batch, V_batch)
            memory_state_initial["kv_mem"][layer_idx] = (K_batch.detach().clone(), V_batch.detach().clone())
        return memory_state, memory_state_initial

    def prepare_batch(self, context_input_ids, query_input_ids, pad_id):
        o = self.owner
        emb_layer = o.model.get_input_embeddings()
        ctx_emb = emb_layer(context_input_ids)
        qry_emb = emb_layer(query_input_ids)

        lm_labels = context_input_ids.clone()
        lm_labels[lm_labels == pad_id] = -100
        mask = (lm_labels != -100)

        x_ctx = ctx_emb
        if o.attn_implementation in ('jvp_flash', 'hvp_semi_manual'):
            pad_n = (-x_ctx.size(1)) % 32
            if pad_n:
                x_ctx = F.pad(x_ctx, [0, 0, 0, pad_n], "constant", 0)
                lm_labels = F.pad(lm_labels, [0, pad_n], "constant", -100)
                mask = F.pad(mask, [0, pad_n], "constant", 0)

        return {
            "x_ctx": x_ctx,
            "lm_labels": lm_labels,
            "mask": mask,
            "qry_emb": qry_emb,
        }

    def build_write_inputs(self, memory_state, batch_ctx):
        return {
            "inputs_embeds": batch_ctx["x_ctx"],
            "lm_labels": batch_ctx["lm_labels"],
            "mask": batch_ctx["mask"],
            "logits_start": 0,
            "label_shift": 1,
            "model_kwargs": {
                "past_key_values": self.owner._build_dynamic_cache(memory_state["kv_mem"]),
                "use_cache": True,
            },
        }

    def build_read_inputs(self, memory_state, batch_ctx):
        o = self.owner
        x_qry = batch_ctx["qry_emb"]
        if o.attn_implementation in ('jvp_flash', 'hvp_semi_manual'):
            x_qry = F.pad(x_qry, [0, 0, 0, (-x_qry.size(1)) % 32], "constant", 0)
        return {
            "inputs_embeds": x_qry,
            "logits_start": 0,
            "pred_len": batch_ctx["qry_emb"].size(1),
            "label_shift": 1,
            "model_kwargs": {
                "past_key_values": self.owner._build_dynamic_cache(memory_state["kv_mem"]),
                "use_cache": True,
            },
        }

    def inner_params(self, memory_state):
        params = []
        for layer_idx in self.owner.kv_mem_layer_ids:
            K_batch, V_batch = memory_state["kv_mem"][layer_idx]
            params.extend([K_batch, V_batch])
        return params

    def assign_inner_params(self, memory_state, new_params):
        p = 0
        for layer_idx in self.owner.kv_mem_layer_ids:
            memory_state["kv_mem"][layer_idx] = (new_params[p], new_params[p + 1])
            p += 2

    def maybe_detach_after_step(self, memory_state):
        if self.owner.grad_mode != "none":
            return
        for layer_idx in self.owner.kv_mem_layer_ids:
            K_batch, V_batch = memory_state["kv_mem"][layer_idx]
            memory_state["kv_mem"][layer_idx] = (
                K_batch.detach().requires_grad_(True),
                V_batch.detach().requires_grad_(True),
            )

    def compute_memory_stats(self, memory_state, memory_state_initial):
        mem_device = next(iter(memory_state["kv_mem"].values()))[0].device
        mem_norm_sq = torch.zeros(next(iter(memory_state["kv_mem"].values()))[0].size(0), device=mem_device)
        delta_mem_norm_sq = torch.zeros_like(mem_norm_sq)
        for layer_idx in self.owner.kv_mem_layer_ids:
            K_batch, V_batch = memory_state["kv_mem"][layer_idx]
            K_init, V_init = memory_state_initial["kv_mem"][layer_idx]
            mem_norm_sq = mem_norm_sq + K_batch.detach().pow(2).sum(dim=(1, 2, 3))
            mem_norm_sq = mem_norm_sq + V_batch.detach().pow(2).sum(dim=(1, 2, 3))
            delta_mem_norm_sq = delta_mem_norm_sq + (K_batch.detach() - K_init).pow(2).sum(dim=(1, 2, 3))
            delta_mem_norm_sq = delta_mem_norm_sq + (V_batch.detach() - V_init).pow(2).sum(dim=(1, 2, 3))
        return mem_norm_sq.sqrt(), delta_mem_norm_sq.sqrt()

    def attach_return_memory(self, output, memory_state):
        output["kv_mem"] = memory_state["kv_mem"]


class GradMemGPT(PreTrainedModel):
    """
    Transformer-decoder backbone + writable prefix memory (n_mem_tokens x d).
    """
    config_class = GradMemGPTConfig

    def __init__(self, config):
        """
        grad_mode: "none" | "first" | "second"
        none: stop grad in inner update. Outer optimizer ignores mem pathway.
            Initial params of self.mem are never trained. Per-sample memory is updated in inner loop.
        first: first-order update. Outer grads flow to mem, but ignore Hessian term (Straight-Through / FOMAML).
            Only outer loop gradients update self.mem, inner loop (second-order) gradients are ignored:
            self.mem.grad = mem_batch.grad.sum(0)
        second: second-order update. Full MAML. Outer grads include second-order term via a differentiable inner step.

        mem_proj_mode: "none" | "proj" | "per_sample"
        none: no linear projection of mem
        proj: one shared nn.Linear trained by the outer loop only, acts like a gate/preconditioner/tuned inner lr
            mem* = W x mem + b
            mem = mem - lr x W^T x grad -- inner loop update of per-sample mem
            -> mem* = W x (mem - lr x W^T x grad) + b = (W x mem + b) - lr x (W x W^T) x grad
            so W x W^T is a preconditioner of how to apply grads
            special cases:
                W = sI -- tuned learning rate
                W is diagonal -- per-dimension learning rates
                full-rank W -- mixing + scaling, more complex preconditioner
                todo: add constraints on W, e.g. W is diagonal, W is low-rank, ...
        per_sample: per-sample fast weights W_i,b_i updated in the inner loop;
            their initial values (self.mem_proj.*) are meta-learned by the outer loop

        inner loop: [write_st][mem][write_end][context]
        outer loop: [read_st][mem][read_end][query][target]

        write_st/write_end/read_st/read_end are parameters aka prompts, that can be used by model to control
            the write/read operation.
        n_ctrl_tokens = 1 means that [write_st] is a single token.

        mem is updated in inner loop, write_ctrl/read_ctrl/model_params/init_mem are trained by outer loop
        """
        super().__init__(config)

        if config.pretrained_model is not None and config.base_config is not None:
            raise ValueError("Only one of pretrained_model or base_config should be provided")
        if config.pretrained_model is None and config.base_config is None:
            raise ValueError("Either pretrained_model or base_config must be provided to instantiate GradMemGPT")

        # initialize base model, attention is eager to support backward pass over backward pass
        if config.pretrained_model is not None:
            self.model = AutoModelForCausalLM.from_pretrained(config.pretrained_model,
                                                              attn_implementation=config.attn_implementation)
        else:
            self.model = AutoModelForCausalLM.from_config(config.base_config,
                                                          attn_implementation=config.attn_implementation)
        self.attn_implementation = config.attn_implementation
        if self.attn_implementation in ("jvp_flash", "hvp_semi_manual") and attn_double_bwd is None:
            raise ImportError(
                f"attn_implementation={self.attn_implementation} requires triton/attn_double_bwd. "
                "Install triton or use a standard attention implementation (e.g. eager)."
            )
        if attn_double_bwd is None and _is_main_process():
            logger.info("triton/attn_double_bwd is not available; using standard attention kernels")

        self.memory_backend = getattr(config, "memory_backend", "prefix")
        self.lora_mem_placement = getattr(config, "lora_mem_placement", "between_layers")
        self.lora_mem_r = getattr(config, "lora_mem_r", 8)
        self.lora_mem_alpha = getattr(config, "lora_mem_alpha", 16)
        self.lora_mem_dropout = getattr(config, "lora_mem_dropout", 0.0)
        self.lora_mem_layers = getattr(config, "lora_mem_layers", None)
        self.kv_mem_layers = getattr(config, "kv_mem_layers", None)
        self._active_lora_memory = None
        self._lora_mem_hooks = []

        # write-phase LoRA (applies only during inner loop), additional params to train for WRITE operation
        self.use_write_lora = getattr(config, "use_write_lora", False)
        self.write_lora_r = getattr(config, "write_lora_r", 8)
        self.write_lora_alpha = getattr(config, "write_lora_alpha", 16)
        self.write_lora_dropout = getattr(config, "write_lora_dropout", 0.0)
        self.write_lora_target_modules = getattr(config, "write_lora_target_modules", None)
        self.freeze_backbone = getattr(config, "freeze_backbone", False)
        if self.use_write_lora:
            self._init_write_lora()
        if self.freeze_backbone:
            self._freeze_backbone_params()

        if self.memory_backend == "lora":
            self._init_lora_memory_between_layers()
        elif self.memory_backend == "kv_cache":
            self._init_kv_cache_memory()

        if self.memory_backend == "prefix":
            self.memory_backend_impl = InputPrefixMemoryBackend(self)
        elif self.memory_backend == "lora":
            self.memory_backend_impl = LoraMemoryBackend(self)
        elif self.memory_backend == "kv_cache":
            self.memory_backend_impl = KVCacheMemoryBackend(self)
        else:
            raise ValueError(f"Unsupported memory_backend={self.memory_backend}. Supported: prefix, lora, kv_cache")

        # store GradMemGPT parameters
        self.n_mem_tokens = config.n_mem_tokens
        self.n_ctrl_tokens = config.n_ctrl_tokens
        self.K = config.K
        self.last_K_second_order = config.last_K_second_order
        self.lr = config.lr
        self.use_adam = config.use_adam
        self.grad_mode = config.grad_mode
        self.inner_clip_value = config.inner_clip_value
        self.inner_clip_norm = config.inner_clip_norm
        self.use_mem_proj = config.use_mem_proj
        self.mem_proj_mode = config.mem_proj_mode
        if self.memory_backend == "lora" and self.mem_proj_mode != "none":
            raise ValueError("mem_proj_mode is not supported for memory_backend='lora'")
        if self.memory_backend == "kv_cache" and self.mem_proj_mode == "per_sample":
            raise ValueError(
                "mem_proj_mode='per_sample' is not supported for memory_backend='kv_cache'; "
                "use mem_proj_mode='none' or 'proj'"
            )
        self.use_write_head = config.use_write_head
        self.add_inner_loss_to_outer = config.add_inner_loss_to_outer
        self.inner_loss_weight = config.inner_loss_weight
        if self.add_inner_loss_to_outer:
            if self.inner_loss_weight is None:
                self.inner_loss_weight = 1.0
        else:
            self.inner_loss_weight = 0.0

        # memory parameters (shape = n_mem_tokens × d)
        n_embd = getattr(self.model.config, 'n_embd', self.model.config.hidden_size)
        if self.memory_backend == "prefix":
            # self.mem are inner loop per-sample params, intial states of mem (self.mem) are meta-learned
            self.mem = nn.Parameter(torch.randn(self.n_mem_tokens, n_embd) * 0.02)

            # optional mem projection linear layer
            if self.mem_proj_mode != "none":
                self.mem_proj = nn.Linear(n_embd, n_embd, bias=True)
                # initialize mem_proj to be identity
                with torch.no_grad():
                    nn.init.eye_(self.mem_proj.weight)
                    self.mem_proj.bias.zero_()
            else:
                self.mem_proj = None
        else:
            self.mem = None
            self.mem_proj = None

        # optional read/write control parameters (shape = n_ctrl_tokens × d)
        if self.n_ctrl_tokens > 0:
            # write ctrl tokens can be trained only by outer loop and only if grads flow through inner loop ("second")
            self.write_st = nn.Parameter(torch.randn(self.n_ctrl_tokens, n_embd) * 0.02)
            self.write_end = nn.Parameter(torch.randn(self.n_ctrl_tokens, n_embd) * 0.02)
            self.read_st = nn.Parameter(torch.randn(self.n_ctrl_tokens, n_embd) * 0.02)
            self.read_end = nn.Parameter(torch.randn(self.n_ctrl_tokens, n_embd) * 0.02)

        if self.use_write_head:
            V = self.model.config.vocab_size
            self.write_head = nn.Linear(n_embd, V, bias=False)

            if hasattr(self.model, 'get_output_embeddings'):
                head_params = self.model.get_output_embeddings().weight
            else:  # fallback to input embeddings
                head_params = self.model.get_input_embeddings().weight
            with torch.no_grad():
                self.write_head.weight.copy_(head_params.detach())

        self.tie_weights()
        self.main_input_name = "input_ids"
        self.model.config.use_cache = False
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

        # turn on gradient checkpointing to save gpu ram
        # currently, gradient checkpointing is not used in inner loop, so it wont save GPU RAM at forward pass.
        # but it's still saves some GPU RAM for training in outer loop
        if getattr(config, "use_gradient_checkpointing", False):
            self.gradient_checkpointing_enable()

        n_memory_params = self._count_memory_parameters()
        if _is_main_process():
            logger.info(f"GradMemGPT memory params (backend={self.memory_backend}): total={n_memory_params}")

    def floating_point_ops(self, inputs):
        # dummy method to satisfy base class and it's invocation by trainer:
        # Trainer supposes that `inputs`` is a tensor, not dict.
        return 0

    def tie_weights(self):
        self.model.tie_weights()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # force {"use_reentrant": False}
        if hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                # fallback for older HF versions
                self.model.gradient_checkpointing_enable()

    def _count_memory_parameters(self):
        if self.memory_backend == "prefix":
            return 0 if self.mem is None else self.mem.numel()
        if self.memory_backend == "lora":
            total = 0
            for p in self.lora_mem_A0.values():
                total += p.numel()
            for p in self.lora_mem_B0.values():
                total += p.numel()
            return total
        if self.memory_backend == "kv_cache":
            total = 0
            for p in self.kv_mem_K0.values():
                total += p.numel()
            for p in self.kv_mem_V0.values():
                total += p.numel()
            return total
        raise ValueError(f"Unsupported memory_backend={self.memory_backend}")

    @staticmethod
    def _parse_lora_targets(value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            targets = [str(v).strip() for v in value if str(v).strip()]
            return targets or None
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned == "" or cleaned.lower() in ("none", "auto"):
                return None
            targets = [v.strip() for v in cleaned.split(",") if v.strip()]
            return targets or None
        return None

    def _resolve_write_lora_targets(self):
        parsed = self._parse_lora_targets(self.write_lora_target_modules)
        if parsed:
            return parsed

        model_type = getattr(self.model.config, "model_type", None)
        targets_by_type = {
            "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "gpt2": ["c_attn", "c_proj", "c_fc"],
            "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        }
        if model_type in targets_by_type:
            return targets_by_type[model_type]

        raise ValueError(
            "write_lora_target_modules is not set and model_type is unknown. "
            "Please provide explicit target modules."
        )

    def _init_write_lora(self):
        if get_peft_model is None or LoraConfig is None or TaskType is None:
            raise ImportError("peft is required for write_lora. Please install the peft package.")

        target_modules = self._resolve_write_lora_targets()
        lora_config = LoraConfig(
            r=self.write_lora_r,
            lora_alpha=self.write_lora_alpha,
            lora_dropout=self.write_lora_dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)
        if not (hasattr(self.model, "disable_adapter_layers") and hasattr(self.model, "enable_adapter_layers")):
            raise RuntimeError("PEFT model does not support adapter toggling; cannot enforce write-only LoRA.")

        # keep base model trainable to preserve previous behavior
        for param in self.model.parameters():
            param.requires_grad = True

    def _freeze_backbone_params(self):
        for _, param in self.model.named_parameters():
            param.requires_grad = False
        if self.use_write_lora:
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True

    @contextmanager
    def _disable_write_lora(self):
        if not self.use_write_lora:
            yield
            return
        self.model.disable_adapter_layers()
        try:
            yield
        finally:
            self.model.enable_adapter_layers()

    def _adam_step(self, p, g, state, step_idx, lr, beta1=0.9, beta2=0.999, eps=1e-8, clip_value=10.0):
        """
        Functional Adam:
        - no in-place math on graph-tracked tensors
        - buffers detached, keep them in state dict
        - bias-correction uses step_idx
        - gradient norm is clipped

        Current impl computes updates under torch.no_grad(), so MAML/second-order paths are cut (by design).
        With grad_mode="second" it will still give first-order behavior.

        WARNING: not tested throughly!
        """
        # ---- 0. init ------------------------------------------------------ #
        if "m" not in state:
            state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        m, v = state["m"], state["v"]

        # ---- 1. clip grad to stabilise very first steps ------------------ #
        if clip_value is not None:
            g = torch.clamp(g, min=-clip_value, max=clip_value)

        # ---- 2. update moments (stay outside autograd graph) ------------- #
        with torch.no_grad():
            m_new = beta1 * m + (1 - beta1) * g
            v_new = beta2 * v + (1 - beta2) * (g * g)

            # bias‑correction with current step
            m_hat = m_new / (1 - beta1 ** step_idx)
            v_hat = v_new / (1 - beta2 ** step_idx)
            v_hat = torch.clamp(v_hat, min=eps)        # avoid √0 / 0

            step = lr * m_hat / (v_hat.sqrt() + eps)

        # ---- 3. write back *detached* buffers ---------------------------- #
        state["m"].copy_(m_new.detach())
        state["v"].copy_(v_new.detach())

        # ---- 4. return new parameter tensor (on graph via g) ------------- #
        return p - step

    def _sgd_step(self, p, g, lr=None, clip_value=None, clip_norm=None):
        """
        Stateless SGD with optional element-wise and per-sample total-norm clipping.
        Works for shapes (B,M,d), (B,d,d), (B,d).

        Args
        ----
        p : torch.Tensor          current parameter tensor
        g : torch.Tensor          gradient wrt p
        clip_value : float|None   element-wise clamp, e.g. 5.0
        clip_norm  : float|None   total-norm clamp, e.g. 1.0
        """

        if clip_value is not None:
            # simple element-wise clamp
            g = torch.clamp(g, -clip_value, clip_value)

        if clip_norm is not None:
            # scale gradient if its 2-norm is too large
            # check grad for each sample separately as we do per-sample optimization
            reduce_dims = tuple(range(1, g.ndim))           # all non-batch dims
            g_norm = g.norm(dim=reduce_dims, keepdim=True)  # (B,1,1)
            scale = clip_norm / (g_norm + 1e-6)
            g = torch.where(g_norm > clip_norm, g * scale, g)

        if lr is None:
            lr = self.lr

        return p - lr * g

    @staticmethod
    def _apply_linear(mem, W, b):
        """
        Functional linear on a batch of memories:
        mem: (B,M,d), W: (B,d,d) or None, b: (B,d) or None
        """
        if W is None:
            return mem
        return torch.baddbmm(b.unsqueeze(1), mem, W.transpose(1, 2))

    def _get_transformer_blocks(self):
        backbone = get_backbone(self.model)
        for attr in ("h", "layers", "block", "blocks"):
            if hasattr(backbone, attr):
                blocks = getattr(backbone, attr)
                if isinstance(blocks, (nn.ModuleList, list, tuple)):
                    return list(blocks)
        raise ValueError("Could not resolve transformer block list for LoRA memory placement")

    @staticmethod
    def _parse_lora_mem_layers(value, n_layers):
        if value is None:
            return list(range(n_layers))
        if isinstance(value, (list, tuple)):
            idx = sorted(set(int(v) for v in value))
            return idx
        if isinstance(value, str):
            cleaned = value.strip().lower()
            if cleaned in ("", "all", "none", "auto"):
                return list(range(n_layers))
            if cleaned.startswith("last_"):
                k = int(cleaned.split("last_", 1)[1])
                k = max(0, min(k, n_layers))
                return list(range(n_layers - k, n_layers))
            idx = sorted(set(int(v.strip()) for v in value.split(",") if v.strip() != ""))
            return idx
        raise ValueError("lora_mem_layers should be None, list[int], or str")

    def _init_lora_memory_between_layers(self):
        blocks = self._get_transformer_blocks()
        n_layers = len(blocks)
        layer_ids = self._parse_lora_mem_layers(self.lora_mem_layers, n_layers)
        for idx in layer_ids:
            if idx < 0 or idx >= n_layers:
                raise ValueError(f"lora_mem_layers has out-of-range index {idx} for {n_layers} layers")
        self.lora_mem_layer_ids = layer_ids

        hidden_size = getattr(self.model.config, "n_embd", getattr(self.model.config, "hidden_size", None))
        if hidden_size is None:
            raise ValueError("Could not infer hidden size from model config")
        self.lora_mem_scale = float(self.lora_mem_alpha) / float(max(1, self.lora_mem_r))
        self.lora_mem_A0 = nn.ParameterDict()
        self.lora_mem_B0 = nn.ParameterDict()
        for layer_idx in self.lora_mem_layer_ids:
            a = nn.Parameter(
                torch.zeros(hidden_size, self.lora_mem_r),
                requires_grad=True,
            )
            b = nn.Parameter(
                torch.zeros(self.lora_mem_r, hidden_size),
                requires_grad=True,
            )
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            nn.init.zeros_(b)
            self.lora_mem_A0[str(layer_idx)] = a
            self.lora_mem_B0[str(layer_idx)] = b

        self._register_lora_mem_hooks(blocks)

    def _init_kv_cache_memory(self):
        blocks = self._get_transformer_blocks()
        n_layers = len(blocks)
        layer_ids = self._parse_lora_mem_layers(self.kv_mem_layers, n_layers)
        for idx in layer_ids:
            if idx < 0 or idx >= n_layers:
                raise ValueError(f"kv_mem_layers has out-of-range index {idx} for {n_layers} layers")
        self.kv_mem_layer_ids = layer_ids

        hidden_size = getattr(self.model.config, "n_embd", getattr(self.model.config, "hidden_size", None))
        if hidden_size is None:
            raise ValueError("Could not infer hidden size from model config")

        num_attn_heads = getattr(
            self.model.config,
            "num_attention_heads",
            getattr(self.model.config, "n_head", None),
        )
        if num_attn_heads is None:
            raise ValueError("Could not infer number of attention heads from model config")

        num_kv_heads = getattr(self.model.config, "num_key_value_heads", num_attn_heads)
        head_dim = getattr(self.model.config, "head_dim", hidden_size // num_attn_heads)

        self.kv_mem_K0 = nn.ParameterDict()
        self.kv_mem_V0 = nn.ParameterDict()
        n_mem_tokens = int(getattr(self.config, "n_mem_tokens"))
        mem_proj_mode = getattr(self.config, "mem_proj_mode", "none")
        for layer_idx in self.kv_mem_layer_ids:
            k = nn.Parameter(
                torch.randn(num_kv_heads, n_mem_tokens, head_dim) * 0.02,
                requires_grad=True,
            )
            v = nn.Parameter(
                torch.randn(num_kv_heads, n_mem_tokens, head_dim) * 0.02,
                requires_grad=True,
            )
            self.kv_mem_K0[str(layer_idx)] = k
            self.kv_mem_V0[str(layer_idx)] = v

        if mem_proj_mode == "proj":
            self.mem_proj_K_l = nn.ModuleDict()
            self.mem_proj_V_l = nn.ModuleDict()
            for layer_idx in self.kv_mem_layer_ids:
                proj_k = nn.Linear(head_dim, head_dim, bias=True)
                proj_v = nn.Linear(head_dim, head_dim, bias=True)
                with torch.no_grad():
                    nn.init.eye_(proj_k.weight)
                    proj_k.bias.zero_()
                    nn.init.eye_(proj_v.weight)
                    proj_v.bias.zero_()
                self.mem_proj_K_l[str(layer_idx)] = proj_k
                self.mem_proj_V_l[str(layer_idx)] = proj_v

    def _build_dynamic_cache(self, kv_mem):
        legacy = []
        for layer_idx in self.kv_mem_layer_ids:
            K_batch, V_batch = kv_mem[layer_idx]
            if self.mem_proj_mode == "proj":
                B, H, M, d = K_batch.shape
                K_batch = self.mem_proj_K_l[str(layer_idx)](K_batch.reshape(B, H * M, d)).reshape(B, H, M, d)
                V_batch = self.mem_proj_V_l[str(layer_idx)](V_batch.reshape(B, H * M, d)).reshape(B, H, M, d)
            legacy.append((K_batch, V_batch))
        return DynamicCache.from_legacy_cache(tuple(legacy))

    def _register_lora_mem_hooks(self, blocks):
        if self._lora_mem_hooks:
            return

        for layer_idx in self.lora_mem_layer_ids:
            module = blocks[layer_idx]

            def _hook_fn(_mod, _inp, output, idx=layer_idx):
                if self._active_lora_memory is None or idx not in self._active_lora_memory:
                    return output

                x = output[0] if isinstance(output, tuple) else output
                A_mem, B_mem = self._active_lora_memory[idx]
                if self.lora_mem_dropout > 0.0:
                    x_in = F.dropout(x, p=self.lora_mem_dropout, training=self.training)
                else:
                    x_in = x
                low_rank = torch.einsum("bsd,bdr->bsr", x_in, A_mem)
                delta = torch.einsum("bsr,brd->bsd", low_rank, B_mem)
                x_new = x + self.lora_mem_scale * delta

                if isinstance(output, tuple):
                    out_list = list(output)
                    out_list[0] = x_new
                    return tuple(out_list)
                return x_new

            self._lora_mem_hooks.append(module.register_forward_hook(_hook_fn))

    @contextmanager
    def _enable_lora_memory(self, memory_params):
        old = self._active_lora_memory
        self._active_lora_memory = memory_params
        try:
            yield
        finally:
            self._active_lora_memory = old

    def forward(self, input_ids, labels=None, return_mem=False):
        context_input_ids = input_ids['context_input_ids']
        query_input_ids = input_ids['query_input_ids']

        pad_id = self.model.config.pad_token_id
        device = context_input_ids.device
        B = context_input_ids.size(0)
        inner_loss = torch.tensor(0.0, device=device)

        backend = self.memory_backend_impl
        memory_state, memory_state_initial = backend.init_memory_state(B)
        batch_ctx = backend.prepare_batch(context_input_ids, query_input_ids, pad_id)
        opt_state = {}

        inner_loop_stats = {
            'inner_grad_norm_mean': torch.tensor(0.0, device=device),
            'inner_grad_norm_max': torch.tensor(-1.0, device=device),
            'inner_grad_norm_min': torch.tensor(1e06, device=device),
        }

        if self.K and context_input_ids.ne(pad_id).any():
            with torch.enable_grad():
                for k in range(self.K):
                    write_batch = backend.build_write_inputs(memory_state, batch_ctx)
                    write_model_kwargs = write_batch.get('model_kwargs', {})
                    with backend.activation_context(memory_state):
                        if self.use_write_head:
                            outs = get_backbone(self.model)(inputs_embeds=write_batch['inputs_embeds'],
                                                            return_dict=True,
                                                            **write_model_kwargs)
                            hidden = outs.last_hidden_state[:, write_batch['logits_start']:, :]
                            logits = self.write_head(hidden)
                        else:
                            outs = self.model(inputs_embeds=write_batch['inputs_embeds'],
                                              return_dict=True,
                                              **write_model_kwargs)
                            logits = outs.logits[:, write_batch['logits_start']:, :]

                    logits_loss = logits[:, :-1]
                    label_shift = write_batch.get('label_shift', 0)
                    labels_loss = write_batch['lm_labels'][:, label_shift:]
                    mask_loss = write_batch['mask'][:, label_shift:]
                    logits_len = logits_loss.size(1)
                    labels_len = labels_loss.size(1)
                    mask_len = mask_loss.size(1)
                    if (logits_len != labels_len) or (labels_len != mask_len) or (labels_len == 0):
                        raise ValueError(
                            "Invalid inner-loop alignment: "
                            f"backend={self.memory_backend}, "
                            f"logits_len={logits_len}, labels_len={labels_len}, mask_len={mask_len}, "
                            f"logits_start={write_batch['logits_start']}, label_shift={label_shift}, "
                            f"mismatch_logits_labels={logits_len != labels_len}, "
                            f"mismatch_labels_mask={labels_len != mask_len}, "
                            f"empty_training_tokens={labels_len == 0}"
                        )

                    inner_loss = nn.functional.cross_entropy(
                        logits_loss.reshape(-1, logits.size(-1)),
                        labels_loss.reshape(-1),
                        ignore_index=-100,
                        reduction='none',
                    ).view(B, -1)
                    seq_len = mask_loss.sum(dim=1).clamp_min(1)
                    inner_loss = (inner_loss * mask_loss).sum(1) / seq_len
                    inner_loss = inner_loss.sum()
                    del outs, logits

                    is_second_order_step = (self.grad_mode == "second") and (k >= (self.K - self.last_K_second_order))
                    create_graph = is_second_order_step
                    retain_graph = create_graph or (self.add_inner_loss_to_outer and (k == self.K - 1))

                    inner_params = backend.inner_params(memory_state)
                    grads = torch.autograd.grad(inner_loss, inner_params,
                                                create_graph=create_graph, retain_graph=retain_graph)

                    g_sq = torch.zeros(B, device=device)
                    for g in grads:
                        g_sq = g_sq + g.reshape(B, -1).pow(2).sum(dim=1)
                    g_norm = g_sq.sqrt().detach()
                    inner_loop_stats['inner_grad_norm_mean'] += g_norm.mean()
                    inner_loop_stats['inner_grad_norm_max'] = max(inner_loop_stats['inner_grad_norm_max'], g_norm.max())
                    inner_loop_stats['inner_grad_norm_min'] = min(inner_loop_stats['inner_grad_norm_min'], g_norm.min())

                    new_params = []
                    for p, g, i in zip(inner_params, grads, range(len(inner_params))):
                        if self.use_adam:
                            p_new = self._adam_step(p, g, opt_state.setdefault(str(i), {}), k + 1, self.lr)
                        else:
                            p_new = self._sgd_step(p, g,
                                                   clip_value=self.inner_clip_value,
                                                   clip_norm=self.inner_clip_norm)
                        new_params.append(p_new)
                    backend.assign_inner_params(memory_state, new_params)
                    backend.maybe_detach_after_step(memory_state)

        if self.K:
            inner_loop_stats['inner_grad_norm_mean'] = inner_loop_stats['inner_grad_norm_mean'] / self.K
            inner_loop_stats['inner_loss'] = inner_loss.detach() / B

        mem_norm, delta_mem_norm = backend.compute_memory_stats(memory_state, memory_state_initial)
        inner_loop_stats['mem_norm_mean'] = mem_norm.mean()
        inner_loop_stats['mem_norm_max'] = mem_norm.max()
        inner_loop_stats['mem_norm_min'] = mem_norm.min()
        inner_loop_stats['delta_mem_norm_mean'] = delta_mem_norm.mean()
        inner_loop_stats['delta_mem_norm_max'] = delta_mem_norm.max()
        inner_loop_stats['delta_mem_norm_min'] = delta_mem_norm.min()

        read_batch = backend.build_read_inputs(memory_state, batch_ctx)
        read_model_kwargs = read_batch.get('model_kwargs', {})
        log_mem_attn_read = (self.attn_implementation == "eager") and (self.memory_backend in ("prefix", "kv_cache"))
        if log_mem_attn_read:
            read_model_kwargs = dict(read_model_kwargs)
            read_model_kwargs["output_attentions"] = True
        with backend.activation_context(memory_state):
            with self._disable_write_lora():
                read_out = self.model(inputs_embeds=read_batch['inputs_embeds'],
                                      return_dict=True,
                                      **read_model_kwargs)
                logits_q = read_out.logits

        if log_mem_attn_read and read_out.attentions is not None:
            if self.memory_backend == "prefix":
                mem_start = self.n_ctrl_tokens
                mem_end = self.n_ctrl_tokens + self.n_mem_tokens
            else:
                mem_start = 0
                mem_end = self.n_mem_tokens

            layer_ratios = []
            for att in read_out.attentions:
                if att is None:
                    continue
                k_len = att.size(-1)
                if not (0 <= mem_start < mem_end <= k_len):
                    raise ValueError(
                        "GradMem: Invalid memory attention span on read: "
                        f"backend={self.memory_backend}, mem_start={mem_start}, mem_end={mem_end}, k_len={k_len}"
                    )
                layer_ratios.append(att[..., mem_start:mem_end].sum(dim=-1).mean())
            if layer_ratios:
                inner_loop_stats['mem_attn_read'] = torch.stack(layer_ratios).mean().detach()

        # LoRA/KV-cache memory backends need one seed token to start autoregressive prediction;
        # they cannot predict the very first token from memory alone when query is empty.
        logits_q = logits_q[:, read_batch['logits_start']:read_batch['logits_start'] + read_batch['pred_len'], :]

        output = {'predictions': logits_q, 'inner_loop_stats': inner_loop_stats}
        if return_mem:
            backend.attach_return_memory(output, memory_state)

        if labels is None:
            return output

        target_logits = output['predictions'][:, :-1]
        target_label_shift = read_batch.get('label_shift', 0)
        # Prefix memory can predict the first target token from memory itself (label_shift=0),
        # while LoRA/KV-cache memory without a prepended seed token cannot (label_shift=1).
        # Backends should produce exactly aligned lengths after this shift.
        target_labels = labels[:, target_label_shift:]
        if target_logits.size(1) != target_labels.size(1):
            raise ValueError(
                f"Mismatched target lengths after alignment: logits_len={target_logits.size(1)}, "
                f"labels_len={target_labels.size(1)}, label_shift={target_label_shift}"
            )
        target_loss = nn.functional.cross_entropy(
            target_logits.reshape(-1, output['predictions'].size(-1)),
            target_labels.reshape(-1),
            ignore_index=-100,
        )

        output['inner_loop_stats']['target_loss'] = target_loss.detach()
        if self.add_inner_loss_to_outer:
            inner_loss_mean = inner_loss / B
            combined_loss = target_loss + self.inner_loss_weight * inner_loss_mean
        else:
            combined_loss = target_loss
        output['loss'] = combined_loss
        return output
