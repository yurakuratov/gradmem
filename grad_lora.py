"""
GradLoRA: test-time low-rank injection into a frozen causal LM.

Two modes are supported (selected via ``GradLoRAConfig.mode``):

* **residual** – adds ``x ← x + B A x`` before the *l*-th transformer layer.
  A ∈ ℝ^{r×d} and B ∈ ℝ^{d×r}.
* **ffn** – applies LoRA to the output projection of the MLP at layer *l*:
  ``(W_out + A B) σ(W_in x)``.  A ∈ ℝ^{d×r} and B ∈ ℝ^{r×d_ff}.

In both modes the zero-initialised matrix ensures the model starts at the
identity.  Per-sample copies of A and B are optimised in an inner loop to
compress context, then used at read time.  The interface (context / query /
predictions dict) matches GradMemGPT so the models can be swapped freely.
"""

import os
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
import attn_double_bwd  # noqa: F401  – registers custom attention kernels

GRAD_VERBOSE = os.environ.get("GRAD_VERBOSE", "0") == "1"


# ── helpers ─────────────────────────────────────────────────────────────────

def _get_backbone(m):
    if hasattr(m, "base_model_prefix") and hasattr(m, m.base_model_prefix):
        return getattr(m, m.base_model_prefix)
    for attr in ("model", "transformer", "gpt_neox", "backbone", "decoder"):
        if hasattr(m, attr):
            return getattr(m, attr)
    raise AttributeError("Could not locate backbone submodule")


def _find_layers(model):
    """Return the ``nn.ModuleList`` of transformer blocks from a CausalLM."""
    backbone = _get_backbone(model)
    for attr in ("layers", "h", "block"):
        obj = getattr(backbone, attr, None)
        if isinstance(obj, nn.ModuleList):
            return obj
    raise AttributeError("Cannot find transformer layer list in model")


def _find_output_proj(layer):
    """Return ``(mlp_module, attr_name, d_ff)`` for the MLP output projection."""
    mlp = getattr(layer, "mlp", getattr(layer, "feed_forward", None))
    if mlp is None:
        raise AttributeError("Cannot find MLP sub-module in layer")
    for name in ("c_proj", "down_proj", "dense_4h_to_h", "fc2", "w2"):
        proj = getattr(mlp, name, None)
        if proj is None:
            continue
        if hasattr(proj, "in_features"):
            d_ff = proj.in_features
        elif hasattr(proj, "nf"):            # transformers Conv1D
            d_ff = proj.weight.shape[0]
        else:
            d_ff = proj.weight.shape[1]
        return mlp, name, d_ff
    raise AttributeError("Cannot find output projection in MLP")


# ── layer wrappers ──────────────────────────────────────────────────────────

class LowRankWrapper(nn.Module):
    """Wraps a single transformer layer.

    When per-sample matrices *A* ``[B,r,d]`` and *B* ``[B,d,r]`` are set via
    :meth:`set_params`, the forward pass applies
    ``hidden_states ← hidden_states + hidden_states @ Aᵀ @ Bᵀ``
    before delegating to the original layer.
    """

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self._A = None
        self._B = None

    def set_params(self, A, B):
        self._A, self._B = A, B

    def clear_params(self):
        self._A = self._B = None

    def forward(self, hidden_states, *args, **kwargs):
        if self._A is not None and self._B is not None:
            proj = torch.bmm(hidden_states, self._A.transpose(1, 2))   # [B,S,r]
            hidden_states = hidden_states + torch.bmm(proj, self._B.transpose(1, 2))
        return self.layer(hidden_states, *args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        try:
            layer = super().__getattr__("layer")
            return getattr(layer, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute '{name}'")


class FFNLoRAWrapper(nn.Module):
    """Wraps the MLP output projection (e.g. ``c_proj`` / ``down_proj``).

    When per-sample A ``[B, d_model, r]`` and B ``[B, r, d_ff]`` are set,
    the forward pass becomes ``layer(x) + x @ Bᵀ @ Aᵀ``, which is equivalent
    to replacing W_out with ``W_out + A @ B``.
    """

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self._A = None
        self._B = None

    def set_params(self, A, B):
        self._A, self._B = A, B

    def clear_params(self):
        self._A = self._B = None

    def forward(self, hidden_states, *args, **kwargs):
        out = self.layer(hidden_states, *args, **kwargs)
        if self._A is not None and self._B is not None:
            proj = torch.bmm(hidden_states, self._B.transpose(1, 2))  # [B,S,r]
            out = out + torch.bmm(proj, self._A.transpose(1, 2))      # [B,S,d]
        return out

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        try:
            layer = super().__getattr__("layer")
            return getattr(layer, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute '{name}'")


# ── config ──────────────────────────────────────────────────────────────────

class GradLoRAConfig(PretrainedConfig):
    model_type = "grad_lora"

    def __init__(self,
                 pretrained_model=None,
                 base_config=None,
                 mode="residual",
                 layer_idx=0,
                 rank=4,
                 K=2,
                 lr=0.01,
                 use_adam=False,
                 grad_mode="second",
                 last_K_second_order=None,
                 inner_clip_value=None,
                 inner_clip_norm=None,
                 use_gradient_checkpointing=False,
                 attn_implementation="eager",
                 early_stop_acc=None,
                 early_stop_check_every=100,
                 **kwargs):
        super().__init__(**kwargs)
        assert mode in ("residual", "ffn"), f"Unknown mode {mode!r}"
        if pretrained_model is not None:
            self.pretrained_model = pretrained_model
            self.base_config = None
        else:
            self.pretrained_model = None
            self.base_config = base_config

        self.mode = mode
        self.layer_idx = layer_idx
        self.rank = rank
        self.K = K
        self.lr = lr
        self.use_adam = use_adam
        self.grad_mode = grad_mode
        self.inner_clip_value = inner_clip_value
        self.inner_clip_norm = inner_clip_norm
        self.last_K_second_order = K if last_K_second_order is None else last_K_second_order
        if grad_mode != "second":
            self.last_K_second_order = 0
        self.last_K_second_order = max(0, min(self.last_K_second_order, K))
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.attn_implementation = attn_implementation
        self.early_stop_acc = early_stop_acc
        self.early_stop_check_every = early_stop_check_every


# ── model ───────────────────────────────────────────────────────────────────

class GradLoRA(PreTrainedModel):
    """Frozen causal LM + test-time low-rank injection at layer *l*.

    Two modes:
      * **residual** — ``x ← x + B A x`` before layer *l*.
      * **ffn** — ``(W_out + A B) σ(W_in x)`` at layer *l*'s MLP.
    """
    config_class = GradLoRAConfig

    def __init__(self, config):
        super().__init__(config)

        if config.pretrained_model is not None and config.base_config is not None:
            raise ValueError("Only one of pretrained_model / base_config")
        if config.pretrained_model is None and config.base_config is None:
            raise ValueError("Either pretrained_model or base_config required")

        if config.pretrained_model is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.pretrained_model,
                attn_implementation=config.attn_implementation)
        else:
            self.model = AutoModelForCausalLM.from_config(
                config.base_config,
                attn_implementation=config.attn_implementation)

        d = getattr(self.model.config, "n_embd", self.model.config.hidden_size)
        self.d_model = d
        self.lora_mode = config.mode
        self.rank = config.rank
        self.layer_idx = config.layer_idx
        self.K = config.K
        self.lr = config.lr
        self.use_adam = config.use_adam
        self.grad_mode = config.grad_mode
        self.last_K_second_order = config.last_K_second_order
        self.inner_clip_value = config.inner_clip_value
        self.inner_clip_norm = config.inner_clip_norm
        self.early_stop_acc = getattr(config, "early_stop_acc", None)
        self.early_stop_check_every = getattr(config, "early_stop_check_every", 100)

        layers = _find_layers(self.model)
        idx = self.layer_idx if self.layer_idx >= 0 else len(layers) + self.layer_idx
        assert 0 <= idx < len(layers), \
            f"layer_idx {config.layer_idx} out of range [0, {len(layers)})"

        if self.lora_mode == "residual":
            wrapper = LowRankWrapper(layers[idx])
            layers[idx] = wrapper
            # A [r, d], B [d, r] — B=0 so initial residual is zero
            self.A_init = nn.Parameter(torch.randn(self.rank, d) * 0.02)
            self.B_init = nn.Parameter(torch.zeros(d, self.rank))
        elif self.lora_mode == "ffn":
            mlp, proj_name, d_ff = _find_output_proj(layers[idx])
            wrapper = FFNLoRAWrapper(getattr(mlp, proj_name))
            setattr(mlp, proj_name, wrapper)
            # A [d, r], B [r, d_ff] — B=0 so initial LoRA is zero
            self.A_init = nn.Parameter(torch.randn(d, self.rank) * 0.02)
            self.B_init = nn.Parameter(torch.zeros(self.rank, d_ff))
        else:
            raise ValueError(f"Unknown mode {self.lora_mode}")
        # Store without nn.Module registration to avoid duplicate parameter paths
        self._wrapped_ref = [wrapper]

        self.tie_weights()
        self.main_input_name = "input_ids"
        self.model.config.use_cache = False
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

        if getattr(config, "use_gradient_checkpointing", False):
            self.gradient_checkpointing_enable()

    # ── boilerplate (same as GradMemGPT) ────────────────────────────────────

    def mem_state_numel(self):
        """Number of optimisable floats in the memory state (A + B)."""
        return self.A_init.numel() + self.B_init.numel()

    def floating_point_ops(self, inputs):
        return 0

    def tie_weights(self):
        self.model.tie_weights()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                self.model.gradient_checkpointing_enable()

    # ── inner-loop optimisers (same signatures as GradMemGPT) ───────────────

    def _adam_step(self, p, g, state, step_idx, lr,
                   beta1=0.9, beta2=0.999, eps=1e-8, clip_value=10.0):
        if "m" not in state:
            state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        m, v = state["m"], state["v"]
        if clip_value is not None:
            g = torch.clamp(g, -clip_value, clip_value)
        with torch.no_grad():
            m_new = beta1 * m + (1 - beta1) * g
            v_new = beta2 * v + (1 - beta2) * (g * g)
            m_hat = m_new / (1 - beta1 ** step_idx)
            v_hat = torch.clamp(v_new / (1 - beta2 ** step_idx), min=eps)
            step = lr * m_hat / (v_hat.sqrt() + eps)
        state["m"].copy_(m_new.detach())
        state["v"].copy_(v_new.detach())
        return p - step

    def _sgd_step(self, p, g, lr=None, clip_value=None, clip_norm=None):
        if clip_value is not None:
            g = torch.clamp(g, -clip_value, clip_value)
        if clip_norm is not None:
            dims = tuple(range(1, g.ndim))
            gn = g.norm(dim=dims, keepdim=True)
            g = torch.where(gn > clip_norm, g * clip_norm / (gn + 1e-6), g)
        return p - (lr if lr is not None else self.lr) * g

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(self, input_ids, labels=None):
        context_input_ids = input_ids["context_input_ids"]
        query_input_ids   = input_ids["query_input_ids"]

        pad_id = self.model.config.pad_token_id
        device = context_input_ids.device
        B = context_input_ids.size(0)

        # per-sample copies of A, B
        A_batch = self.A_init.unsqueeze(0).expand(B, -1, -1).clone()   # [B, r, d]
        B_batch = self.B_init.unsqueeze(0).expand(B, -1, -1).clone()   # [B, d, r]

        if self.grad_mode == "none":
            A_batch = A_batch.detach().requires_grad_(True)
            B_batch = B_batch.detach().requires_grad_(True)
        else:
            A_batch = A_batch.requires_grad_(True)
            B_batch = B_batch.requires_grad_(True)

        opt_state = {}
        inner_loop_stats = {
            "inner_grad_norm_mean": torch.tensor(0.0, device=device),
            "inner_grad_norm_max":  torch.tensor(-1.0, device=device),
            "inner_grad_norm_min":  torch.tensor(1e6, device=device),
        }

        # ── 1. INNER loop — compress context into (A, B) ───────────────────
        _early_stop = False
        if self.K and context_input_ids.ne(pad_id).any():
            with torch.enable_grad():
                lm_labels = context_input_ids.clone()
                lm_labels[lm_labels == pad_id] = -100
                targets = lm_labels[:, 1:]
                mask = (targets != -100)
                seq_len = mask.sum(1).clamp_min(1)

                _iter = range(self.K)
                if GRAD_VERBOSE:
                    from tqdm import tqdm
                    _iter = tqdm(_iter, desc="inner", leave=False)

                for k in _iter:
                    self._wrapped_ref[0].set_params(A_batch, B_batch)
                    logits = self.model(input_ids=context_input_ids).logits

                    inner_loss = F.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                        ignore_index=-100, reduction="none",
                    ).view(B, -1)
                    inner_loss = (inner_loss * mask).sum(1) / seq_len
                    inner_loss = inner_loss.sum()

                    _cur_acc = None
                    _early_stop = False
                    if (self.early_stop_acc is not None
                            and (k + 1) % self.early_stop_check_every == 0):
                        with torch.no_grad():
                            _hits = (logits[:, :-1].argmax(-1) == targets) & mask
                            _cur_acc = (_hits.sum() / mask.sum()).item()
                            _early_stop = _cur_acc >= self.early_stop_acc

                    if GRAD_VERBOSE:
                        _postfix = {"loss": f"{inner_loss.item() / B:.4f}"}
                        if _cur_acc is not None:
                            _postfix["acc"] = f"{_cur_acc:.4f}"
                        _iter.set_postfix(_postfix)

                    del logits

                    if _early_stop:
                        break

                    is_second = ((self.grad_mode == "second")
                                 and k >= self.K - self.last_K_second_order)
                    g_A, g_B = torch.autograd.grad(
                        inner_loss, [A_batch, B_batch],
                        create_graph=is_second)

                    g_cat = torch.cat([g_A.reshape(B, -1),
                                       g_B.reshape(B, -1)], 1)
                    gn = g_cat.norm(dim=1).detach()
                    inner_loop_stats["inner_grad_norm_mean"] += gn.mean()
                    inner_loop_stats["inner_grad_norm_max"] = max(
                        inner_loop_stats["inner_grad_norm_max"], gn.max())
                    inner_loop_stats["inner_grad_norm_min"] = min(
                        inner_loop_stats["inner_grad_norm_min"], gn.min())

                    if self.use_adam:
                        A_batch = self._adam_step(
                            A_batch, g_A,
                            opt_state.setdefault("A", {}), k + 1, self.lr)
                        B_batch = self._adam_step(
                            B_batch, g_B,
                            opt_state.setdefault("B", {}), k + 1, self.lr)
                    else:
                        A_batch = self._sgd_step(
                            A_batch, g_A,
                            clip_value=self.inner_clip_value,
                            clip_norm=self.inner_clip_norm)
                        B_batch = self._sgd_step(
                            B_batch, g_B,
                            clip_value=self.inner_clip_value,
                            clip_norm=self.inner_clip_norm)

                    if self.grad_mode == "none":
                        A_batch = A_batch.detach().requires_grad_(True)
                        B_batch = B_batch.detach().requires_grad_(True)

        if self.K:
            _n = k if _early_stop else self.K
            inner_loop_stats["inner_grad_norm_mean"] /= max(1, _n)
            inner_loop_stats["inner_loss"] = inner_loss.detach() / B
            inner_loop_stats["inner_steps"] = _n

        # ── 2. READ phase — reconstruct from query using optimised (A, B) ──
        self._wrapped_ref[0].set_params(A_batch, B_batch)
        logits_q = self.model(input_ids=query_input_ids).logits
        self._wrapped_ref[0].clear_params()

        output = {"predictions": logits_q,
                  "inner_loop_stats": inner_loop_stats}

        if labels is not None:
            loss = F.cross_entropy(
                logits_q[:, :-1].reshape(-1, logits_q.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100)
            output["loss"] = loss

        return output
