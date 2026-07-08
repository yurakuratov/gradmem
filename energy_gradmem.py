import logging

import torch
from torch import nn
from tqdm.auto import tqdm

from grad_memgpt import GradMemGPT, GradMemGPTConfig, _is_main_process


logger = logging.getLogger(__name__)


class EnergyGradMemConfig(GradMemGPTConfig):
    model_type = "energy_gradmem"

    def __init__(
        self,
        inner_objective="lstm",
        energy_hidden_size=None,
        energy_num_layers=2,
        energy_dropout=0.0,
        energy_future_mode="next_token",
        energy_ce_guidance=False,
        energy_ce_guidance_alpha=0.01,
        energy_inner_ce_weight=0.0,
        energy_pretrain_objective="ce",
        energy_pretrain_steps=0,
        energy_pretrain_batch_size=16,
        energy_pretrain_seq_len=32,
        energy_pretrain_lr=1e-3,
        energy_pretrain_seed=0,
        energy_pretrain_l2_reg=0.0,
        return_energy_state=False,
        **kwargs,
    ):
        if kwargs.get("use_write_head", False):
            raise ValueError("EnergyGradMem does not support use_write_head; the energy objective does not use LM logits")
        super().__init__(**kwargs)
        if inner_objective not in ("lstm", "embedding_l1", "embedding_l2"):
            raise ValueError("inner_objective must be one of: 'lstm', 'embedding_l1', 'embedding_l2'")
        if energy_future_mode not in ("none", "next_token"):
            raise ValueError("energy_future_mode must be one of: 'none', 'next_token'")
        if inner_objective in ("embedding_l1", "embedding_l2") and energy_future_mode != "next_token":
            raise ValueError("embedding_l1/embedding_l2 inner objectives require energy_future_mode='next_token'")
        if energy_pretrain_objective not in ("ce", "embedding_mean_abs_diff", "embedding_l2"):
            raise ValueError("energy_pretrain_objective must be one of: 'ce', 'embedding_mean_abs_diff', 'embedding_l2'")
        self.inner_objective = inner_objective
        self.energy_hidden_size = energy_hidden_size
        self.energy_num_layers = energy_num_layers
        self.energy_dropout = energy_dropout
        self.energy_future_mode = energy_future_mode
        self.energy_ce_guidance = energy_ce_guidance
        self.energy_ce_guidance_alpha = energy_ce_guidance_alpha
        self.energy_inner_ce_weight = energy_inner_ce_weight
        self.energy_pretrain_objective = energy_pretrain_objective
        self.energy_pretrain_steps = energy_pretrain_steps
        self.energy_pretrain_batch_size = energy_pretrain_batch_size
        self.energy_pretrain_seq_len = energy_pretrain_seq_len
        self.energy_pretrain_lr = energy_pretrain_lr
        self.energy_pretrain_seed = energy_pretrain_seed
        self.energy_pretrain_l2_reg = energy_pretrain_l2_reg
        self.return_energy_state = return_energy_state


class EnergyGradMem(GradMemGPT):
    config_class = EnergyGradMemConfig

    def __init__(self, config):
        super().__init__(config)

        model_hidden_size = getattr(self.model.config, "n_embd", getattr(self.model.config, "hidden_size", None))
        if model_hidden_size is None:
            raise ValueError("Could not infer hidden size from model config")

        energy_hidden_size = config.energy_hidden_size or model_hidden_size
        energy_num_layers = int(config.energy_num_layers)
        if energy_num_layers < 1:
            raise ValueError("energy_num_layers must be >= 1")

        self.inner_objective = config.inner_objective
        self.energy_hidden_size = int(energy_hidden_size)
        self.energy_num_layers = energy_num_layers
        self.energy_future_mode = config.energy_future_mode
        self.energy_ce_guidance = bool(getattr(config, "energy_ce_guidance", False))
        self.energy_ce_guidance_alpha = float(getattr(config, "energy_ce_guidance_alpha", 0.01))
        self.energy_inner_ce_weight = float(getattr(config, "energy_inner_ce_weight", 0.0))
        self.energy_pretrain_objective = getattr(config, "energy_pretrain_objective", "ce")
        self.energy_pretrain_steps = int(getattr(config, "energy_pretrain_steps", 0))
        self.energy_pretrain_batch_size = int(getattr(config, "energy_pretrain_batch_size", 16))
        self.energy_pretrain_seq_len = int(getattr(config, "energy_pretrain_seq_len", 32))
        self.energy_pretrain_lr = float(getattr(config, "energy_pretrain_lr", 1e-3))
        self.energy_pretrain_seed = int(getattr(config, "energy_pretrain_seed", 0))
        self.energy_pretrain_l2_reg = float(getattr(config, "energy_pretrain_l2_reg", 0.0))
        self.return_energy_state = bool(getattr(config, "return_energy_state", False))
        dropout = float(config.energy_dropout) if energy_num_layers > 1 else 0.0
        energy_input_size = model_hidden_size * 2 if self.energy_future_mode == "next_token" else model_hidden_size

        if self.inner_objective == "lstm":
            self.energy_encoder = nn.LSTM(
                input_size=energy_input_size,
                hidden_size=self.energy_hidden_size,
                num_layers=energy_num_layers,
                dropout=dropout,
                batch_first=True,
            )
            self.energy_head = nn.Linear(self.energy_hidden_size, 1)

    def _energy_parameters(self):
        params = []
        if hasattr(self, "energy_encoder"):
            params.extend(self.energy_encoder.parameters())
        if hasattr(self, "energy_head"):
            params.extend(self.energy_head.parameters())
        return list(params)

    def set_energy_trainable(self, trainable=True):
        for param in self._energy_parameters():
            param.requires_grad = trainable

    @staticmethod
    def _context_segments(context_input_ids):
        if isinstance(context_input_ids, torch.Tensor):
            if context_input_ids.ndim != 2:
                raise ValueError(
                    "context_input_ids tensor must have shape [B, S]; "
                    f"got {tuple(context_input_ids.shape)}"
                )
            return [context_input_ids]
        if isinstance(context_input_ids, (list, tuple)):
            if len(context_input_ids) == 0:
                raise ValueError("context_input_ids segment list must be non-empty")
            for i, segment in enumerate(context_input_ids):
                if not isinstance(segment, torch.Tensor) or segment.ndim != 2:
                    raise ValueError(
                        "each context segment must be a tensor with shape [B, S]; "
                        f"segment={i}, type={type(segment)}, shape={getattr(segment, 'shape', None)}"
                    )
            return list(context_input_ids)
        raise ValueError("context_input_ids must be a tensor [B, S] or a list of tensors [B, S_i]")

    def _write_context_start(self, batch_ctx):
        if self.memory_backend == "prefix":
            return batch_ctx["mem_offset"]
        return 0

    def _energy_values(self, hidden, energy_state=None):
        if self.inner_objective in ("embedding_l1", "embedding_l2"):
            if hidden.size(-1) % 2 != 0:
                raise ValueError(f"{self.inner_objective} requires an even energy input size")
            pred, target = hidden.chunk(2, dim=-1)
            if self.inner_objective == "embedding_l1":
                return (pred - target).abs().mean(dim=-1), energy_state
            return (pred - target).pow(2).mean(dim=-1), energy_state

        with torch.backends.cudnn.flags(enabled=False):
            encoded, energy_state = self.energy_encoder(hidden, energy_state)
        energy = self.energy_head(encoded).squeeze(-1)
        return energy, energy_state

    def _energy_loss(self, hidden, mask, energy_state):
        # Eval still needs inner-loop gradients. cuDNN RNN backward rejects eval-mode
        # modules, so use the native autograd path for this small objective model.
        energy, energy_state = self._energy_values(hidden, energy_state)

        mask = mask.to(dtype=energy.dtype)
        valid_lengths = mask.sum(dim=1)
        per_sample_energy = (energy * mask).sum(dim=1) / valid_lengths.clamp_min(1.0)
        per_sample_energy = per_sample_energy * (valid_lengths > 0).to(per_sample_energy.dtype)
        return per_sample_energy.sum(), energy_state, energy

    def _run_write_model(self, write_batch, memory_state):
        write_model_kwargs = write_batch.get("model_kwargs", {})
        with self.memory_backend_impl.activation_context(memory_state):
            outs = self.model(
                inputs_embeds=write_batch["inputs_embeds"],
                output_hidden_states=True,
                return_dict=True,
                **write_model_kwargs,
            )
        if outs.hidden_states is None:
            raise ValueError("Base model did not return hidden states for energy objective")
        return outs

    def _write_token_ce(self, write_out, write_batch):
        logits = write_out.logits[:, write_batch["logits_start"]:, :]
        logits_loss = logits[:, :-1]
        label_shift = write_batch.get("label_shift", 0)
        labels_loss = write_batch["lm_labels"][:, label_shift:]
        mask_loss = write_batch["mask"][:, label_shift:]
        if logits_loss.size(1) != labels_loss.size(1) or labels_loss.size(1) != mask_loss.size(1):
            raise ValueError(
                "Invalid CE-guidance alignment: "
                f"logits_len={logits_loss.size(1)}, labels_len={labels_loss.size(1)}, mask_len={mask_loss.size(1)}"
            )
        token_ce = nn.functional.cross_entropy(
            logits_loss.reshape(-1, logits.size(-1)),
            labels_loss.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(labels_loss.size())
        if label_shift == 0:
            # Prefix write logits include the pre-context position. Energy at context
            # token t is conditioned on token t+1, so skip CE for token 0.
            token_ce = token_ce[:, 1:]
            mask_loss = mask_loss[:, 1:]
        return token_ce, mask_loss

    def _write_inner_ce_loss(self, write_out, write_batch):
        logits = write_out.logits[:, write_batch["logits_start"]:, :]
        logits_loss = logits[:, :-1]
        label_shift = write_batch.get("label_shift", 0)
        labels_loss = write_batch["lm_labels"][:, label_shift:]
        mask_loss = write_batch["mask"][:, label_shift:]
        if logits_loss.size(1) != labels_loss.size(1) or labels_loss.size(1) != mask_loss.size(1) or labels_loss.size(1) == 0:
            raise ValueError(
                "Invalid inner CE alignment: "
                f"logits_len={logits_loss.size(1)}, labels_len={labels_loss.size(1)}, mask_len={mask_loss.size(1)}"
            )
        token_ce = nn.functional.cross_entropy(
            logits_loss.reshape(-1, logits.size(-1)),
            labels_loss.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(labels_loss.size())
        return self._masked_token_loss_sum(token_ce, mask_loss)

    @staticmethod
    def _energy_ce_guidance_loss(energy, token_ce, mask):
        energy = energy[:, :token_ce.size(1)]
        mask = mask.to(dtype=energy.dtype)
        per_token = (energy - token_ce.detach().to(dtype=energy.dtype)).abs() * mask
        return per_token.sum() / mask.sum().clamp_min(1.0)

    @staticmethod
    def _masked_token_loss_sum(token_loss, mask):
        mask = mask.to(dtype=token_loss.dtype)
        valid_lengths = mask.sum(dim=1)
        per_sample = (token_loss * mask).sum(dim=1) / valid_lengths.clamp_min(1.0)
        per_sample = per_sample * (valid_lengths > 0).to(per_sample.dtype)
        return per_sample.sum()

    def _sample_energy_pretrain_batch(self, batch_size, seq_len, generator, device):
        vocab_size = int(self.model.config.vocab_size)
        pad_id = self.model.config.pad_token_id
        low = 1 if vocab_size > 1 else 0
        tokens = torch.randint(low, vocab_size, (batch_size, seq_len), generator=generator, device=device)
        if pad_id is not None:
            tokens = torch.where(tokens == pad_id, (tokens + 1) % vocab_size, tokens)
        return tokens

    def _embedding_mean_abs_diff_pretrain_loss(self, batch_size, seq_len, generator, device):
        if self.inner_objective != "lstm":
            raise ValueError("embedding_mean_abs_diff pretraining requires trainable inner_objective='lstm'")
        input_size = self.energy_encoder.input_size
        if input_size % 2 != 0:
            raise ValueError("embedding_mean_abs_diff pretraining requires an even energy encoder input size")
        hidden_size = input_size // 2
        dtype = self.energy_head.weight.dtype
        h1 = torch.randn(batch_size, seq_len, hidden_size, generator=generator, device=device, dtype=dtype)
        h2 = torch.randn(batch_size, seq_len, hidden_size, generator=generator, device=device, dtype=dtype)
        energy_input = torch.cat([h1, h2], dim=-1)
        target = (h1 - h2).abs().mean(dim=-1)
        energy, _ = self._energy_values(energy_input)
        return (energy - target.detach()).pow(2).mean()

    def _embedding_l2_pretrain_loss(self, batch_size, seq_len, generator, device):
        if self.inner_objective != "lstm":
            raise ValueError("embedding_l2 pretraining requires trainable inner_objective='lstm'")
        input_size = self.energy_encoder.input_size
        if input_size % 2 != 0:
            raise ValueError("embedding_l2 pretraining requires an even energy encoder input size")
        hidden_size = input_size // 2
        dtype = self.energy_head.weight.dtype
        h1 = torch.randn(batch_size, seq_len, hidden_size, generator=generator, device=device, dtype=dtype)
        h2 = torch.randn(batch_size, seq_len, hidden_size, generator=generator, device=device, dtype=dtype)
        energy_input = torch.cat([h1, h2], dim=-1)
        target = (h1 - h2).pow(2).mean(dim=-1)
        energy, _ = self._energy_values(energy_input)
        return (energy - target.detach()).pow(2).mean()

    def _ce_pretrain_loss(self, backend, pad_id, generator, device):
        context = self._sample_energy_pretrain_batch(
            self.energy_pretrain_batch_size,
            self.energy_pretrain_seq_len,
            generator,
            device,
        )
        memory_state, _ = backend.init_memory_state(context.size(0))
        batch_ctx = backend.prepare_batch(context, context, pad_id)
        write_batch = backend.build_write_inputs(memory_state, batch_ctx)

        with torch.no_grad():
            write_out = self._run_write_model(write_batch, memory_state)
            ctx_hidden = self._extract_context_hidden(write_out.hidden_states[-1], write_batch, batch_ctx)
            energy_input = self._energy_input(ctx_hidden, context, write_batch["mask"]).detach()
            token_ce, token_ce_mask = self._write_token_ce(write_out, write_batch)

        energy, _ = self._energy_values(energy_input)
        return self._energy_ce_guidance_loss(energy, token_ce, token_ce_mask)

    def _energy_pretrain_l2_regularization(self):
        if self.energy_pretrain_l2_reg <= 0.0:
            ref = next(self.parameters())
            return ref.new_zeros(())
        params = self._energy_parameters()
        if not params:
            ref = next(self.parameters())
            return ref.new_zeros(())
        penalty = params[0].new_zeros(())
        for param in params:
            penalty = penalty + param.pow(2).sum()
        return self.energy_pretrain_l2_reg * penalty

    def pretrain_energy_objective(self):
        if self.energy_pretrain_batch_size < 1 or self.energy_pretrain_seq_len < 2:
            raise ValueError("energy_pretrain_batch_size must be >= 1 and energy_pretrain_seq_len must be >= 2")
        if self.energy_pretrain_steps <= 0:
            return
        if self.inner_objective != "lstm":
            raise ValueError("energy pretraining requires trainable inner_objective='lstm'")

        was_training = self.training
        device = self.model.get_input_embeddings().weight.device
        generator = torch.Generator(device=device)
        generator.manual_seed(self.energy_pretrain_seed)
        optimizer = torch.optim.AdamW(self._energy_parameters(), lr=self.energy_pretrain_lr, weight_decay=0.0)

        try:
            self.eval()
            self.energy_encoder.train()
            self.energy_head.train()
            backend = self.memory_backend_impl
            pad_id = self.model.config.pad_token_id
            show_progress = _is_main_process()
            log_every = max(1, self.energy_pretrain_steps // 10)
            last_loss = None

            if show_progress:
                logger.info(
                    "Energy pretraining started: objective=%s, steps=%d, batch_size=%d, seq_len=%d, lr=%g, l2_reg=%g",
                    self.energy_pretrain_objective,
                    self.energy_pretrain_steps,
                    self.energy_pretrain_batch_size,
                    self.energy_pretrain_seq_len,
                    self.energy_pretrain_lr,
                    self.energy_pretrain_l2_reg,
                )

            progress = tqdm(
                range(self.energy_pretrain_steps),
                desc="energy pretrain",
                disable=not show_progress,
            )
            for step in progress:
                if self.energy_pretrain_objective == "ce":
                    loss = self._ce_pretrain_loss(backend, pad_id, generator, device)
                elif self.energy_pretrain_objective == "embedding_mean_abs_diff":
                    loss = self._embedding_mean_abs_diff_pretrain_loss(
                        self.energy_pretrain_batch_size,
                        self.energy_pretrain_seq_len,
                        generator,
                        device,
                    )
                elif self.energy_pretrain_objective == "embedding_l2":
                    loss = self._embedding_l2_pretrain_loss(
                        self.energy_pretrain_batch_size,
                        self.energy_pretrain_seq_len,
                        generator,
                        device,
                    )
                else:
                    raise ValueError(f"Unsupported energy_pretrain_objective={self.energy_pretrain_objective}")
                loss = loss + self._energy_pretrain_l2_regularization()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().item())
                if show_progress:
                    progress.set_postfix(loss=f"{last_loss:.4f}")
                    if (step + 1) % log_every == 0 or step == 0 or (step + 1) == self.energy_pretrain_steps:
                        logger.info(
                            "Energy pretraining step %d/%d: loss=%.6f",
                            step + 1,
                            self.energy_pretrain_steps,
                            last_loss,
                        )

            if show_progress:
                logger.info("Energy pretraining finished: final_loss=%.6f", last_loss)
        finally:
            self.train(was_training)

    def _validate_context_segments(self, context_segments, batch_size):
        for i, segment in enumerate(context_segments):
            if segment.size(0) != batch_size:
                raise ValueError(
                    f"context segment batch size mismatch at segment={i}: "
                    f"segment_B={segment.size(0)}, query_B={batch_size}"
                )

    @staticmethod
    def _init_inner_loop_stats(device):
        return {
            "inner_grad_norm_mean": torch.tensor(0.0, device=device),
            "inner_grad_norm_max": torch.tensor(-1.0, device=device),
            "inner_grad_norm_min": torch.tensor(1e06, device=device),
        }

    def _extract_context_hidden(self, hidden, write_batch, batch_ctx):
        ctx_start = self._write_context_start(batch_ctx)
        ctx_len = write_batch["mask"].size(1)
        ctx_hidden = hidden[:, ctx_start:ctx_start + ctx_len, :]
        if ctx_hidden.size(1) != ctx_len:
            raise ValueError(
                "Invalid energy hidden-state alignment: "
                f"backend={self.memory_backend}, hidden_len={hidden.size(1)}, "
                f"ctx_start={ctx_start}, ctx_len={ctx_len}"
            )
        return ctx_hidden

    def _future_embeddings(self, segment, mask):
        if self.energy_future_mode == "none":
            return None
        if self.energy_future_mode != "next_token":
            raise ValueError(f"Unsupported energy_future_mode={self.energy_future_mode}")

        emb_layer = self.model.get_input_embeddings()
        target_len = mask.size(1)
        emb_dim = emb_layer.embedding_dim
        future = emb_layer.weight.new_zeros(segment.size(0), target_len, emb_dim)
        if segment.size(1) <= 1:
            return future

        segment_len = min(segment.size(1), target_len)
        next_len = max(0, segment_len - 1)
        if next_len == 0:
            return future

        next_valid = mask[:, 1:segment_len].bool()
        next_emb = emb_layer(segment[:, 1:segment_len])
        future[:, :next_len, :] = next_emb * next_valid.unsqueeze(-1).to(next_emb.dtype)
        return future

    def _energy_input(self, ctx_hidden, segment, mask):
        future = self._future_embeddings(segment, mask)
        if future is None:
            return ctx_hidden
        return torch.cat([ctx_hidden, future.to(dtype=ctx_hidden.dtype)], dim=-1)

    def _inner_grad_options(self, global_step, total_steps, keep_energy_graph=False):
        is_second_order_step = (
            self.grad_mode == "second"
            and global_step >= (total_steps - self.last_K_second_order)
        )
        create_graph = is_second_order_step
        has_future_energy_use = global_step < (total_steps - 1)
        retain_graph = (
            create_graph
            or has_future_energy_use
            or keep_energy_graph
            or (self.add_inner_loss_to_outer and global_step == total_steps - 1)
        )
        return create_graph, retain_graph

    @staticmethod
    def _record_grad_stats(stats, grads, batch_size, device):
        g_sq = torch.zeros(batch_size, device=device)
        for g in grads:
            g_sq = g_sq + g.reshape(batch_size, -1).pow(2).sum(dim=1)
        g_norm = g_sq.sqrt().detach()
        stats["inner_grad_norm_mean"] += g_norm.mean()
        stats["inner_grad_norm_max"] = max(stats["inner_grad_norm_max"], g_norm.max())
        stats["inner_grad_norm_min"] = min(stats["inner_grad_norm_min"], g_norm.min())

    def _updated_inner_params(self, inner_params, grads, opt_state, local_step):
        new_params = []
        for p_idx, (p, g) in enumerate(zip(inner_params, grads)):
            if self.use_adam:
                p_new = self._adam_step(p, g, opt_state.setdefault(str(p_idx), {}), local_step + 1, self.lr)
            else:
                p_new = self._sgd_step(
                    p,
                    g,
                    clip_value=self.inner_clip_value,
                    clip_norm=self.inner_clip_norm,
                )
            new_params.append(p_new)
        return new_params

    def _write_segments(self, context_segments, query_input_ids, memory_state, energy_state):
        backend = self.memory_backend_impl
        pad_id = self.model.config.pad_token_id
        device = query_input_ids.device
        batch_size = query_input_ids.size(0)
        opt_state = {}
        inner_loss = torch.tensor(0.0, device=device)
        inner_energy_loss = torch.tensor(0.0, device=device)
        inner_ce_loss = torch.tensor(0.0, device=device)
        guidance_loss = torch.tensor(0.0, device=device)
        write_steps = 0
        stats = self._init_inner_loop_stats(device)

        if not self.K:
            return memory_state, energy_state, inner_loss, guidance_loss, write_steps, stats

        with torch.enable_grad():
            total_steps = self.K * len(context_segments)
            global_step = 0
            for segment in context_segments:
                batch_ctx = backend.prepare_batch(segment, query_input_ids, pad_id)
                if not segment.ne(pad_id).any():
                    global_step += self.K
                    continue

                for k in range(self.K):
                    write_batch = backend.build_write_inputs(memory_state, batch_ctx)
                    write_out = self._run_write_model(write_batch, memory_state)
                    ctx_hidden = self._extract_context_hidden(write_out.hidden_states[-1], write_batch, batch_ctx)
                    energy_input = self._energy_input(ctx_hidden, segment, write_batch["mask"])
                    energy_loss, energy_state, energy = self._energy_loss(energy_input, write_batch["mask"], energy_state)
                    step_ce_loss = torch.zeros_like(energy_loss)
                    if self.energy_ce_guidance:
                        token_ce, token_ce_mask = self._write_token_ce(write_out, write_batch)
                    if self.energy_inner_ce_weight != 0.0:
                        step_ce_loss = self._write_inner_ce_loss(write_out, write_batch)
                    inner_loss = energy_loss + self.energy_inner_ce_weight * step_ce_loss
                    inner_energy_loss = inner_energy_loss + energy_loss.detach()
                    inner_ce_loss = inner_ce_loss + step_ce_loss.detach()
                    if self.energy_ce_guidance:
                        guidance_loss = guidance_loss + self._energy_ce_guidance_loss(
                            energy,
                            token_ce,
                            token_ce_mask,
                        )
                    del write_out

                    create_graph, retain_graph = self._inner_grad_options(
                        global_step,
                        total_steps,
                        keep_energy_graph=self.energy_ce_guidance,
                    )
                    inner_params = backend.inner_params(memory_state)
                    grads = torch.autograd.grad(
                        inner_loss,
                        inner_params,
                        create_graph=create_graph,
                        retain_graph=retain_graph,
                    )

                    self._record_grad_stats(stats, grads, batch_size, device)
                    new_params = self._updated_inner_params(inner_params, grads, opt_state, k)
                    backend.assign_inner_params(memory_state, new_params)
                    backend.maybe_detach_after_step(memory_state)
                    write_steps += 1
                    global_step += 1

        if write_steps and self.energy_ce_guidance:
            guidance_loss = guidance_loss / write_steps
        if write_steps:
            stats["inner_energy_loss"] = inner_energy_loss / (write_steps * batch_size)
            stats["inner_ce_loss"] = inner_ce_loss / (write_steps * batch_size)
        return memory_state, energy_state, inner_loss, guidance_loss, write_steps, stats

    @staticmethod
    def _finalize_inner_stats(stats, inner_loss, write_steps, batch_size):
        if write_steps:
            stats["inner_grad_norm_mean"] = stats["inner_grad_norm_mean"] / write_steps
            stats["inner_loss"] = inner_loss.detach() / batch_size
        else:
            stats["inner_loss"] = inner_loss.detach()

    def _add_memory_stats(self, stats, memory_state, memory_state_initial):
        mem_norm, delta_mem_norm = self.memory_backend_impl.compute_memory_stats(memory_state, memory_state_initial)
        stats["mem_norm_mean"] = mem_norm.mean()
        stats["mem_norm_max"] = mem_norm.max()
        stats["mem_norm_min"] = mem_norm.min()
        stats["delta_mem_norm_mean"] = delta_mem_norm.mean()
        stats["delta_mem_norm_max"] = delta_mem_norm.max()
        stats["delta_mem_norm_min"] = delta_mem_norm.min()

    def _read_from_memory(self, memory_state, query_input_ids):
        backend = self.memory_backend_impl
        pad_id = self.model.config.pad_token_id
        dummy_context = query_input_ids[:, :1].clone()
        read_ctx = backend.prepare_batch(dummy_context, query_input_ids, pad_id)
        read_batch = backend.build_read_inputs(memory_state, read_ctx)
        read_model_kwargs = read_batch.get("model_kwargs", {})
        log_mem_attn = (self.attn_implementation == "eager") and (self.memory_backend in ("prefix", "kv_cache"))
        if log_mem_attn:
            read_model_kwargs = dict(read_model_kwargs)
            read_model_kwargs["output_attentions"] = True

        with backend.activation_context(memory_state):
            with self._disable_write_lora():
                read_out = self.model(
                    inputs_embeds=read_batch["inputs_embeds"],
                    return_dict=True,
                    **read_model_kwargs,
                )

        logits = read_out.logits[:, read_batch["logits_start"]:read_batch["logits_start"] + read_batch["pred_len"], :]
        return logits, read_batch, read_out if log_mem_attn else None

    def _add_read_attention_stats(self, stats, read_out):
        if read_out is None or read_out.attentions is None:
            return
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
                    "EnergyGradMem: Invalid memory attention span on read: "
                    f"backend={self.memory_backend}, mem_start={mem_start}, mem_end={mem_end}, k_len={k_len}"
                )
            layer_ratios.append(att[..., mem_start:mem_end].sum(dim=-1).mean())
        if layer_ratios:
            stats["mem_attn_read"] = torch.stack(layer_ratios).mean().detach()

    @staticmethod
    def _target_loss(predictions, labels, read_batch):
        target_logits = predictions[:, :-1]
        target_label_shift = read_batch.get("label_shift", 0)
        target_labels = labels[:, target_label_shift:]
        if target_logits.size(1) != target_labels.size(1):
            raise ValueError(
                f"Mismatched target lengths after alignment: logits_len={target_logits.size(1)}, "
                f"labels_len={target_labels.size(1)}, label_shift={target_label_shift}"
            )
        return nn.functional.cross_entropy(
            target_logits.reshape(-1, predictions.size(-1)),
            target_labels.reshape(-1),
            ignore_index=-100,
        )

    def forward(self, input_ids, labels=None, return_mem=False, return_energy_state=False, energy_state=None):
        context_segments = self._context_segments(input_ids["context_input_ids"])
        query_input_ids = input_ids["query_input_ids"]
        if energy_state is None:
            energy_state = input_ids.get("energy_state")

        B = query_input_ids.size(0)
        self._validate_context_segments(context_segments, B)

        backend = self.memory_backend_impl
        memory_state, memory_state_initial = backend.init_memory_state(B)

        memory_state, energy_state, inner_loss, guidance_loss, write_steps, inner_loop_stats = self._write_segments(
            context_segments,
            query_input_ids,
            memory_state,
            energy_state,
        )
        self._finalize_inner_stats(inner_loop_stats, inner_loss, write_steps, B)
        self._add_memory_stats(inner_loop_stats, memory_state, memory_state_initial)

        logits_q, read_batch, read_out = self._read_from_memory(memory_state, query_input_ids)
        self._add_read_attention_stats(inner_loop_stats, read_out)

        output = {"predictions": logits_q, "inner_loop_stats": inner_loop_stats}
        if return_mem:
            backend.attach_return_memory(output, memory_state)
        if return_energy_state or self.return_energy_state:
            output["energy_state"] = energy_state

        if labels is None:
            return output

        target_loss = self._target_loss(output["predictions"], labels, read_batch)
        output["inner_loop_stats"]["target_loss"] = target_loss.detach()
        if self.energy_ce_guidance:
            output["inner_loop_stats"]["energy_ce_guidance_loss"] = guidance_loss.detach()
        if self.add_inner_loss_to_outer:
            combined_loss = target_loss + self.inner_loss_weight * (inner_loss / B)
        else:
            combined_loss = target_loss
        if self.energy_ce_guidance:
            combined_loss = combined_loss + self.energy_ce_guidance_alpha * guidance_loss
        output["loss"] = combined_loss
        return output
