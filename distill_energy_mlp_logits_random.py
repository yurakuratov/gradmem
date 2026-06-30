import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitEnergyMLP(nn.Module):
    def __init__(self, vocab_size, energy_hidden_dim, num_hidden_layers=1, logit_norm="none"):
        super().__init__()
        if logit_norm == "none":
            self.ln = nn.Identity()
        elif logit_norm == "layernorm":
            self.ln = nn.LayerNorm(vocab_size)
        else:
            raise ValueError("logit_norm must be one of: none, layernorm")
        layers = []
        in_dim = vocab_size
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, energy_hidden_dim))
            layers.append(nn.SiLU())
            in_dim = energy_hidden_dim
        layers.append(nn.Linear(in_dim, vocab_size))
        self.net = nn.Sequential(*layers)

    def forward(self, logits):
        return self.net(self.ln(logits))


def make_random_logits(batch_size, seq_len, vocab_size, device, logit_std_min, logit_std_max):
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    log_min = torch.log(torch.tensor(logit_std_min, device=device))
    log_max = torch.log(torch.tensor(logit_std_max, device=device))
    scale = torch.exp(torch.empty(batch_size, seq_len, 1, device=device).uniform_(log_min, log_max))
    shift = torch.randn(batch_size, seq_len, 1, device=device) * scale
    return logits * scale + shift


@torch.no_grad()
def ce_for_all_labels(logits):
    return torch.logsumexp(logits, dim=-1, keepdim=True) - logits


@torch.no_grad()
def evaluate(energy_mlp, args, device):
    energy_mlp.eval()
    total_abs_err = 0.0
    total_sq_err = 0.0
    total_values = 0
    total_target = 0.0
    total_pred = 0.0

    for _ in range(args.val_batches):
        logits = make_random_logits(
            args.batch_size,
            args.seq_len,
            args.vocab_size,
            device,
            args.logit_std_min,
            args.logit_std_max,
        )
        target = ce_for_all_labels(logits)
        pred = energy_mlp(logits)

        total_abs_err += (pred - target).abs().sum().item()
        total_sq_err += ((pred - target) ** 2).sum().item()
        total_target += target.sum().item()
        total_pred += pred.sum().item()
        total_values += target.numel()

    return {
        "mae": total_abs_err / max(total_values, 1),
        "mse": total_sq_err / max(total_values, 1),
        "target_mean": total_target / max(total_values, 1),
        "pred_mean": total_pred / max(total_values, 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Distill a logits-conditioned energy MLP into per-label CE targets on random LM logits."
    )
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-batches", type=int, default=2000)
    parser.add_argument("--val-batches", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--energy-hidden-dim", type=int, default=512)
    parser.add_argument("--energy-head-num-layers", type=int, default=1)
    parser.add_argument("--logit-norm", choices=["none", "layernorm"], default="none")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logit-std-min", type=float, default=0.25)
    parser.add_argument("--logit-std-max", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-path", default="distilled_energy_mlp_logits_random.pt")
    args = parser.parse_args()

    if args.energy_head_num_layers < 1:
        raise ValueError("energy-head-num-layers must be >= 1")
    if args.logit_std_min <= 0 or args.logit_std_max < args.logit_std_min:
        raise ValueError("logit std range must satisfy 0 < min <= max")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    energy_mlp = LogitEnergyMLP(
        args.vocab_size,
        args.energy_hidden_dim,
        args.energy_head_num_layers,
        args.logit_norm,
    ).to(device)
    optimizer = torch.optim.AdamW(energy_mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        energy_mlp.train()
        total_loss = 0.0
        total_values = 0

        for _ in range(args.train_batches):
            logits = make_random_logits(
                args.batch_size,
                args.seq_len,
                args.vocab_size,
                device,
                args.logit_std_min,
                args.logit_std_max,
            )
            with torch.no_grad():
                target = ce_for_all_labels(logits)

            pred = energy_mlp(logits)
            loss = F.smooth_l1_loss(pred, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            num_values = target.numel()
            total_loss += loss.item() * num_values
            total_values += num_values

        val = evaluate(energy_mlp, args, device)
        train_loss = total_loss / max(total_values, 1)
        print(
            f"epoch={epoch:03d} "
            f"train_huber={train_loss:.4f} "
            f"val_mae={val['mae']:.4f} "
            f"val_mse={val['mse']:.4f} "
            f"target_mean={val['target_mean']:.4f} "
            f"pred_mean={val['pred_mean']:.4f}"
        )

    torch.save(
        {
            "args": vars(args),
            "energy_mlp": energy_mlp.state_dict(),
            "energy_input_mode": "logits",
            "energy_output_mode": "per_label_ce",
            "energy_logit_norm": args.logit_norm,
            "vocab_size": args.vocab_size,
            "energy_hidden_dim": args.energy_hidden_dim,
            "energy_head_num_layers": args.energy_head_num_layers,
        },
        args.save_path,
    )
    print(f"saved {args.save_path}")


if __name__ == "__main__":
    main()
