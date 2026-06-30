import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class EnergyMLP(nn.Module):
    def __init__(self, hidden_dim, energy_hidden_dim):
        super().__init__()
        self.ln = nn.LayerNorm(2 * hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_dim, energy_hidden_dim),
            nn.SiLU(),
            nn.Linear(energy_hidden_dim, energy_hidden_dim),
            nn.SiLU(),
            nn.Linear(energy_hidden_dim, 1),
        )

    def forward(self, hidden, label_emb):
        x = torch.cat([hidden, label_emb], dim=-1)
        return self.net(self.ln(x)).squeeze(-1)


def make_random_dataset(num_samples, seq_len, hidden_dim, vocab_size, device):
    hidden = torch.randn(num_samples, seq_len, hidden_dim, device=device)
    labels = torch.randint(0, vocab_size, (num_samples, seq_len), device=device)
    mask = torch.ones(num_samples, seq_len, device=device)
    return hidden, labels, mask


@torch.no_grad()
def compute_ce_targets(hidden, labels, lm_head):
    logits = lm_head(hidden)
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    ).view_as(labels)
    return ce


@torch.no_grad()
def evaluate(energy_mlp, label_embedding, lm_head, loader, device):
    energy_mlp.eval()
    total_abs_err = 0.0
    total_sq_err = 0.0
    total_tokens = 0.0
    total_target = 0.0
    total_pred = 0.0

    for hidden, labels, mask in loader:
        hidden = hidden.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        ce = compute_ce_targets(hidden, labels, lm_head)
        label_emb = label_embedding(labels)
        pred = energy_mlp(hidden, label_emb)

        total_abs_err += ((pred - ce).abs() * mask).sum().item()
        total_sq_err += (((pred - ce) ** 2) * mask).sum().item()
        total_target += (ce * mask).sum().item()
        total_pred += (pred * mask).sum().item()
        total_tokens += mask.sum().item()

    return {
        "mae": total_abs_err / max(total_tokens, 1.0),
        "mse": total_sq_err / max(total_tokens, 1.0),
        "target_mean": total_target / max(total_tokens, 1.0),
        "pred_mean": total_pred / max(total_tokens, 1.0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Distill a label-conditioned energy MLP into CE targets on random data."
    )
    parser.add_argument("--num-samples", type=int, default=100000)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--energy-hidden-dim", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-path", default="distilled_energy_mlp_random.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Frozen synthetic LM components. These define the CE function to distill.
    lm_head = nn.Linear(args.hidden_dim, args.vocab_size).to(device)
    label_embedding = nn.Embedding(args.vocab_size, args.hidden_dim).to(device)
    lm_head.requires_grad_(False)
    label_embedding.requires_grad_(False)

    hidden, labels, mask = make_random_dataset(
        args.num_samples,
        args.seq_len,
        args.hidden_dim,
        args.vocab_size,
        device="cpu",
    )

    split = int(0.9 * args.num_samples)
    train_ds = TensorDataset(hidden[:split], labels[:split], mask[:split])
    val_ds = TensorDataset(hidden[split:], labels[split:], mask[split:])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    energy_mlp = EnergyMLP(args.hidden_dim, args.energy_hidden_dim).to(device)
    optimizer = torch.optim.AdamW(energy_mlp.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        energy_mlp.train()
        total_loss = 0.0
        total_tokens = 0.0

        for hidden_b, labels_b, mask_b in train_loader:
            hidden_b = hidden_b.to(device)
            labels_b = labels_b.to(device)
            mask_b = mask_b.to(device)

            with torch.no_grad():
                ce = compute_ce_targets(hidden_b, labels_b, lm_head)
                label_emb = label_embedding(labels_b)

            pred = energy_mlp(hidden_b, label_emb)
            token_loss = F.smooth_l1_loss(pred, ce, reduction="none")
            loss = (token_loss * mask_b).sum() / mask_b.sum().clamp_min(1)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            num_tokens = mask_b.sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

        train_loss = total_loss / max(total_tokens, 1.0)
        val = evaluate(energy_mlp, label_embedding, lm_head, val_loader, device)
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
            "lm_head": lm_head.state_dict(),
            "label_embedding": label_embedding.state_dict(),
        },
        args.save_path,
    )
    print(f"saved {args.save_path}")


if __name__ == "__main__":
    main()
