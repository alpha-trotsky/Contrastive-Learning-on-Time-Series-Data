import os
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import wandb


# Dependencies and usage:
# pip install torch matplotlib pyyaml wandb
# wandb login
# python synthetic_experiment.py

class SyntheticTimeSeriesDataset(Dataset):
    """
    Generates a single long synthetic 1D time-series (sine waves + noise)
    and returns sliding pairs of windows (x_t, x_{t+1}).
    """

    def __init__(self, total_steps: int, sequence_length: int):
        super().__init__()
        self.sequence_length = sequence_length

        # Create time axis
        t = torch.linspace(0, 100, steps=total_steps + sequence_length + 2)

        # Underlying (noiseless) signal: sum of a few sine waves
        clean_signal = (
            torch.sin(6 * t)
            + 0.5 * torch.sin(1.5 * t + 1.0)
            + 0.3 * torch.sin(15 * t + 2.5)
        )
        # Observed series: add Gaussian noise
        noise = 0.01 * torch.randn_like(clean_signal)
        self.clean_series = clean_signal
        self.series = clean_signal + noise

    def __len__(self) -> int:
        # For each index i we build:
        # x_t     = series[i : i + L]
        # x_{t+1} = series[i + 1 : i + L + 1]
        # so the last valid i is len(series) - L - 1
        return self.series.numel() - self.sequence_length - 1

    def __getitem__(self, idx: int):
        L = self.sequence_length
        x_t = self.series[idx : idx + L]
        x_t1 = self.series[idx + 1 : idx + L + 1]
        return x_t, x_t1


class SupervisedMLP(nn.Module):
    """
    Simple MLP baseline mapping window x_t -> predicted window x̂_{t+1}.
    """

    def __init__(self, sequence_length: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sequence_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sequence_length),
        )

    def forward(self, x):
        # x: (batch, L)
        return self.net(x)


class TimeSeriesEncoder(nn.Module):
    """
    MLP encoder that outputs L2-normalized embeddings for contrastive learning.
    """

    def __init__(self, sequence_length: int, embed_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sequence_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        # x: (batch, L)
        z = self.net(x)
        z = F.normalize(z, dim=-1)
        return z


class LinearProbe(nn.Module):
    """
    Simple linear head on top of frozen encoder embeddings to predict
    the next scalar value (last point in the next window).
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, z):
        # z: (batch, D)
        return self.fc(z).squeeze(-1)


def contrastive_loss(z_t, z_t1, temperature: float):
    """
    CLIP-style symmetric InfoNCE loss.
    z_t, z_t1: (N, D) L2-normalized embeddings from current and future states.
    """
    logits = (z_t @ z_t1.T) / temperature  # (N, N)
    targets = torch.arange(z_t.size(0), device=z_t.device)

    loss_i = F.cross_entropy(logits, targets)
    loss_j = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_i + loss_j)


def load_config():
    config_path = Path(__file__).with_name("config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def train_supervised_and_contrastive(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SyntheticTimeSeriesDataset(
        total_steps=cfg["total_steps"], sequence_length=cfg["sequence_length"]
    )
    dataloader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=True
    )

    sequence_length = cfg["sequence_length"]
    embed_dim = 32  # fixed here; could be moved into config

    supervised_model = SupervisedMLP(sequence_length=sequence_length, hidden_dim=64).to(
        device
    )
    encoder = TimeSeriesEncoder(
        sequence_length=sequence_length, embed_dim=embed_dim, hidden_dim=64
    ).to(device)

    supervised_opt = torch.optim.Adam(
        supervised_model.parameters(), lr=cfg["learning_rate_supervised"]
    )
    contrastive_opt = torch.optim.Adam(
        encoder.parameters(), lr=cfg["learning_rate_contrastive"]
    )

    mse_loss = nn.MSELoss()

    wandb.init(project=cfg["project_name"], config=cfg)

    global_step = 0
    for epoch in range(cfg["num_epochs"]):
        for batch_idx, (x_t, x_t1) in enumerate(dataloader):
            x_t = x_t.to(device)
            x_t1 = x_t1.to(device)

            # Supervised baseline: x_t -> x̂_{t+1}
            supervised_opt.zero_grad()
            pred_next = supervised_model(x_t)
            sup_loss = mse_loss(pred_next, x_t1)
            sup_loss.backward()
            supervised_opt.step()

            # Contrastive encoder: maximize similarity between z_t and z_{t+1}
            contrastive_opt.zero_grad()
            z_t = encoder(x_t)
            z_t1 = encoder(x_t1)
            cont_loss = contrastive_loss(z_t, z_t1, temperature=cfg["temperature"])
            cont_loss.backward()
            contrastive_opt.step()

            global_step += 1

            if batch_idx % cfg["log_interval"] == 0:
                print(
                    f"Epoch {epoch:03d} | Batch {batch_idx:04d} "
                    f"| Supervised MSE: {sup_loss.item():.6f} "
                    f"| Contrastive loss: {cont_loss.item():.6f}"
                )
                wandb.log(
                    {
                        "supervised_mse": sup_loss.item(),
                        "contrastive_loss": cont_loss.item(),
                        "epoch": epoch,
                        "step": global_step,
                    }
                )

    return dataset, supervised_model, encoder, device


def train_linear_probe(cfg, dataset, encoder, device):
    """
    Train a small linear head on top of the frozen encoder embeddings
    to predict the next scalar value. This gives us a 'contrastive model'
    prediction to compare against the supervised baseline.
    """
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    embed_dim = encoder.net[-1].out_features
    probe = LinearProbe(embed_dim=embed_dim).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=cfg["learning_rate_probe"])
    mse_loss = nn.MSELoss()

    dataloader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=True
    )

    num_probe_epochs = cfg["num_probe_epochs"]
    for epoch in range(num_probe_epochs):
        for batch_idx, (x_t, x_t1) in enumerate(dataloader):
            x_t = x_t.to(device)
            x_t1 = x_t1.to(device)
            # Next scalar target: last value in x_{t+1}
            target_next_scalar = x_t1[:, -1]

            with torch.no_grad():
                z_t = encoder(x_t)

            pred_scalar = probe(z_t)
            loss = mse_loss(pred_scalar, target_next_scalar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[Probe] Epoch {epoch} | MSE: {loss.item():.6f}")
        wandb.log({"probe_mse": loss.item(), "probe_epoch": epoch})

    return probe


def roll_out_forecast(cfg, dataset, supervised_model, encoder, probe, device):
    """
    Roll out n-step forecasts for:
      - supervised window predictor
      - contrastive encoder + linear probe
    and compare against ground truth.
    """
    supervised_model.eval()
    encoder.eval()
    probe.eval()

    L = cfg["sequence_length"]
    future_steps = cfg["future_steps"]
    start_idx = cfg["start_index"]

    series = dataset.series.to(device)
    clean_series = dataset.clean_series.to(device)

    # Initial window x_t
    x_window = series[start_idx : start_idx + L].clone().unsqueeze(0)  # (1, L)

    # Ground truth future values: next L-1 already in window; we track from the last point onward
    gt_start = start_idx + L - 1
    gt = series[gt_start : gt_start + future_steps + 1].detach().cpu().numpy()
    clean_gt = (
        clean_series[gt_start : gt_start + future_steps + 1].detach().cpu().numpy()
    )

    # Supervised predictions (next scalar each step)
    sup_window = x_window.clone()
    sup_preds = [sup_window[0, -1].item()]

    # Contrastive + probe predictions
    cont_window = x_window.clone()
    cont_preds = [cont_window[0, -1].item()]

    with torch.no_grad():
        for _ in range(future_steps):
            # Supervised model: predict full next window, take last value
            next_window_sup = supervised_model(sup_window)
            next_val_sup = next_window_sup[0, -1]

            # Update supervised window (shift left, append prediction)
            sup_window = torch.roll(sup_window, shifts=-1, dims=1)
            sup_window[0, -1] = next_val_sup
            sup_preds.append(next_val_sup.item())

            # Contrastive model + probe: encode current window, predict next scalar
            z = encoder(cont_window)
            next_val_cont = probe(z)[0]

            cont_window = torch.roll(cont_window, shifts=-1, dims=1)
            cont_window[0, -1] = next_val_cont
            cont_preds.append(next_val_cont.item())

    sup_preds = torch.tensor(sup_preds).cpu().numpy()
    cont_preds = torch.tensor(cont_preds).cpu().numpy()

    # Time axis (relative)
    steps = list(range(len(gt)))

    plt.figure(figsize=(10, 6))
    plt.plot(steps, gt, label="Ground Truth", color="black")
    plt.plot(steps, clean_gt, label="Noiseless Signal", color="tab:green", alpha=0.8)
    plt.plot(steps, sup_preds, label="Supervised", color="tab:blue")
    plt.plot(steps, cont_preds, label="Contrastive+Probe", color="tab:orange")
    plt.xlabel("Step into the future")
    plt.ylabel("Value")
    plt.title("n-step Forecast: Supervised vs Contrastive (Probe)")
    plt.legend()
    plt.tight_layout()

    fig_path = Path(__file__).with_name("forecast_plot.png")
    plt.savefig(fig_path)
    wandb.log({"forecast_plot": wandb.Image(str(fig_path))})
    plt.show()


def main():
    cfg = load_config()

    # Main training: supervised MLP and contrastive encoder
    dataset, supervised_model, encoder, device = train_supervised_and_contrastive(cfg)

    # Train a simple linear probe on top of the contrastive encoder
    probe = train_linear_probe(cfg, dataset, encoder, device)

    # Roll out forecasts and visualize
    roll_out_forecast(cfg, dataset, supervised_model, encoder, probe, device)


if __name__ == "__main__":
    """
    Usage:
      1. Install dependencies:
           pip install torch matplotlib pyyaml wandb
      2. Log in to Weights & Biases:
           wandb login
      3. From the Synthetic_Proof_of_Concept directory, run:
           python synthetic_experiment.py
    """
    main()

