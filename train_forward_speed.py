from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.models.forward_speed_gru import ForwardSpeedGRU
from src.preprocessing.dataset import load_synchronized_drive


def _candidate_starts(speed: np.ndarray, length: int, minimum_start: float, moving_fraction: float):
    moving = speed >= 1.0
    count = np.convolve(moving.astype(int), np.ones(length, dtype=int), mode="valid")
    starts = np.flatnonzero((count >= moving_fraction * length) & (speed[: len(count)] >= minimum_start))
    return starts


def _features(drive, start: int, length: int) -> np.ndarray:
    selection = slice(start, start + length)
    linear_accel = drive.accel_device_mps2[selection] - drive.gravity_device_mps2[selection]
    elapsed = drive.time_s[selection] - drive.time_s[start]
    initial = np.full(length, drive.reference_speed_mps[start])
    return np.column_stack((drive.gyro_ypr_rps[selection], linear_accel, initial, elapsed / 20.0))


def _prepare(specs: list[dict], settings: dict, maximum_per_session: int, seed: int):
    features, targets, sessions, starts_used = [], [], [], {}
    length = settings["sequence_samples"]
    for number, spec in enumerate(specs):
        drive = load_synchronized_drive(spec["smartphone_csv"], spec["reference_csv"])
        starts = _candidate_starts(
            drive.reference_speed_mps, length, settings["minimum_start_speed_mps"],
            settings["minimum_moving_fraction"],
        )
        rng = np.random.default_rng(seed + number * 997)
        if len(starts) > maximum_per_session:
            starts = np.sort(rng.choice(starts, maximum_per_session, replace=False))
        for start in starts:
            features.append(_features(drive, int(start), length))
            targets.append(drive.reference_speed_mps[start : start + length])
            sessions.append(spec["name"])
        starts_used[spec["name"]] = starts.tolist()
    return np.asarray(features, np.float32), np.asarray(targets, np.float32), sessions, starts_used


def _loss(mean: torch.Tensor, std: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    variance = std.square()
    nll = 0.5 * ((target - mean).square() / variance + torch.log(variance))
    # A pure heteroscedastic NLL can reduce its penalty by inflating uncertainty
    # before learning the conditional mean. The MSE term keeps mean speed as the
    # primary task; NLL still learns sample-dependent uncertainty.
    return torch.mean((target - mean).square()) + 0.10 * nll.mean()


def _epoch(model, loader, optimizer=None):
    model.train(optimizer is not None)
    total, count = 0.0, 0
    for features, target in loader:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            mean, std = model(features, features[:, 0, 6])
            loss = _loss(mean, std, target)
        if optimizer is not None:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total += float(loss.detach()) * len(features)
        count += len(features)
    return total / count


def train(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings, training = config["experiment"], config["training"]
    seed = settings["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    x_train, y_train, train_sessions, train_starts = _prepare(
        settings["train"], settings, settings["train_windows_per_session"], seed
    )
    x_val, y_val, _, val_starts = _prepare(
        [settings["validation"]], settings, settings["validation_windows"], seed + 10000
    )
    mean = x_train.mean(axis=(0, 1))
    std = x_train.std(axis=(0, 1)).clip(min=1e-4)
    # Initial speed and elapsed time already have useful physical scales.
    mean[6:] = 0.0
    std[6:] = 1.0
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=training["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)), batch_size=training["batch_size"])
    model = ForwardSpeedGRU(**config["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    history, best_loss, best_state = [], float("inf"), None
    for epoch in range(1, training["epochs"] + 1):
        train_loss = _epoch(model, train_loader, optimizer)
        val_loss = _epoch(model, val_loader)
        row = {"epoch": epoch, "train_nll": train_loss, "validation_nll": val_loss}
        history.append(row); print(json.dumps(row))
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model_path = Path(config["outputs"]["model"])
    experiment_dir = Path(config["outputs"]["experiment_dir"])
    plots_dir = Path(config["outputs"]["plots_dir"])
    for directory in (model_path.parent, experiment_dir, plots_dir): directory.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_config": config["model"], "feature_mean": mean, "feature_std": std}, model_path)
    report = {
        "best_validation_nll": best_loss,
        "training_windows": len(x_train),
        "validation_windows": len(x_val),
        "training_sessions": sorted(set(train_sessions)),
        "validation_session": settings["validation"]["name"],
        "fresh_test_session": settings["fresh_test"]["name"],
        "fresh_test_was_loaded_during_training": False,
        "train_window_start_indices": train_starts,
        "validation_window_start_indices": val_starts,
        "history": history,
    }
    (experiment_dir / "training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([x["epoch"] for x in history], [x["train_nll"] for x in history], label="Train")
    ax.plot([x["epoch"] for x in history], [x["validation_nll"] for x in history], label="Validation")
    ax.set(title="Forward-speed GRU Gaussian NLL", xlabel="Epoch", ylabel="NLL"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(plots_dir / "training_loss.png", dpi=240); plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/forward_speed_v1.yaml"))
    args = parser.parse_args(); print(json.dumps(train(args.config), indent=2))


if __name__ == "__main__": main()
