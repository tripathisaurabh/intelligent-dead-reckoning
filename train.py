from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

from src.ins.differentiable_iekf import (
    initialize_state,
    rollout,
    vehicle_to_enu_rotation,
)
from src.models.adapter_cnn import AdapterOutput, DilatedCNNAdapter
from src.preprocessing.dataset import load_synchronized_drive
from src.preprocessing.frames import geodetic_to_local_enu
from src.preprocessing.stationary import (
    estimate_biases_from_stationary_run,
    stationary_mask,
)


@dataclass(frozen=True)
class PreparedDrive:
    name: str
    contexts: np.ndarray
    gyro: np.ndarray
    accel: np.ndarray
    dt: np.ndarray
    initial_speed: np.ndarray
    initial_bearing: np.ndarray
    gyro_bias: np.ndarray
    accel_bias: np.ndarray
    target_position: np.ndarray
    target_velocity_delta: np.ndarray
    target_relative_rotation: np.ndarray
    phone_speed_assessment: dict


def mounting_rotation(config: dict) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(
        [config["roll_deg"], config["pitch_deg"], config["yaw_deg"]]
    )
    cx, sx, cy, sy, cz, sz = (
        np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    )
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _velocity(speed: np.ndarray, bearing_deg: np.ndarray) -> np.ndarray:
    bearing = np.deg2rad(bearing_deg)
    return np.column_stack((speed * np.sin(bearing), speed * np.cos(bearing), np.zeros_like(speed)))


def _vehicle_rotation(bearing_deg: np.ndarray) -> np.ndarray:
    bearing = np.deg2rad(bearing_deg)
    result = np.zeros((len(bearing), 3, 3))
    result[:, 0, 0], result[:, 0, 1] = np.sin(bearing), -np.cos(bearing)
    result[:, 1, 0], result[:, 1, 1] = np.cos(bearing), np.sin(bearing)
    result[:, 2, 2] = 1.0
    return result


def prepare_drive(spec: dict, config: dict, maximum_windows: int) -> PreparedDrive:
    drive = load_synchronized_drive(spec["smartphone_csv"], spec["reference_csv"])
    mask = stationary_mask(
        drive.reference_speed_mps,
        drive.gyro_ypr_rps,
        drive.accel_device_mps2,
        gravity_mps2=9.80665,
        maximum_reference_speed_mps=0.5,
        maximum_gyro_norm_rps=0.08,
        gravity_magnitude_tolerance_mps2=0.75,
    )
    bias = estimate_biases_from_stationary_run(
        drive.time_s,
        drive.accel_device_mps2,
        drive.gravity_device_mps2,
        drive.gyro_ypr_rps,
        mask,
        minimum_duration_s=5.0,
    )
    gyro_bias = bias.gyro_ypr_bias_rps[[2, 1, 0]]
    gyro = drive.gyro_ypr_rps[:, [2, 1, 0]]
    accel = drive.accel_device_mps2
    imu = np.column_stack((gyro - gyro_bias, accel - bias.residual_accel_device_bias_mps2))
    reference = geodetic_to_local_enu(drive.reference_lat_deg, drive.reference_lon_deg)
    training = config["training"]
    context, steps = config["model"]["context_samples"], training["rollout_samples"]
    starts = np.arange(context, len(drive.time_s) - steps, training["stride_samples"])
    moving = (
        (drive.reference_speed_mps[starts - 1] >= training["minimum_speed_mps"])
        & (drive.reference_speed_mps[starts + steps - 1] >= training["minimum_speed_mps"])
    )
    starts = starts[moving]
    rng = np.random.default_rng(training["seed"] + sum(map(ord, spec["name"])))
    if len(starts) > maximum_windows:
        starts = np.sort(rng.choice(starts, maximum_windows, replace=False))
    if not len(starts):
        raise ValueError(f"{spec['name']} produced no moving training windows")

    contexts = np.stack(
        [np.stack([imu[index - context + 1:index + 1] for index in range(start, start + steps)]) for start in starts]
    )
    gyro_windows = np.stack([gyro[start:start + steps] for start in starts])
    accel_windows = np.stack([accel[start:start + steps] for start in starts])
    dt = np.stack([np.diff(drive.time_s[start - 1:start + steps]) for start in starts])
    initial_indices, final_indices = starts - 1, starts + steps - 1
    initial_velocity = _velocity(
        drive.reference_speed_mps[initial_indices], drive.reference_heading_deg[initial_indices]
    )
    final_velocity = _velocity(
        drive.reference_speed_mps[final_indices], drive.reference_heading_deg[final_indices]
    )
    initial_rotation = _vehicle_rotation(drive.reference_heading_deg[initial_indices])
    final_rotation = _vehicle_rotation(drive.reference_heading_deg[final_indices])
    return PreparedDrive(
        name=spec["name"],
        contexts=contexts.astype(np.float32),
        gyro=gyro_windows.astype(np.float32),
        accel=accel_windows.astype(np.float32),
        dt=dt.astype(np.float32),
        initial_speed=drive.reference_speed_mps[initial_indices].astype(np.float32),
        initial_bearing=drive.reference_heading_deg[initial_indices].astype(np.float32),
        gyro_bias=np.tile(gyro_bias, (len(starts), 1)).astype(np.float32),
        accel_bias=np.tile(bias.residual_accel_device_bias_mps2, (len(starts), 1)).astype(np.float32),
        target_position=(reference[final_indices] - reference[initial_indices]).astype(np.float32),
        target_velocity_delta=(final_velocity - initial_velocity).astype(np.float32),
        target_relative_rotation=(np.transpose(initial_rotation, (0, 2, 1)) @ final_rotation).astype(np.float32),
        phone_speed_assessment=drive.phone_speed_unit_assessment,
    )


class IncrementDataset(Dataset):
    def __init__(self, drive: PreparedDrive):
        self.values = {
            key: torch.from_numpy(value)
            for key, value in asdict(drive).items()
            if isinstance(value, np.ndarray)
        }

    def __len__(self) -> int:
        return len(self.values["contexts"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.values.items()}


def _adapter_sequence(model: DilatedCNNAdapter, contexts: torch.Tensor) -> AdapterOutput:
    batch, steps, window, channels = contexts.shape
    raw = model(contexts.reshape(batch * steps, window, channels)).reshape(batch, steps, -1)
    return model.decode(raw)


def _loss(
    model: DilatedCNNAdapter,
    batch: dict[str, torch.Tensor],
    rotation_car_to_imu: torch.Tensor,
    config: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    count = len(batch["contexts"])
    car_rotation = rotation_car_to_imu.expand(count, -1, -1)
    state = initialize_state(
        batch["initial_speed"],
        batch["initial_bearing"],
        batch["gyro_bias"],
        batch["accel_bias"],
        car_rotation,
    )
    final = rollout(
        state,
        batch["gyro"],
        batch["accel"],
        batch["dt"],
        _adapter_sequence(model, batch["contexts"]),
    )
    initial_car = vehicle_to_enu_rotation(batch["initial_bearing"])
    final_car = final.rotation_imu_to_enu @ final.rotation_car_to_imu
    predicted_relative_rotation = initial_car.transpose(-1, -2) @ final_car
    position = nn.functional.smooth_l1_loss(final.position_enu_m, batch["target_position"], beta=2.0)
    initial_velocity = torch.stack(
        (
            batch["initial_speed"] * torch.sin(torch.deg2rad(batch["initial_bearing"])),
            batch["initial_speed"] * torch.cos(torch.deg2rad(batch["initial_bearing"])),
            torch.zeros_like(batch["initial_speed"]),
        ), dim=-1,
    )
    velocity = nn.functional.smooth_l1_loss(
        final.velocity_enu_mps - initial_velocity, batch["target_velocity_delta"], beta=1.0
    )
    rotation = nn.functional.smooth_l1_loss(
        predicted_relative_rotation, batch["target_relative_rotation"], beta=0.05
    )
    training = config["training"]
    total = (
        training["position_weight"] * position
        + training["velocity_weight"] * velocity
        + training["rotation_weight"] * rotation
    )
    return total, {
        "total": float(total.detach()),
        "position": float(position.detach()),
        "velocity": float(velocity.detach()),
        "rotation": float(rotation.detach()),
    }


def _epoch(
    model: DilatedCNNAdapter,
    loader: DataLoader,
    car_rotation: torch.Tensor,
    config: dict,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    model.train(optimizer is not None)
    totals = {name: 0.0 for name in ("total", "position", "velocity", "rotation")}
    count = 0
    for batch in loader:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            loss, parts = _loss(model, batch, car_rotation, config)
        if optimizer is not None:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_norm"])
            optimizer.step()
        batch_count = len(batch["contexts"])
        count += batch_count
        for name, value in parts.items():
            totals[name] += value * batch_count
    return {name: value / count for name, value in totals.items()}


def train(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = config["training"]["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train_drive = prepare_drive(
        config["experiment"]["train"], config, config["training"]["maximum_train_windows"]
    )
    validation_drive = prepare_drive(
        config["experiment"]["validation"],
        config,
        config["training"]["maximum_validation_windows"],
    )
    model = DilatedCNNAdapter(config["model"]["channels"], config["model"]["dropout"])
    flat_train = train_drive.contexts.reshape(-1, 6)
    model.set_normalization(np.mean(flat_train, axis=0), np.std(flat_train, axis=0))
    car_rotation = torch.tensor(mounting_rotation(config["mounting"]), dtype=torch.float32)
    train_loader = DataLoader(
        IncrementDataset(train_drive), batch_size=config["training"]["batch_size"], shuffle=True
    )
    validation_loader = DataLoader(
        IncrementDataset(validation_drive), batch_size=config["training"]["batch_size"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    history = []
    best_state, best_loss = None, float("inf")
    for epoch in range(1, config["training"]["epochs"] + 1):
        train_metrics = _epoch(model, train_loader, car_rotation, config, optimizer)
        validation_metrics = _epoch(model, validation_loader, car_rotation, config, None)
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        print(json.dumps(history[-1]))
        if validation_metrics["total"] < best_loss:
            best_loss = validation_metrics["total"]
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)

    outputs = config["outputs"]
    model_path, history_path, split_path, plot_path = map(
        Path, (outputs["model"], outputs["history"], outputs["split"], outputs["loss_plot"])
    )
    for path in (model_path, history_path, split_path, plot_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "model_config": config["model"], "mounting": config["mounting"]},
        model_path,
    )
    history_payload = {
        "history": history,
        "best_validation_loss": best_loss,
        "parameter_count": model.parameter_count,
        "training_windows": len(train_drive.contexts),
        "validation_windows": len(validation_drive.contexts),
    }
    history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")
    split_path.write_text(
        json.dumps(
            {
                "train": train_drive.name,
                "validation": validation_drive.name,
                "test": config["experiment"]["test"]["name"],
                "split_rule": "whole drives; no within-drive train/validation split",
                "phone_speed_unit_assessment": {
                    train_drive.name: train_drive.phone_speed_assessment,
                    validation_drive.name: validation_drive.phone_speed_assessment,
                },
            }, indent=2,
        ), encoding="utf-8",
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot([row["epoch"] for row in history], [row["train"]["total"] for row in history], label="train")
    axis.plot([row["epoch"] for row in history], [row["validation"]["total"] for row in history], label="validation")
    axis.set(title="Differentiable IEKF Adapter Loss", xlabel="Epoch", ylabel="Weighted Huber loss")
    axis.grid(alpha=0.25); axis.legend(); figure.tight_layout(); figure.savefig(plot_path, dpi=240)
    plt.close(figure)
    return history_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Step 4 dilated CNN through the IEKF")
    parser.add_argument("--config", type=Path, default=Path("configs/adapter_cnn.yaml"))
    args = parser.parse_args()
    print(json.dumps(train(args.config), indent=2))


if __name__ == "__main__":
    main()
