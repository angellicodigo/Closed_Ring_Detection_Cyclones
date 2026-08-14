import os
import json
import csv
import torch
import torch.nn as nn
from datetime import datetime

def create_dir(base_path: str) -> str:
    """Creates a unique directory using YYYY-MM-DD-HH-MM-SS when training starts."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = os.path.join(base_path, timestamp)
    
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    return output_dir

def save_config(output_dir: str, config: dict) -> None:
    """Saves hyperparameters and run settings into config.json."""
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

def log_metrics(output_dir: str, epoch: int, metrics: dict) -> None:
    """Logs per-epoch metrics into metrics.csv."""
    csv_path = os.path.join(output_dir, "metrics.csv")
    file_exists = os.path.isfile(csv_path)
    
    fieldnames = ["epoch"] + list(metrics.keys())
    row = {"epoch": epoch + 1, **metrics}
    
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def save_checkpoint(
    model: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    output_dir: str, 
    epoch: int
) -> None:
    """Saves model state dictionary and optimizer state for the epoch."""
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    
    checkpoint_data = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    
    file_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pth")
    torch.save(checkpoint_data, file_path)