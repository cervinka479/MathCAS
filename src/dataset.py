import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from config.schema import DataConfig, FullConfig
from config import load_config



@dataclass
class TensorSplit:
    in_train: torch.Tensor
    out_train: torch.Tensor
    in_val: torch.Tensor
    out_val: torch.Tensor

def to_tensor(data: np.ndarray, unsqueeze: bool = False) -> torch.Tensor:
    tensor = torch.tensor(data, dtype=torch.float32)
    return tensor.unsqueeze(1) if unsqueeze else tensor

def extract_data(config: DataConfig) -> tuple[np.ndarray, np.ndarray]:
    
    df: pd.DataFrame = pd.read_csv(
        config.path_to_data,
        usecols=config.in_cols + config.out_cols,
        nrows=config.num_samples
    )

    inputs = df[config.in_cols].values
    outputs = df[config.out_cols].values
    return inputs, outputs

def split_data(
    inputs: np.ndarray,
    outputs: np.ndarray,
    val_split: float,
    shuffle: bool = True
) -> TensorSplit:
    
    in_train, in_val, out_train, out_val = train_test_split(
        inputs, outputs,
        test_size=val_split,
        random_state=42,
        shuffle=shuffle
    )

    return TensorSplit(
        in_train=to_tensor(in_train),
        out_train=to_tensor(out_train),
        in_val=to_tensor(in_val),
        out_val=to_tensor(out_val),
    )

def create_dataloaders(
    in_train: torch.Tensor,
    out_train: torch.Tensor,
    in_val: torch.Tensor,
    out_val: torch.Tensor,
    batch_size: int
) -> tuple[DataLoader, DataLoader]:
    
    train_loader = DataLoader(
        TensorDataset(in_train, out_train),
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        TensorDataset(in_val, out_val),
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, val_loader

def prepare_dataloaders(config_path: str) -> tuple[DataLoader, DataLoader]:
    config: FullConfig = load_config(config_path)

    inputs, outputs = extract_data(config.data)
    tensor_split = split_data(inputs, outputs, config.data.val_split, config.data.shuffle)

    return create_dataloaders(
        tensor_split.in_train,
        tensor_split.out_train,
        tensor_split.in_val,
        tensor_split.out_val,
        batch_size=config.data.batch_size
    )