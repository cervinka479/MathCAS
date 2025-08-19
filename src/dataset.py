import math
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from config.schema import DataConfig, FullConfig
from config import load_config
from utils.profiler import nvtx_range



class FastTensorDataLoader:
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.dataset_size = tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_batches = (self.dataset_size + batch_size - 1) // batch_size

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.dataset_size, device=self.tensors[0].device)
        else:
            indices = torch.arange(self.dataset_size, device=self.tensors[0].device)
        for i in range(0, self.dataset_size, self.batch_size):
            batch_idx = indices[i:i+self.batch_size]
            yield tuple(t[batch_idx] for t in self.tensors)

    def __len__(self):
        return self.num_batches

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
        out_val=to_tensor(out_val)
    )

def create_dataloaders(
    in_train: torch.Tensor,
    out_train: torch.Tensor,
    in_val: torch.Tensor,
    out_val: torch.Tensor,
    batch_size: int
) -> tuple[DataLoader, DataLoader]:

    with nvtx_range("Create Dataloaders", color="blue"):
        train_loader = DataLoader(
            TensorDataset(in_train, out_train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            TensorDataset(in_val, out_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
    return train_loader, val_loader

def create_fast_dataloaders(
    in_train: torch.Tensor,
    out_train: torch.Tensor,
    in_val: torch.Tensor,
    out_val: torch.Tensor,
    batch_size: int
) -> tuple[FastTensorDataLoader, FastTensorDataLoader]:
    train_loader = FastTensorDataLoader(in_train, out_train, batch_size=batch_size, shuffle=True)
    val_loader = FastTensorDataLoader(in_val, out_val, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def prepare_dataloaders(data_config: DataConfig, epoch: int = 0) -> tuple[DataLoader, DataLoader]:
    from utils.profiler import nvtx_range

    with nvtx_range("Extract Data", color="blue"):
        inputs, outputs = extract_data(data_config)

    with nvtx_range("Split Data", color="blue"):
        tensor_split = split_data(inputs, outputs, data_config.val_split, data_config.shuffle)

    # Sliding window logic
    sliding_window = data_config.sliding_window
    if sliding_window is not None and sliding_window > 0:
        train_size = tensor_split.in_train.shape[0]
        window_size = sliding_window
        num_windows = math.ceil(train_size / window_size)
        window_idx = epoch % num_windows
        start = window_idx * window_size
        end = min(start + window_size, train_size)
        in_train_window = tensor_split.in_train[start:end]
        out_train_window = tensor_split.out_train[start:end]
    else:
        in_train_window = tensor_split.in_train
        out_train_window = tensor_split.out_train

    train_loader, val_loader = create_dataloaders(
        in_train_window,
        out_train_window,
        tensor_split.in_val,
        tensor_split.out_val,
        batch_size=data_config.batch_size
    )
    return train_loader, val_loader