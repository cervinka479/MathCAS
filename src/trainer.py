import os
import torch
import time
import pandas as pd
import random
import numpy as np

from utils.metrics import format_time
from utils.modules import get_loss_function, get_optimizer
from utils.logger import setup_logger
from utils.profiler import nvtx_range, nvtx_mark
from config import load_config
from config.schema import FullConfig
from src.architecture import NeuralNetwork
from src.dataset import prepare_dataloaders

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_one_epoch(
    model: torch.nn.Module,
    loader, # temporary placeholder for DataLoader
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    
    model.train()
    running_loss = 0.0

    nvtx_mark("Training Epoch", color="yellow")
    for batch_idx, (batch_x, batch_y) in enumerate(loader):
        if batch_idx < 10:
            with nvtx_range(f"Batch {batch_idx}", color="red"):
                with nvtx_range("Move to Device", color="blue"):
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                with nvtx_range("Forward Pass", color="red"):
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                with nvtx_range("Backward Pass", color="red"):
                    loss.backward()
                optimizer.step()
                running_loss += loss.item()
        else:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    return running_loss / len(loader)

def evaluate(
    model: torch.nn.Module,
    loader, # temporary placeholder for DataLoader
    criterion: torch.nn.Module,
    device: torch.device
) -> float:
    
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            running_loss += loss.item()

    return running_loss / len(loader)

def train(config_path: str):
    config: FullConfig = load_config(config_path)
    set_seed(config.seed)

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    log_file = os.path.join(config.output_dir, f"{config.name}.log")
    logger = setup_logger(config.verbose, config.save_logs, log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.verbose:
        logger.info(f"Using device: {device}")

    model = NeuralNetwork(config.architecture).to(device)

    train_loader, val_loader = prepare_dataloaders(config_path)

    criterion = get_loss_function(config.training.loss_function)
    optimizer = get_optimizer(
        config.training.optimizer,
        model.parameters(),
        config.training.learning_rate
    )

    logger.info(f"Experiment: {config.name}")
    logger.info(f"Model architecture:\n{model}")
    logger.info(f"Training config: {config.training}")
    logger.info(f"Data config: {config.data}")

    best_val_loss = float("inf")
    best_epoch = -1
    best_train_loss = None
    patience_counter = 0
    train_losses, val_losses, elapsed_times = [], [], []
    start_time = time.time()

    for epoch in range(config.training.epochs):

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_train_loss = train_loss
            torch.save(
                model.state_dict(),
                os.path.join(config.output_dir, f"{config.name}_best_model.pth")
            )
            patience_counter = 0
        else:
            patience_counter += 1

        elapsed_time = time.time() - start_time
        elapsed_times.append(elapsed_time)

        logger.info(
            f"Epoch {epoch + 1}/{config.training.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Time: {format_time(elapsed_time)}"
        )

        if config.training.early_stopping and patience_counter >= config.training.patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    total_time = time.time() - start_time
    logger.info("Training complete.")
    logger.info(f"Best model at epoch {best_epoch} | Train Loss: {best_train_loss:.4f} | Val Loss: {best_val_loss:.4f}")
    logger.info(f"Total training time: {format_time(total_time)}")