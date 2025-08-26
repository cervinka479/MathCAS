import os
import torch
import time
import pandas as pd
import random
import numpy as np
import traceback
from pathlib import Path
from utils.experiment import create_experiment_dir, copy_config

from utils.metrics import format_time
from utils.modules import get_loss_function, get_optimizer, get_scheduler
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
        if batch_idx < 5:
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

def validate(
    model: torch.nn.Module,
    loader, # temporary placeholder for DataLoader
    criterion: torch.nn.Module,
    device: torch.device
) -> float:
    
    model.eval()
    running_loss = 0.0

    nvtx_mark("Validate Epoch", color="yellow")
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            running_loss += loss.item()

    return running_loss / len(loader)

def train(config: FullConfig, exp_dir: Path, logger) -> None:
    """
    Train a neural network model based on the provided configuration.

    Args:
        config (FullConfig): The full configuration object containing architecture, training, and data settings.
        exp_dir (str): Path to the experiment directory for saving models and logs.
        logger: Logger object for logging training progress and information.

    Returns:
        None
    """
    
    set_seed(config.seed)

    # Log hardware/environment info
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        cuda_version = getattr(torch.version, "cuda", "N/A")  # type: ignore
        logger.info(f"CUDA version: {cuda_version}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    with nvtx_range("NN Creation, Move to Device", color="blue"):
        model = NeuralNetwork(config.architecture).to(device)

    # Log model parameter count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {num_params}")

    criterion = get_loss_function(config.training.loss_function)
    optimizer = get_optimizer(
        config.training.optimizer,
        model.parameters(),
        config.training.learning_rate
    )

    # Scheduler setup
    scheduler = None
    if hasattr(config.training, "scheduler") and config.training.scheduler == "ReduceLROnPlateau":
        scheduler = get_scheduler(
            "ReduceLROnPlateau",
            optimizer,
            patience=config.training.scheduler_patience or 2,
            factor=config.training.scheduler_factor or 0.5,
            threshold=config.training.scheduler_threshold or 1e-4,
            verbose=True
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

    try:
        for epoch in range(config.training.epochs):
            train_loader, val_loader = prepare_dataloaders(config.data, epoch=epoch)

            # Log dataset sizes
            if epoch == 0:
                try:
                    train_size = len(train_loader.dataset)  # type: ignore
                    val_size = len(val_loader.dataset)  # type: ignore
                except Exception:
                    train_size = val_size = None
                logger.info(f"Training samples: {train_size} | Validation samples: {val_size} | Batch size: {config.data.batch_size}")

            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = validate(model, val_loader, criterion, device)

            # Step the scheduler
            if scheduler is not None:
                scheduler.step(val_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_train_loss = train_loss
                torch.save(
                    model.state_dict(),
                    os.path.join(exp_dir, "best_model.pt")
                )

                patience_counter = 0
            else:
                patience_counter += 1

            elapsed_time = time.time() - start_time
            elapsed_times.append(elapsed_time)

            # Log epoch information
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch + 1}/{config.training.epochs} | "
                f"Train Loss: {train_loss:.4e} | Val Loss: {val_loss:.4e} | "
                f"LR: {current_lr:.2e} | Time: {format_time(elapsed_time)}"
            )

            # Log scheduler events (ReduceLROnPlateau)
            if scheduler is not None and hasattr(scheduler, 'num_bad_epochs') and scheduler.num_bad_epochs == 0 and epoch > 0:
                logger.info(f"Learning rate reduced to {current_lr:.2e}")

            if config.training.early_stopping and patience_counter >= config.training.patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        total_time = time.time() - start_time
        logger.info("Training complete.")
        logger.info(f"Best model at epoch {best_epoch} | Train Loss: {best_train_loss:.4e} | Val Loss: {best_val_loss:.4e}")
        logger.info(f"Total training time: {format_time(total_time)}")
    except Exception as e:
        logger.error(f"Exception occurred: {e}")
        logger.error(traceback.format_exc())
        raise