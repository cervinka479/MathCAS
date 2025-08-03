from pydantic import BaseModel



class ArchitectureConfig(BaseModel):
    in_size: int
    out_size: int
    hidden_layers: list[int]
    activation: str
    use_dropout: bool
    dropout: float
    dropout_inplace: bool = False
    final_activation: str | None

class TrainingConfig(BaseModel):
    learning_rate: float
    optimizer: str
    loss_function: str
    epochs: int
    early_stopping: bool
    patience: int
    scheduler: str | None = None
    scheduler_patience: int | None = None
    scheduler_factor: float | None = None
    scheduler_threshold: float | None = None

class DataConfig(BaseModel):
    path_to_data: str
    num_samples: int | None = None
    batch_size: int
    in_cols: list[str]
    out_cols: list[str]
    val_split: float
    shuffle: bool
    sliding_window: int | None = None

class FullConfig(BaseModel):
    name: str | None = None
    verbose: bool = True
    save_logs: bool = True
    seed: int = 42
    output_dir: str = "outputs"
    architecture: ArchitectureConfig
    training: TrainingConfig
    data: DataConfig