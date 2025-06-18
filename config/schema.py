from pydantic import BaseModel



class ArchitectureConfig(BaseModel):
    in_size: int
    out_size: int
    hidden_layers: list[int]
    activation: str
    use_dropout: bool
    dropout: float
    final_activation: str | None

class TrainingConfig(BaseModel):
    learning_rate: float
    optimizer: str
    loss_function: str
    epochs: int
    early_stopping: bool
    patience: int

class DataConfig(BaseModel):
    path_to_data: str
    num_samples: int | None = None
    batch_size: int
    in_cols: list[str]
    out_cols: list[str]
    val_split: float
    shuffle: bool

class FullConfig(BaseModel):
    name: str | None = None
    verbose: bool = True
    save_logs: bool = True
    seed: int = 42
    output_dir: str = "outputs"
    architecture: ArchitectureConfig
    training: TrainingConfig
    data: DataConfig