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
    num_samples: int
    batch_size: int
    in_cols: list[str]
    out_cols: list[str]
    val_split: float
    shuffle: bool

class FullConfig(BaseModel):
    architecture: ArchitectureConfig
    training: TrainingConfig
    data: DataConfig