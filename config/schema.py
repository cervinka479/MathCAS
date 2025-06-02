from typing import TypedDict

class ArchitectureConfig(TypedDict):
    in_size: int
    out_size: int
    hidden_layers: list[int]
    activation: str
    use_dropout: bool
    dropout: float
    final_activation: str | None