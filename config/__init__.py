import yaml
from typing import Any

def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)