import yaml
from config.schema import FullConfig



def load_config(config_path: str) -> FullConfig:
    with open(config_path, 'r') as f:
        return FullConfig(**yaml.safe_load(f))