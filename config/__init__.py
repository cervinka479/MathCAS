import os
import yaml
from config.schema import FullConfig



def load_config(config_path: str) -> FullConfig:
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)

        # Infer name if not present
        if 'name' not in raw or raw['name'] is None:
            raw['name'] = os.path.splitext(os.path.basename(config_path))[0]
            
        # Set defaults if not present
        raw.setdefault('verbose', True)
        raw.setdefault('seed', 42)
        raw.setdefault('output_dir', "outputs")
        raw.setdefault('save_logs', False)

        if 'log_file' not in raw or not raw['log_file']:
            raw['log_file'] = os.path.join(raw.get('output_dir', 'outputs'), f"{raw['name']}.log")

        return FullConfig(**raw)