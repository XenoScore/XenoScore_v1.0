from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import yaml

class ConfigError(Exception):
    pass

def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"YAML not found: {p}")
    with open(p, "r") as f:
        return yaml.safe_load(f)

def load_component_config(path: str | Path) -> Dict[str, Any]:
    data = load_yaml(path)
    # Expected structure:
    # components:
    #   - name: InfectionStatus
    #     params: {...}
    #   - name: DonorGenetics
    #     params: {...}
    components = data.get("components")
    if not isinstance(components, list):
        raise ConfigError("Invalid component list in YAML.")
    return data

def load_weights_config(path: str | Path) -> Dict[str, float]:
    data = load_yaml(path)
    weights = data.get("weights", {})
    if not isinstance(weights, dict):
        raise ConfigError("Invalid weights mapping in YAML.")
    # Ensure all weights are floats
    return {str(k): float(v) for k, v in weights.items()}
