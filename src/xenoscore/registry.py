from __future__ import annotations
from typing import Dict, Type

# Global registry for feature components
COMPONENT_REGISTRY: Dict[str, "FeatureComponent"] = {}

def register_component(name: str):
    def _wrap(cls):
        COMPONENT_REGISTRY[name] = cls
        cls.__component_name__ = name
        return cls
    return _wrap

def get_component(name: str):
    if name not in COMPONENT_REGISTRY:
        raise KeyError(f"Component not found: {name}. Registered: {list(COMPONENT_REGISTRY)}")
    return COMPONENT_REGISTRY[name]
