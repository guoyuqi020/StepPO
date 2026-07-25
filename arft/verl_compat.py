from __future__ import annotations

from pathlib import Path


def active_verl_trainer_config_path() -> str:
    import verl

    config_path = Path(verl.__file__).resolve().parent / "trainer" / "config"
    if config_path.is_dir():
        return str(config_path)
    raise FileNotFoundError(f"Could not locate active verl trainer config directory: {config_path}")
