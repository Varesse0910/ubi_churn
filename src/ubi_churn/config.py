"""Project configuration, Kedro-style but without the Kedro dependency.

- `PROJECT_ROOT`: folder holding `conf/` and `data/`.
- `load_params()`: `conf/base/parameters.yml`, overridden by `conf/local/parameters.yml` if present.
- `DataCatalog`: loads / saves the datasets declared in `conf/base/catalog.yml` by name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import polars as pl
import yaml


def _find_project_root() -> Path:
    """Return the project root: the package location first, then the working directory and its parents."""
    candidates: list[Path] = [Path(__file__).resolve().parents[2], Path.cwd().resolve(), *Path.cwd().resolve().parents]
    for folder in candidates:
        if (folder / 'conf' / 'base' / 'catalog.yml').exists():
            return folder
    raise FileNotFoundError('project root not found: no conf/base/catalog.yml above ' + str(Path.cwd()))


PROJECT_ROOT: Path = _find_project_root()
CONF_DIR: Path = PROJECT_ROOT / 'conf'


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`."""
    merged: dict = dict(base)
    for key, value in override.items():
        merged[key] = _deep_update(merged[key], value) if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
    return merged


def load_yaml(name: str) -> dict:
    """Load `conf/base/<name>` merged with `conf/local/<name>` (local wins)."""
    config: dict = yaml.safe_load((CONF_DIR / 'base' / name).read_text(encoding='utf-8')) or {}
    local_path: Path = CONF_DIR / 'local' / name
    if local_path.exists():
        config = _deep_update(config, yaml.safe_load(local_path.read_text(encoding='utf-8')) or {})
    return config


def load_params() -> dict:
    """Return the project parameters (`parameters.yml`)."""
    return load_yaml('parameters.yml')


class DataCatalog:
    """Minimal data catalog: dataset name -> file, with typed load / save.

    Supported `type` values: `polars.CSVDataset`, `polars.ParquetDataset`,
    `pandas.CSVDataset`, `joblib.JoblibDataset`, `folder`.
    """

    def __init__(self, entries: dict[str, dict], root: Path = PROJECT_ROOT) -> None:
        self._entries: dict[str, dict] = entries
        self._root: Path = root

    @classmethod
    def from_config(cls) -> DataCatalog:
        """Build the catalog from `conf/*/catalog.yml`."""
        return cls(load_yaml('catalog.yml'))

    def list(self) -> list[str]:
        """Names of every dataset in the catalog."""
        return list(self._entries)

    def _entry(self, name: str) -> dict:
        if name not in self._entries:
            raise KeyError(f'{name!r} is not in the catalog; available: {self.list()}')
        return self._entries[name]

    def path(self, name: str) -> Path:
        """Absolute path of a dataset."""
        return self._root / self._entry(name)['filepath']

    def exists(self, name: str) -> bool:
        """True if the dataset file exists on disk."""
        return self.path(name).exists()

    def load(self, name: str) -> Any:
        """Load a dataset by name."""
        entry, path = self._entry(name), self.path(name)
        if not path.exists():
            raise FileNotFoundError(f'{name!r} not found at {path}. Run the pipeline first: python -m ubi_churn.pipeline')
        kind, load_args = entry['type'], entry.get('load_args', {})
        if kind == 'polars.CSVDataset':
            return pl.read_csv(path, **load_args)
        if kind == 'polars.ParquetDataset':
            return pl.read_parquet(path, **load_args)
        if kind == 'pandas.CSVDataset':
            return pd.read_csv(path, **load_args)
        if kind == 'joblib.JoblibDataset':
            return joblib.load(path)
        raise ValueError(f'cannot load dataset type {kind!r}')

    def save(self, name: str, data: Any) -> Path:
        """Save a dataset by name (parent folders are created) and return its path."""
        entry, path = self._entry(name), self.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        kind, save_args = entry['type'], entry.get('save_args', {})
        if kind == 'polars.CSVDataset':
            data.write_csv(path, **save_args)
        elif kind == 'polars.ParquetDataset':
            data.write_parquet(path, **save_args)
        elif kind == 'pandas.CSVDataset':
            data.to_csv(path, index=False, **save_args)
        elif kind == 'joblib.JoblibDataset':
            joblib.dump(data, path, **save_args)
        else:
            raise ValueError(f'cannot save dataset type {kind!r}')
        return path
