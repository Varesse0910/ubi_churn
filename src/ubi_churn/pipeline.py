"""Kedro-style pipeline: raw CSV -> typed matches -> player features -> churn model.

Run from the project root (or anywhere after `pip install -e .`):

    python -m ubi_churn.pipeline                  # every node
    python -m ubi_churn.pipeline --nodes features  # a subset, in pipeline order
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

from ubi_churn.config import DataCatalog, load_params
from ubi_churn.utils import features, ml


def prepare_node(catalog: DataCatalog, params: dict) -> None:
    """player_matches_raw -> player_matches (typed)."""
    df = features.prepare_player_matches(catalog.load('player_matches_raw'), params['preprocessing']['integer_columns'])
    path = catalog.save('player_matches', df)
    print(f'[prepare]  {df.shape} -> {path}')


def features_node(catalog: DataCatalog, params: dict) -> None:
    """player_matches -> player_features (one row per player)."""
    df, groups = features.build_player_features(catalog.load('player_matches'), params)
    path = catalog.save('player_features', df)
    print(f'[features] {df.shape} -> {path} | ' + ', '.join(f'{g}: {len(c)}' for g, c in groups.items()))


def train_node(catalog: DataCatalog, params: dict) -> None:
    """player_features -> churn_model (artifact dict)."""
    artifact, df_eval = ml.train_churn_model(catalog.load('player_features'), params)
    path = catalog.save('churn_model', artifact)
    print(f'[train]    {path}\n{df_eval.to_string(index=False)}')


NODES: dict[str, Callable[[DataCatalog, dict], None]] = {
    'prepare': prepare_node,
    'features': features_node,
    'train': train_node,
}


def run(nodes: list[str] | None = None) -> None:
    """Run the selected nodes (all by default) in pipeline order."""
    catalog, params = DataCatalog.from_config(), load_params()
    for name, node in NODES.items():
        if nodes is None or name in nodes:
            node(catalog, params)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--nodes', nargs='+', choices=list(NODES), help='nodes to run (default: all)')
    run(parser.parse_args().nodes)


if __name__ == '__main__':
    main()
