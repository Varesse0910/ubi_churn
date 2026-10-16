"""Sanity tests for the utils. Run with `pytest tests` or `python tests/test_utils.py` (from the project root).

They need the typed match table and the feature table: run `python -m ubi_churn.pipeline` first.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import polars as pl

from ubi_churn.config import DataCatalog, load_params
from ubi_churn.utils import exploration as ex
from ubi_churn.utils import features as fe
from ubi_churn.utils import ml

CATALOG = DataCatalog.from_config()
PARAMS = load_params()


def _matches() -> pl.DataFrame:
    return CATALOG.load('player_matches')


def test_config_and_catalog() -> None:
    assert PARAMS['target'] == 'still_active'
    assert set(PARAMS['churn_model']['feature_info']) <= set(CATALOG.load('player_features').columns)
    for name in ('player_matches_raw', 'player_matches', 'player_features', 'churn_model'):
        assert CATALOG.exists(name), name


def test_prepare_is_reproducible() -> None:
    df = fe.prepare_player_matches(CATALOG.load('player_matches_raw'), PARAMS['preprocessing']['integer_columns'])
    assert df.equals(_matches())
    assert df.select('playerid', 'matchid').unique().height == df.height, '(player, match) must be a key'


def test_feature_table_is_reproducible() -> None:
    df, groups = fe.build_player_features(_matches(), PARAMS)
    assert df.equals(CATALOG.load('player_features'))
    assert df.height == _matches()['playerid'].n_unique()
    assert sum(len(c) for c in groups.values()) == df.width - 2  # + playerid + target


def test_streaks_match_plain_python() -> None:
    df = _matches()
    df_eligible = ex.win_kd_stats(df).filter(pl.col('matches') >= 3).select('playerid', 'matches')
    df_streaks = ex.max_streaks(df, df_eligible)
    for player_id in np.random.default_rng(0).choice(df_streaks['playerid'].to_numpy(), size=50, replace=False):
        results = df.filter(pl.col('playerid') == player_id).sort('match_datetime', 'matchid')['victory'].to_list()
        got = df_streaks.filter(pl.col('playerid') == player_id).select('max_win_streak', 'max_loss_streak').row(0)
        assert ex.python_streaks(results) == got, player_id


def test_features_match_plain_python() -> None:
    df_raw, df_features = _matches(), CATALOG.load('player_features')
    gap_s = PARAMS['features']['session_gap_min'] * 60
    for pid in np.random.default_rng(0).choice(df_features['playerid'].to_numpy(), size=30, replace=False):
        rows = df_raw.filter(pl.col('playerid') == pid).sort('match_datetime', 'matchid').to_dicts()
        feat = df_features.filter(pl.col('playerid') == pid).to_dicts()[0]
        times = [r['match_datetime'] for r in rows]
        assert feat['matches'] == len(rows)
        assert feat['sessions'] == 1 + sum((b - a).total_seconds() > gap_s for a, b in zip(times, times[1:]))
        assert feat['active_days'] == len({t.date() for t in times})
        assert feat['unique_teammates'] == len({t for r in rows for t in (r['teammateid'] or '').split(',') if t})


def test_elo_path() -> None:
    elo = PARAMS['features']['elo']
    assert fe.elo_path([], **elo) == []
    path = fe.elo_path([1, 1, 0], **elo)
    assert path[0] == elo['start'] + elo['k_provisional'] * 0.5  # first win vs an equal opponent
    assert path[1] > path[0] and path[2] < path[1]  # a win raises the rating, a loss lowers it


def test_lobby_arithmetic() -> None:
    df_lobby = ex.add_lobby_columns(_matches())
    assert (df_lobby['gap_with_self'] == 0).all(), 'teammates + opponents + 1 must equal playernb_in_match'


def test_predictor() -> None:
    predictor = ml.ChurnPredictor(CATALOG.load('churn_model'))
    regular = dict(days_since_last_match=2, matches_last_30d=40, matches_last_90d=120, matches=300, tenure_days=250)
    silent = dict(days_since_last_match=200, matches_last_30d=0, matches_last_90d=0, matches=3, tenure_days=1)
    assert predictor.predict_one('a', regular)['churn_risk'] < predictor.predict_one('b', silent)['churn_risk']
    assert predictor.check_inputs(regular) == []
    assert predictor.check_inputs({'matches_last_90d': -6}), 'a negative count must be flagged'
    explanation = predictor.explain(predictor.player_frame('a', regular))
    p_churn = 1 - predictor.predict_one('a', regular)['p_active']
    assert abs(explanation.base_values[0] + explanation.values[0].sum() - p_churn) < 1e-3, 'SHAP must add up to the churn risk'


if __name__ == '__main__':
    tests = [obj for name, obj in dict(globals()).items() if name.startswith('test_') and callable(obj)]
    for test in tests:
        test()
        print('ok  ', test.__name__)
    print(f'{len(tests)} tests passed')
