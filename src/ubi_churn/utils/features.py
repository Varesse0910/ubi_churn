"""Data preparation and feature engineering (notebooks 04 and 06).

Every function is DataFrame in, DataFrame out, with no side effect.

Conventions (kept from the notebooks):
- Win/lose uses only `victory` in TRUE / FALSE; `draw` / `other` are neither and break a streak.
- K/D uses only rows with non-null `kills` and `death`.
- A session is a run of matches separated by less than `session_gap_min` minutes.
- Reference date for recency features = last match of the whole dataset.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #


def prepare_player_matches(df_raw: pl.DataFrame, integer_columns: list[str]) -> pl.DataFrame:
    """Type the raw player-match table.

    Args:
        df_raw: Raw CSV read as strings (`player_matches_raw` in the catalog).
        integer_columns: Columns to cast to Int64 (unparseable values become null).

    Returns:
        Same rows with `match_datetime` as UTC datetime, integer counts and
        `still_active` as Boolean. `victory` stays a string (TRUE / FALSE / draw / other).
    """
    return df_raw.with_columns(
        pl.col('match_datetime').str.to_datetime(time_zone='UTC', strict=False),
        pl.col(integer_columns).cast(pl.Int64, strict=False),
        (pl.col('still_active') == 'TRUE').alias('still_active'),
    )


def reference_date(df_player_matches: pl.DataFrame) -> datetime:
    """Last match of the whole log, used as 'today' for recency features."""
    return df_player_matches['match_datetime'].max()


def win_expr() -> pl.Expr:
    """1 for a victory, 0 for a defeat, null for draw / other."""
    return pl.when(pl.col('victory') == 'TRUE').then(1).when(pl.col('victory') == 'FALSE').then(0).cast(pl.Int8).alias('win')


def kd_expr(mask: pl.Expr) -> pl.Expr:
    """K/D (total kills / total deaths) over the rows selected by `mask`; null without any death."""
    deaths = pl.col('death').filter(mask).sum()
    return pl.when(deaths > 0).then(pl.col('kills').filter(mask).sum() / deaths)


# --------------------------------------------------------------------------- #
# Match-level context
# --------------------------------------------------------------------------- #


def add_match_context(df_player_matches: pl.DataFrame, session_gap_min: int) -> pl.DataFrame:
    """Sort by player and time and add per-match context columns.

    Adds `win`, `gap_s` (seconds since the previous match), `match_n` / `match_n_desc`
    (rank in the player's history), `n_total`, `days_since_first`, `hour`, `is_weekend`,
    `is_night` (UTC 0-6h), `n_teammates`, `kills_ge_deaths`, `new_session`, `session_id`, `prev_win`.
    """
    return (
        df_player_matches.sort('playerid', 'match_datetime', 'matchid')
        .with_columns(
            win_expr(),
            pl.col('match_datetime').diff().over('playerid').dt.total_seconds().alias('gap_s'),
            (pl.int_range(pl.len()).over('playerid') + 1).alias('match_n'),
            pl.len().over('playerid').alias('n_total'),
            (pl.col('match_datetime') - pl.col('match_datetime').min().over('playerid')).dt.total_seconds().truediv(86400).alias('days_since_first'),
            pl.col('match_datetime').dt.hour().alias('hour'),
            (pl.col('match_datetime').dt.weekday() >= 6).alias('is_weekend'),
            pl.col('teammateid').str.split(',').list.len().fill_null(0).alias('n_teammates'),
            (pl.col('kills') >= pl.col('death')).alias('kills_ge_deaths'),
        )
        .with_columns(
            (pl.col('gap_s').is_null() | (pl.col('gap_s') > session_gap_min * 60)).alias('new_session'),
            pl.col('win').shift(1).over('playerid').alias('prev_win'),
            (pl.col('n_total') - pl.col('match_n') + 1).alias('match_n_desc'),
            (pl.col('hour') < 6).alias('is_night'),  # UTC: the player time zone is unknown
        )
        .with_columns(pl.col('new_session').cum_sum().over('playerid').alias('session_id'))
    )


# --------------------------------------------------------------------------- #
# Feature families (one row per player)
# --------------------------------------------------------------------------- #


def build_activity(df_pm: pl.DataFrame, ref_date: datetime) -> pl.DataFrame:
    """Activity and tenure: volume, rhythm, sessions, gaps and recency (`days_since_last_match`, `matches_last_*`)."""
    return (
        df_pm.group_by('playerid')
        .agg(
            pl.col('still_active').first(),
            pl.len().alias('matches'),
            pl.col('match_datetime').min().alias('first_match'),
            pl.col('match_datetime').max().alias('last_match'),
            pl.col('match_datetime').dt.date().n_unique().alias('active_days'),
            pl.col('match_datetime').dt.truncate('1w').n_unique().alias('active_weeks'),
            pl.col('match_datetime').dt.truncate('1w').min().alias('first_week'),
            pl.col('match_datetime').dt.truncate('1w').max().alias('last_week'),
            pl.col('session_id').max().alias('sessions'),
            # gaps between two sessions, in hours (gap inside a session is not a "break")
            (pl.col('gap_s').filter(pl.col('new_session')).median() / 3600).alias('median_break_h'),
            (pl.col('gap_s').max() / 86400).alias('max_gap_days'),
            (pl.col('days_since_first') < 30).sum().alias('matches_first_30d'),
            (pl.col('match_datetime') >= ref_date - timedelta(days=30)).sum().alias('matches_last_30d'),
            (pl.col('match_datetime') >= ref_date - timedelta(days=90)).sum().alias('matches_last_90d'),
        )
        .with_columns(
            ((pl.col('last_match') - pl.col('first_match')).dt.total_seconds() / 86400).alias('tenure_days'),
            ((pl.lit(ref_date) - pl.col('last_match')).dt.total_seconds() / 86400).alias('days_since_last_match'),
        )
        .with_columns(
            (pl.col('matches') / pl.col('active_days')).alias('matches_per_active_day'),
            (pl.col('matches') / pl.col('sessions')).alias('matches_per_session'),
            (pl.col('sessions') / (pl.col('tenure_days') / 7 + 1)).alias('sessions_per_week'),
            # share of the calendar weeks of his lifespan in which he played at least once (regularity)
            (pl.col('active_weeks') / ((pl.col('last_week') - pl.col('first_week')).dt.total_days() / 7 + 1)).alias('active_weeks_share'),
        )
        .drop('first_week', 'last_week')
    )


def build_runs(df_pm: pl.DataFrame) -> pl.DataFrame:
    """Runs of consecutive identical `victory` values per player (draw / other break a run).

    `df_pm` must be sorted by player and time. Returns one row per (player, run).
    """
    return (
        df_pm.with_columns(
            (pl.col('victory') != pl.col('victory').shift(1).over('playerid')).fill_null(True).alias('new_run')
        )
        .with_columns(pl.col('new_run').cum_sum().over('playerid').alias('run_id'))
        .group_by('playerid', 'run_id', 'victory')
        .agg(pl.len().alias('run_length'))
        .sort('playerid', 'run_id')
    )


def build_streaks(df_runs: pl.DataFrame) -> pl.DataFrame:
    """Longest win / loss streak and the signed streak at the last match (+ wins, - losses, 0 draw / other)."""
    return df_runs.group_by('playerid').agg(
        pl.col('run_length').filter(pl.col('victory') == 'TRUE').max().fill_null(0).alias('max_win_streak'),
        pl.col('run_length').filter(pl.col('victory') == 'FALSE').max().fill_null(0).alias('max_loss_streak'),
        pl.when(pl.col('victory').last() == 'TRUE').then(pl.col('run_length').last().cast(pl.Int64))
        .when(pl.col('victory').last() == 'FALSE').then(-pl.col('run_length').last().cast(pl.Int64))
        .otherwise(0).alias('current_streak'),
    )


def build_performance(df_pm: pl.DataFrame, n_early: int, n_edge: int) -> pl.DataFrame:
    """Performance: win rate, K/D, kills / deaths profile, early and first / last window, streaks.

    `win_rate_first/last` and `kd_first/last` are turned into trends in `assemble_player_features`.
    """
    df_perf: pl.DataFrame = (
        df_pm.group_by('playerid')
        .agg(
            pl.col('win').count().alias('decided'),
            pl.col('win').sum().alias('wins'),
            pl.col('win').mean().alias('win_rate'),
            pl.col('kills').count().alias('matches_with_kd'),
            pl.col('kills').sum().alias('kills_sum'),
            pl.col('death').sum().alias('death_sum'),
            pl.col('kills').mean().alias('kills_per_match'),
            pl.col('death').mean().alias('deaths_per_match'),
            pl.col('kills').std().alias('kills_std'),
            pl.col('death').std().alias('deaths_std'),
            pl.col('kills').max().alias('kills_max'),
            (pl.col('kills') == 0).mean().alias('share_zero_kill'),
            pl.col('kills_ge_deaths').mean().alias('share_kills_ge_deaths'),
            pl.col('win').filter(pl.col('match_n') <= n_early).mean().alias('win_rate_early'),
            kd_expr(pl.col('match_n') <= n_early).alias('kd_early'),
            pl.col('win').filter(pl.col('match_n') <= n_edge).mean().alias('win_rate_first'),
            pl.col('win').filter(pl.col('match_n_desc') <= n_edge).mean().alias('win_rate_last'),
            kd_expr(pl.col('match_n') <= n_edge).alias('kd_first'),
            kd_expr(pl.col('match_n_desc') <= n_edge).alias('kd_last'),
            pl.col('win').last().alias('last_result_win'),
        )
        .with_columns(
            pl.when(pl.col('death_sum') > 0).then(pl.col('kills_sum') / pl.col('death_sum')).alias('kd_ratio'),
        )
        .drop('kills_sum', 'death_sum')
    )
    return df_perf.join(build_streaks(build_runs(df_pm)), on='playerid', how='left')


def build_behaviour(df_pm: pl.DataFrame) -> pl.DataFrame:
    """Behaviour: requeue after a loss / win (`tilt_gap`), weekend / night play, longest session."""
    return (
        df_pm.group_by('playerid')
        .agg(
            (~pl.col('new_session')).filter(pl.col('prev_win') == 0).mean().alias('stay_after_loss'),
            (~pl.col('new_session')).filter(pl.col('prev_win') == 1).mean().alias('stay_after_win'),
            pl.col('is_weekend').mean().alias('share_weekend'),
            pl.col('is_night').mean().alias('share_night'),
            pl.col('hour').median().alias('median_hour'),
        )
        .with_columns((pl.col('stay_after_loss') - pl.col('stay_after_win')).alias('tilt_gap'))
        .join(
            df_pm.group_by('playerid', 'session_id').len().group_by('playerid').agg(pl.col('len').max().alias('max_session_matches')),
            on='playerid', how='left',
        )
    )


def build_modes(df_pm: pl.DataFrame, modes: list[str], min_decided_mode: int) -> pl.DataFrame:
    """Game modes and lobbies: mode shares, number of modes, lobby size, per-mode win rate / K/D.

    Per-mode win rate / K/D are null when the player has fewer than `min_decided_mode` decided matches in the mode.
    """
    df_modes: pl.DataFrame = df_pm.group_by('playerid').agg(
        *[(pl.col('gamemode') == mode).mean().alias(f'share_{mode.lower()}') for mode in modes],
        pl.col('gamemode').n_unique().alias('n_modes'),
        pl.col('playernb_in_match').mean().alias('avg_lobby_size'),
        (pl.col('playernb_in_match') == 10).mean().alias('share_standard_lobby'),
        pl.col('n_teammates').mean().alias('avg_teammates'),
    )
    df_mode_perf: pl.DataFrame = (
        df_pm.filter(pl.col('gamemode').is_in(modes))
        .group_by('playerid', 'gamemode')
        .agg(
            pl.col('win').count().alias('decided'),
            pl.col('win').mean().alias('win_rate'),
            pl.when(pl.col('death').sum() > 0).then(pl.col('kills').sum() / pl.col('death').sum()).alias('kd_ratio'),
        )
        .filter(pl.col('decided') >= min_decided_mode)
        .pivot(on='gamemode', index='playerid', values=['win_rate', 'kd_ratio'])
    )
    df_mode_perf = df_mode_perf.rename({c: c.lower() for c in df_mode_perf.columns if c != 'playerid'})
    # pivot orders columns by first appearance after a group_by (not deterministic): fix the order
    ordered: list[str] = [f'{metric}_{mode.lower()}' for metric in ('win_rate', 'kd_ratio') for mode in modes]
    df_mode_perf = df_mode_perf.select('playerid', *[c for c in ordered if c in df_mode_perf.columns])
    return df_modes.join(df_mode_perf, on='playerid', how='left')


def explode_partners(df_pm: pl.DataFrame, column: str) -> pl.DataFrame:
    """One row per (player, partner slot) from a comma-separated id column (`teammateid` / `opponentid`)."""
    return (
        df_pm.select('playerid', pl.col(column).str.split(',').alias('partner'))
        .explode('partner')
        .drop_nulls('partner')
    )


def build_social(df_pm: pl.DataFrame, df_activity: pl.DataFrame) -> pl.DataFrame:
    """Social structure: distinct teammates / opponents, repeat-partner shares, top teammate share."""
    df_team: pl.DataFrame = explode_partners(df_pm, 'teammateid')
    df_opp: pl.DataFrame = explode_partners(df_pm, 'opponentid')
    return (
        df_team.group_by('playerid')
        .agg(pl.len().alias('teammate_slots'), pl.col('partner').n_unique().alias('unique_teammates'))
        .join(
            df_team.group_by('playerid', 'partner').len().group_by('playerid').agg(pl.col('len').max().alias('top_teammate_matches')),
            on='playerid',
        )
        .join(
            df_opp.group_by('playerid').agg(pl.len().alias('opponent_slots'), pl.col('partner').n_unique().alias('unique_opponents')),
            on='playerid', how='left',
        )
        .join(df_activity.select('playerid', 'matches'), on='playerid')
        .with_columns(
            (1 - pl.col('unique_teammates') / pl.col('teammate_slots')).alias('repeat_teammate_share'),
            (1 - pl.col('unique_opponents') / pl.col('opponent_slots')).alias('repeat_opponent_share'),
            (pl.col('top_teammate_matches') / pl.col('matches')).alias('top_teammate_share'),
        )
        .drop('teammate_slots', 'opponent_slots', 'top_teammate_matches', 'matches')
    )


# --------------------------------------------------------------------------- #
# Composite scores: Elo and addiction index
# --------------------------------------------------------------------------- #


def elo_scale_se() -> float:
    """Logit standard error -> Elo points (400 / ln 10)."""
    return 400 / np.log(10)


def elo_path(results: list[int], start: int, k_provisional: int, k_standard: int, provisional_matches: int) -> list[float]:
    """Elo rating after each decided match against an average opponent rated `start`."""
    rating, path = float(start), []
    for i, result in enumerate(results):
        expected = 1 / (1 + 10 ** ((start - rating) / 400))  # opponent = average player at `start`
        rating += (k_provisional if i < provisional_matches else k_standard) * (result - expected)
        path.append(rating)
    return path


def compute_elo_paths(df_pm: pl.DataFrame, elo_params: dict) -> dict[str, list[float]]:
    """Elo trajectory of every player with at least one decided match, in time order.

    Args:
        df_pm: Output of `add_match_context`.
        elo_params: `start`, `k_provisional`, `k_standard`, `provisional_matches`.
    """
    df_decided: pl.DataFrame = (
        df_pm.filter(pl.col('win').is_not_null())
        .group_by('playerid', maintain_order=True)
        .agg(pl.col('win'))
    )
    return {pid: elo_path(wins, **elo_params) for pid, wins in zip(df_decided['playerid'], df_decided['win'])}


def build_elo(df_activity: pl.DataFrame, df_perf: pl.DataFrame, elo_paths: dict[str, list[float]], start: int) -> pl.DataFrame:
    """Elo current form (`elo`, `elo_peak`) and long-run skill with confidence (`elo_skill`, `elo_se`, ...)."""
    scale: float = elo_scale_se()
    return (
        df_activity.select('playerid')
        .join(
            pl.DataFrame({'playerid': list(elo_paths), 'elo': [p[-1] for p in elo_paths.values()], 'elo_peak': [max(p) for p in elo_paths.values()]}),
            on='playerid', how='left',
        )
        # players without any decided match stay at the starting rating
        .with_columns(pl.col('elo').fill_null(start), pl.col('elo_peak').fill_null(start))
        .join(df_perf.select('playerid', 'wins', 'decided'), on='playerid')
        .with_columns((pl.col('decided') - pl.col('wins')).alias('losses'))
        .with_columns(
            (start + scale * ((pl.col('wins') + 1) / (pl.col('losses') + 1)).log()).alias('elo_skill'),
            (scale * (1 / (pl.col('wins') + 1) + 1 / (pl.col('losses') + 1)).sqrt()).alias('elo_se'),
        )
        .with_columns(
            (1 - pl.col('elo_se') / (scale * np.sqrt(2))).alias('elo_confidence'),
            (pl.col('elo_skill') - 1.96 * pl.col('elo_se')).alias('elo_low'),
            (pl.col('elo_skill') + 1.96 * pl.col('elo_se')).alias('elo_high'),
        )
        .drop('wins', 'losses', 'decided')
    )


def build_addiction(df_activity: pl.DataFrame, df_behaviour: pl.DataFrame, min_matches: int, components: list[str]) -> pl.DataFrame:
    """Addiction index (0-100): mean percentile rank of `components` among players with `min_matches`+ matches."""
    return (
        df_activity.select('playerid', 'matches', 'sessions_per_week', 'matches_per_active_day', 'matches_per_session', 'active_weeks_share')
        .join(df_behaviour.select('playerid', 'max_session_matches', 'share_night', 'stay_after_loss'), on='playerid')
        .filter(pl.col('matches') >= min_matches)
        .with_columns(*[(pl.col(c).rank() / pl.col(c).count()).alias(f'{c}_pct') for c in components])
        .with_columns(
            (100 * pl.mean_horizontal([f'{c}_pct' for c in components])).alias('addiction_index'),  # ignores a null component
        )
        .select('playerid', 'addiction_index')
    )


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def build_feature_groups(mode_columns: list[str], social_columns: list[str]) -> dict[str, list[str]]:
    """Feature names per family. The recency group leaks `still_active`."""
    return {
        'recency (leaks the target)': ['last_match', 'days_since_last_match', 'matches_last_30d', 'matches_last_90d'],
        'activity': ['matches', 'first_match', 'tenure_days', 'active_days', 'active_weeks', 'active_weeks_share', 'sessions', 'sessions_per_week',
                     'matches_per_active_day', 'matches_per_session', 'median_break_h', 'max_gap_days', 'matches_first_30d'],
        'performance': ['decided', 'wins', 'win_rate', 'matches_with_kd', 'kd_ratio', 'kills_per_match', 'deaths_per_match', 'kills_std', 'deaths_std',
                        'kills_max', 'share_zero_kill', 'share_kills_ge_deaths', 'win_rate_early', 'kd_early', 'win_rate_trend', 'kd_trend',
                        'last_result_win', 'max_win_streak', 'max_loss_streak', 'current_streak', 'elo', 'elo_peak', 'elo_skill', 'elo_se',
                        'elo_confidence', 'elo_low', 'elo_high'],
        'behaviour': ['stay_after_loss', 'stay_after_win', 'tilt_gap', 'share_weekend', 'share_night', 'median_hour', 'max_session_matches'],
        'modes and lobbies': [c for c in mode_columns if c != 'playerid'],
        'social': [c for c in social_columns if c != 'playerid'],
        'composite scores': ['addiction_index'],
    }


def assemble_player_features(
    df_activity: pl.DataFrame,
    df_perf: pl.DataFrame,
    df_behaviour: pl.DataFrame,
    df_modes: pl.DataFrame,
    df_social: pl.DataFrame,
    df_elo: pl.DataFrame,
    df_addiction: pl.DataFrame,
    n_edge: int,
    target: str,
) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    """Join the feature families into one row per player and check the result.

    Returns:
        The feature table (`playerid`, target, then features ordered by family) and the family mapping.
    """
    df: pl.DataFrame = (
        df_activity.join(df_perf, on='playerid', how='left')
        .join(df_behaviour, on='playerid', how='left')
        .join(df_modes, on='playerid', how='left')
        .join(df_social, on='playerid', how='left')
        .join(df_elo, on='playerid', how='left')
        .join(df_addiction, on='playerid', how='left')
        .with_columns(
            (pl.col('win_rate_last') - pl.col('win_rate_first')).alias('win_rate_trend'),
            (pl.col('kd_last') - pl.col('kd_first')).alias('kd_trend'),
        )
        # trends only when the first and last windows do not overlap
        .with_columns(
            pl.when(pl.col('matches') >= 2 * n_edge).then(pl.col('win_rate_trend')).alias('win_rate_trend'),
            pl.when(pl.col('matches') >= 2 * n_edge).then(pl.col('kd_trend')).alias('kd_trend'),
        )
        .drop('win_rate_first', 'win_rate_last', 'kd_first', 'kd_last')
        .sort('playerid')
    )
    groups: dict[str, list[str]] = build_feature_groups(df_modes.columns, df_social.columns)
    grouped: list[str] = [c for cols in groups.values() for c in cols]
    missing_or_extra: set[str] = set(df.columns) ^ set(grouped + ['playerid', target])
    assert not missing_or_extra, f'feature groups do not match the table: {missing_or_extra}'
    assert df['playerid'].n_unique() == df.height, 'duplicated players in the feature table'
    return df.select('playerid', target, *grouped), groups


def build_player_features(df_player_matches: pl.DataFrame, params: dict) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    """Full feature pipeline: typed match table -> one row per player (notebook 04).

    Args:
        df_player_matches: Output of `prepare_player_matches`.
        params: Project parameters (`load_params()`), uses `features` and `target`.
    """
    p: dict = params['features']
    ref_date: datetime = reference_date(df_player_matches)
    df_pm: pl.DataFrame = add_match_context(df_player_matches, p['session_gap_min'])
    df_activity: pl.DataFrame = build_activity(df_pm, ref_date)
    df_perf: pl.DataFrame = build_performance(df_pm, p['n_early'], p['n_edge'])
    df_behaviour: pl.DataFrame = build_behaviour(df_pm)
    df_modes: pl.DataFrame = build_modes(df_pm, p['modes'], p['min_decided_mode'])
    df_social: pl.DataFrame = build_social(df_pm, df_activity)
    df_elo: pl.DataFrame = build_elo(df_activity, df_perf, compute_elo_paths(df_pm, p['elo']), p['elo']['start'])
    df_addiction: pl.DataFrame = build_addiction(df_activity, df_behaviour, p['addiction']['min_matches'], p['addiction']['components'])
    df, groups = assemble_player_features(
        df_activity, df_perf, df_behaviour, df_modes, df_social, df_elo, df_addiction, p['n_edge'], params['target'],
    )
    assert df.height == df_player_matches['playerid'].n_unique(), 'every player must have one feature row'
    return df, groups


# --------------------------------------------------------------------------- #
# Model-selection features (notebook 06)
# --------------------------------------------------------------------------- #


def in_window(lo: int, hi: int) -> pl.Expr:
    """Match played between `lo` (included) and `hi` (excluded) days before the end of the log (needs `days_ago`)."""
    return (pl.col('days_ago') >= lo) & (pl.col('days_ago') < hi)


def build_recent_windows(df_player_matches: pl.DataFrame, ref_date: datetime) -> pl.DataFrame:
    """Activity / performance in the last 7 / 14 / 30 days before `ref_date` and trend vs the period before.

    All columns are prefixed `r_`. Players with no match in a window get 0 matches and null rates.
    """
    df_m: pl.DataFrame = df_player_matches.sort('playerid', 'match_datetime', 'matchid').with_columns(
        ((pl.lit(ref_date) - pl.col('match_datetime')).dt.total_seconds() / 86400).alias('days_ago'),
        win_expr(),
    )
    return df_m.group_by('playerid').agg(
        in_window(0, 7).sum().alias('r_matches_7d'),
        in_window(0, 14).sum().alias('r_matches_14d'),
        in_window(14, 28).sum().alias('r_matches_prev14d'),
        in_window(0, 30).sum().alias('r_matches_30d'),
        in_window(30, 60).sum().alias('r_matches_prev30d'),
        pl.col('match_datetime').filter(in_window(0, 14)).dt.date().n_unique().alias('r_active_days_14d'),
        pl.col('match_datetime').filter(in_window(0, 30)).dt.date().n_unique().alias('r_active_days_30d'),
        pl.col('match_datetime').filter(in_window(0, 56)).dt.truncate('1w').n_unique().alias('r_active_weeks_8w'),
        pl.col('win').filter(in_window(0, 30)).mean().alias('r_win_rate_30d'),
        (pl.col('kills').filter(in_window(0, 30)).sum() / pl.col('death').filter(in_window(0, 30)).sum().clip(lower_bound=1)).alias('r_kd_30d'),
        (pl.col('gamemode') == 'PVP_RANKED').filter(in_window(0, 30)).mean().alias('r_share_ranked_30d'),
        pl.col('gamemode').filter(in_window(0, 30)).n_unique().alias('r_n_modes_30d'),
    ).with_columns(
        ((pl.col('r_matches_14d') + 1) / (pl.col('r_matches_prev14d') + 1)).alias('r_trend_14d'),
        ((pl.col('r_matches_30d') + 1) / (pl.col('r_matches_prev30d') + 1)).alias('r_trend_30d'),
    )


def build_early_window(
    df_player_matches: pl.DataFrame,
    ref_date: datetime,
    n_window: int,
    session_gap_min: int,
    target: str,
) -> pl.DataFrame:
    """Features from each player's first `n_window` days only (early-warning model).

    Players whose first match is less than `n_window` days before `ref_date` are dropped
    (incomplete window). Feature columns are prefixed `w_` and cast to Float64.
    """
    gap_new_session: pl.Expr = pl.col('gap_s').is_null() | (pl.col('gap_s') > session_gap_min * 60)
    df_win: pl.DataFrame = (
        df_player_matches.sort('playerid', 'match_datetime', 'matchid')
        .with_columns(
            win_expr(),
            pl.col('match_datetime').diff().over('playerid').dt.total_seconds().alias('gap_s'),
            (pl.col('match_datetime') - pl.col('match_datetime').min().over('playerid')).dt.total_seconds().truediv(86400).alias('days_since_first'),
            pl.col('teammateid').str.split(',').list.len().fill_null(0).alias('n_teammates'),
        )
        .filter(pl.col('days_since_first') < n_window)  # the window: nothing after it is used
        .with_columns(
            gap_new_session.alias('new_session'),
            (~gap_new_session).shift(-1).over('playerid').alias('next_in_session'),
            (pl.col('match_datetime').dt.hour() < 6).alias('is_night'),
            (pl.col('match_datetime').dt.weekday() >= 6).alias('is_weekend'),
        )
        .with_columns(pl.col('new_session').cum_sum().over('playerid').alias('session_id'))
    )
    df_early: pl.DataFrame = (
        df_win.group_by('playerid')
        .agg(
            pl.col(target).first(),
            pl.col('match_datetime').min().alias('first_match'),
            pl.len().alias('w_matches'),
            pl.col('match_datetime').dt.date().n_unique().alias('w_active_days'),
            pl.col('session_id').max().alias('w_sessions'),
            (pl.col('gap_s').filter(pl.col('new_session')).median() / 3600).alias('w_median_break_h'),
            (pl.col('days_since_first') < 1).sum().alias('w_matches_day1'),
            pl.col('win').drop_nulls().mean().alias('w_win_rate'),
            (pl.col('kills').sum() / pl.col('death').sum().clip(lower_bound=1)).alias('w_kd_ratio'),
            pl.col('kills').mean().alias('w_kills_per_match'),
            pl.col('death').mean().alias('w_deaths_per_match'),
            (pl.col('gamemode') == 'PVP_RANKED').mean().alias('w_share_pvp_ranked'),
            (pl.col('gamemode') == 'PVP_UNRANKED').mean().alias('w_share_pvp_unranked'),
            (pl.col('gamemode') == 'PVP').mean().alias('w_share_pvp'),
            pl.col('gamemode').n_unique().alias('w_n_modes'),
            pl.col('playernb_in_match').mean().alias('w_avg_lobby_size'),
            pl.col('n_teammates').mean().alias('w_avg_teammates'),
            pl.col('is_night').mean().alias('w_share_night'),
            pl.col('is_weekend').mean().alias('w_share_weekend'),
            pl.col('next_in_session').filter(pl.col('win') == 0).mean().alias('w_stay_after_loss'),
            pl.col('next_in_session').filter(pl.col('win') == 1).mean().alias('w_stay_after_win'),
            (pl.col('teammateid').n_unique() / pl.len()).alias('w_teammate_diversity'),
        )
        .filter(pl.col('first_match') <= ref_date - timedelta(days=n_window))
    )
    early_features: list[str] = [c for c in df_early.columns if c.startswith('w_')]
    return df_early.with_columns(pl.col(early_features).cast(pl.Float64))
