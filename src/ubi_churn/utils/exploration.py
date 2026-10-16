"""Exploration helpers (notebooks 01-05): profiling, data quality, player stats, group comparison and plots.

Tables are polars DataFrames; plots use matplotlib / seaborn (pandas only at the plotting boundary).
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from scipy.stats import mannwhitneyu

from ubi_churn.utils.features import build_runs

ACTIVITY_ORDER: list[str] = ['Active', 'Inactive']

# --------------------------------------------------------------------------- #
# Generic profiling
# --------------------------------------------------------------------------- #




def null_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Null count and percentage per column, most missing first."""
    return (
        df.null_count()
        .transpose(include_header=True, header_name='column', column_names=['null_count'])
        .with_columns((pl.col('null_count') / df.height * 100).round(3).alias('null_pct'))
        .sort('null_count', descending=True)
    )


def share_by_category(df: pl.DataFrame, column: str, decimals: int = 1) -> pl.DataFrame:
    """Rows per distinct value of `column` with their percentage share, largest first."""
    return (
        df.group_by(column)
        .agg(pl.len().alias('row_count'))
        .with_columns((pl.col('row_count') / df.height * 100).round(decimals).alias('share_pct'))
        .sort('row_count', descending=True)
    )


def iqr_outliers(df: pl.DataFrame, columns: Sequence[str], k: float = 1.5) -> pl.DataFrame:
    """Tukey fences (Q1 - k IQR, Q3 + k IQR) and number / share of rows outside them, per column."""
    rows: list[dict] = []
    for column in columns:
        values: pl.Series = df[column].drop_nulls()
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        low, high = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
        rows.append({'column': column, 'low': low, 'high': high, 'n_outliers': int(((values < low) | (values > high)).sum())})
    return pl.DataFrame(rows).with_columns((pl.col('n_outliers') / df.height * 100).round(2).alias('outlier_pct'))


def dataset_scope(df_player_matches: pl.DataFrame) -> dict:
    """Headline scope numbers: rows, duplicated (player, match) pairs, players, matches, period."""
    first_match_at, last_match_at = df_player_matches['match_datetime'].min(), df_player_matches['match_datetime'].max()
    return {
        'rows': df_player_matches.height,
        'duplicated (player, match)': df_player_matches.height - df_player_matches.select('playerid', 'matchid').unique().height,
        'distinct players': df_player_matches['playerid'].n_unique(),
        'distinct matches': df_player_matches['matchid'].n_unique(),
        'first match': first_match_at,
        'last match': last_match_at,
        'span (days)': (last_match_at - first_match_at).days,
        'unparseable dates': df_player_matches['match_datetime'].null_count(),
    }


def print_dict(values: dict) -> None:
    """Print a dict as an aligned `key : value` block."""
    width: int = max(len(str(k)) for k in values)
    for key, value in values.items():
        print(f'{str(key):<{width}} : {value}')


# --------------------------------------------------------------------------- #
# Scope (notebook 01)
# --------------------------------------------------------------------------- #




def count_ids(column: str, separator: str = ',') -> pl.Expr:
    """Number of ids in a delimited string column; null or empty strings count as 0."""
    ids: pl.Expr = pl.col(column).fill_null('')
    return pl.when(ids == '').then(0).otherwise(ids.str.count_matches(separator) + 1)






def matches_per_gamemode(df_player_matches: pl.DataFrame) -> pl.DataFrame:
    """Distinct matches and rows per game mode, with the share of matches, largest first."""
    return (
        df_player_matches.group_by('gamemode')
        .agg(pl.col('matchid').n_unique().alias('match_count'), pl.len().alias('row_count'))
        .with_columns((pl.col('match_count') / pl.col('match_count').sum() * 100).round(2).alias('share_pct'))
        .sort('match_count', descending=True)
    )


def victory_per_gamemode(df_player_matches: pl.DataFrame) -> pl.DataFrame:
    """Rows per (game mode, victory value) with the share inside the game mode."""
    return (
        df_player_matches.group_by('gamemode', 'victory')
        .agg(pl.len().alias('row_count'))
        .with_columns((pl.col('row_count') / pl.col('row_count').sum().over('gamemode') * 100).round(2).alias('share_in_gamemode_pct'))
    )


def build_games_per_day(df_player_matches: pl.DataFrame) -> pl.DataFrame:
    """Distinct matches per calendar day."""
    return (
        df_player_matches.group_by(pl.col('match_datetime').dt.date().alias('match_date'))
        .agg(pl.col('matchid').n_unique().alias('game_count'))
        .sort('match_date')
    )












# --------------------------------------------------------------------------- #
# Data quality (notebook 02)
# --------------------------------------------------------------------------- #


def add_lobby_columns(df_player_matches: pl.DataFrame) -> pl.DataFrame:
    """Teammate / opponent counts, self-reference flags and gaps vs `playernb_in_match`.

    Adds `n_teammate`, `n_opponent`, `self_in_teammate`, `self_in_opponent`, `n_listed`,
    `gap_without_self`, `gap_with_self` (0 = lobby size is consistent) and `team_size_gap`
    (player's team size - opponents).
    """
    return df_player_matches.with_columns(
        count_ids('teammateid').cast(pl.Int64).alias('n_teammate'),
        count_ids('opponentid').cast(pl.Int64).alias('n_opponent'),
        pl.col('teammateid').fill_null('').str.contains(pl.col('playerid'), literal=True).alias('self_in_teammate'),
        pl.col('opponentid').fill_null('').str.contains(pl.col('playerid'), literal=True).alias('self_in_opponent'),
    ).with_columns(
        (pl.col('n_teammate') + pl.col('n_opponent')).alias('n_listed'),
    ).with_columns(
        (pl.col('n_listed') - pl.col('playernb_in_match')).alias('gap_without_self'),
        (pl.col('n_listed') + 1 - pl.col('playernb_in_match')).alias('gap_with_self'),
        (pl.col('n_teammate') + 1 - pl.col('n_opponent')).alias('team_size_gap'),
    )


def team_balance_by_mode(df_lobby: pl.DataFrame) -> pl.DataFrame:
    """Per game mode: share of balanced teams, modal team sizes and range of the team size gap."""
    return (
        df_lobby.group_by('gamemode')
        .agg(
            pl.len().alias('rows'),
            (pl.col('team_size_gap') == 0).mean().round(3).alias('share_balanced'),
            (pl.col('n_teammate') + 1).mode().first().alias('team_size_mode'),
            pl.col('n_opponent').mode().first().alias('opponent_size_mode'),
            pl.col('team_size_gap').min().alias('gap_min'),
            pl.col('team_size_gap').max().alias('gap_max'),
        )
        .sort('rows', descending=True)
    )


# --------------------------------------------------------------------------- #
# Player stats and streaks (notebook 03)
# --------------------------------------------------------------------------- #


def activity_label(column: str = 'still_active') -> pl.Expr:
    """'Active' / 'Inactive' label from a Boolean column, named `activity`."""
    return pl.when(pl.col(column)).then(pl.lit('Active')).otherwise(pl.lit('Inactive')).alias('activity')


def win_kd_stats(df_player_matches: pl.DataFrame, by: Sequence[str] = ('playerid',)) -> pl.DataFrame:
    """Wins, losses, win rate and K/D per group.

    `decided` = wins + losses (draw / other excluded); `kd_ratio` = total kills / total deaths,
    null without any death; `matches_with_kd` = matches with non-null kills.
    """
    return (
        df_player_matches.group_by(*by)
        .agg(
            pl.len().alias('matches'),
            (pl.col('victory') == 'TRUE').sum().alias('wins'),
            (pl.col('victory') == 'FALSE').sum().alias('losses'),
            pl.col('kills').sum().alias('kills_sum'),
            pl.col('death').sum().alias('death_sum'),
            pl.col('kills').count().alias('matches_with_kd'),
        )
        .with_columns((pl.col('wins') + pl.col('losses')).alias('decided'))
        .with_columns(
            (pl.col('wins') / pl.col('decided')).alias('win_rate'),
            pl.when(pl.col('death_sum') > 0).then(pl.col('kills_sum') / pl.col('death_sum')).alias('kd_ratio'),
        )
    )


def max_streaks(df_player_matches: pl.DataFrame, df_eligible: pl.DataFrame) -> pl.DataFrame:
    """Longest win and loss streak of each eligible player (0 = never won / never lost).

    Matches are ordered by `match_datetime` then `matchid`; draw / other break a streak.

    Args:
        df_player_matches: Typed player-match table.
        df_eligible: Players to keep, with columns `playerid` and `matches`.
    """
    df_runs: pl.DataFrame = build_runs(
        df_player_matches.join(df_eligible.select('playerid'), on='playerid').sort('playerid', 'match_datetime', 'matchid')
    )
    return (
        df_eligible.join(
            df_runs.filter(pl.col('victory') == 'TRUE').group_by('playerid').agg(pl.col('run_length').max().alias('max_win_streak')),
            on='playerid', how='left',
        )
        .join(
            df_runs.filter(pl.col('victory') == 'FALSE').group_by('playerid').agg(pl.col('run_length').max().alias('max_loss_streak')),
            on='playerid', how='left',
        )
        .with_columns(pl.col('max_win_streak', 'max_loss_streak').fill_null(0))
        .sort('matches', descending=True)
    )


def python_streaks(results: Sequence[str]) -> tuple[int, int]:
    """Reference implementation in plain python: (longest TRUE run, longest FALSE run)."""
    best: dict[str, int] = {'TRUE': 0, 'FALSE': 0}
    current, previous = 0, None
    for result in results:
        current = current + 1 if result == previous else 1
        previous = result
        if result in best:
            best[result] = max(best[result], current)
    return best['TRUE'], best['FALSE']


# --------------------------------------------------------------------------- #
# Group comparison (notebooks 04-05)
# --------------------------------------------------------------------------- #


def numeric_columns(df: pl.DataFrame, exclude: Sequence[str] = ()) -> list[str]:
    """Numeric and Boolean columns of `df`, minus `exclude`."""
    return [c for c, t in df.schema.items() if (t.is_numeric() or t == pl.Boolean) and c not in exclude]


def compare_groups(df: pl.DataFrame, features: Sequence[str], target: str = 'still_active', min_size: int = 5) -> pl.DataFrame:
    """Compare each feature between target == True and target == False.

    `auc` = P(feature of a random positive > feature of a random negative) from the
    Mann-Whitney U statistic (0.5 = no difference); `effect` = |auc - 0.5|. Features with
    fewer than `min_size` non-null values in a group are skipped.
    """
    rows: list[dict] = []
    for feature in features:
        positive = df.filter(pl.col(target))[feature].drop_nulls().to_numpy().astype(float)
        negative = df.filter(~pl.col(target))[feature].drop_nulls().to_numpy().astype(float)
        if len(positive) < min_size or len(negative) < min_size:
            continue
        u, p = mannwhitneyu(positive, negative)
        rows.append({
            'feature': feature, 'n_active': len(positive), 'n_inactive': len(negative),
            'median_active': float(np.median(positive)), 'median_inactive': float(np.median(negative)),
            'auc': float(u / (len(positive) * len(negative))), 'p_value': float(p),
        })
    return pl.DataFrame(rows).with_columns((pl.col('auc') - 0.5).abs().alias('effect')).sort('effect', descending=True)


def spearman_with_target(df: pl.DataFrame, features: Sequence[str], target: str) -> pd.Series:
    """Spearman correlation of every feature with the target, sorted ascending (NaN dropped)."""
    pdf: pd.DataFrame = df.select(target, *features).to_pandas().astype(float)
    return pdf.drop(columns=target).corrwith(pdf[target], method='spearman').dropna().sort_values()


def redundant_pairs(corr_matrix: pd.DataFrame, threshold: float = 0.9) -> pd.Series:
    """Feature pairs with |correlation| >= threshold, strongest first."""
    pairs: pd.Series = corr_matrix.where(np.tril(np.ones(corr_matrix.shape, dtype=bool), k=-1)).stack()
    return pairs[pairs.abs() >= threshold].sort_values(key=abs, ascending=False)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


def set_plot_style() -> None:
    """Seaborn whitegrid theme used by every notebook."""
    sns.set_theme(style='whitegrid')


def split_violin(
    ax: plt.Axes,
    data: pd.DataFrame,
    y: str,
    x: str | None = None,
    hue: str = 'activity',
    order: Sequence[str] | None = None,
    hue_order: Sequence[str] | None = None,
    log_scale: bool = False,
) -> plt.Axes:
    """Split violin with quartile lines (two hue levels side by side), the style used across the notebooks."""
    sns.violinplot(
        data=data, x=x, y=y, hue=hue, order=order, hue_order=hue_order if hue_order is not None else ACTIVITY_ORDER,
        split=True, inner='quartile', cut=0, density_norm='width', gap=0.05, log_scale=log_scale, ax=ax,
    )
    return ax


def split_violins_grid(
    df: pl.DataFrame,
    features: Sequence[str],
    log: Sequence[str] = (),
    ncols: int = 3,
    reference: dict[str, float] | None = None,
    hue: str = 'activity',
    hue_order: Sequence[str] = ACTIVITY_ORDER,
) -> plt.Figure:
    """One split violin per feature (e.g. Active left vs Inactive right) in a grid.

    Features in `log` are drawn on a log scale (non-positive values dropped); `reference`
    draws a red dashed line at the given value.
    """
    nrows: int = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows))
    axes = np.ravel(axes)
    pdf: pd.DataFrame = df.to_pandas()
    pdf['all'] = ''
    for i, (ax, feature) in enumerate(zip(axes, features)):
        data: pd.DataFrame = pdf.dropna(subset=[feature])
        if feature in log:
            data = data[data[feature] > 0]
        split_violin(ax, data, y=feature, x='all', hue=hue, hue_order=hue_order, log_scale=feature in log)
        if reference and feature in reference:
            ax.axhline(reference[feature], color='red', linestyle='--')
        ax.set_xlabel('')
        ax.set_title(feature + (' (log)' if feature in log else ''))
        if i == 0:
            ax.legend(title='')
        elif ax.get_legend() is not None:
            ax.get_legend().remove()
    for ax in axes[len(features):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def violin_pair(
    frames: Sequence[tuple[pl.DataFrame, str]],
    y: str,
    reference: float | None = None,
    suptitle: str = '',
) -> plt.Figure:
    """Side-by-side single violins of `y` for several DataFrames (e.g. all players vs 10+ matches)."""
    fig, axes = plt.subplots(1, len(frames), figsize=(5 * len(frames), 5), sharey=True)
    for ax, (df_plot, title) in zip(np.ravel(axes), frames):
        sns.violinplot(data=df_plot.to_pandas(), y=y, inner='quartile', cut=0, color='steelblue', ax=ax)
        if reference is not None:
            ax.axhline(reference, color='red', linestyle='--')
        ax.set_title(title)
    fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def plot_target_correlation(corr: pd.Series, highlight: set[str], title: str) -> plt.Figure:
    """Horizontal bar chart of correlations; features in `highlight` are drawn in red."""
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(corr) + 1))
    ax.barh(corr.index, corr.values, color=['tab:red' if c in highlight else 'steelblue' for c in corr.index])
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(title)
    return fig


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, block_sizes: Sequence[int], title: str, offset: int = 1) -> plt.Figure:
    """Lower-triangle Spearman heatmap with black lines between feature families.

    Args:
        corr_matrix: Square correlation matrix.
        block_sizes: Number of features per family, in matrix order (after the first `offset` rows).
        title: Figure title.
        offset: Leading rows that are not part of a family (e.g. the target).
    """
    fig, ax = plt.subplots(figsize=(20, 17))
    sns.heatmap(
        corr_matrix, mask=np.triu(np.ones_like(corr_matrix, dtype=bool), k=1), cmap='RdBu_r', center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.2, cbar_kws={'shrink': 0.6, 'label': 'Spearman'}, ax=ax,
    )
    position: int = offset
    for size in block_sizes:
        position += size
        ax.axhline(position, color='black', linewidth=1)
        ax.axvline(position, color='black', linewidth=1)
    ax.set_title(title)
    return fig
