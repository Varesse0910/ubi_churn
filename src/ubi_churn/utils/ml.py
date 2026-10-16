"""Machine learning helpers (notebooks 06-08): model zoo, evaluation, explanation, final model and predictor.

The model predicts P(still_active); `churn risk = 1 - P(still_active)`.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import QuantileTransformer

if TYPE_CHECKING:
    import shap

AT_RISK_LABEL: str = 'At risk (likely churn)'
ACTIVE_LABEL: str = 'Likely still active'

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


def quantile_normal() -> QuantileTransformer:
    """Rank-based scaling to a normal distribution (robust to the heavy tails of volume features)."""
    return QuantileTransformer(n_quantiles=100, output_distribution='normal')


def make_random_forest(seed: int, n_estimators: int = 500, min_samples_leaf: int = 5) -> RandomForestClassifier:
    """Random forest used everywhere in the project."""
    return RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, n_jobs=-1, random_state=seed)


def make_models(single_feature: str, seed: int, rf_params: dict | None = None) -> dict[str, tuple[Any, list[str] | None]]:
    """Model zoo of notebook 06, simplest to most flexible.

    Returns:
        name -> (estimator, columns to restrict to or None for all features).
    """
    rf_params = rf_params or {}
    return {
        'dummy (prior)': (make_pipeline(DummyClassifier(strategy='prior')), None),
        f'1 feature ({single_feature})': (make_pipeline(SimpleImputer(strategy='median'), quantile_normal(), LogisticRegression(max_iter=1000)), [single_feature]),
        'logistic regression': (make_pipeline(SimpleImputer(strategy='median'), quantile_normal(), LogisticRegression(C=0.1, max_iter=2000)), None),
        'random forest': (make_pipeline(SimpleImputer(strategy='median'), make_random_forest(seed, **rf_params)), None),
        'gradient boosting': (HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=20, l2_regularization=1.0, random_state=seed), None),
    }


def make_candidate_models(features: list[str], single_feature: str, seed: int, rf_params: dict | None = None) -> dict[str, tuple[Pipeline, list[str]]]:
    """Candidates of notebook 07 (final model choice): name -> (pipeline, columns)."""
    rf_params = rf_params or {}
    return {
        'dummy (prior)': (make_pipeline(DummyClassifier(strategy='prior')), features),
        f'1 feature ({single_feature})': (make_pipeline(SimpleImputer(strategy='median'), quantile_normal(), LogisticRegression()), [single_feature]),
        'logistic regression': (make_pipeline(SimpleImputer(strategy='median'), quantile_normal(), LogisticRegression(C=0.1, max_iter=2000)), features),
        'random forest': (make_pipeline(SimpleImputer(strategy='median'), make_random_forest(seed, **rf_params)), features),
    }


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def pick_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold on `score` maximising F1 for the positive class of `y`."""
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-9, None)
    return float(t[int(np.argmax(f1))])


def holdout_split(y: np.ndarray, test_size: float, seed: int, split: str = 'stratified', first_match: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Train / test row indices: stratified random split, or `cohort` (latest joiners = test, needs `first_match`)."""
    if split == 'stratified':
        return train_test_split(np.arange(len(y)), test_size=test_size, stratify=y, random_state=seed)
    if split == 'cohort':
        order = np.argsort(first_match)
        cut = int(len(y) * (1 - test_size))
        return order[:cut], order[cut:]
    raise ValueError(f'unknown split {split!r}')


def evaluate_models(
    df: pl.DataFrame,
    features: list[str],
    single_feature: str,
    target: str,
    seed: int,
    test_size: float = 0.2,
    n_folds: int = 5,
    split: str = 'stratified',
    rf_params: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Cross-validate every model of `make_models` on the training part, then score the holdout.

    Metrics: CV ROC-AUC / PR-AUC (mean +/- std), holdout ROC-AUC / PR-AUC, and holdout precision /
    recall at the F1-optimal threshold chosen on out-of-fold predictions.

    Returns:
        One row per model, and `fitted`: name -> {'model', 'proba', 'cols'} plus `_data` holding the
        train / holdout matrices and holdout player ids.
    """
    df = df.sort('playerid')
    X, y = df.select(features).to_pandas(), df[target].to_numpy().astype(int)
    first_match = df['first_match'].to_numpy() if 'first_match' in df.columns else None
    idx_tr, idx_te = holdout_split(y, test_size, seed, split, first_match)
    cv = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    rows, fitted = [], {}
    for name, (model, cols) in make_models(single_feature, seed, rf_params).items():
        Xtr, Xte = X.iloc[idx_tr][cols or features], X.iloc[idx_te][cols or features]
        res = cross_validate(model, Xtr, y[idx_tr], cv=cv, scoring=['roc_auc', 'average_precision'])
        oof = cross_val_predict(model, Xtr, y[idx_tr], cv=cv, method='predict_proba')[:, 1]
        thr = pick_threshold(y[idx_tr], oof)
        model.fit(Xtr, y[idx_tr])
        proba = model.predict_proba(Xte)[:, 1]
        pred = proba >= thr
        tp = int((pred & (y[idx_te] == 1)).sum())
        rows.append({
            'model': name,
            'cv_roc_auc': f"{res['test_roc_auc'].mean():.3f} +/- {res['test_roc_auc'].std():.3f}",
            'cv_pr_auc': f"{res['test_average_precision'].mean():.3f} +/- {res['test_average_precision'].std():.3f}",
            'test_roc_auc': round(roc_auc_score(y[idx_te], proba), 3),
            'test_pr_auc': round(average_precision_score(y[idx_te], proba), 3),
            'threshold': round(thr, 2),
            'precision': round(tp / max(pred.sum(), 1), 3),
            'recall': round(tp / max((y[idx_te] == 1).sum(), 1), 3),
        })
        fitted[name] = {'model': model, 'proba': proba, 'cols': cols or features}
    fitted['_data'] = {'X_te': X.iloc[idx_te], 'y_te': y[idx_te], 'ids_te': df['playerid'].to_numpy()[idx_te], 'X_tr': X.iloc[idx_tr], 'y_tr': y[idx_tr]}
    return pd.DataFrame(rows), fitted


def _holdout_scores(fitted: dict, model_name: str) -> pl.DataFrame:
    d = fitted['_data']
    return pl.DataFrame({'playerid': d['ids_te'], 'y': d['y_te'], 'proba': fitted[model_name]['proba']})


def _auc_row(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        'roc_auc': round(roc_auc_score(y, p), 3) if 0 < y.sum() < len(y) else np.nan,
        'pr_auc': round(average_precision_score(y, p), 3) if y.sum() else np.nan,
    }


def segment_report(fitted: dict, model_name: str, label: str, df_players: pl.DataFrame, min_matches: int) -> pd.DataFrame:
    """Holdout ROC-AUC / PR-AUC for players with fewer than / at least `min_matches` lifetime matches."""
    m = _holdout_scores(fitted, model_name).join(df_players.select('playerid', 'matches'), on='playerid')
    rows = []
    for seg, sub in [(f'< {min_matches} matches', m.filter(pl.col('matches') < min_matches)), (f'{min_matches}+ matches', m.filter(pl.col('matches') >= min_matches))]:
        y, p = sub['y'].to_numpy(), sub['proba'].to_numpy()
        rows.append({'set': label, 'segment': seg, 'players': len(y), 'active share': round(y.mean(), 3), **_auc_row(y, p)})
    return pd.DataFrame(rows)


def silent_flag(target: str, silent_days: int) -> pl.Expr:
    """Flagged active but no match for `silent_days`+ days: suspected label noise. Named `silent`."""
    return (pl.col(target) & (pl.col('days_since_last_match') >= silent_days)).alias('silent')


def noise_check(fitted: dict, model_name: str, label: str, df_silent: pl.DataFrame) -> pd.DataFrame:
    """Holdout ROC-AUC / PR-AUC with and without the silent 'active' players.

    Args:
        df_silent: `playerid` and Boolean `silent` (see `silent_flag`).
    """
    m = _holdout_scores(fitted, model_name).join(df_silent.select('playerid', 'silent'), on='playerid')
    rows = []
    for name, sub in [('all holdout players', m), ('without silent "active"', m.filter(~pl.col('silent')))]:
        y, p = sub['y'].to_numpy(), sub['proba'].to_numpy()
        rows.append({'set': label, 'players': name, 'n': len(y), 'active share': round(y.mean(), 3), **_auc_row(y, p)})
    return pd.DataFrame(rows)


def permutation_importance_table(fitted: dict, model_name: str, seed: int, groups: dict[str, str] | None = None, n_repeats: int = 20) -> pd.DataFrame:
    """Drop of holdout ROC-AUC when each feature is shuffled; `group` from `groups` (default 'behaviour')."""
    d, f = fitted['_data'], fitted[model_name]
    pi = permutation_importance(f['model'], d['X_te'][f['cols']], d['y_te'], scoring='roc_auc', n_repeats=n_repeats, random_state=seed)
    df_imp = pd.DataFrame({'feature': f['cols'], 'auc_drop': pi.importances_mean, 'std': pi.importances_std}).sort_values('auc_drop', ascending=False)
    df_imp['group'] = df_imp['feature'].map(groups or {}).fillna('behaviour')
    return df_imp


def evaluate_candidates(
    models: dict[str, tuple[Pipeline, list[str]]],
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    test_size: float = 0.2,
    cv_splits: int = 5,
    cv_repeats: int = 3,
) -> pd.DataFrame:
    """Repeated stratified CV on the training part + stratified holdout, one row per candidate (notebook 07)."""
    idx_tr, idx_te = holdout_split(y, test_size, seed)
    cv = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=seed)
    rows = []
    for name, (model, cols) in models.items():
        res = cross_validate(model, X.iloc[idx_tr][cols], y[idx_tr], cv=cv, scoring=['roc_auc', 'average_precision'])
        model.fit(X.iloc[idx_tr][cols], y[idx_tr])
        p = model.predict_proba(X.iloc[idx_te][cols])[:, 1]
        rows.append({
            'model': name,
            'cv_roc_auc': f"{res['test_roc_auc'].mean():.3f} +/- {res['test_roc_auc'].std():.3f}",
            'cv_pr_auc': f"{res['test_average_precision'].mean():.3f} +/- {res['test_average_precision'].std():.3f}",
            'holdout_roc_auc': round(roc_auc_score(y[idx_te], p), 3),
            'holdout_pr_auc': round(average_precision_score(y[idx_te], p), 3),
        })
    return pd.DataFrame(rows)


def out_of_fold_proba(model: Pipeline, X: pd.DataFrame, y: np.ndarray, seed: int, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold P(positive) from a stratified K-fold."""
    return cross_val_predict(model, X, y, cv=StratifiedKFold(n_splits, shuffle=True, random_state=seed), method='predict_proba')[:, 1]


def threshold_table(churned: np.ndarray, risk: np.ndarray, thresholds: list[float]) -> pd.DataFrame:
    """Players flagged, precision and recall of the churn class at each churn-risk threshold."""
    rows = []
    for t in thresholds:
        pred = risk >= t
        hits = int((pred & (churned == 1)).sum())
        rows.append({'threshold': round(t, 2), 'players flagged': int(pred.sum()), 'precision': round(hits / max(pred.sum(), 1), 3), 'recall': round(hits / churned.sum(), 3)})
    return pd.DataFrame(rows).drop_duplicates('threshold').sort_values('threshold')


# --------------------------------------------------------------------------- #
# Final model (notebook 07)
# --------------------------------------------------------------------------- #


def fit_final_model(X: pd.DataFrame, y: np.ndarray, seed: int, rf_params: dict | None = None) -> Pipeline:
    """Median imputation + random forest, fitted on all players."""
    return make_pipeline(SimpleImputer(strategy='median'), make_random_forest(seed, **(rf_params or {}))).fit(X, y)


def build_model_artifact(
    model: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_info: dict[str, list[str] | tuple[str, str]],
    threshold: float,
    metrics: dict,
    note: str,
) -> dict:
    """Everything the predictor needs: model, feature list / labels, defaults, ranges, threshold, metrics."""
    features: list[str] = list(feature_info)
    return {
        'model': model,
        'features': features,
        'feature_info': {k: tuple(v) for k, v in feature_info.items()},
        'defaults': X.median().to_dict(),
        'ranges': {c: (float(X[c].min()), float(X[c].max())) for c in features},
        'threshold_churn_risk': threshold,
        'metrics': metrics,
        'trained_on': {'players': len(y), 'active_share': float(y.mean()), 'date': date.today().isoformat()},
        'note': note,
    }


def train_churn_model(df_players: pl.DataFrame, params: dict) -> tuple[dict, pd.DataFrame]:
    """Evaluate the candidates, fit the random forest on all players and build the artifact (pipeline node)."""
    p: dict = params['churn_model']
    features: list[str] = list(p['feature_info'])
    X: pd.DataFrame = df_players.select(features).to_pandas()
    y: np.ndarray = df_players[params['target']].to_numpy().astype(int)
    df_eval: pd.DataFrame = evaluate_candidates(
        make_candidate_models(features, params['model_selection']['single_feature'], p['seed'], p['random_forest']),
        X, y, p['seed'], p['test_size'], p['cv_splits'], p['cv_repeats'],
    )
    model: Pipeline = fit_final_model(X, y, p['seed'], p['random_forest'])
    artifact: dict = build_model_artifact(
        model, X, y, p['feature_info'], p['threshold_churn_risk'],
        df_eval.set_index('model').loc['random forest'].to_dict(), p['note'],
    )
    return artifact, df_eval


# --------------------------------------------------------------------------- #
# Explanation (SHAP)
# --------------------------------------------------------------------------- #


def shap_churn(model: Pipeline, X: pd.DataFrame, feature_names: list[str] | None = None) -> shap.Explanation:
    """SHAP values on the churn risk (class still_active = False) for an imputer + tree pipeline.

    Missing values are imputed as in the model, so the explanation matches the prediction.
    """
    import shap  # heavy import, only when explaining

    imputer, forest = model.steps[0][1], model.steps[-1][1]
    X_imp = pd.DataFrame(imputer.transform(X), columns=list(X.columns), index=X.index)
    sv = shap.TreeExplainer(forest)(X_imp)
    churn_idx: int = list(forest.classes_).index(0)  # still_active = False
    return shap.Explanation(
        values=sv.values[:, :, churn_idx], base_values=sv.base_values[:, churn_idx],
        data=X_imp.values, feature_names=feature_names or list(X.columns),
    )


def plot_shap_summary(explanation: shap.Explanation, title: str, max_display: int = 12, bar: bool = True) -> None:
    """Beeswarm (and optionally bar) plot of a SHAP explanation on the churn risk."""
    import shap

    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    plt.title(title, loc='left')
    plt.tight_layout()
    plt.show()
    if bar:
        shap.plots.bar(explanation, max_display=max_display, show=False)
        plt.title('Mean absolute impact on churn risk', loc='left')
        plt.tight_layout()
        plt.show()


# --------------------------------------------------------------------------- #
# Predictor (notebook 08)
# --------------------------------------------------------------------------- #


class ChurnPredictor:
    """Wraps the saved model artifact: predict, check inputs and explain."""

    def __init__(self, artifact: dict) -> None:
        self.artifact: dict = artifact
        self.model: Pipeline = artifact['model']
        self.features: list[str] = artifact['features']
        self.feature_info: dict = artifact['feature_info']
        self.defaults: dict = artifact['defaults']
        self.ranges: dict = artifact['ranges']
        self.threshold: float = artifact['threshold_churn_risk']
        self.labels: list[str] = [self.feature_info[f][0] for f in self.features]

    def summary(self) -> str:
        """One-line description of the model."""
        t = self.artifact['trained_on']
        return f"model trained on {t['players']} players ({t['date']}) | churn-risk threshold {self.threshold:.2f}"

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add `p_active`, `churn_risk` and `prediction` to a table with (some of) the model features.

        Missing columns / values are allowed: the model imputes the training median.
        """
        X = df.reindex(columns=self.features).astype(float)
        p_active = self.model.predict_proba(X)[:, 1]
        out = df.copy()
        out['p_active'] = p_active.round(3)
        out['churn_risk'] = (1 - p_active).round(3)
        out['prediction'] = np.where(out['churn_risk'] >= self.threshold, AT_RISK_LABEL, ACTIVE_LABEL)
        return out

    def player_frame(self, player_id: str, values: dict) -> pd.DataFrame:
        """One-row table (`player_id` + every model feature, NaN when not given)."""
        return pd.DataFrame([{'player_id': player_id, **{f: values.get(f, np.nan) for f in self.features}}])

    def predict_one(self, player_id: str, values: dict) -> pd.Series:
        """Predict a single player from a dict of feature values."""
        return self.predict(self.player_frame(player_id, values)).iloc[0]

    def check_inputs(self, values: dict) -> list[str]:
        """Warnings for values outside the training range (e.g. a negative count)."""
        warnings: list[str] = []
        for f in self.features:
            value = values.get(f)
            if value is None or pd.isna(value):
                continue
            lo, hi = self.ranges[f]
            if value < lo or value > hi:
                warnings.append(f'{self.feature_info[f][0]} = {value:g} is outside the training range {lo:.3g} - {hi:.3g}')
        return warnings

    def explain(self, df: pd.DataFrame) -> shap.Explanation:
        """SHAP explanation of the churn risk for each row of `df`."""
        return shap_churn(self.model, df.reindex(columns=self.features).astype(float), self.labels)

    def plot_waterfall(self, df_row: pd.DataFrame, title: str) -> None:
        """Waterfall for one player: red bars push the churn risk up, blue bars push it down."""
        import shap

        shap.plots.waterfall(self.explain(df_row)[0], max_display=len(self.features), show=False)
        plt.title(title, loc='left')
        plt.tight_layout()
        plt.show()


def append_predictions(df: pd.DataFrame, path: Path) -> None:
    """Append predictions to the CSV log with a timestamp (header written on first use)."""
    df = df.copy()
    df.insert(0, 'timestamp', datetime.now().isoformat(timespec='seconds'))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode='a', header=not path.exists(), index=False)


def load_predictions(path: Path) -> pd.DataFrame:
    """The prediction log, or an empty DataFrame if nothing was stored yet."""
    return pd.read_csv(path) if path.exists() else pd.DataFrame()
