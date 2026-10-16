# CLAUDE.md

Guide for answering questions about this repository and working on it.

## What the project is

Churn analysis of a team-based PvP shooter. Input: a player-match log (`data/01_raw/playersmatches_test.csv`,
95,733 rows, one row = one player in one match, 840 players, 2025-03-04 to 2026-03-03). Target: `still_active`
(TRUE/FALSE, constant per player, 41.9% TRUE). Output: a random forest that predicts P(still_active) from
12 player-level features; churn risk = 1 - P(still_active), "at risk" when churn risk >= 0.5.

The project is Kedro-style **without the Kedro dependency**: config in `conf/`, data by layer in `data/`,
code in `src/ubi_churn/`, analysis in `notebooks/`.

## Where to look

| Question about... | Look in |
|---|---|
| a file path / dataset name | `conf/base/catalog.yml` (never hard-code paths; use `DataCatalog`) |
| a constant (session gap, Elo K, threshold, model features) | `conf/base/parameters.yml` |
| loading / saving data, project root | `src/ubi_churn/config.py` (`DataCatalog`, `load_params`) |
| how the data is rebuilt end to end | `src/ubi_churn/pipeline.py` (nodes `prepare` -> `features` -> `train`) |
| how a player feature is computed | `src/ubi_churn/utils/features.py` (`build_*` functions, `build_player_features`) |
| EDA tables, data-quality checks, plots | `src/ubi_churn/utils/exploration.py` |
| models, evaluation, SHAP, final model, predictor | `src/ubi_churn/utils/ml.py` (`evaluate_models`, `train_churn_model`, `ChurnPredictor`) |
| findings and numbers | the markdown "Observation :" cells of `notebooks/01`-`08` |
| the review report | `reports/churn_analysis.qmd` (source), `data/08_reporting/churn_analysis.html` (built) |

Notebook map: 01 scope, 02 data quality, 03 win rate / K/D / streaks, 04 player features (writes
`player_features`), 05 active vs inactive, 06 model selection, 07 final model (writes `churn_model`),
08 predictor form (appends to `predictions_log`).

When answering, cite code as `path:line` and quote numbers from notebook outputs or recompute them;
do not guess figures.

## Commands

Run from the project root (Windows, Python 3.11):

```bash
python -m ubi_churn.pipeline                         # needs src/ on the path: run from src/, or pip install -e .
python -m ubi_churn.pipeline --nodes features train  # a subset
python tests/test_utils.py                           # 8 sanity tests (pytest is not installed)
python reports/build_report.py --quarto <path>       # rebuild the HTML report (needs Quarto + plotly)
```

Notebooks run from `notebooks/` (their first cell adds `../src` to `sys.path`). Notebook 06 takes about
8 minutes; the others under a minute.

## Key facts and caveats (do not lose these when answering)

- **Recency drives the model.** `days_since_last_match` alone gives about 0.78 ROC-AUC; the full model about
  0.82 (CV). Skill (win rate, K/D, Elo) barely differs between active and inactive players.
- **Recency is measured from the end of the log** (2026-03-03), not from today. Inputs to the predictor must
  be computed the same way.
- **The label is an assumption**: `still_active` is treated as a snapshot taken after the log ends.
  91 players flagged active have not played for 90+ days (label noise); without them holdout ROC-AUC is about 0.90.
- Recency features (`days_since_last_match`, `matches_last_30d`, `matches_last_90d`) leak the target for
  descriptive analysis (notebooks 04-05) but are used on purpose by the model (notebook 06 explains why).
- Data quirks: `victory` holds TRUE / FALSE / draw / other (draw and other are neither win nor loss and break a
  streak); 153 null kills/death, 152 of them on 2025-12-09 (logging incident); deaths are capped at 9;
  PVP lobbies are not fixed 5v5; OPERATIONS has 5 rows.
- A session = matches less than 30 minutes apart.
- `player_features` is reproducible bit for bit by the pipeline; the per-mode pivot columns are explicitly
  ordered because a random forest is sensitive to column order.

## Conventions when changing code

- Polars for data work, pandas only at the sklearn / plotting boundary.
- Utils are typed, documented, side-effect free (DataFrame in, DataFrame out); constants go in
  `parameters.yml`, paths in `catalog.yml`.
- Notebook markdown is short and plain ("Observation :" bullets, simple sentences), like notebook 01.
- After changing feature or model code: run the pipeline, then `python tests/test_utils.py`.
- Commits and PRs: no AI attribution lines (no `Co-Authored-By: Claude`, no "Generated with" footer).
