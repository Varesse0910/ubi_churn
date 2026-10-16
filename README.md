# Ubi player churn

Exploration of a player-match log and a churn model (`still_active`), organised Kedro-style:
configuration in `conf/`, data by layer in `data/`, reusable code in `src/`, analysis in `notebooks/`.

```
ubi_churn/
├── conf/
│   ├── base/
│   │   ├── catalog.yml       # every dataset (name -> file, type, load args)
│   │   └── parameters.yml    # every constant (sessions, Elo, model, threshold, features...)
│   └── local/                # personal overrides (same file names), not shared
├── data/
│   ├── 01_raw/               # playersmatches_test.csv (source, never modified)
│   ├── 02_intermediate/      # player_matches.parquet (typed match table)
│   ├── 04_feature/           # player_features.parquet (one row per player, 72 columns)
│   ├── 06_models/            # churn_model.joblib (model + features + threshold + metrics)
│   ├── 07_model_output/      # predictions_log.csv, *_scored.csv
│   └── 08_reporting/
├── notebooks/                # 01 -> 08, the analysis (read the markdown observations)
├── src/ubi_churn/
│   ├── config.py             # PROJECT_ROOT, load_params(), DataCatalog
│   ├── pipeline.py           # nodes: prepare -> features -> train
│   └── utils/
│       ├── features.py       # data preparation + feature engineering
│       ├── exploration.py    # EDA tables, data-quality checks, group comparison, plots
│       └── ml.py             # model zoo, evaluation, SHAP, final model, ChurnPredictor
└── tests/test_utils.py       # sanity tests (plain python recomputation, reproducibility, predictor)
```

## Quick start

```bash
pip install -r requirements.txt      # or: pip install -e .   (then `ubi-churn` runs the pipeline)
python -m ubi_churn.pipeline         # run from src/ (or anywhere after pip install -e .)
python tests/test_utils.py           # or: pytest tests
```

The pipeline rebuilds everything downstream of the raw CSV:

| node       | input                | output             |
|------------|----------------------|--------------------|
| `prepare`  | `player_matches_raw` | `player_matches`   |
| `features` | `player_matches`     | `player_features`  |
| `train`    | `player_features`    | `churn_model`      |

Run a subset with `python -m ubi_churn.pipeline --nodes features train`.

## Notebooks

| notebook | question | reads | writes |
|---|---|---|---|
| 01 Scope and dataset | what is in the data, what is the scope | `player_matches` | |
| 02 Data quality and outliers | nulls, duplicates, lobby consistency, outliers | `player_matches` | |
| 03 Win rate, K/D and streaks | per-player performance and streaks | `player_matches` | |
| 04 Player features | build one row per player (same code as the `features` node) | `player_matches` | `player_features` |
| 05 Active vs inactive | which features differ between the two groups | `player_features` | |
| 06 Churn model | compare feature sets / models, segments, label noise, SHAP | `player_features`, `player_matches` | |
| 07 Train and save model | final 12-feature random forest (same code as the `train` node) | `player_features` | `churn_model` |
| 08 Churn predictor | form + batch scoring + SHAP explanation, logs predictions | `churn_model` | `predictions_log` |

Every notebook starts with the same setup cell:

```python
sys.path.insert(0, str(Path.cwd().parent / 'src'))  # not needed after `pip install -e .`
from ubi_churn.config import DataCatalog, load_params
catalog = DataCatalog.from_config()
params = load_params()
df = catalog.load('player_matches')
```

To change a constant (session gap, Elo K factor, churn threshold, model features...), edit
`conf/base/parameters.yml` (or override it in `conf/local/parameters.yml`), then re-run the pipeline.
