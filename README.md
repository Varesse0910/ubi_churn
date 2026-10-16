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
│   └── 08_reporting/         # churn_analysis.html (the review report)
├── notebooks/                # 01 -> 08, the analysis (read the markdown observations)
├── reports/                  # churn_analysis.qmd (report source) + build_report.py
├── src/ubi_churn/
│   ├── config.py             # PROJECT_ROOT, load_params(), DataCatalog
│   ├── pipeline.py           # nodes: prepare -> features -> train
│   └── utils/
│       ├── features.py       # data preparation + feature engineering
│       ├── exploration.py    # EDA tables, data-quality checks, group comparison, plots
│       └── ml.py             # model zoo, evaluation, SHAP, final model, ChurnPredictor
├── tests/test_utils.py       # sanity tests (plain python recomputation, reproducibility, predictor)
└── CLAUDE.md                 # project guide for Claude Code (ask questions about the code)
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

## Reading the HTML report

`data/08_reporting/churn_analysis.html` is the review of the whole analysis: bottom line, how it works,
hypotheses tested, results, findings, reusability and next steps.

**Open it**
- From your computer: double-click the file (or right-click > Open with > your browser). It is a single
  self-contained file, so it works offline and you can send it by email as is.
- From GitHub: GitHub shows the HTML source, not the page. Open the file in the repo, click
  **Download raw file**, then open the downloaded file in your browser (or clone the repo).

**Find your way**
- Start with **Bottom line** at the top: the verdict and the key numbers in five bullets.
- Use the **On this page** menu on the left to jump to a section.
- Charts are interactive: hover a bar or a point to see its exact value.
- **What to do next** at the end lists the actions.

**Rebuild it** (after the data or the model changes). It needs [Quarto](https://quarto.org) and `plotly`:

```bash
python reports/build_report.py                          # quarto on PATH
python reports/build_report.py --quarto C:/path/to/quarto.exe
```

Every number in the report is recomputed from the catalog with the project code, so a rebuild follows the
latest pipeline run.

## Ask questions about the code (Claude Code)

`CLAUDE.md` at the root describes the project for [Claude Code](https://claude.com/claude-code): what the
data is, where each piece of logic lives, the commands, and the caveats that matter (recency, label
assumption, data quirks). Claude Code reads it automatically, so you can ask questions straight away.

1. Install Claude Code (`npm install -g @anthropic-ai/claude-code`) or use the VS Code extension.
2. Open a terminal in the project root (`ubi_churn/`) and run `claude`.
3. Ask in plain words, for example:
   - "How is `sessions_per_week` computed?"
   - "Why does the model rely so much on `days_since_last_match`?"
   - "Which notebook shows that skill does not explain churn?"
   - "What do I change to use a 60-minute session gap, and what must I re-run?"
   - "Walk me through what happens when I click Predict and save in notebook 08."

Keep `CLAUDE.md` up to date when the project changes (new feature, new notebook, new caveat).
