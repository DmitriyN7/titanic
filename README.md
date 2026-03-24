# Titanic Survival Prediction

Compact Kaggle Titanic solution focused on **feature engineering + ensembling**.

## Key Results

- Best tracked setup: **Stacking (CatBoost ×2 + KNN + XGBoost) with RidgeClassifierCV meta-model**.
- Cross-validation: **0.839463** (5-fold StratifiedKFold).
- Public leaderboard: **0.77033**.

## What is in this repository

- `titanic.ipynb` — EDA, experiments and table with results
- `main.py` — end-to-end pipeline:
  - feature engineering,
  - model training,
  - generation of submission files.
- `train.csv`, `test.csv` — Titanic data.
- Generated submissions (2 stands for second iteration):
  - `2catboost.csv`
  - `2stack_ridge.csv`
  - `2stack_logreg.csv`
  - `2blend_plus.csv`
- `pyproject.toml`, `uv.lock` — dependencies and lockfile.
- `Notes.md` — experiment notes.

## Feature Engineering (summary):

- title extraction from passenger name;
- missing-value handling for `Embarked`, `Fare`, `Age`;
- engineered features such as `Deck`, family-size features, `TicketGroup`, `LogFare`, `FarePerPerson`, `IsChild`, `AgeClass`;
- separate feature matrices for:
  - CatBoost (categorical strings);
  - sklearn/XGBoost (numeric-encoded categories).

## How to run

### Requirements

- Python **3.13+**
- `uv` (recommended)

### Install dependencies

```bash
uv sync
```

### Run pipeline

```bash
uv run python main.py
```

After running, the repository will contain/refresh these output files:

- `2catboost.csv`
- `2stack_ridge.csv`
- `2stack_logreg.csv`
- `2blend_plus.csv`
