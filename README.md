# quant_yanfu

This project predicts next-day stock returns using prior-day features. The dataset spans 2016-01-01 to 2020-12-31, organized by date.

## Repository Structure

```
quant_yanfu/
├─ data/                          # Raw data (unzipped)
│  └─ project_5year/              # Daily folders: YYYY/MM/DD/data_matrix.csv
├─ src/                           # Code
│  └─ eda/                         # EDA notebook and utilities
│     ├─ eda_overview.ipynb        # EDA notebook (generates statistics)
│     └─ eda_utils.py              # Reusable EDA helpers
├─ res/                           # Results (no code here)
│  └─ data-statics/               # Data statistics outputs
│     ├─ column_dtypes.csv         # Column names and dtypes
│     ├─ column_missingness.csv    # Missingness per column
│     ├─ metadata.csv              # Per-id start/end date coverage
│     ├─ y_hist.png                # Sampled y distribution plot
│     └─ y_summary.csv             # y summary statistics
└─ scripts/                        # Bash scripts (for long-running jobs)
```

## Data Notes

- Each `data_matrix.csv` file contains daily samples per stock `id`.
- Key columns include:
  - `id`, `DateTime`, `industry`, `weight`, `y`
  - `f_0~f_9` (normalized fundamentals)
  - `beta`, `indbeta`
  - `r_0~r_19`, `dv_0~dv_19` (past 30-min returns and turnover)
- Missing values exist and should be handled in modeling.

## EDA

Run the EDA notebook:
- `src/eda/eda_overview.ipynb`

It exports statistics to:
- `res/data-statics/`

## Next Steps

- Define modeling baselines and evaluation (weighted correlation).
- Add training/evaluation scripts when models are long-running.
