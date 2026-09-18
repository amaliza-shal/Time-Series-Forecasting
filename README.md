# Mobile Network Traffic Forecasting — Milan CDR Grid

Forecasting per-square Internet traffic on the Milan Telecom Italia Call Detail Record
(CDR) grid dataset (Nov 1, 2013 – Jan 1, 2014, 10-minute intervals, 10,000 grid squares),
targeting the evaluation week of **Dec 16–22, 2013**.

The entire project lives in **two Jupyter notebooks**, structured to mirror the report's
own section headings — the notebooks' markdown cells are written to be pasted close to
verbatim into the final report:

- [`notebooks/01_data_and_eda.ipynb`](notebooks/01_data_and_eda.ipynb) — **Dataset and Data
  Preparation**, **Exploratory Analysis**. Loads the raw `.txt` files with a two-pass,
  chunked, memory-efficient strategy, ranks all 10,000 squares by total traffic,
  reconstructs clean 10-minute series for the top-3 squares plus two landmark squares
  (4159, 4556), and explores their statistical structure (distribution, seasonality, STL
  decomposition, ACF/PACF, ADF stationarity test).
- [`notebooks/02_modeling_and_results.ipynb`](notebooks/02_modeling_and_results.ipynb) —
  **Methodology**, **Results and Discussion**. A literature-grounded justification for
  three architecturally distinct forecasting models — SARIMA (statistical), LSTM
  (recurrent), and a hand-implemented TCN (dilated causal convolutions) — each trained
  per-square on the top-3 squares and evaluated with walk-forward one-step-ahead
  forecasting over the target week, always conditioned on true history.

There is no `scripts/` folder or shared helper module by design: every function is
defined once near the top of the notebook that uses it, and reused throughout that
notebook rather than re-derived per section.

## Project structure

```
data/
  raw/          raw daily .txt files (already present, not modified)
  processed/    written by notebook 1 — parquet time series used by notebook 2
notebooks/
  01_data_and_eda.ipynb
  02_modeling_and_results.ipynb
figures/        all PNG figures from both notebooks
results/        all CSV tables from both notebooks (rankings, metrics, timing, ADF, etc.)
report/         placeholder for the final assembled PDF report (not yet written)
requirements.txt
```

## Setup

Requires a dedicated Python 3.11 environment (the raw dataset is ~20 GB and the
statsmodels/torch stack is most reliably tested there).

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python -m ipykernel install --user --name ts-forecasting --display-name "ts-forecasting"
```

`torch` installs faster and smaller from the CPU-only wheel index:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Running

Run the two notebooks **in order**, using the `ts-forecasting` kernel — notebook 2 loads
artifacts that notebook 1 writes to `data/processed/`, `figures/`, and `results/`, and
never re-reads the raw `.txt` files itself.

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_and_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_modeling_and_results.ipynb
```

(Or open them in Jupyter/VS Code and run all cells in order.)

**Expect notebook 1 to take a while** — it performs two full chunked passes over the
~20 GB raw corpus (a ranking scan and an extraction scan). Notebook 2 is much faster
(it only trains 9 small per-square models: 3 architectures × top-3 squares).

## Outputs

- `data/processed/timeseries_selected_squares.parquet` — reconstructed 10-minute series
  for the top-3 + landmark squares.
- `results/*.csv` — square ranking, memory-usage table, STL/ADF results, hyperparameter
  grid-search logs, per-square MAE/MAPE/RMSE tables, and training/inference timing.
- `figures/*.png` — all EDA and results figures, including the 9 actual-vs-predicted
  plots (3 models × top-3 squares).

The `report/` folder is intentionally left empty — since both notebooks already mirror
the report's section structure, assembling the final PDF is expected to be mostly
copying and lightly editing the notebooks' markdown and saved figures/tables.
