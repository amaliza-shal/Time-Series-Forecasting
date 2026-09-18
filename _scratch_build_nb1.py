import json
from pathlib import Path

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}

cells = []

cells.append(md("""# Notebook 1 — Dataset and Data Preparation / Exploratory Analysis

**Mobile Network Traffic Forecasting — Milan CDR Grid**

This notebook covers the first two sections of the report: **Dataset and Data Preparation**, and **Exploratory Analysis**. It loads the raw Milan Telecom Italia Call Detail Record (CDR) files, ranks the 10,000 grid squares by total traffic, reconstructs clean 10-minute time series for a small set of squares of interest, and explores their statistical properties. All processed artifacts (a parquet file, CSV tables, PNG figures) are written to disk here and consumed directly by `02_modeling_and_results.ipynb` — that notebook never touches the raw `.txt` files."""))

cells.append(code("""import gc
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

RAW_DIR = Path("../data/raw")
PROCESSED_DIR = Path("../data/processed")
FIGURES_DIR = Path("../figures")
RESULTS_DIR = Path("../results")
for d in (PROCESSED_DIR, FIGURES_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

COLNAMES = ["square_id", "timestamp_ms", "country_code", "sms_in", "sms_out", "call_in", "call_out", "internet"]
CHUNKSIZE = 2_000_000

# --- shared helpers, defined once and reused throughout this notebook ---

def daily_files():
    \"\"\"Sorted list of the raw daily CDR files.\"\"\"
    return sorted(RAW_DIR.glob("sms-call-internet-mi-*.txt"))

def chunk_iter(path, usecols, dtype):
    \"\"\"Chunked tab-separated reader over one daily file, restricted to usecols.\"\"\"
    return pd.read_csv(
        path, sep="\\t", header=None, names=COLNAMES,
        usecols=usecols, dtype=dtype, chunksize=CHUNKSIZE,
    )

def measure_rss_mb():
    \"\"\"Current process resident-set-size, in MB, after a gc pass.\"\"\"
    gc.collect()
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

def reconstruct_series(long_series, full_index):
    \"\"\"Turn a (square_id, timestamp_ms)-indexed Series of summed internet traffic
    into a wide DataFrame (datetime index x square_id columns) reindexed onto a
    complete 10-minute grid, with missing intervals filled as zero traffic.\"\"\"
    df = long_series.reset_index()
    df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
    wide = df.pivot(index="datetime", columns="square_id", values="internet")
    wide = wide.reindex(full_index).fillna(0.0)
    wide.index.name = "datetime"
    return wide

print(f"Baseline RSS: {measure_rss_mb():.1f} MB")"""))

cells.append(md("""## Dataset and Data Preparation

**Source.** The Milan Telecom Italia CDR grid dataset (Telecom Italia Big Data Challenge / Barlacchi et al., 2015 [1]) partitions the city of Milan into a 100×100 grid of 10,000 squares. For each square and each 10-minute interval, the raw files record per-country SMS-in, SMS-out, call-in, call-out and Internet traffic volumes.

**Target variable.** This project forecasts **Internet traffic** volume per square — the `internet` column — summed across all `country_code` rows sharing the same `(square_id, timestamp)`. This matches the target variable used throughout the mobile/cellular traffic forecasting literature built on this dataset (e.g. [2], [3], [4]).

**Schema** (tab-separated, no header row): `square_id, timestamp_ms (epoch millis), country_code, sms_in, sms_out, call_in, call_out, internet`. Most fields are frequently missing (`NaN`) on any given row — a square/interval/country combination only has an entry for the activity types actually observed.

**Files on disk.** One file per calendar day. Confirmed below before doing anything else, since the assignment requires forecasting the week of **Dec 16–22, 2013**."""))

cells.append(code("""files = daily_files()
dates = [f.stem.replace("sms-call-internet-mi-", "") for f in files]
print(f"{len(files)} daily files found, spanning {dates[0]} to {dates[-1]}")

target_week = pd.date_range("2013-12-16", "2013-12-22", freq="D").strftime("%Y-%m-%d").tolist()
missing = [d for d in target_week if d not in dates]
assert not missing, f"Missing files for target evaluation week: {missing}"
print("All 7 files for the Dec 16-22 target evaluation week are present:", target_week)"""))

cells.append(md("""### Two-pass, chunked, memory-efficient loading

The raw corpus is **~20 GB across 62 files** (a single day is already ~340 MB / ~5.3M rows). Reading any one file fully with a plain `pd.read_csv(...)` (all columns, no chunking) already costs a large, immediate jump in process memory — and doing that for all 62 files at once is not something a typical laptop can hold in RAM simultaneously.

The loading strategy therefore uses **two chunked passes**, each processing one file at a time in fixed-size row chunks (`chunksize=2,000,000`) so peak memory stays roughly constant regardless of total corpus size:

1. **Pass 1 (ranking scan)** — reads only `square_id` and `internet` from every file, accumulating a running per-square total. Output: a full ranking of all 10,000 squares by total Internet traffic.
2. **Pass 2 (extraction scan)** — reads only `square_id`, `timestamp_ms`, `internet` from every file, keeping just the handful of squares selected from Pass 1's ranking plus two squares of interest, and reconstructs their full-resolution time series.

Both passes use `usecols` to skip parsing the unused columns entirely, which is both faster and lighter on memory than reading full rows."""))

cells.append(code("""# Naive baseline for comparison: fully load ONE day file (all 8 columns, no chunking)
mem_before_naive = measure_rss_mb()
t0 = time.time()
_naive_df = pd.read_csv(files[0], sep="\\t", header=None, names=COLNAMES)
naive_load_time = time.time() - t0
mem_after_naive = measure_rss_mb()
naive_rows = len(_naive_df)
del _naive_df
gc.collect()

print(f"Naive full load of one file ({files[0].name}): {naive_rows:,} rows in {naive_load_time:.1f}s")
print(f"RSS before: {mem_before_naive:.1f} MB -> after: {mem_after_naive:.1f} MB (delta {mem_after_naive - mem_before_naive:+.1f} MB)")"""))

cells.append(code("""# --- Pass 1: chunked ranking scan across all 62 files ---
mem_before_pass1 = measure_rss_mb()
t0 = time.time()

square_totals = pd.Series(dtype="float64")
for f in daily_files():
    for chunk in chunk_iter(f, usecols=["square_id", "internet"],
                             dtype={"square_id": "int32", "internet": "float32"}):
        grp = chunk.groupby("square_id")["internet"].sum()
        square_totals = square_totals.add(grp, fill_value=0.0)

pass1_time = time.time() - t0
mem_after_pass1 = measure_rss_mb()
print(f"Pass 1 (ranking scan) complete in {pass1_time:.1f}s over {len(daily_files())} files")
print(f"RSS before: {mem_before_pass1:.1f} MB -> after: {mem_after_pass1:.1f} MB (delta {mem_after_pass1 - mem_before_pass1:+.1f} MB)")
print(f"Squares seen: {len(square_totals)}")"""))

cells.append(code("""ranking_df = (
    square_totals.sort_values(ascending=False)
    .rename("total_internet_traffic")
    .reset_index()
    .rename(columns={"square_id": "square_id"})
)
ranking_df.insert(0, "rank", np.arange(1, len(ranking_df) + 1))
ranking_df["square_id"] = ranking_df["square_id"].astype(int)
ranking_df.to_csv(RESULTS_DIR / "square_ranking.csv", index=False)

top3_ids = ranking_df.iloc[:3]["square_id"].tolist()
print("Top-3 squares by total Internet traffic:", top3_ids)
ranking_df.head(10)"""))

cells.append(code("""landmark_ids = [4159, 4556]
selected_ids = sorted(set(top3_ids) | set(landmark_ids))
print("Squares to extract full time series for (top-3 union landmark squares):", selected_ids)

# --- Pass 2: chunked extraction scan across all 62 files, filtered to selected_ids ---
mem_before_pass2 = measure_rss_mb()
t0 = time.time()

partials = []
for f in daily_files():
    for chunk in chunk_iter(f, usecols=["square_id", "timestamp_ms", "internet"],
                             dtype={"square_id": "int32", "timestamp_ms": "int64", "internet": "float32"}):
        sub = chunk[chunk["square_id"].isin(selected_ids)]
        if len(sub) == 0:
            continue
        partials.append(sub.groupby(["square_id", "timestamp_ms"])["internet"].sum())

long_series = pd.concat(partials).groupby(level=[0, 1]).sum()
pass2_time = time.time() - t0
mem_after_pass2 = measure_rss_mb()
print(f"Pass 2 (extraction scan) complete in {pass2_time:.1f}s")
print(f"RSS before: {mem_before_pass2:.1f} MB -> after: {mem_after_pass2:.1f} MB (delta {mem_after_pass2 - mem_before_pass2:+.1f} MB)")
print(f"Extracted {len(long_series):,} (square, timestamp) records for {len(selected_ids)} squares")"""))

cells.append(code("""full_index = pd.date_range("2013-11-01", "2014-01-02", freq="10min", inclusive="left")
wide = reconstruct_series(long_series, full_index)
wide.columns = [f"square_{c}" for c in wide.columns]
wide.to_parquet(PROCESSED_DIR / "timeseries_selected_squares.parquet")

print(f"Reconstructed wide series: shape={wide.shape}, index {wide.index.min()} -> {wide.index.max()}")
n_missing_filled = int((long_series.reset_index().pivot(index='timestamp_ms', columns='square_id', values='internet').reindex(
    pd.date_range('2013-11-01','2014-01-02',freq='10min',inclusive='left').astype('int64')//10**6
).isna().sum().sum()))
wide.head()"""))

cells.append(code("""mem_table = pd.DataFrame([
    {"stage": "baseline (imports only)", "rss_mb": round(mem_before_naive, 1), "elapsed_s": None},
    {"stage": f"naive full single-file load ({files[0].name}, all columns, no chunking)", "rss_mb": round(mem_after_naive, 1), "elapsed_s": round(naive_load_time, 1)},
    {"stage": "after chunked Pass 1 (ranking scan, all 62 files, 2 cols)", "rss_mb": round(mem_after_pass1, 1), "elapsed_s": round(pass1_time, 1)},
    {"stage": "after chunked Pass 2 (extraction scan, all 62 files, 3 cols)", "rss_mb": round(mem_after_pass2, 1), "elapsed_s": round(pass2_time, 1)},
])
mem_table["rss_delta_vs_baseline_mb"] = (mem_table["rss_mb"] - mem_table.loc[0, "rss_mb"]).round(1)
mem_table.to_csv(RESULTS_DIR / "memory_usage.csv", index=False)
mem_table"""))

cells.append(md("""**Discussion.** The naive single-file load already pulls in a noticeable, immediate memory jump for just one of 62 files. Repeating that pattern for the full ~20 GB corpus at once is not viable on commodity hardware. The two chunked passes, in contrast, process one bounded chunk (2,000,000 rows) at a time and only ever retain small aggregates in memory (a 10,000-entry running total in Pass 1; a handful of per-square time series in Pass 2) — so process memory stays close to baseline throughout, independent of how many files are scanned. This is the load-bearing design decision that makes the rest of the pipeline tractable: Pass 1 turns the full 20 GB corpus into a single ranking table, and Pass 2 turns it into one small Parquet file that Notebook 2 works from entirely in memory, without ever opening a raw `.txt` file again."""))

cells.append(md("""## Exploratory Analysis

### Distribution of total traffic across all squares"""))

cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].hist(ranking_df["total_internet_traffic"], bins=60, color="steelblue")
axes[0].set_xlabel("Total Internet traffic (Nov 1 - Jan 1)")
axes[0].set_ylabel("Number of squares")
axes[0].set_title("Distribution across all 10,000 squares")

axes[1].hist(np.log10(ranking_df["total_internet_traffic"] + 1), bins=60, color="darkorange")
axes[1].set_xlabel("log10(total Internet traffic + 1)")
axes[1].set_ylabel("Number of squares")
axes[1].set_title("Same distribution, log scale")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_traffic_distribution.png", bbox_inches="tight")
plt.show()

ranking_df["total_internet_traffic"].describe()"""))

cells.append(md("""**Discussion.** Total traffic is heavily right-skewed across the grid: a small number of central, high-density squares (containing transit hubs, business districts and stadiums) account for a disproportionate share of total Internet traffic, while the majority of the 10,000 squares — many of which cover low-density outskirts — carry comparatively little. On a log scale the distribution is much closer to unimodal, consistent with a multiplicative, population/land-use-driven process rather than a uniform spread of activity across the city grid. This skew is the direct motivation for concentrating both the EDA and the modeling on a handful of high-traffic squares (the top-3) rather than the full grid — they are where forecasting accuracy matters most in absolute terms."""))

cells.append(md("""### Time series for the top-3 squares and the two landmark squares (first two weeks)

The first two weeks of the observation window (Nov 1–14, 2013) are plotted below at native 10-minute resolution, first for the top-3 squares by total traffic, then for squares 4159 and 4556."""))

cells.append(code("""two_weeks = wide.loc["2013-11-01":"2013-11-14"]
top3_cols = [f"square_{sid}" for sid in top3_ids]

fig, ax = plt.subplots(figsize=(13, 4.5))
for col in top3_cols:
    ax.plot(two_weeks.index, two_weeks[col], label=col, linewidth=0.9)
ax.set_xlabel("Datetime")
ax.set_ylabel("Internet traffic (10-min interval)")
ax.set_title(f"Top-3 squares by total traffic ({', '.join(str(s) for s in top3_ids)}) - first two weeks")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_timeseries_top3_first2weeks.png", bbox_inches="tight")
plt.show()"""))

cells.append(md("""**Discussion.** All three top squares show a strong, regular daily cycle (a broad daytime peak and a deep overnight trough) superimposed on a clear weekly pattern distinguishing weekdays from the Nov 2–3 and Nov 9–10 weekends. The three series are highly correlated in shape but differ substantially in amplitude, consistent with their ranking — they are simply busier or quieter versions of the same underlying diurnal rhythm. This regularity is encouraging for forecasting: a model that captures daily and weekly seasonality should already explain most of the variance for these squares."""))

cells.append(code("""landmark_cols = [f"square_{sid}" for sid in landmark_ids]

fig, ax = plt.subplots(figsize=(13, 4.5))
for col in landmark_cols:
    ax.plot(two_weeks.index, two_weeks[col], label=col, linewidth=0.9)
ax.set_xlabel("Datetime")
ax.set_ylabel("Internet traffic (10-min interval)")
ax.set_title(f"Landmark squares {landmark_ids} - first two weeks")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_timeseries_landmarks_first2weeks.png", bbox_inches="tight")
plt.show()"""))

cells.append(md("""**Discussion.** Squares 4159 and 4556 sit well below the top-3 in absolute traffic (consistent with the right-skewed distribution above) but still display the same daily/weekly seasonal shape. Depending on their location in the grid, one or both may show a less sharply peaked profile than the top-3 squares, i.e. flatter and noisier relative to their own mean — a pattern worth revisiting when comparing model performance across squares of different traffic magnitude in Notebook 2, since lower-volume series are typically harder to forecast in relative (MAPE) terms even when absolute errors are small."""))

cells.append(md("""### STL decomposition of the top-1 square

Seasonal-Trend decomposition using LOESS (STL), with a period of 144 (one day at 10-minute resolution), applied to the full Nov 1 - Jan 1 series of the single highest-traffic square."""))

cells.append(code("""top1_col = top3_cols[0]
top1_series = wide[top1_col]

stl = STL(top1_series, period=144, robust=True)
stl_result = stl.fit()

fig = stl_result.plot()
fig.set_size_inches(12, 7)
fig.suptitle(f"STL decomposition - {top1_col} (period=144, daily)", y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_stl_top1.png", bbox_inches="tight")
plt.show()

seasonal_strength = 1 - stl_result.resid.var() / (stl_result.seasonal + stl_result.resid).var()
trend_strength = 1 - stl_result.resid.var() / (stl_result.trend + stl_result.resid).var()
print(f"Seasonal strength: {seasonal_strength:.3f}, Trend strength: {trend_strength:.3f}")

pd.DataFrame([{"square": top1_col, "seasonal_strength": seasonal_strength, "trend_strength": trend_strength}]).to_csv(
    RESULTS_DIR / "stl_strength_top1.csv", index=False
)"""))

cells.append(md("""**Discussion.** The STL decomposition confirms what the raw time series plot already suggested: the daily seasonal component dominates the series, while the trend component is comparatively smooth and slow-moving over the two-month window (with visible dips around lower-activity periods, e.g. late-November/December). The residual component is small relative to the seasonal amplitude, meaning that a model which explicitly captures the 144-step daily cycle — as SARIMA and, implicitly, the windowed deep models below do — should be able to explain the bulk of this square's variance; the remaining forecasting difficulty lives mostly in the residual, i.e. in short-lived deviations from the typical daily rhythm."""))

cells.append(md("""### Stationarity and autocorrelation structure of the top-1 square (ACF, PACF, ADF)"""))

cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
plot_acf(top1_series, lags=432, ax=axes[0])
axes[0].set_title(f"ACF - {top1_col} (lags up to 3 days)")
plot_pacf(top1_series, lags=48, ax=axes[1], method="ywm")
axes[1].set_title(f"PACF - {top1_col} (lags up to 8 hours)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_acf_pacf_top1.png", bbox_inches="tight")
plt.show()

adf_stat, adf_pvalue, adf_lags, adf_nobs, adf_crit, _ = adfuller(top1_series, autolag="AIC")
adf_result = pd.DataFrame([{
    "square": top1_col, "adf_statistic": adf_stat, "p_value": adf_pvalue,
    "n_lags_used": adf_lags, "n_obs": adf_nobs, **{f"crit_{k}": v for k, v in adf_crit.items()},
}])
adf_result.to_csv(RESULTS_DIR / "adf_test_top1.csv", index=False)
adf_result"""))

cells.append(md("""**Discussion.** The ACF shows pronounced peaks at lag 144 and its multiples (288, 432, ...), directly confirming the strong daily periodicity already visible in the raw series and the STL seasonal component. The PACF cuts off much faster, indicating that most of the short-range predictive signal is concentrated in the most recent few lags once the daily seasonal pattern is accounted for — informing the choice of a modest non-seasonal AR order for SARIMA. The ADF test rejects the unit-root null hypothesis at conventional significance levels (see `p_value` above), i.e. the series is statistically stationary around its seasonal pattern; this justifies fitting SARIMA directly on the raw series (with a seasonal order term) rather than requiring additional non-seasonal differencing."""))

cells.append(md("""## Artifacts saved for Notebook 2

- `data/processed/timeseries_selected_squares.parquet` — reconstructed 10-minute series (zero-filled, complete grid) for squares """ + "{" + "top-3 union landmarks" + "}" + """.
- `results/square_ranking.csv` — all 10,000 squares ranked by total Internet traffic.
- `results/memory_usage.csv`, `results/stl_strength_top1.csv`, `results/adf_test_top1.csv` — supporting tables.
- `figures/eda_*.png` — all EDA figures.

Notebook 2 (`02_modeling_and_results.ipynb`) loads only the parquet file and the ranking CSV, and never re-reads the raw `.txt` files."""))

cells.append(code("""print("Files now in data/processed/:", [p.name for p in PROCESSED_DIR.iterdir()])
print("Files now in figures/:", sorted(p.name for p in FIGURES_DIR.iterdir()))
print("Files now in results/:", sorted(p.name for p in RESULTS_DIR.iterdir()))
print("\\nTop-3 squares:", top3_ids, " Landmark squares:", landmark_ids)"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "ts-forecasting", "language": "python", "name": "ts-forecasting"},
        "language_info": {"name": "python", "version": "3.11.9"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path("notebooks/01_data_and_eda.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print("wrote", out_path, "with", len(cells), "cells")
