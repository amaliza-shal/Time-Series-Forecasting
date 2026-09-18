import json
import re
from pathlib import Path

import pandas as pd

NB1 = json.loads(Path("notebooks/01_data_and_eda.ipynb").read_text(encoding="utf-8"))
NB2 = json.loads(Path("notebooks/02_modeling_and_results.ipynb").read_text(encoding="utf-8"))

GITHUB_URL = "https://github.com/amaliza-shal/Time-Series-Forecasting"


def md_cell(nb, index):
    cell = nb["cells"][index]
    assert cell["cell_type"] == "markdown", f"cell {index} is not markdown"
    return "".join(cell["source"])


def promote(text, levels=1):
    """Demote report-level headers by `levels`: e.g. levels=1: '### '->'## ', '## '->'# '."""
    for _ in range(levels):
        out = []
        for line in text.split("\n"):
            if line.startswith("### "):
                out.append("## " + line[4:])
            elif line.startswith("## "):
                out.append("# " + line[3:])
            else:
                out.append(line)
        text = "\n".join(out)
    return text


def img(path, width_pct=90):
    return f'\n<div class="figure"><img src="{path}" style="width:{width_pct}%;"></div>\n'


RENAME_COLS = {
    "rss_mb": "RSS (MB)",
    "elapsed_s": "Elapsed (s)",
    "rss_delta_vs_baseline_mb": "Delta vs baseline (MB)",
    "stage": "Stage",
}


def csv_table(path, max_rows=None, float_fmt="{:.2f}", int_cols=()):
    df = pd.read_csv(path)
    if max_rows:
        df = df.head(max_rows)
    for c in df.columns:
        if c in int_cols:
            df[c] = df[c].astype(int)
    df = df.rename(columns=RENAME_COLS)
    df = df.fillna("-")
    return df.to_markdown(index=False, floatfmt=".3f")


parts = []

# ---------------------------------------------------------------- Title page
parts.append("""# Mobile Network Traffic Forecasting on the Milan CDR Grid

**Forecasting Internet traffic for the week of December 16-22, 2013, on the Milan Telecom
Italia Call Detail Record (CDR) grid, using SARIMA, LSTM, and a hand-implemented Temporal
Convolutional Network.**

<div class="titlemeta">
Data: Milan Telecom Italia Big Data Challenge CDR grid, Nov 1, 2013 - Jan 1, 2014, 10-minute
intervals, 10,000 grid squares [1].<br>
Evaluation target: walk-forward one-step-ahead forecasts, Dec 16-22, 2013, top-3 squares by
total traffic.
</div>

<div style="page-break-after: always;"></div>
""")

# ---------------------------------------------------------------- Introduction
parts.append("""# Introduction

Mobile network operators need short-horizon traffic forecasts to drive capacity planning, load
balancing, and energy-saving decisions (e.g. sleeping under-used cells overnight). This report
addresses a concrete version of that problem: **how do different sequential forecasting models
compare for one-step-ahead mobile network traffic forecasting, and how does their performance
vary across geographical areas with different traffic characteristics?**

The dataset is the Milan Telecom Italia Call Detail Record (CDR) grid [1]: the city of Milan is
partitioned into 10,000 squares, and for each square and each 10-minute interval the data
records SMS, call, and Internet traffic volumes, from November 1, 2013 to January 1, 2014. This
report forecasts **Internet traffic**, one square at a time, for the week of **December 16-22,
2013**, using three architecturally distinct models - a classical statistical model (SARIMA), a
recurrent neural network (LSTM), and a hand-implemented convolutional model (a Temporal
Convolutional Network, TCN) - trained and evaluated independently on the three highest-traffic
squares in the grid.

The work is organized as two notebooks that this report mirrors section-for-section: the first
covers data preparation and exploratory analysis, establishing the traffic characteristics
(seasonality, volatility, stationarity) that motivate the modeling choices; the second covers
methodology, model implementation, and results, ending with a walk-forward evaluation that
conditions every one-step prediction on true past observations. The remainder of this report
follows that structure: Related Work reviews the literature the model choices are grounded in;
Dataset and Data Preparation describes the data and the memory-efficient loading strategy used
to process it; Exploratory Analysis characterizes the traffic patterns across squares; Methodology
details each model's design, input representation, and tuning; Results and Discussion reports
walk-forward forecasting performance, timing, and a specific failure case; and Conclusion and
Future Work synthesizes the comparison and discusses what a next iteration would improve. The
project's code and notebooks are available at the repository linked in the References [6].

<div style="page-break-after: always;"></div>
""")

# ---------------------------------------------------------------- Related Work
lit_full = md_cell(NB2, 2)
marker = "**Resulting model trio and rationale"
lit_review_text, _, model_trio_text = lit_full.partition(marker)
lit_review_text = lit_review_text.replace("## Methodology\n\n### Literature review", "# Related Work")
model_trio_text = marker + model_trio_text
parts.append(lit_review_text.strip())
parts.append('<div style="page-break-after: always;"></div>\n')

# ---------------------------------------------------------------- Dataset and Data Preparation
parts.append(promote(md_cell(NB1, 2)))
parts.append(promote(md_cell(NB1, 4)))
parts.append("**Memory measured at each stage of the loading pipeline:**\n")
parts.append(csv_table("results/memory_usage.csv"))
parts.append(promote(md_cell(NB1, 11)))
parts.append("\n**Top 10 squares by total Internet traffic (Nov 1, 2013 - Jan 1, 2014):**\n")
parts.append(csv_table("results/square_ranking.csv", max_rows=10, int_cols=("rank", "square_id")))
parts.append(
    "\nThe **top-3 squares** carried forward into the rest of this report are **5161, 5059, "
    "and 5259**. Two additional landmark squares, **4159** and **4556**, are also examined in "
    "the Exploratory Analysis section below.\n"
)
parts.append('<div style="page-break-after: always;"></div>\n')

# ---------------------------------------------------------------- Exploratory Analysis
parts.append(promote(md_cell(NB1, 12)))
parts.append(img("../figures/eda_traffic_distribution.png"))
parts.append(promote(md_cell(NB1, 14)))

parts.append(promote(md_cell(NB1, 15)))
parts.append(img("../figures/eda_timeseries_top3_first2weeks.png"))
parts.append(promote(md_cell(NB1, 17)))
parts.append(img("../figures/eda_timeseries_landmarks_first2weeks.png"))
parts.append(promote(md_cell(NB1, 19)))

parts.append(promote(md_cell(NB1, 20)))
parts.append(img("../figures/eda_stl_top1.png"))
parts.append("\n**STL decomposition strength (square 5161):**\n")
parts.append(csv_table("results/stl_strength_top1.csv"))
parts.append(promote(md_cell(NB1, 22)))

parts.append(promote(md_cell(NB1, 23)))
parts.append(img("../figures/eda_acf_pacf_top1.png"))
parts.append("\n**Augmented Dickey-Fuller test (square 5161):**\n")
parts.append(csv_table("results/adf_test_top1.csv", float_fmt="{:.4f}"))
parts.append(promote(md_cell(NB1, 25)))
parts.append('<div style="page-break-after: always;"></div>\n')

# ---------------------------------------------------------------- Methodology
parts.append("# Methodology\n")
parts.append(promote(model_trio_text, levels=0))
parts.append(promote(md_cell(NB2, 3)))

parts.append("## Model 1: SARIMA\n")
parts.append(promote(md_cell(NB2, 6)))
parts.append(promote(md_cell(NB2, 8)))
parts.append("\n**SARIMA grid search (final, Fourier-term approach) - AIC by (p,d,q), per square:**\n")
parts.append(csv_table("results/sarima_grid_search.csv", float_fmt="{:.1f}"))

parts.append("\n## Model 2: LSTM\n")
parts.append(promote(md_cell(NB2, 14)))
parts.append("\n**LSTM hyperparameter grid search (validation MSE, top-1 square):**\n")
parts.append(csv_table("results/lstm_grid_search.csv", float_fmt="{:.4f}"))

parts.append("\n## Model 3: TCN (hand-implemented, dilated causal convolutions)\n")
parts.append(promote(md_cell(NB2, 19)))
parts.append("\n**TCN hyperparameter grid search (validation MSE, top-1 square):**\n")
parts.append(csv_table("results/tcn_grid_search.csv", float_fmt="{:.4f}"))
parts.append(promote(md_cell(NB2, 21)))
parts.append('<div style="page-break-after: always;"></div>\n')

# ---------------------------------------------------------------- Results and Discussion
parts.append("# Results and Discussion\n")

metrics_frames = []
for sid in (5161, 5059, 5259):
    df = pd.read_csv(f"results/metrics_square_{sid}.csv")
    df.insert(0, "square_id", sid)
    metrics_frames.append(df)
metrics_all = pd.concat(metrics_frames, ignore_index=True)

for sid in (5161, 5059, 5259):
    parts.append(f"\n## Square {sid}\n")
    sub = metrics_all[metrics_all["square_id"] == sid][["model", "MAE", "MAPE", "RMSE"]]
    parts.append(sub.to_markdown(index=False, floatfmt=".2f"))
    for model_name in ("SARIMA", "LSTM", "TCN"):
        parts.append(img(f"../figures/results_{model_name}_{sid}.png", width_pct=80))

parts.append("\n## Training and inference timing\n")
timing = pd.read_csv("results/timing.csv")
timing_pivot = timing.pivot_table(index=["model", "phase"], values="seconds", aggfunc="mean").round(2)
timing_pivot = timing_pivot.reset_index()
parts.append(timing_pivot.to_markdown(index=False, floatfmt=".2f"))
parts.append(
    f"\n*Hardware: {timing['hardware'].iloc[0]}, {timing['device'].iloc[0]}. Each value is a "
    "**single, unrepeated measurement** per model/square (see the Timing methodology note in "
    "Methodology above), averaged only across the 3 squares for display here - not across "
    "repeated trials. SARIMA's walk-forward inference is a sequential Kalman-filter recursion "
    "(1,008 steps), while LSTM/TCN inference is a single batched pass over true-history windows "
    "(see the walk-forward equivalence note in Methodology).*\n"
)

parts.append(promote(md_cell(NB2, 26)))
parts.append("\n## Where a model performs poorly\n")
parts.append(img("../figures/results_worst_case.png"))
parts.append(promote(md_cell(NB2, 28)))
parts.append('<div style="page-break-after: always;"></div>\n')

# ---------------------------------------------------------------- Conclusion and Future Work
# NOTE: this section is filled in by hand after inspecting the fresh results below,
# since specific numbers (winning configs, MAPE, timing) can change once the grid
# search runs at full (20-epoch) scale instead of the old 5-epoch proxy.
CONCLUSION_PLACEHOLDER = "%%CONCLUSION_AND_FUTURE_WORK%%"
parts.append(CONCLUSION_PLACEHOLDER)
parts.append('<div style="page-break-after: always;"></div>\n')

# ---------------------------------------------------------------- References
parts.append(f"""# References

[1] G. Barlacchi, M. De Nadai, R. Larcher, et al., "A multi-source dataset of urban life in
the city of Milan and the Province of Trentino," *Scientific Data*, vol. 2, no. 150055, 2015.

[2] A. Azari, P. Papapetrou, S. Denic, and G. Peters, "Cellular Traffic Prediction and
Classification: A Comparative Evaluation of LSTM and ARIMA," in *Proc. 22nd Int. Conf.
Discovery Science (DS)*, Split, Croatia, 2019, pp. 129-144.

[3] W. Wang, C. Zhou, H. He, W. Wu, W. Zhuang, and X. Shen, "Cellular Traffic Load Prediction
with LSTM and Gaussian Process Regression," in *Proc. IEEE Int. Conf. Commun. (ICC)*,
Dublin, Ireland, Jun. 2020, pp. 1-6.

[4] C. Zhang and P. Patras, "Long-Term Mobile Traffic Forecasting Using Deep Spatio-Temporal
Neural Networks," in *Proc. ACM MobiHoc*, Los Angeles, CA, 2018, pp. 231-240.

[5] S. Bai, J. Z. Kolter, and V. Koltun, "An Empirical Evaluation of Generic Convolutional and
Recurrent Networks for Sequence Modeling," *arXiv:1803.01271*, 2018.

[6] Project repository: {GITHUB_URL}

[7] Demo video: *[add video link here]*.
""")

report_md = "\n\n".join(parts)
Path("report/report.md").write_text(report_md, encoding="utf-8")
print(f"wrote report/report.md ({len(report_md)} chars) - CONCLUSION_PLACEHOLDER still needs filling")
