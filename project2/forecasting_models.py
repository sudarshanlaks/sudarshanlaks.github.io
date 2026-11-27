"""
PROJECT 2 - FORECASTING MODELS (FMCG)

Question: is there a simple forecasting method that beats the current statistical
forecast, and is the current forecast better than a naive baseline?

What this script does, step by step
-----------------------------------
1. Load the demand file (one row per SKU x channel x month) and the product master.
2. Check the data (duplicates, missing values, negatives, unknown SKUs) and keep
   only months where actual sales are known (April 2023 - September 2026).
3. Run five simple forecasting methods with a rolling "one month ahead" backtest.
4. Add the two forecasts that already exist in the data (statistical, consensus).
5. Calculate WAPE, bias and forecast value add vs the naive baseline, overall and by
   category, and pick the best simple method per category.
6. Save a fact table for the dashboard and Excel, plus summary tables and a chart.

The five methods
----------------
naive          : next month = last month's actual
seasonal_naive : next month = same month last year
moving_avg_3   : next month = average of the last 3 months
ses            : simple exponential smoothing (alpha = 0.3)
seasonal_ses   : remove the yearly pattern, apply SES, put the pattern back

How the backtest works (a "rolling origin" test)
------------------------------------------------
For every SKU x channel series and every test month from January 2025 to
September 2026, a method only sees actuals BEFORE that month and forecasts that
month. The forecast is then compared with the real value. Nothing from the future
is used. January 2025 is the first test month because the seasonal methods need at
least 12 months of history.

Keep this file next to demand_monthly.csv and products.csv.
Run:  python forecasting_models.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

TEST_START = pd.Timestamp("2025-01-01")
ALPHA = 0.3
METHODS = ["naive", "seasonal_naive", "moving_avg_3", "ses", "seasonal_ses", "statistical", "consensus"]

# ---------------------------------------------------------------- 1. LOAD
dem = pd.read_csv(HERE / "demand_monthly.csv")
prod = pd.read_csv(HERE / "products.csv")[["sku", "description"]]
dem["month"] = pd.to_datetime(dem["month"] + "-01")

# ---------------------------------------------------------------- 2. CHECK AND CLEAN
print("Rows loaded:", len(dem))
print("Duplicate SKU-channel-month rows:", dem.duplicated(["month", "sku", "channel"]).sum())
print("Missing forecasts:", dem[["stat_forecast", "consensus_forecast"]].isna().sum().sum())
print("Negative values:", (dem[["actual_units", "stat_forecast", "consensus_forecast"]] < 0).sum().sum())
print("SKUs missing from product master:", len(set(dem["sku"]) - set(prod["sku"])))

hist = dem[dem["actual_units"].notna()].copy()           # drop Oct-Dec 2026: no actuals yet
print("Rows with known actuals:", len(hist), "| future rows dropped:", len(dem) - len(hist))
hist = hist.sort_values(["sku", "channel", "month"])


# ---------------------------------------------------------------- 3. METHODS
def naive(y, i):
    return y[i - 1]


def seasonal_naive(y, i):
    return y[i - 12]


def moving_avg_3(y, i):
    return y[i - 3:i].mean()


def ses(y, i, alpha=ALPHA):
    level = y[0]
    for v in y[1:i]:
        level = alpha * v + (1 - alpha) * level
    return level


def seasonal_ses(y, i, months):
    """Remove the yearly pattern, smooth, then put the pattern back."""
    past = pd.Series(y[:i])
    m = pd.Series(months[:i])
    idx = past.groupby(m.values).mean() / past.mean()     # seasonal index per calendar month
    deseason = past.values / m.map(idx).values
    level = deseason[0]
    for v in deseason[1:]:
        level = ALPHA * v + (1 - ALPHA) * level
    return level * idx.get(months[i], 1.0)


# ---------------------------------------------------------------- 4. BACKTEST
rows = []
for (sku, ch), g in hist.groupby(["sku", "channel"]):
    g = g.reset_index(drop=True)
    y = g["actual_units"].to_numpy(float)
    mon = g["month"].dt.month.to_numpy()
    for i in range(len(g)):
        if g.loc[i, "month"] < TEST_START or i < 13:
            continue
        rows.append(dict(
            month=g.loc[i, "month"].strftime("%Y-%m"), sku=sku, category=g.loc[i, "category"], channel=ch,
            actual=y[i],
            naive=naive(y, i), seasonal_naive=seasonal_naive(y, i),
            moving_avg_3=moving_avg_3(y, i), ses=ses(y, i),
            seasonal_ses=seasonal_ses(y, i, mon),
            statistical=g.loc[i, "stat_forecast"],           # already in the dataset
            consensus=g.loc[i, "consensus_forecast"]))       # planner-adjusted, already in the dataset
wide = pd.DataFrame(rows)
wide[METHODS] = wide[METHODS].round(0)                       # forecasts are whole units
wide = wide.merge(prod, on="sku", how="left")
print("Test rows (SKU x channel x month):", len(wide))

# ---------------------------------------------------------------- 5. KPIs
def kpis(df):
    """WAPE and bias for every method on a group of rows."""
    a = df["actual"].sum()
    out = {}
    for m in METHODS:
        err = df[m] - df["actual"]
        out[m] = {"wape": err.abs().sum() / a, "bias": err.sum() / a}
    return pd.DataFrame(out).T


overall = kpis(wide)
overall["fva_vs_naive"] = overall.loc["naive", "wape"] - overall["wape"]   # >0 = better than naive
overall.index.name = "method"

by_cat = pd.concat({c: kpis(g) for c, g in wide.groupby("category")}, names=["category", "method"]).reset_index()
simple = [m for m in METHODS if m not in ("statistical", "consensus")]
best = (by_cat[by_cat["method"].isin(simple)].sort_values("wape").groupby("category").head(1)
        .set_index("category")[["method", "wape", "bias"]].rename(columns={"method": "best_simple_method", "wape": "best_simple_wape", "bias": "best_simple_bias"}))
stat = by_cat[by_cat["method"] == "statistical"].set_index("category")[["wape"]].rename(columns={"wape": "statistical_wape"})
best = best.join(stat)
best["improvement_pts"] = best["statistical_wape"] - best["best_simple_wape"]   # >0 = simple method is better

# ---------------------------------------------------------------- 6. SAVE
cols = ["month", "sku", "description", "category", "channel", "actual"] + METHODS
wide[cols].to_csv(OUT / "fact_forecast_methods.csv", index=False)          # -> dashboard and Excel
overall.to_csv(OUT / "method_comparison_overall.csv")
by_cat.to_csv(OUT / "method_comparison_by_category.csv", index=False)
best.to_csv(OUT / "best_method_by_category.csv")

fig, ax = plt.subplots(figsize=(8, 4.5))
overall["wape"].plot.bar(ax=ax, color="#257de0")
ax.set_ylabel("WAPE (lower is better)")
ax.set_title("Forecast error by method, Jan 2025 - Sep 2026")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
plt.xticks(rotation=30, ha="right"); plt.tight_layout()
fig.savefig(OUT / "method_comparison.png", dpi=150)

pd.options.display.float_format = "{:,.3f}".format
print("\nOVERALL\n", overall)
print("\nWAPE BY CATEGORY\n", by_cat.pivot(index="method", columns="category", values="wape").reindex(METHODS))
print("\nBEST SIMPLE METHOD PER CATEGORY vs CURRENT STATISTICAL FORECAST\n", best)
