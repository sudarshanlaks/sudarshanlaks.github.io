"""
PROJECT 1 - FORECAST ACCURACY (FMCG)

What this script does, step by step
-----------------------------------
1. Load the demand file (one row per SKU x channel x month).
2. Keep only months where we know the real sales (Jan 2024 - Sep 2026).
3. Calculate the error of each forecast, row by row.
4. Roll the errors up (overall, by category, channel, month, SKU) into the
   standard planning KPIs: WAPE, Bias, Forecast Value Add (FVA).
5. Save a clean "fact table" for Power BI and small summary tables for checking.

Keep this file in the same folder as demand_monthly.csv and products.csv.
Run:  python forecast_accuracy.py
"""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- 1. LOAD
demand = pd.read_csv(HERE / "demand_monthly.csv")
products = pd.read_csv(HERE / "products.csv")[["sku", "description", "abc_class"]]

# ---------------------------------------------------------------- 2. FILTER
# Future months (Oct-Dec 2026) have no actuals yet, so they cannot be scored.
# 2023 is used by the forecast model as history, so we score from 2024 onwards.
df = demand[demand["actual_units"].notna() & (demand["month"] >= "2024-01")].copy()
df["month"] = pd.to_datetime(df["month"] + "-01")
df = df.merge(products, on="sku", how="left")

# ---------------------------------------------------------------- 3. ROW-LEVEL ERRORS
# Signed error: forecast minus actual. Positive = we forecast too much.
df["err_stat"] = df["stat_forecast"] - df["actual_units"]
df["err_cons"] = df["consensus_forecast"] - df["actual_units"]
# Absolute error: ignores direction, so over- and under-forecasts do not cancel.
df["abs_err_stat"] = df["err_stat"].abs()
df["abs_err_cons"] = df["err_cons"].abs()


# ---------------------------------------------------------------- 4. KPIs
def kpis(g: pd.DataFrame) -> pd.Series:
    """Volume-weighted KPIs for any group of rows."""
    actual = g["actual_units"].sum()
    wape_stat = g["abs_err_stat"].sum() / actual   # WAPE = sum|A-F| / sum A
    wape_cons = g["abs_err_cons"].sum() / actual
    return pd.Series({
        "actual_units": actual,
        "wape_stat": wape_stat,
        "wape_consensus": wape_cons,
        "bias_stat": g["err_stat"].sum() / actual,       # Bias = sum(F-A) / sum A
        "bias_consensus": g["err_cons"].sum() / actual,
        "accuracy_consensus": 1 - wape_cons,             # Accuracy = 1 - WAPE
        "fva": wape_stat - wape_cons,                    # >0: planner overlay helped
    })


overall = kpis(df).to_frame("value")
by_category = df.groupby("category").apply(kpis)
by_channel = df.groupby("channel").apply(kpis)
by_month = df.groupby("month").apply(kpis)
by_sku = df.groupby(["sku", "description", "category", "abc_class"]).apply(kpis)
by_sku["rank_worst_wape"] = by_sku["wape_consensus"].rank(ascending=False).astype(int)

# ---------------------------------------------------------------- 5. SAVE
fact_cols = ["month", "sku", "description", "category", "channel", "abc_class",
             "actual_units", "stat_forecast", "consensus_forecast", "promo_flag",
             "err_stat", "err_cons", "abs_err_stat", "abs_err_cons"]
df[fact_cols].to_csv(OUT / "fact_forecast_accuracy.csv", index=False)   # -> Power BI
overall.to_csv(OUT / "kpi_overall.csv")
by_category.to_csv(OUT / "kpi_by_category.csv")
by_channel.to_csv(OUT / "kpi_by_channel.csv")
by_month.to_csv(OUT / "kpi_by_month.csv")
by_sku.sort_values("rank_worst_wape").to_csv(OUT / "kpi_by_sku.csv")

pd.options.display.float_format = "{:,.3f}".format
print("Rows scored:", len(df))
print("\nOVERALL\n", overall)
print("\nBY CATEGORY\n", by_category[["wape_stat", "wape_consensus", "bias_consensus", "fva"]])
print("\nBY CHANNEL\n", by_channel[["wape_stat", "wape_consensus", "bias_consensus", "fva"]])
print("\n10 WORST SKUs (consensus WAPE)\n",
      by_sku.sort_values("rank_worst_wape").head(10)[["wape_consensus", "bias_consensus"]])
