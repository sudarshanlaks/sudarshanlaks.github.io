"""
Project 3 - Inventory health
Question: where is stock tied up, and is the safety stock buffer bigger than it needs to be?

Run:  python inventory_health.py
Needs: pandas, numpy, matplotlib   (inventory_supply_monthly.csv and products.csv in the same folder)
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("output", exist_ok=True)

# Settings you can change
SCORE_START = "2024-01"       # first month used for analysis (Apr-May 2023 is start-up, see step 2)
EXCESS_MONTHS = 1             # stock above safety stock + this many months of demand counts as excess (one replenishment cycle)

# ---------------------------------------------------------------- 1. LOAD
inv = pd.read_csv("inventory_supply_monthly.csv")
prod = pd.read_csv("products.csv")
print("1. LOAD")
print("   inventory rows:", len(inv), "| SKUs:", inv.sku.nunique(), "| months:", inv.month.min(), "to", inv.month.max())

# ---------------------------------------------------------------- 2. CLEAN AND CHECK
print("2. CLEAN AND CHECK")
print("   duplicate sku-month rows:", int(inv.duplicated(["sku", "month"]).sum()))
print("   missing values:", int(inv.isna().sum().sum()))
print("   negative stock rows:", int((inv.closing_inventory < 0).sum()))
print("   SKUs not in product master:", int((~inv.sku.isin(prod.sku)).sum()))
startup = inv[inv.lost_sales_units > 0]
print("   rows with lost sales:", len(startup), "- all in months:", sorted(startup.month.unique()), "(start-up months, excluded)")
df = inv.merge(prod[["sku", "description", "unit_cost", "abc_class", "lead_time_months", "shelf_life_months"]], on="sku", how="left")
fcst = df[df.is_forecast == 1]
print("   projected rows (Oct-Dec 2026, kept for the forward view only):", len(fcst))
df = df[df.month >= SCORE_START].copy()
print("   rows used for analysis:", int((df.is_forecast == 0).sum()), "actual rows,", df[df.is_forecast == 0].month.min(), "to", df[df.is_forecast == 0].month.max())

# ---------------------------------------------------------------- 3. CREATE KPIs
print("3. CREATE KPIs")
df["inventory_value"] = df.closing_inventory * df.unit_cost
df["safety_stock_value"] = df.safety_stock * df.unit_cost
df["buffer_ratio"] = df.closing_inventory / df.safety_stock                  # 1.0 = exactly at safety stock
df["target_max_units"] = df.safety_stock + EXCESS_MONTHS * df.demand_units    # above this = excess
df["excess_units"] = (df.closing_inventory - df.target_max_units).clip(lower=0)
df["excess_value"] = df.excess_units * df.unit_cost
df["status"] = np.where(df.closing_inventory < df.safety_stock, "Below safety stock",
                np.where(df.closing_inventory > df.target_max_units, "Excess", "Healthy"))
df["headroom_units"] = df.closing_inventory - df.safety_stock

actual = df[df.is_forecast == 0].copy()
latest_month = actual.month.max()
now = actual[actual.month == latest_month]

def kpis(g):
    return pd.Series({
        "inventory_value": g.inventory_value.sum(),
        "safety_stock_value": g.safety_stock_value.sum(),
        "safety_stock_share": g.safety_stock_value.sum() / g.inventory_value.sum(),
        "days_of_cover": g.closing_inventory.sum() / g.demand_units.sum() * 30,
        "excess_value": g.excess_value.sum(),
        "skus_excess": int((g.status == "Excess").sum()),
        "skus_below_ss": int((g.status == "Below safety stock").sum()),
        "skus": g.sku.nunique(),
    })

overall = kpis(now)
by_cat = now.groupby("category").apply(kpis)
by_abc = now.groupby("abc_class").apply(kpis)
print("   latest month:", latest_month)
print("   inventory value EUR %.2fM | safety stock EUR %.2fM (%.0f%%) | days of cover %.0f | excess EUR %.2fM in %d SKUs"
      % (overall.inventory_value/1e6, overall.safety_stock_value/1e6, overall.safety_stock_share*100,
         overall.days_of_cover, overall.excess_value/1e6, overall.skus_excess))

# How often was the safety stock buffer touched? (every month from SCORE_START to latest)
n_below = int((actual.closing_inventory < actual.safety_stock).sum())
print("   SKU-months below safety stock since", SCORE_START, ":", n_below, "of", len(actual))

# Unused buffer: lowest headroom each SKU ever had above its safety stock
sku_hist = actual.groupby("sku").agg(
    min_buffer_ratio=("buffer_ratio", "min"),
    min_headroom_units=("headroom_units", "min"),
    unit_cost=("unit_cost", "first"), category=("category", "first"),
    abc_class=("abc_class", "first"), description=("description", "first"))
sku_hist["unused_buffer_value"] = sku_hist.min_headroom_units * sku_hist.unit_cost
print("   lowest buffer ratio across all SKUs: %.2f (1.00 would mean stock touched the safety stock)" % sku_hist.min_buffer_ratio.min())
print("   median of each SKU's lowest buffer ratio: %.2f" % sku_hist.min_buffer_ratio.median())
print("   unused buffer (lowest headroom x cost): EUR %.2fM" % (sku_hist.unused_buffer_value.sum()/1e6))
unused_cat = sku_hist.groupby("category").unused_buffer_value.sum()

# Forward view
proj = pd.concat([df[df.is_forecast == 1]])
print("   projected months below safety stock (Oct-Dec 2026):", int((proj.closing_inventory < proj.safety_stock).sum()))

# ---------------------------------------------------------------- 4. SAVE OUTPUTS
print("4. SAVE OUTPUTS")
cols = ["month", "sku", "description", "category", "abc_class", "demand_units", "receipts_units", "closing_inventory",
        "safety_stock", "days_of_cover", "unit_cost", "inventory_value", "safety_stock_value", "buffer_ratio",
        "excess_units", "excess_value", "status", "is_forecast"]
out = df[cols].copy()
for c in ["inventory_value", "safety_stock_value", "excess_value"]: out[c] = out[c].round(2)
out["buffer_ratio"] = out.buffer_ratio.round(3)
out.to_csv("output/fact_inventory_health.csv", index=False)

snap = now[["sku", "description", "category", "abc_class", "closing_inventory", "safety_stock", "days_of_cover",
            "inventory_value", "excess_value", "status"]].merge(
    sku_hist[["min_buffer_ratio", "unused_buffer_value"]].round(2), left_on="sku", right_index=True)
snap.sort_values("unused_buffer_value", ascending=False).to_csv("output/sku_snapshot_latest.csv", index=False)

cat = by_cat.copy(); cat["unused_buffer_value"] = unused_cat
cat.round(3).to_csv("output/category_summary.csv")
by_abc.round(3).to_csv("output/abc_summary.csv")
trend = actual.groupby("month").apply(kpis)[["inventory_value", "safety_stock_value", "days_of_cover"]]
trend.round(2).to_csv("output/monthly_trend.csv")

print(by_cat.round(2).to_string())
print(by_abc.round(2).to_string())
print(snap.status.value_counts().to_string())

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(trend.index, trend.inventory_value/1e6, color="#257de0", lw=2, label="Inventory value")
ax[0].plot(trend.index, trend.safety_stock_value/1e6, color="#8c8c8c", lw=2, label="Safety stock value")
ax[0].set_xticks(trend.index[::6]); ax[0].set_title("Inventory vs safety stock, EUR million"); ax[0].legend(); ax[0].set_ylim(0)
ax[1].bar(by_cat.index, by_cat.days_of_cover, color="#00b894"); ax[1].set_title("Days of cover, " + latest_month)
plt.tight_layout(); plt.savefig("output/inventory_health.png", dpi=130)
print("Done. Files are in the output folder.")
