"""
Project 4 - OTIF and supplier delivery performance
Question: are suppliers delivering on time and in full, and which suppliers hold us back from the 95% OTIF target?

Run:  python otif_analysis.py
Needs: pandas, numpy, matplotlib   (purchase_orders.csv and products.csv in the same folder)
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("output", exist_ok=True)
TARGET = 0.95
REPORT_DATE = pd.Timestamp("2026-09-30")   # today, for open orders

# ---------------------------------------------------------------- 1. LOAD
po = pd.read_csv("purchase_orders.csv", parse_dates=["order_date", "promised_date", "received_date"])
prod = pd.read_csv("products.csv")
print("1. LOAD")
print("   PO lines:", len(po), "| POs:", po.po_number.nunique(), "| suppliers:", po.supplier.nunique(), "| SKUs:", po.sku.nunique())
print("   orders placed:", po.order_date.min().date(), "to", po.order_date.max().date())

# ---------------------------------------------------------------- 2. CLEAN AND CHECK
print("2. CLEAN AND CHECK")
n0 = len(po)
dups = po.duplicated(keep="first")
print("   duplicate lines removed:", int(dups.sum()))
po = po[~dups].copy()
print("   missing supplier or SKU:", int(po.supplier.isna().sum() + po.sku.isna().sum()), "| SKUs not in product master:", int((~po.sku.isin(prod.sku)).sum()))
bad_date = po.received_date.notna() & (po.received_date < po.order_date)
no_qty = po.received_date.notna() & po.received_qty.isna()
print("   received before it was ordered (date error):", int(bad_date.sum()))
print("   received but no received quantity:", int(no_qty.sum()))
exceptions = po[bad_date | no_qty].copy()
exceptions["issue"] = np.where(bad_date[bad_date | no_qty], "Received date before order date", "Received quantity missing")
exceptions.to_csv("output/exceptions.csv", index=False)
po = po[~(bad_date | no_qty)].copy()                       # excluded from KPIs, listed in exceptions.csv
open_po = po[po.received_date.isna()].copy()
done = po[po.received_date.notna()].copy()
print("   lines kept: %d (received %d, still open %d) from %d original rows" % (len(po), len(done), len(open_po), n0))

# ---------------------------------------------------------------- 3. CREATE KPIs
print("3. CREATE KPIs")
done = done.merge(prod[["sku", "category"]], on="sku", how="left")
done["on_time"] = done.received_date <= done.promised_date
done["in_full"] = done.received_qty >= done.ordered_qty
done["otif"] = done.on_time & done.in_full
done["days_late"] = (done.received_date - done.promised_date).dt.days.clip(lower=0)
done["short_units"] = (done.ordered_qty - done.received_qty).clip(lower=0)
done["result"] = np.select([done.otif, ~done.on_time & done.in_full, done.on_time & ~done.in_full], ["On time and in full", "Late only", "Short only"], "Late and short")
done["promised_month"] = done.promised_date.dt.strftime("%Y-%m")
done["line_value"] = done.ordered_qty * done.unit_cost
def kpi(g):
    late = g[~g.on_time]
    return pd.Series({"lines": len(g), "on_time_pct": g.on_time.mean(), "in_full_pct": g.in_full.mean(), "otif_pct": g.otif.mean(),
                      "avg_days_late": late.days_late.mean() if len(late) else 0.0, "short_units": g.short_units.sum(),
                      "gap_to_target_pts": (TARGET - g.otif.mean()) * 100})
overall = kpi(done)
print("   all lines: on time %.1f%% | in full %.1f%% | OTIF %.1f%% (target %.0f%%) | late lines average %.1f days late"
      % (overall.on_time_pct * 100, overall.in_full_pct * 100, overall.otif_pct * 100, TARGET * 100, overall.avg_days_late))
scorecard = done.groupby("supplier").apply(kpi).sort_values("otif_pct")
scorecard["meets_target"] = scorecard.otif_pct >= TARGET
print(scorecard[["lines", "on_time_pct", "in_full_pct", "otif_pct", "avg_days_late"]].round(3).to_string())
monthly = done.groupby("promised_month").apply(kpi)
by_cat = done.groupby("category").apply(kpi).sort_values("otif_pct")
by_year = done.groupby(done.promised_date.dt.year).apply(kpi)
print("   by year:", by_year.otif_pct.round(3).to_dict())
print("   share of missed lines: ", done[~done.otif].result.value_counts(normalize=True).round(3).to_dict())
miss = done[~done.otif]; k_share = (miss.supplier.isin(["Kestrel Ingredients", "Purelon Chemicals"])).mean()
print("   Kestrel Ingredients and Purelon Chemicals together cause %.0f%% of missed lines (they place %.0f%% of lines)"
      % (k_share * 100, done.supplier.isin(["Kestrel Ingredients", "Purelon Chemicals"]).mean() * 100))
q4 = done[done.promised_date.dt.month.isin([10, 11, 12])]; rest = done[~done.promised_date.dt.month.isin([10, 11, 12])]
print("   OTIF Oct-Dec %.1f%% vs rest of year %.1f%%" % (q4.otif.mean() * 100, rest.otif.mean() * 100))
open_po["days_overdue"] = (REPORT_DATE - open_po.promised_date).dt.days.clip(lower=0)
open_po["status"] = np.where(open_po.promised_date <= REPORT_DATE, "Open, overdue", "Open, due later")
open_po["line_value"] = open_po.ordered_qty * open_po.unit_cost
open_summary = open_po.groupby(["status", "supplier"]).agg(lines=("sku", "count"), value=("line_value", "sum"), max_days_overdue=("days_overdue", "max")).reset_index()
print("   open lines:", len(open_po), "| overdue:", int((open_po.status == "Open, overdue").sum()),
      "worth EUR %.0f" % open_po[open_po.status == "Open, overdue"].line_value.sum())

# ---------------------------------------------------------------- 4. SAVE OUTPUTS
print("4. SAVE OUTPUTS")
cols = ["po_number", "po_line", "sku", "description", "category", "supplier", "order_date", "promised_date", "received_date", "ordered_qty",
        "received_qty", "days_late", "on_time", "in_full", "otif", "result", "promised_month", "unit_cost"]
out = done[cols].copy()
for c in ["order_date", "promised_date", "received_date"]: out[c] = out[c].dt.strftime("%Y-%m-%d")
for c in ["on_time", "in_full", "otif"]: out[c] = out[c].astype(int)
out.to_csv("output/fact_po_lines.csv", index=False)
scorecard.round(4).to_csv("output/supplier_scorecard.csv")
monthly.round(4).to_csv("output/monthly_otif.csv"); by_cat.round(4).to_csv("output/category_summary.csv")
o = open_po[["po_number", "po_line", "sku", "description", "supplier", "order_date", "promised_date", "ordered_qty", "unit_cost", "status", "days_overdue"]].copy()
for c in ["order_date", "promised_date"]: o[c] = o[c].dt.strftime("%Y-%m-%d")
o.sort_values(["status", "days_overdue"], ascending=[False, False]).to_csv("output/open_po_lines.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(monthly.index, monthly.otif_pct * 100, color="#257de0", lw=2); ax[0].axhline(TARGET * 100, color="#f47230", ls="--")
ax[0].set_xticks(monthly.index[::6]); ax[0].set_title("OTIF % by promised month (target 95%)")
ax[1].barh(scorecard.index, scorecard.otif_pct * 100, color="#00b894"); ax[1].axvline(TARGET * 100, color="#f47230", ls="--"); ax[1].set_title("OTIF % by supplier")
plt.tight_layout(); plt.savefig("output/otif.png", dpi=130)
print("Done. Files are in the output folder.")
