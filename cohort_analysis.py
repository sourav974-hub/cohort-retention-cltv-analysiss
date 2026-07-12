"""
cohort_analysis.py
Project 2: SaaS/E-Commerce Cohort Retention & CLTV Analysis
------------------------------------------------------------
Week 1: Data cleaning (dedup, missing IDs, refund filtering, cohort month)
Week 2: Cohort retention matrix (Pandas pivot_table)
Week 3: CLTV calculation by cohort/channel/region
Week 4: Visualization (retention heatmap, decay curves) + insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# ----------------------------------------------------------------------
# WEEK 1: DATA CLEANING AND WRANGLING
# ----------------------------------------------------------------------
df = pd.read_csv("raw_transactions.csv", parse_dates=["transaction_date", "signup_date"])

print("Raw rows:", len(df))

# 1. Drop exact duplicate transaction rows
df = df.drop_duplicates()

# 2. Drop rows with missing user_id (can't attribute the transaction to a cohort)
df = df.dropna(subset=["user_id"])

# 3. Filter out refunded/failed transactions — they don't represent real revenue
df = df[df["status"] == "completed"].copy()

print("Cleaned rows:", len(df))

# 4. Calculate "Cohort Month" = month of each user's first transaction
df["transaction_month"] = df["transaction_date"].dt.to_period("M")
df["cohort_month"] = df.groupby("user_id")["transaction_month"].transform("min")

# 5. Cohort index = number of months between transaction and cohort month
def month_diff(row):
    return (row["transaction_month"].year - row["cohort_month"].year) * 12 + \
           (row["transaction_month"].month - row["cohort_month"].month)

df["cohort_index"] = df.apply(month_diff, axis=1)

df.to_csv("clean_transactions.csv", index=False)

# ----------------------------------------------------------------------
# WEEK 2: COHORT RETENTION MATRIX
# ----------------------------------------------------------------------
cohort_data = df.groupby(["cohort_month", "cohort_index"])["user_id"].nunique().reset_index()
cohort_pivot = cohort_data.pivot(index="cohort_month", columns="cohort_index", values="user_id")

cohort_sizes = cohort_pivot.iloc[:, 0]
retention_matrix = cohort_pivot.divide(cohort_sizes, axis=0) * 100

retention_matrix.to_csv("retention_matrix_pct.csv")
cohort_pivot.to_csv("retention_matrix_counts.csv")

# Average retention decay curve across all cohorts (Month 0 - Month 6)
avg_retention = retention_matrix.iloc[:, 0:7].mean(axis=0)

# ----------------------------------------------------------------------
# WEEK 3: CLTV CALCULATION
# ----------------------------------------------------------------------
user_revenue = df.groupby("user_id").agg(
    total_revenue=("amount", "sum"),
    n_purchases=("amount", "count"),
    acquisition_channel=("acquisition_channel", "first"),
    region=("region", "first"),
    cohort_month=("cohort_month", "first"),
    active_months=("cohort_index", "max"),
).reset_index()

user_revenue["aov"] = user_revenue["total_revenue"] / user_revenue["n_purchases"]
user_revenue["active_months"] = user_revenue["active_months"] + 1  # inclusive of month 0
user_revenue["purchase_freq_per_month"] = user_revenue["n_purchases"] / user_revenue["active_months"]

# Historical CLTV (observed, 12-month capped) = AOV x monthly purchase freq x 12 months
user_revenue["cltv_12mo"] = (
    user_revenue["aov"] * user_revenue["purchase_freq_per_month"] * 12
)

cltv_by_channel = user_revenue.groupby("acquisition_channel")["cltv_12mo"].agg(
    ["mean", "median", "count"]
).sort_values("mean", ascending=False)

cltv_by_region = user_revenue.groupby("region")["cltv_12mo"].agg(
    ["mean", "median", "count"]
).sort_values("mean", ascending=False)

user_revenue.to_csv("user_cltv.csv", index=False)
cltv_by_channel.to_csv("cltv_by_channel.csv")
cltv_by_region.to_csv("cltv_by_region.csv")

# ----------------------------------------------------------------------
# WEEK 4: VISUALIZATION
# ----------------------------------------------------------------------

# --- Heatmap ---
fig, ax = plt.subplots(figsize=(13, 8))
heatmap_data = retention_matrix.iloc[:, 0:9]  # Month 0 to Month 8
heatmap_data.index = heatmap_data.index.astype(str)
sns.heatmap(
    heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu",
    vmin=0, vmax=100, cbar_kws={"label": "Retention %"}, ax=ax
)
ax.set_title("Monthly Cohort Retention Heatmap (%)", fontsize=14, fontweight="bold")
ax.set_xlabel("Months Since Acquisition")
ax.set_ylabel("Acquisition Cohort (Month)")
plt.tight_layout()
plt.savefig("retention_heatmap.png", dpi=150)
plt.close()

# --- Decay curve ---
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(avg_retention.index, avg_retention.values, marker="o", linewidth=2, color="#2b6cb0")
ax.set_title("Average Retention Decay Curve (All Cohorts)", fontsize=14, fontweight="bold")
ax.set_xlabel("Months Since Acquisition")
ax.set_ylabel("Average Retention (%)")
ax.set_ylim(0, 100)
for x, y in zip(avg_retention.index, avg_retention.values):
    ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 8), ha="center")
plt.tight_layout()
plt.savefig("retention_decay_curve.png", dpi=150)
plt.close()

# --- CLTV by channel bar chart ---
fig, ax = plt.subplots(figsize=(9, 6))
cltv_by_channel["mean"].sort_values().plot(kind="barh", ax=ax, color="#38a169")
ax.set_title("Average 12-Month CLTV by Acquisition Channel", fontsize=14, fontweight="bold")
ax.set_xlabel("CLTV (₹ / $ equivalent)")
plt.tight_layout()
plt.savefig("cltv_by_channel.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# PRINT KEY INSIGHTS
# ----------------------------------------------------------------------
print("\n=== KEY INSIGHTS ===")
print(f"Month 0 -> Month 1 average retention drop: {avg_retention[0] - avg_retention[1]:.1f} pts")
print(f"Month 1 -> Month 2 average retention drop: {avg_retention[1] - avg_retention[2]:.1f} pts")
print("\nTop CLTV channel:", cltv_by_channel['mean'].idxmax(), f"(${cltv_by_channel['mean'].max():.2f})")
print("Bottom CLTV channel:", cltv_by_channel['mean'].idxmin(), f"(${cltv_by_channel['mean'].min():.2f})")
print("\nRetention matrix (%):")
print(retention_matrix.iloc[:, 0:7].round(1))
print("\nCLTV by channel:")
print(cltv_by_channel.round(2))
