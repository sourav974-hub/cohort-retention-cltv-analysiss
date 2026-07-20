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

# 1. Drop exact duplicate transaction rows.
#    In a real pipeline these usually come from double-fired webhooks or
#    re-processed batch jobs. Leaving them in would double-count revenue
#    and inflate both retention counts and CLTV.
df = df.drop_duplicates()

# 2. Drop rows with missing user_id.
#    A transaction we can't tie to a user is useless for cohort analysis —
#    we have no way to know which acquisition month it belongs to, so we
#    can't safely impute it either. Safer to drop than guess.
df = df.dropna(subset=["user_id"])

# 3. Filter out refunded/failed transactions.
#    Refunds represent revenue that was reversed — including them would
#    overstate both the CLTV and the apparent purchase frequency of users
#    who later got refunded.
df = df[df["status"] == "completed"].copy()

print("Cleaned rows:", len(df))

# 4. Calculate "Cohort Month" = the month of each user's FIRST transaction.
#    This is the anchor date every other calculation is relative to —
#    it's what lets us ask "how many months after joining did this user
#    still purchase?" instead of just looking at raw calendar months.
df["transaction_month"] = df["transaction_date"].dt.to_period("M")
df["cohort_month"] = df.groupby("user_id")["transaction_month"].transform("min")

# 5. Cohort index = number of months between a transaction and the user's
#    cohort month. Month 0 = the signup/first-purchase month itself,
#    Month 1 = one month later, etc. This is the x-axis of the retention
#    matrix built in Week 2.
#    NOTE: computed as a vectorized operation on the Period columns rather
#    than a row-wise .apply() — this scales much better as the transaction
#    log grows, since .apply() effectively loops over every row in Python.
df["cohort_index"] = (
    (df["transaction_month"].dt.year - df["cohort_month"].dt.year) * 12
    + (df["transaction_month"].dt.month - df["cohort_month"].dt.month)
)

df.to_csv("clean_transactions.csv", index=False)

# ----------------------------------------------------------------------
# WEEK 2: COHORT RETENTION MATRIX
# ----------------------------------------------------------------------
# Count how many DISTINCT users from each cohort_month are still
# transacting at each cohort_index (0, 1, 2, ... months later).
cohort_data = df.groupby(["cohort_month", "cohort_index"])["user_id"].nunique().reset_index()

# Pivot into a matrix: rows = cohort month, columns = months since acquisition,
# values = number of users still active. This is the "raw counts" version.
cohort_pivot = cohort_data.pivot(index="cohort_month", columns="cohort_index", values="user_id")

# Convert counts into percentages relative to each cohort's own Month 0 size.
# This is what makes cohorts of very different sizes comparable — e.g. a
# cohort of 50 users and a cohort of 500 users can both be read as "% retained".
cohort_sizes = cohort_pivot.iloc[:, 0]
retention_matrix = cohort_pivot.divide(cohort_sizes, axis=0) * 100

retention_matrix.to_csv("retention_matrix_pct.csv")
cohort_pivot.to_csv("retention_matrix_counts.csv")

# Average retention decay curve across all cohorts (Month 0 - Month 6).
# Averaging smooths out noise from any single cohort and shows the
# "typical" churn curve, which is what the Week 4 decay chart plots.
avg_retention = retention_matrix.iloc[:, 0:7].mean(axis=0)

# ----------------------------------------------------------------------
# WEEK 3: CLTV CALCULATION
# ----------------------------------------------------------------------
# Collapse the transaction-level data down to one row per user with the
# aggregates we need for a lifetime-value estimate.
user_revenue = df.groupby("user_id").agg(
    total_revenue=("amount", "sum"),
    n_purchases=("amount", "count"),
    acquisition_channel=("acquisition_channel", "first"),
    region=("region", "first"),
    cohort_month=("cohort_month", "first"),
    active_months=("cohort_index", "max"),
).reset_index()

# Average Order Value: how much does this user spend per purchase, on average?
user_revenue["aov"] = user_revenue["total_revenue"] / user_revenue["n_purchases"]

# +1 because cohort_index is zero-based (Month 0 counts as "1 active month").
user_revenue["active_months"] = user_revenue["active_months"] + 1

# How many purchases does this user make per active month, on average?
user_revenue["purchase_freq_per_month"] = user_revenue["n_purchases"] / user_revenue["active_months"]

# Historical CLTV projection: if this user keeps buying at their observed
# AOV and frequency for a full 12 months, what's their total value?
# This is a simple historical-run-rate model (not a probabilistic/BG-NBD
# model) — appropriate for an MVP, but worth noting as a limitation if this
# were a production system, since it assumes past behavior continues linearly.
user_revenue["cltv_12mo"] = (
    user_revenue["aov"] * user_revenue["purchase_freq_per_month"] * 12
)

# Segment CLTV by channel and region so the Finance Director can compare
# "value of a customer" against "cost to acquire a customer" (CAC) per segment.
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

# --- Heatmap: the core "at-a-glance" deliverable for the Product Manager ---
# Darker/higher values = more retained users. Reading down a column shows
# whether retention at a given month-since-acquisition is improving or
# worsening across newer cohorts (a sign of product changes taking effect).
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

# --- Decay curve: the "single number" summary of the heatmap ---
# Useful for exec reporting where a full heatmap is too dense — this answers
# "on average, how much of a cohort do we still have after N months?"
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

# --- CLTV by channel bar chart: for the Finance Director's CAC comparison ---
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
