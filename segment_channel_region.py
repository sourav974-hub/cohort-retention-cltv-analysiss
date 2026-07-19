"""
segment_channel_region.py
Day 5 addition: Combined channel x region CLTV segmentation.

The Week 3 analysis in cohort_analysis.py segments CLTV by channel and by
region SEPARATELY. This script goes one level deeper and asks: does the
*combination* of channel and region reveal patterns that get averaged away
when you only look at one dimension at a time?

Run after cohort_analysis.py (it reads user_cltv.csv, which that script produces).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

user_revenue = pd.read_csv("user_cltv.csv")

# Group by BOTH acquisition_channel and region simultaneously, rather than
# one at a time. This can surface segments that look average in the
# single-dimension view but are actually a standout (good or bad) combination.
combined = user_revenue.groupby(["acquisition_channel", "region"])["cltv_12mo"].agg(
    ["mean", "count"]
).reset_index()

# Reshape into a channel x region matrix for the heatmap.
pivot = combined.pivot(index="acquisition_channel", columns="region", values="mean")
pivot.to_csv("cltv_by_channel_and_region.csv")

fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(
    pivot, annot=True, fmt=".0f", cmap="Greens", ax=ax,
    cbar_kws={"label": "Avg 12-mo CLTV"}
)
ax.set_title("Average 12-Month CLTV by Channel x Region", fontsize=13, fontweight="bold")
ax.set_xlabel("Region")
ax.set_ylabel("Acquisition Channel")
plt.tight_layout()
plt.savefig("cltv_channel_region_heatmap.png", dpi=150)
plt.close()

best = combined.loc[combined["mean"].idxmax()]
worst = combined.loc[combined["mean"].idxmin()]

print("CLTV by channel x region:")
print(pivot.round(2))
print(f"\nBest combination: {best['acquisition_channel']} x {best['region']} "
      f"(${best['mean']:.2f}, n={int(best['count'])})")
print(f"Worst combination: {worst['acquisition_channel']} x {worst['region']} "
      f"(${worst['mean']:.2f}, n={int(worst['count'])})")
