"""
generate_data.py
Creates a synthetic e-commerce/SaaS transaction log for cohort & CLTV analysis.
This simulates what would normally come from a production database export.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_USERS = 2000
START_DATE = pd.Timestamp("2025-01-01")
END_DATE = pd.Timestamp("2025-12-31")
CHANNELS = ["Organic Search", "Paid Social", "Referral", "Email", "Direct"]
REGIONS = ["North", "South", "East", "West"]

# --- Assign each user a signup date, channel, and region ---
signup_days = np.random.randint(0, (END_DATE - START_DATE).days, N_USERS)
signup_dates = START_DATE + pd.to_timedelta(signup_days, unit="D")

users = pd.DataFrame({
    "user_id": [f"U{i:05d}" for i in range(N_USERS)],
    "signup_date": signup_dates,
    "acquisition_channel": np.random.choice(CHANNELS, N_USERS, p=[0.35, 0.25, 0.15, 0.15, 0.10]),
    "region": np.random.choice(REGIONS, N_USERS),
})

# Give each user a "churn month" (how many months after signup they stop buying)
# Paid Social users churn faster (worse retention); Referral users churn slower.
channel_churn_bias = {
    "Organic Search": 6, "Paid Social": 3, "Referral": 9, "Email": 5, "Direct": 5
}
users["expected_lifetime_months"] = users["acquisition_channel"].map(channel_churn_bias)
users["lifetime_months"] = np.clip(
    np.random.poisson(users["expected_lifetime_months"]), 1, 12
)

# Average order value varies by region slightly, purchase frequency varies by channel
region_aov_bias = {"North": 1.1, "South": 0.9, "East": 1.0, "West": 1.05}
users["base_aov"] = np.random.gamma(shape=5, scale=8, size=N_USERS) * users["region"].map(region_aov_bias)
users["purchase_freq_per_month"] = np.clip(np.random.normal(2.2, 0.6, N_USERS), 0.5, 5)

# --- Generate transaction-level rows for each user across their lifetime ---
records = []
for _, u in users.iterrows():
    n_months_active = int(u["lifetime_months"])
    for m in range(n_months_active):
        month_date = u["signup_date"] + pd.DateOffset(months=m)
        if month_date > END_DATE:
            break
        n_purchases_this_month = np.random.poisson(u["purchase_freq_per_month"])
        for _ in range(n_purchases_this_month):
            day_offset = np.random.randint(0, 28)
            txn_date = month_date + pd.Timedelta(days=day_offset)
            amount = max(5, np.random.normal(u["base_aov"], u["base_aov"] * 0.2))
            status = np.random.choice(["completed", "refunded"], p=[0.95, 0.05])
            records.append({
                "user_id": u["user_id"],
                "transaction_date": txn_date,
                "amount": round(amount, 2),
                "status": status,
                "acquisition_channel": u["acquisition_channel"],
                "region": u["region"],
            })

transactions = pd.DataFrame(records)
transactions = transactions.merge(users[["user_id", "signup_date"]], on="user_id", how="left")

# Introduce some raw-data messiness (missing user_id, duplicate rows) — realistic EDA challenge
dupe_rows = transactions.sample(frac=0.01, random_state=1)
transactions = pd.concat([transactions, dupe_rows], ignore_index=True)
missing_idx = transactions.sample(frac=0.005, random_state=2).index
transactions.loc[missing_idx, "user_id"] = None

transactions.to_csv("raw_transactions.csv", index=False)
users.to_csv("users_reference.csv", index=False)

print(f"Generated {len(transactions):,} transaction rows for {N_USERS:,} users.")
print(transactions.head())
