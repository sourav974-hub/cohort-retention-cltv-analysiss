# Project 2: SaaS/E-Commerce Cohort Retention & CLTV Analysis

## Overview
This project analyzes a synthetic transaction log for a SaaS/e-commerce business to
answer two questions: **when do users churn**, and **which acquisition channels and
regions produce the most valuable long-term customers**.

> Note: `raw_transactions.csv` and `users_reference.csv` are synthetic data generated
> by `generate_data.py` for demonstration purposes (per the assignment's data-privacy
> rule, real datasets should never be committed to GitHub — link or regenerate instead).

## How to run
```bash
pip install pandas numpy matplotlib seaborn
python generate_data.py       # creates raw_transactions.csv
python cohort_analysis.py     # cleans data, builds cohort matrix, calculates CLTV, plots charts
```

## Pipeline

**Week 1 — Data Cleaning:** Removed duplicate transaction rows, dropped rows with
missing `user_id`, filtered out refunded transactions, and derived each user's
"Cohort Month" (month of first purchase).

**Week 2 — Cohort Retention Matrix:** Used `groupby` + `pivot_table` to build a
matrix of retained users by cohort month × months-since-acquisition, then converted
to a retention percentage relative to each cohort's Month 0 size.

**Week 3 — CLTV Calculation:** Computed Average Order Value (AOV) and purchase
frequency per user, then estimated 12-month CLTV = AOV × monthly purchase frequency
× 12, segmented by acquisition channel and region.

**Week 4 — Visualization:** Retention heatmap, average retention decay curve, and
CLTV-by-channel bar chart (see `.png` files in this repo).

## Key Findings

- **Retention drops sharply in Month 1** — cohorts lose roughly **21 percentage
  points** of users between Month 0 and Month 1, and a further **~9 points** by
  Month 2. This is the single biggest drop-off point in the customer lifecycle.
  **Recommendation:** implement an automated re-engagement email/push sequence
  targeted at users approaching the end of their first month.

- **Organic Search and Referral users have the highest 12-month CLTV** (~$1,008
  and ~$997 respectively), while **Paid Social users have the lowest** (~$967),
  despite Paid Social likely having the highest acquisition cost per user.
  **Recommendation:** the Finance Director should weigh CAC against these CLTV
  figures — Paid Social may need a lower CAC ceiling to stay profitable, while
  budget could be shifted toward Referral incentive programs, which show both
  strong retention and strong lifetime value.

- **Retention stabilizes after Month 4–5** in most cohorts (the decay curve
  flattens), suggesting that users who survive the first ~4 months are meaningfully
  more likely to become long-term, high-value customers — a natural point for a
  loyalty or upsell campaign.

## Challenges Faced

- **Handling messy real-world-style data.** The synthetic dataset intentionally
  includes duplicate rows and missing `user_id` values (mimicking double-fired
  tracking pixels and broken joins). Getting the cleaning order right mattered —
  dropping duplicates *before* deriving `cohort_month` avoids inflating any single
  user's transaction count, which would otherwise skew both retention and CLTV.

- **Choosing a CLTV formula appropriate for an MVP.** A full probabilistic CLTV
  model (e.g., BG/NBD + Gamma-Gamma) was out of scope for a 4-week roadmap, so a
  simpler historical run-rate model (`AOV × monthly purchase frequency × 12`) was
  used instead. This is a reasonable approximation for users with enough purchase
  history, but it's worth noting as a real limitation: it assumes recent behavior
  continues linearly, which will overstate CLTV for users who are actually about
  to churn.

- **Deciding what "retention" means at Month 0.** Every cohort is 100% retained
  at Month 0 by definition (it's their first purchase month), which can make the
  heatmap look artificially strong on the left edge. The decay curve chart was
  added specifically to make the *shape* of the drop-off after Month 0 easier to
  read than the heatmap alone.

- **Keeping the analysis reproducible without committing data.** Since raw
  `.csv` files can't be pushed to GitHub, the whole pipeline had to be
  script-driven and deterministic (`np.random.seed(42)` in `generate_data.py`)
  so that anyone re-running it gets the same numbers referenced in this README.

## Files
| File | Description |
|---|---|
| `generate_data.py` | Generates the synthetic raw transaction log |
| `cohort_analysis.py` | Full cleaning → cohort matrix → CLTV → visualization pipeline |
| `segment_channel_region.py` | Combined channel × region CLTV segmentation (deeper dive beyond single-dimension segments) |
| `week1_eda.ipynb` | Jupyter notebook version of the Week 1 exploratory data analysis |
| `requirements.txt` | Python dependencies needed to run the scripts |
| `retention_matrix_pct.csv` | Cohort retention % matrix |
| `user_cltv.csv` | Per-user CLTV, AOV, purchase frequency |
| `cltv_by_channel.csv`, `cltv_by_region.csv` | CLTV segmented by channel/region |
| `cltv_by_channel_and_region.csv` | CLTV segmented by channel AND region combined |
| `retention_heatmap.png` | Cohort retention heatmap |
| `retention_decay_curve.png` | Average retention decay curve |
| `cltv_by_channel.png` | CLTV by acquisition channel |
| `cltv_channel_region_heatmap.png` | CLTV by channel × region combined |

## Suggested `.gitignore`
```
data/
*.csv
.ipynb_checkpoints/
__pycache__/
*.sqlite
.env
```
(Per the internship's data-privacy rule, raw `.csv`/`.sqlite` files should not be
pushed to GitHub in a real submission — this README documents how to regenerate
them instead.)
