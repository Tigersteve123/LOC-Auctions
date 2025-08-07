from run_auction import Auction
from tests import test_scenario
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# fixed model setup
N = 10
total_funds = 100
rate_floor = 0.02
scenario = "positive_corr"
num_runs = 100

# fixed penalty function (linear form)
def q_func(s, A):
    return 2 * (1 - A) * s

# sweep over flat audit probabilities from 0.00 to 1.00 in steps of 0.05
audit_probs = np.arange(0.0, 1.05, 0.05)

results_summary = []

for audit_p in audit_probs:
    def p_func(s, p=audit_p):
        return p  # flat audit rate

    all_profits, all_allocs, all_As, all_rates = [], [], [], []

    for run in range(num_runs):
        np.random.seed(run)
        try:
            df = pd.DataFrame(test_scenario(scenario, q_func, p_func, total_funds, rate_floor, N))
            if df.empty:
                continue
            all_profits.append(df["profit"].mean())
            all_allocs.append(df["allocated"].sum())
            all_As.append(df["A"].mean())
            all_rates.append(df["rate"].mean())
        except Exception:
            continue

    if len(all_profits) == 0:
        continue

    results_summary.append({
        "AuditProb": audit_p,
        "AvgProfit": np.mean(all_profits),
        "AvgAllocation": np.mean(all_allocs),
        "AvgA": np.mean(all_As),
        "AvgRate": np.mean(all_rates)
    })

# Create results dataframe
summary_df = pd.DataFrame(results_summary)
print("\n=== Audit Probability Sweep Summary ===")
print(summary_df)

# Optional: Save to CSV
# summary_df.to_csv("audit_prob_sweep_summary.csv", index=False)

# plotting results
outcomes = ["AvgProfit", "AvgAllocation", "AvgA", "AvgRate"]

for outcome in outcomes:
    plt.figure(figsize=(7, 5))
    plt.plot(summary_df["AuditProb"], summary_df[outcome], marker='o')
    plt.title(f"{outcome} vs Audit Probability")
    plt.xlabel("Audit Probability")
    plt.ylabel(outcome)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
