
from run_auction import Auction
from tests import test_scenario
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

# parameter sweep range
N_values = [5, 10]
total_funds_values = [50, 100]
rate_floor_values = [0.01, 0.05]

# q and p functions
def q_func_linear(s, A): return 2 * (1 - A) * s
def q_func_quadratic(s, A): return 2 * (1 - A)**2 * s
def p_func_linear(s): return min(0.01 * s, 1)
def p_func_flat(s): return 0.05

q_funcs = [("linear", q_func_linear), ("quadratic", q_func_quadratic)]
p_funcs = [("linear", p_func_linear), ("flat", p_func_flat)]

# simulation settings 
scenario = "positive_corr"
num_runs = 100
results_summary = []

# parameter sweep 
for N, funds, floor, (q_label, q_func), (p_label, p_func) in product(N_values, total_funds_values, rate_floor_values, q_funcs, p_funcs):
    all_profits, all_allocs, all_As, all_rates = [], [], [], []

    for run in range(num_runs):
        np.random.seed(run)
        try:
            df = pd.DataFrame(test_scenario(scenario, q_func, p_func, funds, floor, N))
            if df.empty:
                continue
            all_profits.append(df["profit"].mean())
            all_allocs.append(df["allocated"].sum())
            all_As.append(df["A"].mean())
            all_rates.append(df["rate"].mean())
        except Exception as e:
            continue

    if len(all_profits) == 0:
        continue

    results_summary.append({
        "N": N,
        "TotalFunds": funds,
        "RateFloor": floor,
        "QFunc": q_label,
        "PFunc": p_label,
        "AvgProfit": np.mean(all_profits),
        "AvgAllocation": np.mean(all_allocs),
        "AvgA": np.mean(all_As),
        "AvgRate": np.mean(all_rates)
    })

# combined dataframe
summary_df = pd.DataFrame(results_summary)
print("\n=== Parameter Sweep Summary ===")
print(summary_df)

# save results to CSV
# summary_df.to_csv("auction_grid_summary.csv", index=False)

# plotting outcomes vs parameters
outcomes = ["AvgProfit", "AvgAllocation", "AvgA", "AvgRate"]
parameters = ["N", "TotalFunds", "RateFloor"]

for outcome in outcomes:
    for param in parameters:
        plt.figure(figsize=(6, 4))
        for name, group in summary_df.groupby("RateFloor"):
            plt.scatter(group[param], group[outcome], label=f'RateFloor={name}', alpha=0.7)
        plt.title(f"{outcome} vs {param}")
        plt.xlabel(param)
        plt.ylabel(outcome)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
