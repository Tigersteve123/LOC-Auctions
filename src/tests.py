from run_auction import Auction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

seed = 42
np.random.seed(seed)

def make_mu_gamma_lists(N, correlation_type):
    base_returns = np.random.uniform(0.05, 0.2, size=N)  # c1 values
    mus = []
    Gammas = []

    if correlation_type == "positive_corr":
        # High return coefficient -> high cost coefficient
        cost_coeffs = base_returns * 2 + 0.1  # e.g., c2 = 2 * c1 + 0.1
    elif correlation_type == "negative_corr":
        # High return coefficient -> low cost coefficient
        cost_coeffs = 0.6 - 2 * base_returns  # c2 = decreasing in c1
    elif correlation_type == "no_corr":
        # Independent cost coefficients
        cost_coeffs = np.random.uniform(0.2, 0.5, size=N)
    else:
        raise ValueError("Unknown correlation_type")

    for c1, c2 in zip(base_returns, cost_coeffs):
        # Expected return function: mu(s) = c1 * ln(s + 1)
        mus.append(lambda s, c1=c1: c1 * np.log(s + 1))
        # Cost function: Gamma(s, A) = c2 * s * (exp(A) - 1)
        Gammas.append(lambda s, A, c2=c2: c2 * s * (np.exp(A) - 1))

    return mus, Gammas


def test_scenario(correlation_type, q_func, p_func, total_funds, rate_floor, N):
    mus, Gammas = make_mu_gamma_lists(N, correlation_type)
    ps = [p_func] * N

    auction = Auction(
        N=N,
        q_func=q_func,
        Gammas=Gammas,
        mus=mus,
        ps=ps,
        total_funds=total_funds,
        rate_floor=rate_floor,
        guess_rate_floors=True
    )
    auction.run_auction()
    return auction.get_results()


if __name__ == "__main__":
    N = 5
    total_funds = 100
    rate_floor = 0.01
    q_func = lambda s, A: 2 * (1 - A) * s
    p_func = lambda s: min(0.01 * s, 1)
    num_runs = 1000

    scenario = "positive_corr"  #change to "negative_corr" or "no_corr" for other scenarios

    print(f"Parameters:\nN={N}, total_funds={total_funds}, rate_floor={rate_floor}, scenario={scenario}")

    all_profits = []
    all_allocations = []

    skipped = 0
    for run in range(num_runs):
        np.random.seed(run) #ensure reproducibility
        df = pd.DataFrame(test_scenario("positive_corr", q_func, p_func, total_funds, rate_floor, N))

        if df.empty:
            skipped += 1
            continue #skip if no winners
        all_profits.extend(df["profit"].tolist())
        all_allocations.extend(df["allocated"].tolist())
    
    print(f"\nSkipped {skipped} auctions out of {num_runs} due to no qualifying bids.")

        

    #histogram of profits 
    plt.hist(all_profits, bins = 30, alpha = 0.7)
    plt.title("Profit Distribution")
    plt.xlabel("Profit")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    #histogram of allocations
    plt.hist(all_allocations, bins = 30, alpha = 0.7)
    plt.title("Allocation of Funds")
    plt.xlabel("Allocated Funds")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()



    # for scenario in ["positive_corr", "negative_corr", "no_corr"]:
    #     print(f"\n--- {scenario.upper()} ---")
    #     df = pd.DataFrame(test_scenario(scenario, q_func, p_func, total_funds, rate_floor, N))
    #     print(df)

