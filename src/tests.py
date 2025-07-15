from run_auction import Auction
import numpy as np
import pandas as pd

def make_mu_gamma_funcs(correlation_type):
    # boost return and reduce cost
    if correlation_type == "high_corr":
        return (
            lambda s: 0.25,                      # flat 25% return
            lambda s, A: 0.2 * s * (1 + A)       # cheaper cost of compliance
        )
    elif correlation_type == "low_corr":
        return (
            lambda s: 0.20 + 0.005 * s,          # slightly rising return
            lambda s, A: 0.2 * s * (1 - 0.5 * A) # less penalty from A
        )
    elif correlation_type == "no_corr":
        return (
            lambda s: 0.22,                      # flat-ish return
            lambda s, A: 0.15 * s * (1 + 0.25 * A)
        )



def test_scenario(correlation_type):
    N = 5
    
    total_funds = 100
    rate_floor = 0.01  # loosened floor to let more bids qualify

    mu_func, Gamma_func = make_mu_gamma_funcs(correlation_type)

    # gentler audit and penalty settings
    q_func = lambda s, A: 2 * (1 - A) * s
    p_func = lambda s: 0.1

    auction = Auction(
        N=N,
        q_func=q_func,
        Gammas=[Gamma_func] * N,
        mus=[mu_func] * N,
        ps=[p_func] * N,
        total_funds=total_funds,
        rate_floor=rate_floor
    )
    auction.run_auction()
    return auction.get_results()

if __name__ == "__main__":
    print("HIGH CORRELATION:")
    print(pd.DataFrame(test_scenario("high_corr")))

    print("\nLOW CORRELATION:")
    print(pd.DataFrame(test_scenario("low_corr")))

    print("\nNO CORRELATION:")
    print(pd.DataFrame(test_scenario("no_corr")))
