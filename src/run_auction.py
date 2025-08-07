import numpy as np

from fi import FI

class Auction:
    def __init__(self, N, q_func, Gammas, mus, ps, total_funds, rate_floor, guess_rate_floors=False):
        r_lows = [0] * N
        if guess_rate_floors:
            r_lows = [rate_floor + np.random.normal(scale=.01) for i in range(N)]
        assert len(Gammas) == len(mus) == len(ps) == N
        self.FIs = [
            FI(Gammas[i], q_func, ps[i], mus[i], r_lows[i])
            for i in range(N)
        ]
        self.q_func = q_func
        self.total_funds = total_funds
        self.rate_floor = rate_floor
        self.auction_participants = []
        self.winners = []

    def run_auction(self, print_results=False):
        self.auction_participants.clear()
        self.winners.clear()

        if print_results: print("\n--- Entry Decisions ---")
        for i, fi in enumerate(self.FIs):
            if fi.calculate_optimal_entry():
                if print_results: print(f"FI {i}: ENTERED with m={fi.optimal_m:.2f}, r={fi.optimal_r:.4f}, A={fi.optimal_A:.2f}")
                self.auction_participants.append(fi)
            else:
                if print_results: print(f"FI {i}: DID NOT ENTER (Expected profit < 0)")

        # Filter bids at or above rate floor
        eligible_bids = [
            fi for fi in self.auction_participants if fi.optimal_r >= self.rate_floor
        ]

        # Sort by rate descending, then m descending
        eligible_bids.sort(key=lambda fi: (fi.optimal_r, fi.optimal_m), reverse=True)

        # Determine winners to satisfy total funds
        allocated_funds = 0
        clearing_bids = []
        for fi in eligible_bids:
            if allocated_funds >= self.total_funds:
                break
            clearing_bids.append(fi)
            allocated_funds += fi.optimal_m

        # No winners? Bail out
        if not clearing_bids:
            if print_results: print("\nNo qualifying bids met the rate floor.")
            return

        # Determine uniform clearing rate: lowest rate among winners
        clearing_rate = min(fi.optimal_r for fi in clearing_bids)

        # Allocate funds
        remaining_funds = self.total_funds
        for fi in clearing_bids:
            allocated = min(fi.optimal_m, remaining_funds)
            fi.set_auction_winnings(s=allocated, r=clearing_rate)  # use clearing rate
            fi.calculate_profit()
            self.winners.append(fi)
            remaining_funds -= allocated
            if remaining_funds <= 0:
                break

    def get_results(self):
        return [{
            "allocated": fi.s,
            "rate": fi.r,
            "A": fi.A,
            "profit": fi.profit
        } for fi in self.winners]
