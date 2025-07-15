from fi import FI

class Auction:
    def __init__(self, N, q_func, Gammas, mus, ps, total_funds, rate_floor):
        assert len(Gammas) == len(mus) == len(ps) == N
        self.FIs = [
            FI(Gammas[i], q_func, ps[i], mus[i])
            for i in range(N)
        ]
        self.q_func = q_func
        self.total_funds = total_funds
        self.rate_floor = rate_floor
        self.auction_participants = []
        self.winners = []

    def run_auction(self):
        self.auction_participants.clear()
        self.winners.clear()

        # entry decisions 
        print("\n--- Entry Decisions ---")
        for i, fi in enumerate(self.FIs):
            if fi.calculate_optimal_entry():
                print(f"FI {i}: ENTERED with m={fi.optimal_m:.2f}, r={fi.optimal_r:.4f}, A={fi.optimal_A:.2f}")
                self.auction_participants.append(fi)
            else:
                print(f"FI {i}: DID NOT ENTER (Expected profit < 0)")

        # sort bids
        self.auction_participants.sort(
            key=lambda fi: (fi.optimal_r, fi.optimal_m), reverse=True
        )

        # fund allocation
        remaining_funds = self.total_funds
        for fi in self.auction_participants:
            if fi.optimal_r < self.rate_floor:
                continue
            if remaining_funds <= 0:
                break
            allocated = min(fi.optimal_m, remaining_funds)
            fi.set_auction_winnings(s=allocated, r=fi.optimal_r)
            fi.calculate_profit()
            self.winners.append(fi)
            remaining_funds -= allocated

    def get_results(self):
        return [{
            "allocated": fi.s,
            "rate": fi.r,
            "A": fi.A,
            "profit": fi.profit
        } for fi in self.winners]
