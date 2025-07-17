import numpy as np
from scipy.optimize import minimize

class FI:
    def __init__(self, Gamma_func, q_func, p_func, mu_func, f=0):
        self.Gamma = Gamma_func
        self.q = q_func
        self.p = p_func
        self.mu = mu_func
        self.f = f
        self.optimal_A = None
        self.optimal_m = None
        self.optimal_r = None
        self.s = 0
        self.r = 0
        self.A = 0
        self.profit = 0
        self.entry_decision = True

    def calculate_expected_profit_risk_neutral(self, A, s, r):
        gross_returns = self.mu(s) * s
        funding_cost = self.Gamma(s, A)
        bid_cost = s * r
        audit_penalty = self.q(s, A)
        base_profit = gross_returns - funding_cost - bid_cost - self.f
        expected_profit = (1 - self.p(s)) * base_profit + self.p(s) * (base_profit - audit_penalty)
        return expected_profit

    def calculate_optimal_action(self, s, r):
        def objective(A):
            return -self.calculate_expected_profit_risk_neutral(A, s, r)
        res = minimize(objective, x0=[0.5], bounds=[(0, 1)], method='L-BFGS-B')
        if res.success:
            self.optimal_A = res.x[0]
            return res.x[0]
        else:
            raise RuntimeError("Optimization of A failed")

    def calculate_optimal_bid(self):
        def objective(x):
            A, r, m = x
            if not (0 <= A <= 1 and r >= 0 and m >= 0):
                return np.inf
            s = m
            return -self.calculate_expected_profit_risk_neutral(A, s, r)
        x0 = [0.5, 0.05, 5]  # lower guess for m, firms weren't entering with high m
        #x0 = [0.5, 0.05, 10]
        bounds = [(0, 1), (0.01, 0.5), (1, 20)]  # cap r and m
        #bounds = [(0, 1), (0.01, 1.0), (1e-3, None)]
        res = minimize(objective, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.success:
            self.optimal_A = res.x[0]
            self.optimal_r = res.x[1]
            self.optimal_m = res.x[2]
            return (self.optimal_m, self.optimal_r)
        else:
            raise RuntimeError("Joint optimization of (A, r, m) failed")

    def calculate_optimal_entry(self):
        if self.optimal_A is None or self.optimal_m is None or self.optimal_r is None:
            self.calculate_optimal_bid()
        expected_profit = self.calculate_expected_profit_risk_neutral(
            self.optimal_A, self.optimal_m, self.optimal_r
        )
        self.entry_decision = expected_profit >= 0
        return self.entry_decision

    def set_auction_winnings(self, s, r):
        self.s = s
        self.r = r

    def calculate_profit(self):
        real_A = self.calculate_optimal_action(self.s, self.r)
        self.A = real_A
        self.profit = self.calculate_expected_profit_risk_neutral(self.A, self.s, self.r)
        return self.profit
