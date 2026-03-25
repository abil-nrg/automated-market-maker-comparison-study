import numpy as np
from .agents import NoisyAgent, InformedAgent

class Market:
    def __init__(self, amm, T, true_p, noisy_ratio):
        self.amm = amm
        self.T = T
        self.true_p = true_p
        self.noisy_ratio = noisy_ratio
        self.n = len(true_p)

    def run(self):
        price_hist = []
        spread_hist = []
        profit_hist = []

        for _ in range(self.T):
            p = self.amm.get_prices()

            # spread (for outcome 0)
            r = np.zeros(self.n)
            r[0] = 1
            spread = self.amm.get_spread(r)

            # agent
            if np.random.rand() < self.noisy_ratio:
                agent = NoisyAgent()
            else:
                agent = InformedAgent(self.true_p)

            dq = agent.get_trade(p, self.n)
            if np.any(dq != 0):
                self.amm.execute_trade(dq)

            # profit tracking
            expected_liability = np.sum(self.amm.q * self.true_p)
            profit = self.amm.collected_cash - expected_liability

            price_hist.append(p.copy())
            spread_hist.append(spread)
            profit_hist.append(profit)

        # settle
        winner = np.random.choice(self.n, p=self.true_p)
        realized = self.amm.collected_cash - self.amm.q[winner]

        return (
            np.array(price_hist),
            np.array(spread_hist),
            np.array(profit_hist),
            realized
        )