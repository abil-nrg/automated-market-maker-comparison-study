import numpy as np

class InformedAgent:
    def __init__(self, true_p, confidence=0.5, threshold=0.02):
        self.true_p = true_p
        self.confidence = confidence #std 
        self.threshold = threshold

    def get_trade(self, current_prices, n):
        # probability with some noise
        perceived = self.true_p + np.random.normal(0, self.confidence, n)
        perceived = np.clip(perceived, 0.01, 0.99)
        perceived /= perceived.sum()

        diff = perceived - current_prices
        trade = np.zeros(n)

        best_idx = np.argmax(np.abs(diff))
        max_diff = diff[best_idx]

        # if under/over valued beyond thresh, buy/sell 1 unit
        if np.abs(max_diff) > self.threshold:
            trade[best_idx] = 1.0 if max_diff > 0 else -1.0
        return trade

class NoisyAgent:
    def get_trade(self, current_prices, n_outcomes):
        trade = np.zeros(n_outcomes)
        idx = np.random.randint(0, n_outcomes)
        # Randomly buy or sell a small amount
        trade[idx] = np.random.uniform(-0.5, 0.5)
        return trade