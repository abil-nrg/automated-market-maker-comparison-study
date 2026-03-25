import numpy as np

class MarketMaker:
    def  __init__(self, n):
        self.q = np.zeros(n)
        self.collected_cash = 0.0

    def price_bundle(self):
        raise NotImplementedError
    
    def execute_trade(self, r):
        cost = self.price_bundle(r)
        self.q += r
        self.collected_cash += cost
        return cost
    
    def get_spread(self, r):
        ask = self.price_bundle(r)
        bid = - self.price_bundle(-r)
        return ask - bid

class LMSR(MarketMaker):
    def __init__(self, n, b):
        super().__init__(n)
        self.b = b

    def cost(self, q_state):
        q_scaled = q_state / self.b
        m = np.max(q_scaled)
        return self.b * (m + np.log(np.sum(np.exp(q_scaled - m)))) 
    
    def price_bundle(self, r):
        return self.cost(self.q + r) - self.cost(self.q) 
    
    def get_prices(self): #instantenous price
        exps = np.exp(self.q / self.b)
        return exps/np.sum(exps)
    

    
class CFMM(MarketMaker):
    def __init__(self, n, k_constant=1000.0):
        super().__init__(n)
        self.k_constant = k_constant

    def solve_c(self, q_vector):
        """
        Use Newton-Raphson to get the roots of a polynomial
        is convex so will be quick!
        """
        c = np.max(q_vector) + self.k_constant**(1/len(q_vector))
        for _ in range(20):
            terms = c - q_vector
            f = np.prod(terms) - self.k_constant
            prod_terms = np.prod(terms)
            df = np.sum(prod_terms / terms)
            c_new = c - f / df
            if abs(c_new - c) < 1e-8:
                return c_new
            c = c_new
        return c

    def price_bundle(self, r):
        c_now = self.solve_c(self.q)
        c_next = self.solve_c(self.q + r)
        return c_next - c_now

    def get_prices(self):
        c = self.solve_c(self.q)
        inv_terms = 1.0 / (c - self.q)
        return inv_terms / np.sum(inv_terms)