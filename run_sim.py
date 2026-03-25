from MarketSim import LMSR, CFMM, InformedAgent, NoisyAgent, Market


n_outcomes = 2
lmsr_b = 50
cfmm_k = 1000
T = 1000 #num of time steps


lmsr_amm = LMSR(n_outcomes, b=lmsr_b)
cfmm_amm = CFMM(n_outcomes, k_constant=cfmm_k)

agent_noisy = NoisyAgent()
agent_informed = InformedAgent()

market = Market(n_outcomes)

market.run()

