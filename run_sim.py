import matplotlib
matplotlib.use('Agg') 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from MarketSim import Market, LMSR, CFMM
import time

def run_sweep(n_runs=100, T=500, noise_ratio=0.8):
    n_values = [2, 4, 8, 16, 32, 50, 64] 
    b_values = [25, 50, 100, 200, 400, 800]
    
    sweep_results = []

    start_sim_time = time.time()

    for n in n_values:
        print(f"Running sim for n={n}")
        true_p = np.ones(n) / n
        true_p[0] += 0.2
        true_p /= true_p.sum()

        for b in b_values:
            k_equiv = b**n
            
            lmsr_pnls, cfmm_pnls = [], []
            lmsr_spreads, cfmm_spreads = [], []

            for _ in range(n_runs):
                m1 = Market(LMSR(n, b), T, true_p, noise_ratio)
                m2 = Market(CFMM(n, k_equiv), T, true_p, noise_ratio)

                _, s1, _, r1 = m1.run()
                _, s2, _, r2 = m2.run()

                lmsr_pnls.append(r1)
                cfmm_pnls.append(r2)
                lmsr_spreads.append(np.mean(s1))
                cfmm_spreads.append(np.mean(s2))

            theory_lmsr = -b * np.log(n)
            theory_cfmm = -b 

            sweep_results.append({
                'n': n, 'b': b, 'k': k_equiv,
                'lmsr_mean': np.mean(lmsr_pnls),
                'cfmm_mean': np.mean(cfmm_pnls),
                'lmsr_spread_avg': np.mean(lmsr_spreads),
                'cfmm_spread_avg': np.mean(cfmm_spreads),
                'lmsr_spread_std': np.std(lmsr_spreads), 
                'cfmm_spread_std': np.std(cfmm_spreads),
                'lmsr_spread_raw': lmsr_spreads, 
                'cfmm_spread_raw': cfmm_spreads,
                'theory_lmsr_limit': theory_lmsr,
                'theory_cfmm_limit': theory_cfmm
            })
        
        print(f"n={n} complete. Total elapsed: {time.time() - start_sim_time:.2f} seconds")

    return pd.DataFrame(sweep_results)

start_time = time.time()
df = run_sweep(n_runs=50)

# PLOT
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# PnL vs Theoretical Limits (n=4)
n_ref = 4
subset = df[df['n'] == n_ref].sort_values('b')
ax[0].plot(subset['b'], subset['lmsr_mean'], 'o-', label='LMSR Mean PnL', color='blue')
ax[0].fill_between(subset['b'], subset['theory_lmsr_limit'], 0, alpha=0.1, color='blue', label='LMSR Risk Zone')
ax[0].plot(subset['b'], subset['cfmm_mean'], 's-', label='CFMM Mean PnL', color='orange')
ax[0].fill_between(subset['b'], subset['theory_cfmm_limit'], 0, alpha=0.1, color='orange', label='CFMM Risk Zone')
ax[0].axhline(0, color='black', lw=1)
ax[0].set_title(f"PnL vs. Risk Bounds (n={n_ref})")
ax[0].set_xlabel("Liquidity (b)")
ax[0].set_ylabel("Profit and Loss")
ax[0].legend()

# Avg Spread with 1-std band (Fixed b=100)
b_ref = 100
subset_n = df[df['b'] == b_ref].sort_values('n')

# LMSR Averages and Band
ax[1].plot(subset_n['n'], subset_n['lmsr_spread_avg'], 'o-', label='LMSR Avg Spread', color='blue')
ax[1].fill_between(subset_n['n'], 
                 subset_n['lmsr_spread_avg'] - subset_n['lmsr_spread_std'], 
                 subset_n['lmsr_spread_avg'] + subset_n['lmsr_spread_std'], 
                 alpha=0.2, color='blue')

# CFMM Averages and Band
ax[1].plot(subset_n['n'], subset_n['cfmm_spread_avg'], 's-', label='CFMM Avg Spread', color='orange')
ax[1].fill_between(subset_n['n'], 
                 subset_n['cfmm_spread_avg'] - subset_n['cfmm_spread_std'], 
                 subset_n['cfmm_spread_avg'] + subset_n['cfmm_spread_std'], 
                 alpha=0.2, color='orange')

ax[1].set_title(f"Spread Comparison with 1-Std Band (b={b_ref})")
ax[1].set_xlabel("Number of Outcomes (n)")
ax[1].set_ylabel("Average Spread")
ax[1].legend()

plt.tight_layout()
plt.savefig('sweep_results_expanded.png', dpi=300)
df.to_csv('sim_data.csv', index=False)

print(f"Done! Saved 'sweep_results_expanded.png' and 'sim_data.csv'")
print("--- %s seconds to complete entire process ---" % (time.time() - start_time))