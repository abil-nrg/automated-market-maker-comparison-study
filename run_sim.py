import matplotlib
matplotlib.use('Agg') 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from MarketSim import Market, LMSR, CFMM
import time

def run_master_sweep(n_runs=30, T=500):
    n_values = [2, 4, 8, 32, 64]
    b_values = [50, 200, 800]
    noise_ratios = [0.3, 0.6, 0.9]
    confidences = [0.05, 0.2, 0.5, 0.8] 
    
    sweep_results = []
    
    config_count = 0
    total_configs = len(n_values) * len(b_values) * len(noise_ratios) * len(confidences)

    for n in n_values:
        true_p = np.ones(n) / n
        true_p[0] += 0.2
        true_p /= true_p.sum()

        for b in b_values:
            k_equiv = b**n
            for nr in noise_ratios:
                for conf in confidences:
                    config_count += 1
                    print(f"[{config_count}/{total_configs}] n={n}, b={b}, NR={nr}, Conf={conf}")
                    
                    lmsr_pnls, cfmm_pnls = [], []

                    for _ in range(n_runs):
                        m1 = Market(LMSR(n, b), T, true_p, nr, agent_confidence=conf)
                        m2 = Market(CFMM(n, k_equiv), T, true_p, nr, agent_confidence=conf)

                        lmsr_pnls.append(m1.run()[-1]) # Assuming last element is PnL
                        cfmm_pnls.append(m2.run()[-1])

                    sweep_results.append({
                        'n': n, 'b': b, 'noise_ratio': nr, 'confidence': conf,
                        'lmsr_mean': np.mean(lmsr_pnls),
                        'cfmm_mean': np.mean(cfmm_pnls),
                        'lmsr_std': np.std(lmsr_pnls),
                        'cfmm_std': np.std(cfmm_pnls)
                    })
        
    return pd.DataFrame(sweep_results)

start_time = time.time()
df = run_master_sweep(n_runs=30)
df.to_csv('master_grid_data.csv', index=False)

#plots
g = sns.FacetGrid(df, col="n", row="b", margin_titles=True)
g.map_dataframe(sns.scatterplot, x="noise_ratio", y="lmsr_mean", hue="confidence", palette="viridis")
g.add_legend()
g.set_axis_labels("Noise Trader Proportion", "Mean PnL")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Market Maker Profitability across N, B, Noise, and Confidence')

plt.savefig('master_grid_analysis.png', dpi=300)

print(f"Done! Saved 'master_grid_analysis.png' and 'master_grid_data.csv'")
print("--- %s seconds total ---" % (time.time() - start_time))