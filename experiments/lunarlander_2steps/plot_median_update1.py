import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde  
import tikzplotlib


# Baseline median reward is obtained using standard 2-step Q-learning in another file 
# with conventional cumulative reward and Markovian sampling.
baseline = 222.265  
MULTI_STEP = 2
lambda_list = [0.0, 0.3, 0.5, 0.7, 0.9, 0.99]
seed_values = [0, 5, 10, 15, 20]
testing_dir = "testing"

# Plot 
plt.figure(figsize=(10, 6))
lambda_values = []
median_values = []

for lam_val in lambda_list:
    all_rewards = []

    for seed in seed_values:
        filename = f"Testing_result_{MULTI_STEP}GM{lam_val}_seed{seed}.npy"
        file_path = os.path.join(testing_dir, filename)
        
        if os.path.exists(file_path):
            rewards = np.load(file_path)
            all_rewards.extend(rewards)
        else:
            print(f"[Warning] File not found: {file_path}")

    all_rewards = np.array(all_rewards)
    if len(all_rewards) == 0:
        continue

    median_val = np.median(all_rewards)
    lambda_values.append(lam_val)
    median_values.append(median_val)

    # Compute KDE on all rewards
    if len(all_rewards) > 1:
        kde = gaussian_kde(all_rewards)
        kde_vals = kde(all_rewards)
    else:
        kde_vals = np.ones_like(all_rewards)

    # Filter out low rewards for cleaner visualization 
    mask = all_rewards > 150
    filtered_rewards = all_rewards[mask]
    filtered_kde_vals = kde_vals[mask]

    # Add jitter for visualization
    xs = lam_val + np.random.uniform(-0.02, 0.02, size=len(filtered_rewards))
    
    plt.scatter(xs, filtered_rewards, c=filtered_kde_vals, cmap='viridis', alpha=0.5, edgecolors='none')

# Plot baseline and median Curve
plt.axhline(y=baseline, color='red', linestyle='--', label='Conventional RL baseline')
plt.plot(lambda_values, median_values, marker='o', color='black', linestyle='-', label='Median score')


plt.xlabel("Lambda")
plt.ylabel("Cumulative Rewards")
plt.title(f"Reward distributions across λ (Multi-step={MULTI_STEP})")
plt.xticks(lambda_list, [str(lam) for lam in lambda_list])
plt.colorbar(label="Reward density")
plt.legend()

# Export plot to TikZ and PNG
tikzplotlib.save(f"Lunar_{MULTI_STEP}step_median.tex")
plt.savefig(f"Lunar_{MULTI_STEP}step_median.png")
plt.show()
