import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import tikzplotlib


MULTI_STEP = 5
lambda_list = [0.0, 0.3, 0.5, 0.7, 0.9, 0.99]
seed_values = [0, 5, 10, 15, 20]
testing_dir = "testing_results"

plt.figure(figsize=(10, 6))
lambda_values = []
mean_values = []
baseline = None  # will be set by λ=0

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

    mean_val = np.mean(all_rewards)
    lambda_values.append(lam_val)
    mean_values.append(mean_val)

    # Set baseline as λ=0
    if lam_val == 0.0:
        baseline = mean_val

    # Compute KDE before filtering
    if len(all_rewards) > 1:
        kde = gaussian_kde(all_rewards)
        kde_vals = kde(all_rewards)
    else:
        kde_vals = np.ones_like(all_rewards)

    # Apply filtering only for visualization
    mask = all_rewards > 150
    filtered_rewards = all_rewards[mask]
    filtered_kde_vals = kde_vals[mask]
    xs = lam_val + np.random.uniform(-0.02, 0.02, size=len(filtered_rewards))

    # Plot scatter with density
    plt.scatter(xs, filtered_rewards, c=filtered_kde_vals, cmap='viridis', alpha=0.5, edgecolors='none')

# Plot baseline line and mean trend
if baseline is not None:
    plt.axhline(y=baseline, color='red', linestyle='--', label='Baseline (λ=0)')

plt.plot(lambda_values, mean_values, marker='o', color='black', linestyle='-', label='Median reward')

# Plot settings
plt.xlabel("Lambda")
plt.ylabel("Cumulative Rewards")
plt.title(f"Reward distributions across λ (Multi-step={MULTI_STEP})")
plt.xticks(lambda_list, [str(l) for l in lambda_list])
plt.colorbar(label="Reward density")
plt.legend()

# Save to TikZ and image file
tikzplotlib.save(f"Cartpole_{MULTI_STEP}step_mean0.tex")
plt.savefig(f"Cartpole_{MULTI_STEP}step_mean0.png")
plt.show()
