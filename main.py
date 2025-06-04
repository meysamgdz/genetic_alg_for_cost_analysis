import numpy as np
import matplotlib.pyplot as plt
from network_planner import NetworkPlanner
from helper import *


# Network parameters
xy, frequency, pathloss_exponent = (120, 60), 5.4e9, 1.67
num_aps = np.arange(20, 23)

# Calculate noise and threshold
noise_power = calculate_noise(band_width=160e6)
threshold = calculate_threshold(
    tx_power=23,
    noise=noise_power,
    localization_acc=0.5
)

score_tot = []
cable_len = []

for num_ap in num_aps:
    planner = NetworkPlanner(
        xy=xy,
        frequency=frequency,
        pl_exponent=pathloss_exponent,
        num_ap=num_ap
    )

    # Run optimization
    scores, ap_locs, pl_cov, pl_nth = planner.genetic_optimization(
        threshold=threshold,
        init_iter=100,
        dropout=0.8,
        nth=3
    )

    # Get best solution
    best_idx = np.argmax(scores)

    # Generate visualizations
    planner.generate_heatmap(ap_locs[best_idx], nth=0)
    planner.generate_heatmap(ap_locs[best_idx], nth=3)

    # Store results
    score_tot.append(scores[best_idx])
    cable_len.append(np.array(ap_locs[best_idx])[:, :, 0, 0].sum())

# Plot results
fig, ax = plt.subplots()
ax.plot(num_aps, score_tot, 'b-*')
ax.set(ylabel='Coverage Score', xlabel='Number of APs')

fig, ax = plt.subplots()
ax.plot(num_aps, cable_len, 'b-o')
ax.set(ylabel='Required Cable Length (meters)', xlabel='Number of APs')

plt.show()