# Genetic Algorithm for Cost Analysis of Wireless Networks in Industrial Environments
The algorithm in this repo helps computing cost of 5G network implementation in an industrial environment. That is, 
how many Access Points (APs) and how many meters of cable are require for a full-coverage of AVG paths in a factor.

# Wireless Network Access Point Planner

A Python tool for optimizing the placement of wireless access points (APs) in industrial or facility environments to ensure optimal coverage and localization accuracy.

## Features

- Path loss modeling for wireless signal propagation
- Multiple AP distribution strategies:
  - Random placement
  - Uniform distribution along paths
  - Lattice/grid distribution
  - Genetic algorithm optimization
- Coverage visualization with heatmaps
- Performance metrics calculation:
  - Coverage score
  - Required cable length
  - Localization accuracy

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/wireless-network-planner.git
   cd wireless-network-planner

2. Install required packages:
    ```bash
   pip install numpy matplotlib

## Basic Example
A basic example is provides in the `main.py` file.

## Key Components
Here are the key components of the repository:
### `network_planner.py`
The core module containing:
- NetworkPlanner class with all planning functionality
- Path loss calculation methods
- AP distribution strategies:
  - `uniform_ap_distribution()`
  - `lattice_distribution()`
- Optimization algorithms:
  - `genetic_optimization()`
- Visualization tools:
  - `generate_heatmap()`
  - `map_pathloss()`

### `helper.py`
Utility functions for:
- Noise power calculation (calculate_noise())
- Path loss threshold calculation (calculate_threshold())

### `main.py`
Example usage demonstrating:
- Network parameter configuration
- Optimization workflow
- Result visualization
