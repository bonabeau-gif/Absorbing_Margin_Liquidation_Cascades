"""Finite-size convergence check for the equal-exposure avalanche distribution.

Writes results/avalanche_finiteN_sweep.csv. The experiment lives in amlc.experiments.run_avalanche_size_sweep.
"""
from amlc.experiments import run_avalanche_size_sweep

if __name__ == "__main__":
    run_avalanche_size_sweep()
