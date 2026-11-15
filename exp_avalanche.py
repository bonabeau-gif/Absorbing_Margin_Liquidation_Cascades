"""Precursor avalanche-size statistics (account counts) before the macroscopic collapse (Sec. IX).

Usage: python exp_avalanche.py <equal|lognormal|mu>. Writes results/avalanches_<kind>.csv.
The experiment lives in amlc.experiments.run_avalanche.
"""
import sys
from amlc.experiments import run_avalanche

if __name__ == "__main__":
    run_avalanche(sys.argv[1])
