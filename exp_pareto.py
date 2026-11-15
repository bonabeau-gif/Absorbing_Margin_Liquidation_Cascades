"""Collapse-field statistics with Pareto-distributed exposures independent of leverage (Sec. VIII).

Usage: python exp_pareto.py <mu>. Writes results/pareto_raw_mu<mu>.npz and appends to
results/finiteN_pareto.csv. The experiment lives in amlc.experiments.run_pareto.
"""
import sys
from amlc.experiments import run_pareto

if __name__ == "__main__":
    run_pareto(sys.argv[1])
