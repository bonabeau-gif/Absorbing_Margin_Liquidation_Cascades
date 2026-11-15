"""Finite-cohort collapse statistics with lognormal exposures (Sec. VII) and parameter-free Daniels-type predictions.

Writes results/finiteN_lognormal.csv and results/finiteN_theory_constants.csv.
The experiment lives in amlc.experiments.run_finiteN.
"""
from amlc.experiments import run_finiteN

if __name__ == "__main__":
    run_finiteN()
