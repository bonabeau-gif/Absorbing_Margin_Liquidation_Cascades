"""Path-resolved experiments: cohort-size convergence, ramp-rate sharpening, impact-memory sweep (Sec. X).

Usage: python exp_path.py <size|ramp|memory>. Writes results/path_<which>.csv and
results/path_curves_<which>.npz. The experiments live in amlc.experiments.run_path_study.
"""
import sys
from amlc.experiments import run_path_study

if __name__ == "__main__":
    run_path_study(sys.argv[1])
