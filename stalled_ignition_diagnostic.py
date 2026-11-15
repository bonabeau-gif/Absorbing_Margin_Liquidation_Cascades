"""Matched-cohort diagnostic of stalled ignition under transient impact.

Writes results/stalled_ignition_control2.csv (file name as in the original script).
The experiment lives in amlc.experiments.run_stalled_ignition.
"""
from amlc.experiments import run_stalled_ignition

if __name__ == "__main__":
    run_stalled_ignition()
