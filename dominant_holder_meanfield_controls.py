"""Event-definition-independent dominant-holder test and binary mean-field benchmark (Sec. X population).

Reads results/path_size.csv; writes results/dominant_holder_event_independent.csv and
results/path_model_meanfield_reference.csv. The controls live in amlc.experiments.run_dominant_holder_controls.
"""
from amlc.experiments import run_dominant_holder_controls

if __name__ == "__main__":
    run_dominant_holder_controls()
