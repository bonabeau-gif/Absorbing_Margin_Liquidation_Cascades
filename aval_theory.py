"""Borel-mixture prediction for the integrated precursor avalanche distribution (equal exposures).

The implementation now lives in ``amlc/aval_theory.py``; this module re-exports it.
"""
from amlc.aval_theory import *  # noqa: F401,F403

if __name__ == "__main__":
    import pandas as pd
    d = pd.read_csv("results/avalanches_equal.csv"); d = d[d.count_all > 30]
    for lo, hi, pdf in zip(d.bin_lo, d.bin_hi, d.pdf_all):
        Ss = np.arange(lo, hi); th = integrated_borel(Ss).mean()
        print(int(lo), int(hi), "sim %.3e  theory %.3e  ratio %.3f" % (pdf, th, pdf/th))
