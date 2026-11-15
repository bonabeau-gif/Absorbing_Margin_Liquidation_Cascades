"""Borel-mixture prediction for the integrated precursor avalanche distribution (equal exposures)."""
import numpy as np
from scipy.special import gammaln
from .model import *
def integrated_borel(S, l=2.0, s=0.16, m=0.25, chi=0.0, Lam=0.30, n=20000):
    Bp = folds(l, s, m, chi, Lam)[0][0]
    B = np.linspace(1e-9, Bp, n); R = Lam*IMPACT["square-root"][1](B)/dt_dB(B, l, s, m, chi)
    R = np.clip(R, 1e-12, 1.0)
    S = np.asarray(S, float)[:, None]
    logp = -R*S + (S-1)*np.log(R*S) - gammaln(S+1)
    D = np.trapezoid((1-R)*np.exp(logp), B, axis=1)
    norm_ = np.trapezoid(1-R, B)
    return D/norm_
