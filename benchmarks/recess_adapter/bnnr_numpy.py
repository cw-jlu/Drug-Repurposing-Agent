"""NumPy port of BNNR (Yang et al., 2019) for the upstream RECeSS runner.

benchscofi's ``BNNR`` shells out to Octave with the original MATLAB sources
(https://github.com/BioinformaticsCSU/BNNR, ``BNNR.m`` and ``svt.m``).  Octave is
not available here, so ``bnnr`` and ``svt`` below are line-by-line ports of
those two files, and ``BNNRNumpy`` subclasses benchscofi's ``BNNR`` so that
preprocessing (``X_s``, ``X_p``, ``A_sp`` construction), default parameters and
output orientation are inherited unchanged; only the Octave call is replaced.

Octave command reproduced from benchscofi ``BNNR.model_fit``::

    T = [Wrr, Wdr'; Wdr, Wdd]            % Wrr = X_s (drugs), Wdr = A_sp, Wdd = X_p
    [WW, iter] = BNNR(alpha, beta, T, double(T ~= 0), tol1, tol2, maxiter, 0, 1)
    M_recovery = WW((t1-dn+1):t1, 1:dr)   % diseases x drugs; transposed on load

benchscofi formats ``alpha``/``beta`` with ``%d`` and ``tol1``/``tol2`` with
``%f``; that formatting is replicated so the effective parameters match.

Optional exact-result cache: if ``BNNR_NUMPY_CACHE_DIR`` is set, the solver
output is stored under the SHA-256 of the input matrix and parameters and
reused for byte-identical inputs (the solver is deterministic).  This only
avoids recomputation when BNNRnp and B4 fit the same training folds; it never
changes a result.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import warnings

import numpy as np
from benchscofi.BNNR import BNNR
from stanscofi.models import BasicModel


def svt(y: np.ndarray, x: float) -> np.ndarray:
    """Singular value thresholding (port of ``svt.m``)."""
    s, v, d = np.linalg.svd(y, full_matrices=False)  # [S, V, D] = svd(Y, 'econ')
    v_new = np.zeros_like(v)
    non_zero = v > x
    v_new[non_zero] = v[non_zero] - x
    return (s * v_new) @ d  # S * diag(v_new) * D'


def bnnr(alpha: float, beta: float, t: np.ndarray, tr_index: np.ndarray, tol1: float,
         tol2: float, maxiter: int, a: float, b: float) -> tuple[np.ndarray, int]:
    """Bounded nuclear norm regularisation (port of ``BNNR.m``)."""
    t = np.asarray(t, dtype=float)
    tr_index = np.asarray(tr_index, dtype=float)
    x = t.copy()
    w = x.copy()
    y = x.copy()
    i = 1
    stop1 = 1.0
    stop2 = 1.0
    n_iter = 0
    t_obs = t * tr_index
    while stop1 > tol1 or stop2 > tol2:
        tran = (1 / beta) * (y + alpha * t_obs) + x
        w = tran - (alpha / (alpha + beta)) * (tran * tr_index)
        w[w < a] = a
        w[w > b] = b
        x_1 = svt(w - 1 / beta * y, 1 / beta)
        y = y + beta * (x_1 - w)
        stop1_0 = stop1
        stop1 = np.linalg.norm(x_1 - x, "fro") / np.linalg.norm(x, "fro")
        stop2 = abs(stop1 - stop1_0) / max(1, abs(stop1_0))
        x = x_1
        i = i + 1
        if i < maxiter:
            n_iter = i - 1
        else:
            n_iter = maxiter
            warnings.warn("reach maximum iteration~~do not converge!!!")
            break
    return w, n_iter


def _octave_int(value) -> int:
    return int("%d" % value)


def _octave_float(value) -> float:
    return float("%f" % value)


def cached_bnnr(alpha, beta, t, tr_index, tol1, tol2, maxiter, a, b):
    """``bnnr`` with an optional content-addressed disk cache (see module docstring)."""
    folder = os.environ.get("BNNR_NUMPY_CACHE_DIR")
    if not folder:
        return bnnr(alpha, beta, t, tr_index, tol1, tol2, maxiter, a, b)
    t = np.ascontiguousarray(t, dtype=float)
    tr_index = np.ascontiguousarray(tr_index, dtype=float)
    digest = hashlib.sha256()
    digest.update(repr((t.shape, alpha, beta, tol1, tol2, maxiter, a, b)).encode())
    digest.update(t.tobytes())
    digest.update(tr_index.tobytes())
    path = Path(folder) / f"bnnr_{digest.hexdigest()}.npz"
    if path.is_file():
        with np.load(path) as cached:
            return cached["w"].copy(), int(cached["n_iter"])
    w, n_iter = bnnr(alpha, beta, t, tr_index, tol1, tol2, maxiter, a, b)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(temporary, w=w, n_iter=n_iter)
    os.replace(temporary, path)
    return w, n_iter


class BNNRNumpy(BNNR):
    """benchscofi ``BNNR`` with the Octave subprocess replaced by the NumPy port."""

    def __init__(self, params=None):
        params = params if params is not None else self.default_parameters()
        BasicModel.__init__(self, params)
        self.scalerS, self.scalerP = None, None
        self.name = "BNNRnp"
        self.estimator = None
        self.BNNR_filepath = None

    def model_fit(self, X_s, X_p, A_sp):
        wdd, wdr, wrr = X_p, A_sp, X_s
        t = np.block([[wrr, wdr.T], [wdr, wdd]])
        ww, n_iter = cached_bnnr(_octave_int(self.alpha), _octave_int(self.beta), t,
                          (t != 0).astype(float), _octave_float(self.tol1),
                          _octave_float(self.tol2), int(self.maxiter), 0, 1)
        t1 = t.shape[0]
        dn, dr = wdr.shape
        m_recovery = ww[t1 - dn:t1, :dr]
        self.estimator = {"niter": int(n_iter), "predictions": m_recovery.T.copy()}

    def model_predict_proba(self, X_s, X_p, A_sp):
        return self.estimator["predictions"].copy()


# Name expected by the upstream runner's ``getattr`` dispatch.
BNNRnp = BNNRNumpy
