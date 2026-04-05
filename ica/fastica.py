"""
FastICA — Hyvärinen (1999)
===========================
Fixed-point algorithm for Independent Component Analysis based on
negentropy maximisation.

Two variants are provided:
  * ``FastICACustom``  — from-scratch implementation (deflation mode)
  * ``FastICASklearn`` — thin wrapper around ``sklearn.decomposition.FastICA``
    used as a reference baseline.

Reference:
    Hyvärinen, A. (1999). Fast and robust fixed-point algorithms for
    independent component analysis. IEEE Transactions on Neural Networks,
    10(3), 626-634.
"""

import numpy as np
from sklearn.decomposition import FastICA as _SklearnFastICA


# ---------------------------------------------------------------------------
# From-scratch implementation (deflation)
# ---------------------------------------------------------------------------

class FastICACustom:
    """
    FastICA implemented from scratch using the deflation (one-by-one) strategy.

    The negentropy contrast function ``g`` can be one of:
      * ``'logcosh'``  — tanh nonlinearity (default, robust)
      * ``'exp'``      — exponential, good for super-Gaussian sources
      * ``'cube'``     — cubic, equivalent to kurtosis maximisation

    Parameters
    ----------
    n_components : int or None
        Number of components. Defaults to min(n_samples, n_features).
    max_iter : int
        Maximum fixed-point iterations per component.
    tol : float
        Convergence threshold (change in weight vector).
    g : {'logcosh', 'exp', 'cube'}
        Nonlinearity for negentropy approximation.
    whiten : bool
        Whether to whiten data first.
    random_state : int or None
    verbose : bool
    """

    _G_FUNCS = {
        "logcosh": (
            lambda u: np.tanh(u),
            lambda u: 1 - np.tanh(u) ** 2,
        ),
        "exp": (
            lambda u: u * np.exp(-(u ** 2) / 2),
            lambda u: (1 - u ** 2) * np.exp(-(u ** 2) / 2),
        ),
        "cube": (
            lambda u: u ** 3,
            lambda u: 3 * u ** 2,
        ),
    }

    def __init__(
        self,
        n_components: int = None,
        max_iter: int = 500,
        tol: float = 1e-6,
        g: str = "logcosh",
        whiten: bool = True,
        random_state: int = None,
        verbose: bool = False,
    ):
        if g not in self._G_FUNCS:
            raise ValueError(f"g must be one of {list(self._G_FUNCS)}.")
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.g = g
        self.whiten = whiten
        self.random_state = random_state
        self.verbose = verbose

        self.W_ = None
        self.whitening_matrix_ = None
        self.mean_ = None
        self.n_iter_ = []

    # ------------------------------------------------------------------

    def _whiten(self, X: np.ndarray) -> np.ndarray:
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_
        cov = X_c.T @ X_c / (X_c.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1][: self.n_components]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
        self.whitening_matrix_ = (eigvecs / np.sqrt(eigvals)).T
        return X_c @ self.whitening_matrix_.T

    def fit(self, X: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        if self.n_components is None:
            self.n_components = d

        X_proc = self._whiten(X) if self.whiten else (X - X.mean(axis=0))
        k = X_proc.shape[1]

        g_fn, g_prime = self._G_FUNCS[self.g]
        W = np.zeros((self.n_components, k))
        self.n_iter_ = []

        for c in range(self.n_components):
            w = rng.standard_normal(k)
            w /= np.linalg.norm(w)

            for i in range(self.max_iter):
                proj = X_proc @ w          # (n,)
                gw = g_fn(proj)            # (n,)
                gpw = g_prime(proj)        # (n,)
                w_new = (X_proc * gw[:, None]).mean(axis=0) - gpw.mean() * w

                # Gram-Schmidt deflation against previously found components
                for j in range(c):
                    w_new -= (w_new @ W[j]) * W[j]

                norm = np.linalg.norm(w_new)
                if norm < 1e-12:
                    w_new = rng.standard_normal(k)
                    norm = np.linalg.norm(w_new)
                w_new /= norm

                delta = np.abs(np.abs(w_new @ w) - 1)
                w = w_new

                if delta < self.tol:
                    if self.verbose:
                        print(f"[FastICA] Component {c}: converged at iter {i}.")
                    break

            W[c] = w
            self.n_iter_.append(i + 1)

        self.W_ = W
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_c = X - self.mean_ if self.mean_ is not None else X
        if self.whiten and self.whitening_matrix_ is not None:
            X_c = X_c @ self.whitening_matrix_.T
        return X_c @ self.W_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def mixing_matrix_(self) -> np.ndarray:
        return np.linalg.pinv(self.W_)


# ---------------------------------------------------------------------------
# sklearn baseline wrapper
# ---------------------------------------------------------------------------

class FastICASklearn:
    """
    Thin wrapper around ``sklearn.decomposition.FastICA`` used as a baseline.

    Exposes the same ``fit / transform / fit_transform`` interface and
    forwards all keyword arguments to the sklearn constructor.
    """

    def __init__(self, n_components: int = None, **sklearn_kwargs):
        self.n_components = n_components
        self._model = _SklearnFastICA(n_components=n_components, **sklearn_kwargs)
        self.W_ = None
        self.mixing_matrix_ = None

    def fit(self, X: np.ndarray):
        self._model.fit(X)
        self.W_ = self._model.components_
        self.mixing_matrix_ = self._model.mixing_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._model.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        result = self._model.fit_transform(X)
        self.W_ = self._model.components_
        self.mixing_matrix_ = self._model.mixing_
        return result
