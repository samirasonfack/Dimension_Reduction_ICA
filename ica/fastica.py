"""
FastICA — Hyvärinen (1999)
===========================
Algorithme à point fixe pour l'ICA basé sur la maximisation de la néguentropie.

Convention (cours)
------------------
    x = W y          (W : matrice de mélange)
    y = V x          (V : matrice séparatrice)

La règle de mise à jour FastICA (déflation, composante par composante) :

    v_new = E[x g(vᵀx)] − E[g'(vᵀx)] v

où g est la nonlinéarité (logcosh, exp ou cube) et v est une ligne de V.

Two variants:
  * ``FastICACustom``  — implémentation from scratch (mode déflation)
  * ``FastICASklearn`` — wrapper ``sklearn.decomposition.FastICA`` (baseline)

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

        self.V_ = None          # matrice séparatrice (k × d)
        self.W_ = None          # matrice de mélange  (d × k) = pinv(V_)
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
        # V : matrice séparatrice — chaque ligne vⱼ est un vecteur de séparation
        V = np.zeros((self.n_components, k))
        self.n_iter_ = []

        for c in range(self.n_components):
            v = rng.standard_normal(k)
            v /= np.linalg.norm(v)

            for i in range(self.max_iter):
                proj = X_proc @ v            # vᵀx,  shape (n,)
                gv = g_fn(proj)              # g(vᵀx)
                gpv = g_prime(proj)          # g'(vᵀx)

                # Règle FastICA : v_new = E[x g(vᵀx)] − E[g'(vᵀx)] v
                v_new = (X_proc * gv[:, None]).mean(axis=0) - gpv.mean() * v

                # Déflation de Gram-Schmidt
                for j in range(c):
                    v_new -= (v_new @ V[j]) * V[j]

                norm = np.linalg.norm(v_new)
                if norm < 1e-12:
                    v_new = rng.standard_normal(k)
                    norm = np.linalg.norm(v_new)
                v_new /= norm

                delta = np.abs(np.abs(v_new @ v) - 1)
                v = v_new

                if delta < self.tol:
                    if self.verbose:
                        print(f"[FastICA] Composante {c}: convergée à l'iter {i}.")
                    break

            V[c] = v
            self.n_iter_.append(i + 1)

        self.V_ = V
        self.W_ = np.linalg.pinv(V)   # matrice de mélange estimée
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """y = V x"""
        X_c = X - self.mean_ if self.mean_ is not None else X
        if self.whiten and self.whitening_matrix_ is not None:
            X_c = X_c @ self.whitening_matrix_.T
        return X_c @ self.V_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# sklearn baseline wrapper
# ---------------------------------------------------------------------------

class FastICASklearn:
    """
    Wrapper ``sklearn.decomposition.FastICA`` — baseline de référence.

    Expose la même interface fit / transform / fit_transform
    et aligne la notation du cours :
        V_ : matrice séparatrice  (= components_  de sklearn)
        W_ : matrice de mélange   (= mixing_      de sklearn)
    """

    def __init__(self, n_components: int = None, **sklearn_kwargs):
        self.n_components = n_components
        self._model = _SklearnFastICA(n_components=n_components, **sklearn_kwargs)
        self.V_ = None   # matrice séparatrice
        self.W_ = None   # matrice de mélange

    def fit(self, X: np.ndarray):
        self._model.fit(X)
        self.V_ = self._model.components_   # y = V x
        self.W_ = self._model.mixing_       # x = W y
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._model.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        result = self._model.fit_transform(X)
        self.V_ = self._model.components_
        self.W_ = self._model.mixing_
        return result
