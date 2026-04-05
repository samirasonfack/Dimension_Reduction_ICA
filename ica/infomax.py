"""
Infomax ICA — Bell & Sejnowski (1995)
======================================
Batch gradient ascent on the Infomax mutual information criterion.

Reference:
    Bell, A. J., & Sejnowski, T. J. (1995). An information-maximization
    approach to blind separation and blind deconvolution.
    Neural Computation, 7(6), 1129-1159.
"""

import numpy as np


class InfomaxICA:
    """
    Infomax ICA via batch natural gradient ascent.

    The algorithm maximises the log-likelihood of a generative model whose
    latent sources follow a super-Gaussian (logistic sigmoid) distribution,
    which corresponds to maximising the output entropy of a sigmoid nonlinearity.

    Parameters
    ----------
    n_components : int
        Number of independent components to extract.
    learning_rate : float
        Step size for the natural gradient update.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence threshold on the Frobenius norm of the weight update.
    whiten : bool
        Whether to whiten (sphere) the data before running ICA.
    random_state : int or None
        Seed for reproducibility.
    verbose : bool
        Print convergence information.
    """

    def __init__(
        self,
        n_components: int = None,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
        whiten: bool = True,
        random_state: int = None,
        verbose: bool = False,
    ):
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.whiten = whiten
        self.random_state = random_state
        self.verbose = verbose

        self.W_ = None          # unmixing matrix  (n_components × n_features)
        self.whitening_matrix_ = None
        self.mean_ = None
        self.n_iter_ = 0
        self.loss_curve_ = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _whiten(self, X: np.ndarray):
        """PCA whitening: zero-mean + unit variance + decorrelated."""
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_
        cov = X_c.T @ X_c / (X_c.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Keep only the top n_components directions
        idx = np.argsort(eigvals)[::-1][: self.n_components]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
        self.whitening_matrix_ = (eigvecs / np.sqrt(eigvals)).T  # (k × d)
        return X_c @ self.whitening_matrix_.T  # (n × k)

    def _log_likelihood(self, W: np.ndarray, X: np.ndarray) -> float:
        """Infomax log-likelihood (per sample, averaged)."""
        n = X.shape[0]
        y = X @ W.T          # (n × k)
        # log |det W| + sum of log sigmoid derivative (= log sig + log(1-sig))
        sign, logabsdet = np.linalg.slogdet(W)
        log_py = np.sum(np.log(self._sigmoid(y) * (1 - self._sigmoid(y)) + 1e-12))
        return logabsdet + log_py / n

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray):
        """
        Fit the Infomax ICA model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        """
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape

        if self.n_components is None:
            self.n_components = d

        if self.whiten:
            X_proc = self._whiten(X)
            k = self.n_components
        else:
            X_proc = X - X.mean(axis=0)
            k = self.n_components

        # Initialise unmixing matrix orthogonally
        W = rng.standard_normal((k, k if self.whiten else d))
        W, _ = np.linalg.qr(W)

        lr = self.learning_rate
        self.loss_curve_ = []

        for i in range(self.max_iter):
            y = X_proc @ W.T          # (n × k)
            phi = 1 - 2 * self._sigmoid(y)  # natural-gradient score (n × k)

            # Natural gradient: ΔW ∝ (I + φ yᵀ) W
            grad = (np.eye(k) + phi.T @ y / n) @ W
            W_new = W + lr * grad

            delta = np.linalg.norm(W_new - W, "fro")
            W = W_new

            ll = self._log_likelihood(W, X_proc)
            self.loss_curve_.append(ll)

            if self.verbose and (i % 100 == 0 or i == self.max_iter - 1):
                print(f"[Infomax] iter {i:4d}  LL={ll:.6f}  ΔW={delta:.2e}")

            if delta < self.tol:
                if self.verbose:
                    print(f"[Infomax] converged at iteration {i}.")
                break

        self.W_ = W
        self.n_iter_ = i + 1
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X onto independent components."""
        X_c = X - self.mean_ if self.mean_ is not None else X
        if self.whiten and self.whitening_matrix_ is not None:
            X_c = X_c @ self.whitening_matrix_.T
        return X_c @ self.W_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def mixing_matrix_(self) -> np.ndarray:
        """Pseudo-inverse of the unmixing matrix."""
        return np.linalg.pinv(self.W_)
