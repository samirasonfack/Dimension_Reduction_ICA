"""
Infomax ICA — Bell & Sejnowski (1995)
======================================
Batch natural gradient ascent on the Infomax criterion.

Convention (cours)
------------------
    x = W y          (W : matrice de mélange,  x : observé, y : sources)
    y = V x          (V : matrice séparatrice, V ≈ W⁻¹)

Le gradient naturel de la log-vraisemblance Infomax par rapport à V est :

    ΔV ∝ (I − E[tanh(y) yᵀ]) V

où y = Vx et tanh est la nonlinéarité adaptée aux sources super-gaussiennes.

Reference:
    Bell, A. J., & Sejnowski, T. J. (1995). An information-maximization
    approach to blind separation and blind deconvolution.
    Neural Computation, 7(6), 1129-1159.
"""

import numpy as np


class InfomaxICA:
    """
    Infomax ICA via gradient naturel batch.

    Modèle : x = W y  →  y = V x  (V : matrice séparatrice)

    Parameters
    ----------
    n_components : int or None
        Nombre de composantes indépendantes.
    learning_rate : float
        Pas du gradient naturel (η).
    max_iter : int
        Nombre maximum d'itérations.
    tol : float
        Seuil de convergence sur ‖ΔV‖_F.
    whiten : bool
        Blanchiment PCA des données avant ICA.
    random_state : int or None
    verbose : bool
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

        # Attributs publics — notation du cours
        self.V_ = None          # matrice séparatrice  (k × d)
        self.W_ = None          # matrice de mélange   (d × k) = pinv(V_)
        self.whitening_matrix_ = None
        self.mean_ = None
        self.n_iter_ = 0
        self.loss_curve_ = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _whiten(self, X: np.ndarray) -> np.ndarray:
        """Blanchiment PCA : zero-mean, décorrélé, variance unitaire."""
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_
        cov = X_c.T @ X_c / (X_c.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1][: self.n_components]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
        self.whitening_matrix_ = (eigvecs / np.sqrt(eigvals)).T   # (k × d)
        return X_c @ self.whitening_matrix_.T                      # (n × k)

    def _log_likelihood(self, V: np.ndarray, X: np.ndarray) -> float:
        """Log-vraisemblance Infomax (moyenne par échantillon)."""
        n = X.shape[0]
        y = X @ V.T
        _, logabsdet = np.linalg.slogdet(V)
        # log g'(y) = log(1 - tanh²(y)) = log sech²(y)
        log_gprime = np.sum(np.log(1.0 - np.tanh(y) ** 2 + 1e-12))
        return logabsdet + log_gprime / n

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray):
        """
        Ajuste le modèle ICA sur X.

        Parameters
        ----------
        X : ndarray (n_samples, n_features)
        """
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape

        if self.n_components is None:
            self.n_components = d

        X_proc = self._whiten(X) if self.whiten else (X - X.mean(axis=0))
        k = X_proc.shape[1]

        # Initialisation orthogonale de V
        V, _ = np.linalg.qr(rng.standard_normal((k, k)))

        self.loss_curve_ = []

        for i in range(self.max_iter):
            y = X_proc @ V.T          # y = V x,  shape (n, k)

            # Gradient naturel : ΔV = η (I − E[tanh(y) yᵀ]) V
            phi_yt = np.tanh(y).T @ y / n   # (k × k)
            grad = (np.eye(k) - phi_yt) @ V

            V_new = V + self.learning_rate * grad

            delta = np.linalg.norm(V_new - V, "fro")
            V = V_new

            ll = self._log_likelihood(V, X_proc)
            self.loss_curve_.append(ll)

            if self.verbose and (i % 100 == 0 or i == self.max_iter - 1):
                print(f"[Infomax] iter {i:4d}  LL={ll:.6f}  ΔV={delta:.2e}")

            if delta < self.tol:
                if self.verbose:
                    print(f"[Infomax] convergé à l'itération {i}.")
                break

        self.V_ = V
        self.W_ = np.linalg.pinv(V)   # matrice de mélange estimée
        self.n_iter_ = i + 1
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Projette X sur les composantes indépendantes : y = V x."""
        X_c = X - self.mean_ if self.mean_ is not None else X
        if self.whiten and self.whitening_matrix_ is not None:
            X_c = X_c @ self.whitening_matrix_.T
        return X_c @ self.V_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
