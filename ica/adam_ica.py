"""
Adam-ICA — Optimiseur Adam appliqué à Infomax
===============================================
Adaptive Moment Estimation appliqué au critère Infomax en mini-batch.

Convention (cours)
------------------
    x = W y          (W : matrice de mélange)
    y = V x          (V : matrice séparatrice)

Gradient naturel Infomax sur un mini-batch B_t :

    g_t = (I − (1/|B_t|) tanh(y) yᵀ) V,   y = V X_{B_t}^T

Mise à jour Adam (gradient *ascent*) :

    m_t = β₁ m_{t-1} + (1 − β₁) g_t          ← 1er moment (momentum)
    v_t = β₂ v_{t-1} + (1 − β₂) g_t²         ← 2ème moment (variance)
    m̂_t = m_t / (1 − β₁ᵗ)                    ← correction du biais
    v̂_t = v_t / (1 − β₂ᵗ)
    V_{t+1} = V_t + α m̂_t / (√v̂_t + ε)

References:
    Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic
    optimization. ICLR 2015.

    Bell, A. J., & Sejnowski, T. J. (1995). Neural Computation, 7(6).
"""

import numpy as np


class AdamICA:
    """
    Adam-ICA : optimiseur Adam appliqué au critère Infomax.

    Modèle : x = W y  →  y = V x  (V : matrice séparatrice)

    Parameters
    ----------
    n_components : int or None
    learning_rate : float
        Pas Adam (α). Valeur typique : 1e-3.
    beta1 : float
        Décroissance exponentielle du 1er moment (β₁). Défaut : 0.9.
    beta2 : float
        Décroissance exponentielle du 2ème moment (β₂). Défaut : 0.999.
    epsilon : float
        Constante de stabilité numérique (ε). Défaut : 1e-8.
    batch_size : int
        Taille du mini-batch.
    n_epochs : int
        Nombre d'epochs.
    tol : float
        Arrêt anticipé si ‖ΔV‖_F < tol.
    whiten : bool
    random_state : int or None
    verbose : bool
    """

    def __init__(
        self,
        n_components: int = None,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        batch_size: int = 32,
        n_epochs: int = 50,
        tol: float = 1e-7,
        whiten: bool = True,
        random_state: int = None,
        verbose: bool = False,
    ):
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.tol = tol
        self.whiten = whiten
        self.random_state = random_state
        self.verbose = verbose

        self.V_ = None          # matrice séparatrice (k × d)
        self.W_ = None          # matrice de mélange  (d × k) = pinv(V_)
        self.whitening_matrix_ = None
        self.mean_ = None
        self.loss_curve_ = []

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

    def _batch_gradient(self, V: np.ndarray, X_batch: np.ndarray) -> np.ndarray:
        """
        Gradient naturel Infomax sur un mini-batch.

        g = (I − (1/m) tanh(y) yᵀ) V,   y = V x
        """
        m = X_batch.shape[0]
        y = X_batch @ V.T              # (m, k)
        phi_yt = np.tanh(y).T @ y / m  # (k, k)
        return (np.eye(V.shape[0]) - phi_yt) @ V

    def _log_likelihood(self, V: np.ndarray, X_batch: np.ndarray) -> float:
        y = X_batch @ V.T
        _, logdet = np.linalg.slogdet(V)
        return logdet + np.sum(np.log(1.0 - np.tanh(y) ** 2 + 1e-12)) / X_batch.shape[0]

    def fit(self, X: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        if self.n_components is None:
            self.n_components = d

        X_proc = self._whiten(X) if self.whiten else (X - X.mean(axis=0))
        k = X_proc.shape[1]

        V, _ = np.linalg.qr(rng.standard_normal((k, k)))

        # État Adam : moments du 1er et 2ème ordre
        m_t = np.zeros_like(V)
        v_t = np.zeros_like(V)
        t = 0

        β1, β2, ε, α = self.beta1, self.beta2, self.epsilon, self.learning_rate
        n_batches = max(1, n // self.batch_size)
        self.loss_curve_ = []

        for epoch in range(self.n_epochs):
            perm = rng.permutation(n)
            X_shuf = X_proc[perm]
            epoch_loss = 0.0

            for b in range(n_batches):
                X_batch = X_shuf[b * self.batch_size: (b + 1) * self.batch_size]
                t += 1

                # Gradient naturel Infomax
                g_t = self._batch_gradient(V, X_batch)

                # Mise à jour des moments Adam
                m_t = β1 * m_t + (1 - β1) * g_t
                v_t = β2 * v_t + (1 - β2) * g_t ** 2

                # Correction du biais
                m_hat = m_t / (1 - β1 ** t)
                v_hat = v_t / (1 - β2 ** t)

                # Mise à jour de V (gradient ascent)
                V_new = V + α * m_hat / (np.sqrt(v_hat) + ε)

                delta = np.linalg.norm(V_new - V, "fro")
                V = V_new

                epoch_loss += self._log_likelihood(V, X_batch)

                if delta < self.tol:
                    if self.verbose:
                        print(f"[Adam-ICA] Arrêt anticipé — epoch {epoch}, batch {b}.")
                    self.V_ = V
                    self.W_ = np.linalg.pinv(V)
                    return self

            self.loss_curve_.append(epoch_loss / n_batches)
            if self.verbose:
                print(f"[Adam-ICA] Epoch {epoch:3d}  avg LL={self.loss_curve_[-1]:.6f}")

        self.V_ = V
        self.W_ = np.linalg.pinv(V)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """y = V x"""
        X_c = X - self.mean_ if self.mean_ is not None else X
        if self.whiten and self.whitening_matrix_ is not None:
            X_c = X_c @ self.whitening_matrix_.T
        return X_c @ self.V_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
