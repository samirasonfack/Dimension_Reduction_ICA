"""
SGD-ICA — Gradient Stochastique appliqué à Infomax
=====================================================
Mini-batch stochastic gradient ascent on the Infomax log-likelihood.

Convention (cours)
------------------
    x = W y          (W : matrice de mélange)
    y = V x          (V : matrice séparatrice)

Mise à jour stochastique sur un mini-batch B_t :

    ΔV = η_t (I − (1/|B_t|) tanh(V X_Bt^T) (V X_Bt^T)^T) V

Avantage principal : chaque pas ne nécessite que |B_t| ≪ n exemples,
ce qui rend l'algorithme scalable aux grands datasets et aux flux.
"""

import numpy as np


class SGDICA:
    """
    SGD-ICA : descente de gradient stochastique sur le critère Infomax.

    Modèle : x = W y  →  y = V x  (V : matrice séparatrice)

    Parameters
    ----------
    n_components : int or None
    learning_rate : float
        Pas initial η₀.
    batch_size : int
        Taille du mini-batch |B_t|.
    n_epochs : int
        Nombre de passes sur le dataset.
    lr_schedule : {'constant', 'inv_sqrt', 'cosine'}
        Schedule du pas :
        - ``'constant'``  : η_t = η₀
        - ``'inv_sqrt'``  : η_t = η₀ / √(t+1)
        - ``'cosine'``    : cosine annealing sur le total des steps
    tol : float
        Arrêt anticipé si ‖ΔV‖_F < tol.
    whiten : bool
    random_state : int or None
    verbose : bool
    """

    def __init__(
        self,
        n_components: int = None,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        n_epochs: int = 50,
        lr_schedule: str = "constant",
        tol: float = 1e-7,
        whiten: bool = True,
        random_state: int = None,
        verbose: bool = False,
    ):
        if lr_schedule not in ("constant", "inv_sqrt", "cosine"):
            raise ValueError("lr_schedule must be 'constant', 'inv_sqrt', or 'cosine'.")
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr_schedule = lr_schedule
        self.tol = tol
        self.whiten = whiten
        self.random_state = random_state
        self.verbose = verbose

        self.V_ = None          # matrice séparatrice
        self.W_ = None          # matrice de mélange = pinv(V_)
        self.whitening_matrix_ = None
        self.mean_ = None
        self.loss_curve_ = []

    # ------------------------------------------------------------------

    def _get_lr(self, step: int, total_steps: int) -> float:
        η0 = self.learning_rate
        if self.lr_schedule == "constant":
            return η0
        if self.lr_schedule == "inv_sqrt":
            return η0 / np.sqrt(step + 1)
        return η0 * 0.5 * (1 + np.cos(np.pi * step / total_steps))

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

        ΔV = (I − (1/m) tanh(y) yᵀ) V,   y = V x_batch^T
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
        self.loss_curve_ = []

        n_batches = max(1, n // self.batch_size)
        total_steps = self.n_epochs * n_batches
        step = 0

        for epoch in range(self.n_epochs):
            perm = rng.permutation(n)
            X_shuf = X_proc[perm]
            epoch_loss = 0.0

            for b in range(n_batches):
                X_batch = X_shuf[b * self.batch_size: (b + 1) * self.batch_size]
                lr = self._get_lr(step, total_steps)

                grad = self._batch_gradient(V, X_batch)
                V_new = V + lr * grad

                delta = np.linalg.norm(V_new - V, "fro")
                V = V_new
                step += 1

                epoch_loss += self._log_likelihood(V, X_batch)

                if delta < self.tol:
                    if self.verbose:
                        print(f"[SGD-ICA] Arrêt anticipé — epoch {epoch}, batch {b}.")
                    self.V_ = V
                    self.W_ = np.linalg.pinv(V)
                    return self

            self.loss_curve_.append(epoch_loss / n_batches)
            if self.verbose:
                print(f"[SGD-ICA] Epoch {epoch:3d}  avg LL={self.loss_curve_[-1]:.6f}  lr={lr:.2e}")

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
