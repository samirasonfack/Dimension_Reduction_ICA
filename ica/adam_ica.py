"""
Adam-ICA — Adam Optimizer applied to Infomax
==============================================
Adaptive Moment Estimation (Adam) applied to the Infomax ICA criterion.

Adam combines momentum (first moment) with adaptive per-parameter learning
rates (second moment), which typically yields faster and more stable
convergence than plain SGD, especially in early training.

Reference for Adam:
    Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic
    optimization. ICLR 2015.

Infomax reference:
    Bell, A. J., & Sejnowski, T. J. (1995). An information-maximization
    approach to blind separation. Neural Computation, 7(6), 1129-1159.
"""

import numpy as np


class AdamICA:
    """
    Adam-ICA: Adam optimizer applied to mini-batch Infomax ICA.

    Parameters
    ----------
    n_components : int or None
        Number of independent components.
    learning_rate : float
        Adam step size (α). Typical value: 1e-3.
    beta1 : float
        Exponential decay for first moment (momentum). Default: 0.9.
    beta2 : float
        Exponential decay for second moment (RMSProp). Default: 0.999.
    epsilon : float
        Numerical stability constant. Default: 1e-8.
    batch_size : int
        Mini-batch size.
    n_epochs : int
        Number of training epochs.
    tol : float
        Early-stop threshold on weight-update norm.
    whiten : bool
        Whiten data before ICA.
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

        self.W_ = None
        self.whitening_matrix_ = None
        self.mean_ = None
        self.loss_curve_ = []

    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def _whiten(self, X: np.ndarray) -> np.ndarray:
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_
        cov = X_c.T @ X_c / (X_c.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1][: self.n_components]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
        self.whitening_matrix_ = (eigvecs / np.sqrt(eigvals)).T
        return X_c @ self.whitening_matrix_.T

    def _batch_gradient(self, W: np.ndarray, X_batch: np.ndarray) -> np.ndarray:
        """Natural gradient of Infomax on a mini-batch."""
        m = X_batch.shape[0]
        y = X_batch @ W.T
        phi = 1 - 2 * self._sigmoid(y)
        k = W.shape[0]
        return (np.eye(k) + phi.T @ y / m) @ W

    def fit(self, X: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        if self.n_components is None:
            self.n_components = d

        X_proc = self._whiten(X) if self.whiten else (X - X.mean(axis=0))
        k = X_proc.shape[1]

        W, _ = np.linalg.qr(rng.standard_normal((k, k)))

        # Adam state: first and second moment estimates
        m_t = np.zeros_like(W)   # first moment
        v_t = np.zeros_like(W)   # second moment
        t = 0                     # global step counter

        β1, β2, ε = self.beta1, self.beta2, self.epsilon
        α = self.learning_rate
        n_batches = max(1, n // self.batch_size)
        self.loss_curve_ = []

        for epoch in range(self.n_epochs):
            perm = rng.permutation(n)
            X_shuf = X_proc[perm]
            epoch_loss = 0.0

            for b in range(n_batches):
                X_batch = X_shuf[b * self.batch_size: (b + 1) * self.batch_size]
                t += 1

                # Gradient of the Infomax criterion (we ascend, so grad → +)
                grad = self._batch_gradient(W, X_batch)

                # Adam moment updates
                m_t = β1 * m_t + (1 - β1) * grad
                v_t = β2 * v_t + (1 - β2) * grad ** 2

                # Bias correction
                m_hat = m_t / (1 - β1 ** t)
                v_hat = v_t / (1 - β2 ** t)

                # Parameter update (gradient *ascent*)
                W_new = W + α * m_hat / (np.sqrt(v_hat) + ε)

                delta = np.linalg.norm(W_new - W, "fro")
                W = W_new

                # Track log-likelihood on this batch
                sign, logdet = np.linalg.slogdet(W)
                y = X_batch @ W.T
                ll = logdet + np.sum(
                    np.log(self._sigmoid(y) * (1 - self._sigmoid(y)) + 1e-12)
                ) / X_batch.shape[0]
                epoch_loss += ll

                if delta < self.tol:
                    if self.verbose:
                        print(f"[Adam-ICA] Early stop at epoch {epoch}, batch {b}.")
                    self.W_ = W
                    return self

            self.loss_curve_.append(epoch_loss / n_batches)
            if self.verbose:
                print(f"[Adam-ICA] Epoch {epoch:3d}  avg LL={self.loss_curve_[-1]:.6f}")

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
