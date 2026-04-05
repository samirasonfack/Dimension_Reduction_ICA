"""
SGD-ICA — Stochastic Gradient Descent applied to Infomax
==========================================================
Mini-batch stochastic gradient ascent on the Infomax log-likelihood.
This approach scales to large datasets and streaming data because it
processes one mini-batch at a time rather than the entire dataset.

Compared to batch Infomax:
  - Each gradient step uses a mini-batch of size ``batch_size``.
  - A learning-rate schedule (constant, 1/t, or cosine) is supported.
  - Useful for large / streaming datasets.
"""

import numpy as np


class SGDICA:
    """
    Mini-batch SGD applied to the Infomax ICA criterion.

    Parameters
    ----------
    n_components : int or None
        Number of independent components.
    learning_rate : float
        Initial step size.
    batch_size : int
        Number of samples per mini-batch.
    n_epochs : int
        Number of full passes over the data.
    lr_schedule : {'constant', 'inv_sqrt', 'cosine'}
        Learning-rate decay schedule.
        - ``'constant'`` : η_t = η_0
        - ``'inv_sqrt'`` : η_t = η_0 / sqrt(t+1)
        - ``'cosine'``   : cosine annealing over total steps
    tol : float
        Stop early when the norm of the weight update is below this value.
    whiten : bool
        Whether to whiten the data.
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

        self.W_ = None
        self.whitening_matrix_ = None
        self.mean_ = None
        self.loss_curve_ = []

    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def _get_lr(self, step: int, total_steps: int) -> float:
        η0 = self.learning_rate
        if self.lr_schedule == "constant":
            return η0
        if self.lr_schedule == "inv_sqrt":
            return η0 / np.sqrt(step + 1)
        # cosine
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

    def _batch_gradient(self, W: np.ndarray, X_batch: np.ndarray) -> np.ndarray:
        """Natural gradient of the Infomax LL w.r.t. W on a mini-batch."""
        m = X_batch.shape[0]
        y = X_batch @ W.T                 # (m × k)
        phi = 1 - 2 * self._sigmoid(y)    # (m × k)  score function
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
                grad = self._batch_gradient(W, X_batch)
                W_new = W + lr * grad

                delta = np.linalg.norm(W_new - W, "fro")
                W = W_new
                step += 1

                # Log-likelihood on this batch
                sign, logdet = np.linalg.slogdet(W)
                y = X_batch @ W.T
                ll = logdet + np.sum(np.log(self._sigmoid(y) * (1 - self._sigmoid(y)) + 1e-12)) / X_batch.shape[0]
                epoch_loss += ll

                if delta < self.tol:
                    if self.verbose:
                        print(f"[SGD-ICA] Early stop at epoch {epoch}, batch {b}.")
                    self.W_ = W
                    return self

            self.loss_curve_.append(epoch_loss / n_batches)
            if self.verbose:
                print(f"[SGD-ICA] Epoch {epoch:3d}  avg LL={self.loss_curve_[-1]:.6f}  lr={lr:.2e}")

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
