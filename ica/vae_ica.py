"""
VAE-ICA — Variational Autoencoder pour la séparation de sources
================================================================

Convention (cours)
------------------
    x = W s          (W : matrice de mélange, inconnue)
    s ≈ ẑ            (l'encodeur apprend la séparation non-linéaire)

Architecture
------------
    Encodeur q_φ(z|x) : MLP  →  μ_φ(x), log σ²_φ(x)
    Reparamétrage      : z = μ + σ · ε,   ε ~ N(0, I)
    Décodeur p_θ(x|z)  : MLP  →  x̂

Loss (ELBO + indépendance)
--------------------------
    L = Reconstruction  +  β · KL  +  λ · HSIC

    ─ Reconstruction : MSE(x, x̂)
    ─ KL             : KL(q_φ(z|x) ‖ N(0,I))    régularise z vers gaussien
    ─ HSIC           : mesure la dépendance entre les dimensions z_j
                       → minimiser HSIC = pousser vers l'indépendance

Pourquoi HSIC et pas juste le KL ?
------------------------------------
Le prior N(0,I) suppose des dimensions indépendantes, mais ne le garantit
pas dans les z appris. Le terme HSIC force *explicitement* l'indépendance.

Implémentation : PyTorch avec autograd (gradients exacts, rapide).

Référence HSIC :
    Gretton et al. (2005). Measuring Statistical Dependence with
    Hilbert-Schmidt Norms. ALT 2005.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# HSIC (PyTorch) — gradient analytique via autograd
# ---------------------------------------------------------------------------

def _hsic_torch(Z: "torch.Tensor", sigma: float = 1.0) -> "torch.Tensor":
    """
    HSIC total entre toutes les paires de dimensions de Z.

    Z : (n, k) Tensor
    HSIC(a, b) = 1/(n−1)² · tr(K_a H K_b H)

    Le kernel RBF 1D K_a[i,j] = exp(−(z_ia − z_ja)² / 2σ²).
    H = I − 11ᵀ/n  (matrice de centrage).
    """
    import torch
    n, k = Z.shape
    H = torch.eye(n, device=Z.device) - torch.ones(n, n, device=Z.device) / n

    total = torch.tensor(0.0, device=Z.device)
    for a in range(k):
        za = Z[:, a]
        D_a = (za.unsqueeze(1) - za.unsqueeze(0)) ** 2  # (n, n)
        K_a = torch.exp(-D_a / (2 * sigma ** 2))

        for b in range(a + 1, k):
            zb = Z[:, b]
            D_b = (zb.unsqueeze(1) - zb.unsqueeze(0)) ** 2
            K_b = torch.exp(-D_b / (2 * sigma ** 2))
            total = total + torch.trace(K_a @ H @ K_b @ H) / (n - 1) ** 2

    return total


# ---------------------------------------------------------------------------
# Modules PyTorch
# ---------------------------------------------------------------------------

class _Encoder(nn.Module):
    """MLP : d → hidden → 2k  (μ et log σ² concaténés)"""

    def __init__(self, d_in: int, d_hidden: int, k: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 2 * k),
        )
        self._k = k

    def forward(self, x):
        out = self.net(x)
        mu      = out[:, : self._k]
        log_var = torch.clamp(out[:, self._k:], -10, 2)
        return mu, log_var


class _Decoder(nn.Module):
    """MLP : k → hidden → d"""

    def __init__(self, k: int, d_hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, z):
        return self.net(z)


# ---------------------------------------------------------------------------
# VAE-ICA
# ---------------------------------------------------------------------------

class VAEICA:
    """
    VAE-ICA : encodeur variationnel pour la séparation de sources.

    L'encodeur apprend q_φ(z|x) = N(μ_φ(x), diag(σ²_φ(x))).
    Le décodeur apprend p_θ(x|z).
    La loss pénalise la dépendance entre les z_j via HSIC.

    Modèle : x = Ws  →  ẑ = g_φ(x) ≈ s   (séparation non-linéaire)

    Parameters
    ----------
    n_components : int
        Dimension de l'espace latent (= nombre de sources).
    hidden_dim : int
        Taille des couches cachées des MLP.
    learning_rate : float
        Pas de l'optimiseur Adam.
    beta : float
        Poids du terme KL.
    lambda_hsic : float
        Poids du terme HSIC (indépendance).
    batch_size : int
    n_epochs : int
    sigma_hsic : float
        Bandwidth du kernel RBF pour HSIC.
    random_state : int or None
    verbose : bool
    """

    def __init__(
        self,
        n_components: int = None,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        beta: float = 1.0,
        lambda_hsic: float = 1.0,
        batch_size: int = 64,
        n_epochs: int = 50,
        sigma_hsic: float = 1.0,
        random_state: int = None,
        verbose: bool = False,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch requis : pip install torch")

        self.n_components = n_components
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.beta = beta
        self.lambda_hsic = lambda_hsic
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.sigma_hsic = sigma_hsic
        self.random_state = random_state
        self.verbose = verbose

        self.encoder_ = None
        self.decoder_ = None
        self.mean_ = None
        self.std_ = None
        self.loss_curve_ = []
        self.V_ = None    # approximation linéaire de la séparatrice
        self.W_ = None

    # ------------------------------------------------------------------
    # Reparameterization trick
    # ------------------------------------------------------------------

    @staticmethod
    def _reparametrize(mu, log_var):
        """z = μ + σ · ε,  ε ~ N(0, I)"""
        import torch
        sigma = torch.exp(0.5 * log_var)
        eps   = torch.randn_like(sigma)
        return mu + sigma * eps

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _loss(self, x, x_hat, mu, log_var, z):
        import torch
        n = x.shape[0]

        # Reconstruction MSE
        L_rec = nn.functional.mse_loss(x_hat, x, reduction="mean")

        # KL divergence : KL(N(μ,σ²) || N(0,1))
        L_kl = 0.5 * torch.mean(mu ** 2 + torch.exp(log_var) - log_var - 1)

        # HSIC : indépendance entre les dimensions de z
        L_hsic = _hsic_torch(z, self.sigma_hsic)

        total = L_rec + self.beta * L_kl + self.lambda_hsic * L_hsic
        return total, L_rec.item(), L_kl.item(), L_hsic.item()

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray):
        import torch

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)

        n, d = X.shape
        if self.n_components is None:
            self.n_components = d

        # Normalisation
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0) + 1e-8
        X_norm = (X - self.mean_) / self.std_

        k = self.n_components
        self.encoder_ = _Encoder(d, self.hidden_dim, k)
        self.decoder_ = _Decoder(k, self.hidden_dim, d)

        optimizer = optim.Adam(
            list(self.encoder_.parameters()) + list(self.decoder_.parameters()),
            lr=self.learning_rate,
        )

        X_t = torch.tensor(X_norm, dtype=torch.float32)
        n_batches = max(1, n // self.batch_size)
        self.loss_curve_ = []

        self.encoder_.train()
        self.decoder_.train()

        for epoch in range(self.n_epochs):
            perm = rng.permutation(n)
            X_shuf = X_t[perm]
            epoch_loss = 0.0

            for b in range(n_batches):
                xb = X_shuf[b * self.batch_size: (b + 1) * self.batch_size]

                optimizer.zero_grad()

                mu, log_var = self.encoder_(xb)
                z           = self._reparametrize(mu, log_var)
                x_hat       = self.decoder_(z)

                loss, L_rec, L_kl, L_hsic = self._loss(xb, x_hat, mu, log_var, z)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            self.loss_curve_.append(epoch_loss / n_batches)
            if self.verbose:
                print(f"[VAE-ICA] Epoch {epoch:3d}  "
                      f"loss={self.loss_curve_[-1]:.4f}  "
                      f"rec={L_rec:.4f}  kl={L_kl:.4f}  hsic={L_hsic:.4f}")

        # Approximation linéaire de V pour le calcul de l'indice d'Amari
        Z_all = self.transform(X)
        self.V_ = np.linalg.lstsq(X_norm, Z_all, rcond=None)[0].T   # (k, d)
        self.W_ = np.linalg.pinv(self.V_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode x → μ (mode de q_φ(z|x)), sans bruit."""
        import torch
        X_norm = (X - self.mean_) / self.std_
        X_t    = torch.tensor(X_norm, dtype=torch.float32)
        self.encoder_.eval()
        with torch.no_grad():
            mu, _ = self.encoder_(X_t)
        return mu.numpy()

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
