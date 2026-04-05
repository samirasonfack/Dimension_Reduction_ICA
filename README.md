# ICA Stochastique — Réduction de Dimension

Implémentation et comparaison d'algorithmes **ICA (Analyse en Composantes Indépendantes)** classiques, stochastiques et variationnels, appliqués au problème de séparation aveugle de sources (BSS).

---

## Convention de notation

| Symbole | Signification | Dimension |
|---|---|---|
| **x** | Signal observé (mélange) | ℝ^d |
| **s** | Sources indépendantes | ℝ^k |
| **W** | Matrice de **mélange** : `x = Ws` | d × k |
| **V** | Matrice **séparatrice** : `ŝ = Vx` | k × d |
| n | Nombre d'échantillons | — |
| k | Nombre de sources | — |
| d | Dimension de x (nb capteurs) | — |

L'objectif de l'ICA est d'estimer **V** telle que `VW ≈ I` (à permutation et échelle près), en exploitant uniquement l'indépendance des sources.

---

## Contexte

L'ICA traditionnelle repose sur des algorithmes déterministes (FastICA, Infomax batch). L'**optimisation stochastique** permet d'étendre l'ICA aux données massives et aux flux en remplaçant le calcul de gradient sur l'ensemble du dataset par des mises à jour sur mini-batches. L'approche **variationnelle (VAE-ICA)** généralise à des mélanges non-linéaires.

---

## Algorithmes implémentés

### Classiques (référence)

| Algorithme | Fichier | Nonlinéarité g | Description |
|---|---|---|---|
| **Infomax (batch)** | `ica/infomax.py` | `tanh` (sech²) | Gradient naturel batch sur Infomax (Bell & Sejnowski, 1995) |
| **FastICA (custom)** | `ica/fastica.py` | `logcosh` (tanh) / `exp` / `cube` | Algorithme à point fixe de Newton, déflation (Hyvärinen, 1999) |
| **FastICA (sklearn)** | `ica/fastica.py` | configurable | Wrapper `sklearn.decomposition.FastICA` — baseline |

### Stochastiques (contributions principales)

| Algorithme | Fichier | Nonlinéarité g | Description |
|---|---|---|---|
| **SGD-ICA** | `ica/sgd_ica.py` | `tanh` | Mini-batch SGD sur Infomax ; schedules `constant / inv_sqrt / cosine` |
| **Adam-ICA** | `ica/adam_ica.py` | `tanh` | Optimiseur Adam sur Infomax ; moments adaptatifs, bias-correction |

### Variationnel (extension non-linéaire)

| Algorithme | Fichier | Description |
|---|---|---|
| **VAE-ICA** | `ica/vae_ica.py` | Encodeur MLP apprend q_φ(z\|x), loss = reconstruction + β·KL + λ·HSIC |

---

## Structure du projet

```
Reduction_dimension_projet/
├── ica/                        # Implémentations ICA
│   ├── __init__.py
│   ├── infomax.py              # Infomax batch — g = tanh (sech²)
│   ├── fastica.py              # FastICA custom (g ∈ {logcosh, exp, cube}) + sklearn
│   ├── sgd_ica.py              # SGD-ICA stochastique — g = tanh
│   ├── adam_ica.py             # Adam-ICA stochastique — g = tanh
│   └── vae_ica.py              # VAE-ICA variationnel (PyTorch)
├── experiments/                # Benchmark et visualisations
│   ├── __init__.py
│   ├── benchmark.py            # Génération de données, Amari index, runner
│   └── visualization.py        # Plots : sources, convergence, Amari scores
├── notebooks/
│   └── demo.ipynb              # Notebook de démonstration complet
├── data/                       # Données (exclues du git par défaut)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```bash
git clone <url>
cd Reduction_dimension_projet

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

---

## Démarrage rapide

### Algorithmes classiques et stochastiques

```python
import numpy as np
from ica import InfomaxICA, FastICACustom, SGDICA, AdamICA

X = np.random.randn(1000, 3)   # (n_samples × n_features)

# Infomax batch — V_ : séparatrice,  W_ : mélange estimé
model = InfomaxICA(n_components=3, learning_rate=0.01, max_iter=500, random_state=42)
S_hat = model.fit_transform(X)   # ŝ = V x
print(model.V_.shape)            # (k, d)
print(model.W_.shape)            # (d, k) = pinv(V_)

# FastICA — choix de la nonlinéarité g
model = FastICACustom(n_components=3, g='logcosh', random_state=42)

# SGD stochastique
model = SGDICA(n_components=3, learning_rate=0.01, batch_size=64,
               n_epochs=100, lr_schedule='cosine', random_state=42)

# Adam stochastique
model = AdamICA(n_components=3, learning_rate=1e-3, batch_size=64,
                n_epochs=100, random_state=42)
```

### VAE-ICA (non-linéaire)

```python
from ica import VAEICA

model = VAEICA(
    n_components=3,
    hidden_dim=64,
    learning_rate=1e-3,
    beta=1.0,          # poids KL
    lambda_hsic=5.0,   # poids HSIC (indépendance)
    n_epochs=100,
    random_state=42,
)
S_hat = model.fit_transform(X)
```

### Benchmark complet

```python
from experiments.benchmark import run_benchmark
from experiments.visualization import plot_sources, plot_convergence, plot_amari_scores

results = run_benchmark(n_samples=3000, n_components=3, verbose=True)
plot_amari_scores(results)
plot_convergence(results)
```

---

## Détails algorithmiques

### Critère Infomax (Infomax, SGD-ICA, Adam-ICA)

Maximisation de la log-vraisemblance avec nonlinéarité **g = tanh** (prior sech²) :

$$\mathcal{L}(V) = \log|\det V| + \frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{k} \log(1 - \tanh^2(v_j^\top x_i))$$

Gradient naturel (identique pour les 3 algorithmes, différent dans la mise à jour) :

$$\Delta V = \left(I - \frac{1}{m}\tanh(Vx)\,(Vx)^\top\right) V$$

### FastICA — Règle de Newton

Algorithme à point fixe, composante par composante. La nonlinéarité **g** choisie définit implicitement la distribution des sources :

| g | g'(u) | g''(u) | Distribution implicite |
|---|---|---|---|
| `logcosh` | tanh(u) | 1 − tanh²(u) = sech²(u) | Logistique (super-gaussienne) |
| `exp` | u·exp(−u²/2) | (1−u²)·exp(−u²/2) | Super-gaussienne |
| `cube` | u³ | 3u² | Kurtosis (super-gaussienne) |

Mise à jour (déflation de Gram-Schmidt) :

$$v_{\text{new}} = \mathbb{E}[x\, g(v^\top x)] - \mathbb{E}[g'(v^\top x)]\, v$$

### Adam-ICA vs SGD-ICA

Adam ajoute deux accumulateurs de moments au gradient naturel :

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t \qquad \text{(direction lissée)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2 \qquad \text{(variance adaptative)}$$
$$V_{t+1} = V_t + \alpha\, \hat{m}_t / (\sqrt{\hat{v}_t} + \varepsilon)$$

### VAE-ICA — Approche variationnelle

Encodeur MLP apprend $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma^2_\phi(x))$.  
Reparameterization trick : $z = \mu + \sigma \cdot \varepsilon$, $\varepsilon \sim \mathcal{N}(0,I)$.

$$\mathcal{L} = \underbrace{\|x - \hat{x}\|^2}_{\text{reconstruction}} + \beta\underbrace{\,\text{KL}(q_\phi(z|x)\,\|\,\mathcal{N}(0,I))}_{\text{régularisation}} + \lambda\underbrace{\,\text{HSIC}(z_1,\ldots,z_k)}_{\text{indépendance}}$$

Le terme **HSIC** (Hilbert-Schmidt Independence Criterion) pénalise explicitement la dépendance entre dimensions de z, là où le KL seul ne le garantit pas.

### Indice d'Amari

Métrique invariante aux permutations et aux échelles, mesure la qualité de séparation :

$$\text{Amari}(V, W) = \frac{1}{2k(k-1)}\left(\sum_i\frac{\sum_j |p_{ij}|}{\max_j|p_{ij}|} - 1 + \sum_j\frac{\sum_i|p_{ij}|}{\max_i|p_{ij}|} - 1\right)$$

où $P = VW^\top$. Une valeur proche de **0** indique une séparation parfaite.

---

## Références

- Bell, A. J., & Sejnowski, T. J. (1995). *An information-maximization approach to blind separation and blind deconvolution*. Neural Computation, 7(6), 1129–1159.
- Hyvärinen, A. (1999). *Fast and robust fixed-point algorithms for independent component analysis*. IEEE Trans. Neural Networks, 10(3), 626–634.
- Kingma, D. P., & Ba, J. (2015). *Adam: A method for stochastic optimization*. ICLR 2015.
- Gretton, A. et al. (2005). *Measuring Statistical Dependence with Hilbert-Schmidt Norms*. ALT 2005.
- Amari, S., Cichocki, A., & Yang, H. H. (1996). *A new learning algorithm for blind signal separation*. NIPS 1995.
- Khemakhem, I. et al. (2020). *Variational Autoencoders and Nonlinear ICA: A Unifying Framework*. AISTATS 2020.
