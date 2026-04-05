# ICA Stochastique — Réduction de Dimension

Implémentation et comparaison d'algorithmes **ICA (Analyse en Composantes Indépendantes)** classiques et stochastiques, appliqués au problème de séparation aveugle de sources (BSS).

---

## Contexte

L'ICA traditionnelle repose sur des algorithmes déterministes (FastICA, Infomax batch). L'**optimisation stochastique** permet d'étendre l'ICA aux données massives et aux flux en remplaçant le calcul de gradient sur l'ensemble du dataset par des mises à jour sur mini-batches.

---

## Algorithmes implémentés

### Classiques (référence)

| Algorithme | Fichier | Description |
|---|---|---|
| **Infomax (batch)** | `ica/infomax.py` | Gradient naturel batch sur le critère Infomax (Bell & Sejnowski, 1995) |
| **FastICA (custom)** | `ica/fastica.py` | Algorithme à point fixe, déflation, nonlinéarités `logcosh / exp / cube` (Hyvärinen, 1999) |
| **FastICA (sklearn)** | `ica/fastica.py` | Wrapper `sklearn.decomposition.FastICA` — baseline de référence |

### Stochastiques (contributions principales)

| Algorithme | Fichier | Description |
|---|---|---|
| **SGD-ICA** | `ica/sgd_ica.py` | Descente de gradient stochastique (mini-batch) sur le critère Infomax ; schedules `constant / inv_sqrt / cosine` |
| **Adam-ICA** | `ica/adam_ica.py` | Optimiseur Adam (Kingma & Ba, 2015) appliqué au critère Infomax ; moments adaptatifs, bias-correction |

---

## Structure du projet

```
Reduction_dimension_projet/
├── ica/                        # Implémentations ICA
│   ├── __init__.py
│   ├── infomax.py              # Infomax batch (Bell & Sejnowski 1995)
│   ├── fastica.py              # FastICA custom + sklearn baseline
│   ├── sgd_ica.py              # SGD-ICA stochastique
│   └── adam_ica.py             # Adam-ICA stochastique
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
# Cloner le dépôt
git clone <url>
cd Reduction_dimension_projet

# Créer un environnement virtuel et installer les dépendances
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

---

## Démarrage rapide

### Utilisation de l'API Python

```python
import numpy as np
from ica import InfomaxICA, FastICACustom, SGDICA, AdamICA

# Données synthétiques (n_samples × n_features)
X = np.random.randn(1000, 3)

# --- Algorithme classique ---
model = InfomaxICA(n_components=3, learning_rate=0.01, max_iter=500, random_state=42)
S_hat = model.fit_transform(X)

# --- SGD stochastique ---
model = SGDICA(n_components=3, learning_rate=0.01, batch_size=64,
               n_epochs=100, lr_schedule='cosine', random_state=42)
S_hat = model.fit_transform(X)

# --- Adam stochastique ---
model = AdamICA(n_components=3, learning_rate=1e-3, batch_size=64,
                n_epochs=100, random_state=42)
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

### Notebook interactif

```bash
jupyter notebook notebooks/demo.ipynb
```

---

## Détails algorithmiques

### Critère Infomax

Tous les algorithmes basés sur Infomax maximisent la log-vraisemblance :

$$\mathcal{L}(W) = \log|\det W| + \frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{k} \log g'(w_j^\top x_i)$$

où $g(\cdot)$ est la sigmoïde logistique et $W$ est la matrice de démélange.

Le **gradient naturel** est :

$$\Delta W \propto \left(I + \varphi(y) y^\top\right) W, \quad \varphi(y) = 1 - 2\sigma(y)$$

### SGD-ICA

Chaque itération porte sur un mini-batch $\mathcal{B}_t \subset \{x_1,\ldots,x_n\}$ :

$$W_{t+1} = W_t + \eta_t \left(I + \frac{1}{|\mathcal{B}_t|}\varphi(W_t X_{\mathcal{B}_t}^\top) X_{\mathcal{B}_t}\right) W_t$$

Schedules disponibles pour $\eta_t$ : constant, $1/\sqrt{t}$, cosine annealing.

### Adam-ICA

Utilise les estimateurs de moments adaptatifs d'Adam avec bias-correction :

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$W_{t+1} = W_t + \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$

### Indice d'Amari

Métrique invariante aux permutations et aux échelles, mesure la qualité de séparation :

$$\text{Amari}(W, A) = \frac{1}{2k(k-1)}\left(\sum_i\frac{\sum_j |p_{ij}|}{\max_j|p_{ij}|} - 1 + \sum_j\frac{\sum_i|p_{ij}|}{\max_i|p_{ij}|} - 1\right)$$

où $P = WA^\top$. Une valeur proche de **0** indique une séparation parfaite.

---

## Références

- Bell, A. J., & Sejnowski, T. J. (1995). *An information-maximization approach to blind separation and blind deconvolution*. Neural Computation, 7(6), 1129–1159.
- Hyvärinen, A. (1999). *Fast and robust fixed-point algorithms for independent component analysis*. IEEE Trans. Neural Networks, 10(3), 626–634.
- Kingma, D. P., & Ba, J. (2015). *Adam: A method for stochastic optimization*. ICLR 2015.
- Amari, S., Cichocki, A., & Yang, H. H. (1996). *A new learning algorithm for blind signal separation*. NIPS 1995.
