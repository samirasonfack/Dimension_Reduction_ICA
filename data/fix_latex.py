import json

with open('c:/Users/phadi/Documents/Development/Reduction_dimension_projet/notebooks/demo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ── Cell 20 : introduction VAE ───────────────────────────────────────────────
nb['cells'][20]['source'] = [
    "## 8. VAE-ICA — Séparation variationnelle non-linéaire\n",
    "\n",
    "Le VAE-ICA étend l'ICA classique à des **mélanges non-linéaires** :\n",
    "- L'**encodeur** MLP apprend $q_\\phi(z|x) = \\mathcal{N}(\\mu_\\phi(x), \\sigma^2_\\phi(x))$\n",
    "- Le **décodeur** MLP apprend $p_\\theta(x|z)$\n",
    "- La **loss** combine reconstruction + KL + HSIC (indépendance)\n",
    "\n",
    "$$\\mathcal{L} = \\underbrace{\\|x - \\hat{x}\\|^2}_{\\text{reconstruction}}"
    " + \\beta \\underbrace{\\text{KL}(q_\\phi \\| \\mathcal{N}(0,I))}_{\\text{gaussien}}"
    " + \\lambda \\underbrace{\\text{HSIC}(z_1, \\ldots, z_k)}_{\\text{indépendance}}$$\n",
    "\n",
    "**Tension fondamentale** : le KL pousse $z$ vers une gaussienne (VAE), "
    "mais l'ICA requiert des sources **non-gaussiennes**. "
    "On contrôle cela via $\\beta$ faible et $\\lambda$ élevé.",
]

# ── Cell 22 : effet beta ─────────────────────────────────────────────────────
nb['cells'][22]['source'] = [
    "### 8a. Effet de $\\beta$ sur la distribution de $z$\n",
    "\n",
    "- $\\beta$ élevé $\\Rightarrow$ KL dominant $\\Rightarrow$ $z \\sim \\mathcal{N}(0,I)$ "
    "(gaussien) $\\Rightarrow$ **incompatible ICA**\n",
    "- $\\beta$ faible $\\Rightarrow$ $z$ libre $\\Rightarrow$ peut être non-gaussien "
    "$\\Rightarrow$ **compatible ICA**",
]

# ── Cell 24 : effet lambda ───────────────────────────────────────────────────
nb['cells'][24]['source'] = [
    "### 8b. Effet de $\\lambda_{\\text{HSIC}}$ sur l'indépendance des composantes\n",
    "\n",
    "La matrice de corrélation de $z$ doit tendre vers $I$ (diagonale) quand $\\lambda$ augmente.\n",
    "\n",
    "$$\\text{HSIC}(z_a, z_b) = \\frac{1}{(n-1)^2} \\text{tr}(K_a H K_b H)$$\n",
    "\n",
    "où $K_a$ est la matrice de kernel RBF de la dimension $a$ et "
    "$H = I - \\frac{1}{n}\\mathbf{1}\\mathbf{1}^\\top$ est la matrice de centrage.",
]

# ── Cell 26 : convergence ────────────────────────────────────────────────────
nb['cells'][26]['source'] = [
    "### 8c. Courbe de convergence et composantes récupérées\n",
    "\n",
    "On compare les sources vraies $s_j$ avec les composantes estimées "
    "$\\hat{z}_c = \\mu_\\phi(x)_c$ (mode de $q_\\phi$, sans bruit $\\varepsilon$).\n",
    "\n",
    "L'alignement est fait par corrélation maximale (ICA ne garantit ni l'ordre ni le signe) :\n",
    "\n",
    "$$c^* = \\arg\\max_c |\\text{corr}(s_j, \\hat{z}_c)|$$",
]

with open('c:/Users/phadi/Documents/Development/Reduction_dimension_projet/notebooks/demo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('LaTeX corrige dans les cellules 20, 22, 24, 26')
