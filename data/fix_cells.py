import json

with open('c:/Users/phadi/Documents/Development/Reduction_dimension_projet/notebooks/demo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ── Cell 23 : boucle betas allégée ──────────────────────────────────────────
nb['cells'][23]['source'] = [
    "from scipy import stats\n",
    "\n",
    "betas = [0.0, 0.5, 2.0]   # 3 valeurs seulement\n",
    "fig, axes = plt.subplots(len(betas), 3, figsize=(12, 3 * len(betas)))\n",
    "fig.suptitle('Distribution de z selon β (λ_HSIC=5, 20 epochs)', fontsize=13, fontweight='bold')\n",
    "\n",
    "for row, beta in enumerate(betas):\n",
    "    m = VAEICA(n_components=3, hidden_dim=32, learning_rate=1e-3,\n",
    "               beta=beta, lambda_hsic=5.0, n_epochs=20, batch_size=128,\n",
    "               random_state=42, verbose=False)\n",
    "    Z = m.fit_transform(X)\n",
    "    for col in range(3):\n",
    "        ax = axes[row, col]\n",
    "        z_j = Z[:, col]\n",
    "        ax.hist(z_j, bins=40, density=True, alpha=0.6, color=colors[col])\n",
    "        xg = np.linspace(z_j.min(), z_j.max(), 200)\n",
    "        ax.plot(xg, stats.norm.pdf(xg, z_j.mean(), z_j.std()), 'k--', lw=1.5)\n",
    "        if col == 0:\n",
    "            ax.set_ylabel(f'β={beta}', fontsize=10, fontweight='bold')\n",
    "        if row == 0:\n",
    "            ax.set_title(f'z_{col+1}', fontsize=10)\n",
    "        ax.set_yticks([])\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()",
]

# ── Cell 25 : boucle lambdas allégée ────────────────────────────────────────
nb['cells'][25]['source'] = [
    "lambdas = [0.0, 5.0, 20.0]\n",
    "fig, axes = plt.subplots(1, len(lambdas), figsize=(12, 3.5))\n",
    "fig.suptitle('Corrélation de z selon λ_HSIC (β=0.5, 20 epochs)', fontsize=12, fontweight='bold')\n",
    "\n",
    "for ax, lam in zip(axes, lambdas):\n",
    "    m = VAEICA(n_components=3, hidden_dim=32, learning_rate=1e-3,\n",
    "               beta=0.5, lambda_hsic=lam, n_epochs=20, batch_size=128,\n",
    "               random_state=42, verbose=False)\n",
    "    Z = m.fit_transform(X)\n",
    "    corr = np.corrcoef(Z.T)\n",
    "    off = corr[np.triu_indices(3, k=1)]\n",
    "    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')\n",
    "    ax.set_title(f'λ={lam}\\nmoy|corr|={np.abs(off).mean():.3f}', fontsize=10)\n",
    "    ax.set_xticks([0,1,2]); ax.set_xticklabels(['z1','z2','z3'])\n",
    "    ax.set_yticks([0,1,2]); ax.set_yticklabels(['z1','z2','z3'])\n",
    "    for i in range(3):\n",
    "        for j in range(3):\n",
    "            ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center', fontsize=9)\n",
    "\n",
    "plt.colorbar(im, ax=axes[-1], fraction=0.046)\n",
    "plt.tight_layout()\n",
    "plt.show()",
]

# ── Cell 35 : VAE-ICA EEG — sous-échantillonnage ────────────────────────────
nb['cells'][35]['source'] = [
    "# Sous-échantillonnage : HSIC sur n×n est O(n²) → on limite à 5000 points\n",
    "N_SUB = 5000\n",
    "idx = np.random.default_rng(42).choice(len(X_eeg), N_SUB, replace=False)\n",
    "X_eeg_sub = X_eeg[idx]\n",
    "\n",
    "t0 = time.perf_counter()\n",
    "vae_eeg = VAEICA(\n",
    "    n_components=n_components_eeg,\n",
    "    hidden_dim=64,\n",
    "    learning_rate=1e-3,\n",
    "    beta=0.3,\n",
    "    lambda_hsic=2.0,\n",
    "    batch_size=128,\n",
    "    n_epochs=15,        # réduit pour la RAM\n",
    "    sigma_hsic=2.0,     # bandwidth large = noyau plus lisse\n",
    "    random_state=42,\n",
    "    verbose=True,\n",
    ")\n",
    "vae_eeg.fit(X_eeg_sub)          # entraînement sur sous-échantillon\n",
    "S_vae_eeg = vae_eeg.transform(X_eeg)  # inférence sur tout le signal\n",
    "t_vae = time.perf_counter() - t0\n",
    "\n",
    "print(f'\\nVAE-ICA  —  {t_vae:.2f}s  |  shape: {S_vae_eeg.shape}')\n",
    "\n",
    "fig, axes = plt.subplots(n_components_eeg, 1, figsize=(14, n_components_eeg * 0.9), sharex=True)\n",
    "fig.suptitle(f'VAE-ICA — {n_components_eeg} composantes EEG', fontweight='bold', fontsize=12)\n",
    "for i, ax in enumerate(axes):\n",
    "    ax.plot(t_eeg[:3000], S_vae_eeg[:3000, i], lw=0.5, color='#8e44ad')\n",
    "    ax.set_ylabel(f'IC{i+1}', fontsize=7, rotation=0, labelpad=25)\n",
    "    ax.set_yticks([])\n",
    "axes[-1].set_xlabel('Temps (s)')\n",
    "plt.tight_layout()\n",
    "plt.show()",
]

with open('c:/Users/phadi/Documents/Development/Reduction_dimension_projet/notebooks/demo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Cellules allégées OK')
print(f'Total cellules: {len(nb["cells"])}')
