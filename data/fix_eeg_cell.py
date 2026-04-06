import json

with open('c:/Users/phadi/Documents/Development/Reduction_dimension_projet/notebooks/demo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'][35]['source'] = [
    "# Sous-echantillonnage agressif : HSIC est O(batch^2 * k^2)\n",
    "N_SUB = 2000     # echantillons d'entrainement\n",
    "idx = np.random.default_rng(42).choice(len(X_eeg), N_SUB, replace=False)\n",
    "X_eeg_sub = X_eeg[idx]\n",
    "\n",
    "t0 = time.perf_counter()\n",
    "vae_eeg = VAEICA(\n",
    "    n_components=n_components_eeg,\n",
    "    hidden_dim=32,        # plus leger\n",
    "    learning_rate=1e-3,\n",
    "    beta=0.3,\n",
    "    lambda_hsic=1.0,\n",
    "    batch_size=64,        # batch plus petit = moins de calcul kernel\n",
    "    n_epochs=10,          # peu d'epochs\n",
    "    sigma_hsic=2.0,\n",
    "    random_state=42,\n",
    "    verbose=True,\n",
    ")\n",
    "vae_eeg.fit(X_eeg_sub)\n",
    "\n",
    "# Inference sur un sous-ensemble pour la visu\n",
    "N_INFER = 10000\n",
    "S_vae_eeg = vae_eeg.transform(X_eeg[:N_INFER])\n",
    "t_vae = time.perf_counter() - t0\n",
    "\n",
    "print(f'VAE-ICA  --  {t_vae:.2f}s  |  shape: {S_vae_eeg.shape}')\n",
    "\n",
    "fig, axes = plt.subplots(n_components_eeg, 1, figsize=(14, n_components_eeg * 0.9), sharex=True)\n",
    "fig.suptitle(f'VAE-ICA -- {n_components_eeg} composantes EEG', fontweight='bold', fontsize=12)\n",
    "for i, ax in enumerate(axes):\n",
    "    ax.plot(t_eeg[:3000], S_vae_eeg[:3000, i], lw=0.5, color='#8e44ad')\n",
    "    ax.set_ylabel(f'IC{i+1}', fontsize=7, rotation=0, labelpad=25)\n",
    "    ax.set_yticks([])\n",
    "axes[-1].set_xlabel('Temps (s)')\n",
    "plt.tight_layout()\n",
    "plt.show()",
]

# Aussi alleger la cellule de comparaison (cell 37) pour n'utiliser que N_INFER samples
nb['cells'][37]['source'] = [
    "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
    "\n",
    "# Correlation signal brut (reference)\n",
    "corr_raw = np.corrcoef(X_eeg[:N_INFER, :n_components_eeg].T)\n",
    "im0 = axes[0].imshow(np.abs(corr_raw), vmin=0, vmax=1, cmap='hot_r')\n",
    "axes[0].set_title(f'EEG brut\\n(moy|corr|={np.abs(corr_raw[np.triu_indices(n_components_eeg,k=1)]).mean():.3f})')\n",
    "plt.colorbar(im0, ax=axes[0])\n",
    "\n",
    "# FastICA\n",
    "corr_fastica = np.corrcoef(S_fastica_eeg[:N_INFER].T)\n",
    "off_f = np.abs(corr_fastica[np.triu_indices(n_components_eeg, k=1)]).mean()\n",
    "im1 = axes[1].imshow(np.abs(corr_fastica), vmin=0, vmax=1, cmap='hot_r')\n",
    "axes[1].set_title(f'FastICA sklearn\\n(moy|corr|={off_f:.3f})  {t_fastica:.1f}s')\n",
    "plt.colorbar(im1, ax=axes[1])\n",
    "\n",
    "# VAE-ICA\n",
    "corr_vae = np.corrcoef(S_vae_eeg.T)\n",
    "off_v = np.abs(corr_vae[np.triu_indices(n_components_eeg, k=1)]).mean()\n",
    "im2 = axes[2].imshow(np.abs(corr_vae), vmin=0, vmax=1, cmap='hot_r')\n",
    "axes[2].set_title(f'VAE-ICA\\n(moy|corr|={off_v:.3f})  {t_vae:.1f}s')\n",
    "plt.colorbar(im2, ax=axes[2])\n",
    "\n",
    "for ax in axes:\n",
    "    ax.set_xlabel('Composante'); ax.set_ylabel('Composante')\n",
    "\n",
    "fig.suptitle('Matrice de correlation absolue entre composantes ICA (EEG reel)', fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(f'EEG brut   corr moy = {np.abs(corr_raw[np.triu_indices(n_components_eeg,k=1)]).mean():.4f}')\n",
    "print(f'FastICA    corr moy = {off_f:.4f}   ({t_fastica:.2f}s)')\n",
    "print(f'VAE-ICA    corr moy = {off_v:.4f}   ({t_vae:.2f}s)')",
]

with open('c:/Users/phadi/Documents/Development/Reduction_dimension_projet/notebooks/demo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Cellules EEG allegees OK')
print('  N_SUB train  : 2000')
print('  batch_size   : 64')
print('  n_epochs     : 10')
print('  n_batches    : 2000/64 = 31')
print('  total passes : 31 x 10 = 310')
print('  N_INFER      : 10000 samples pour inference/visu')
