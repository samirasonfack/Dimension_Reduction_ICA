import json

with open('c:/Users/phadi/Documents/Development/Reduction_dimension_projet/notebooks/demo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ── Cellule 35 : VAE-ICA sur EEG avec sous-echantillonnage ──
nb['cells'][35]['source'] = """# Sous-echantillonnage agressif : HSIC est O(batch^2 * k^2)
N_SUB = 2000     # echantillons d'entrainement
N_INFER = 10000  # echantillons pour l'inference / plots
idx = np.random.default_rng(42).choice(len(X_eeg), N_SUB, replace=False)
X_eeg_sub = X_eeg[idx]

t0 = time.perf_counter()
vae_eeg = VAEICA(
    n_components=n_components_eeg,
    hidden_dim=32,        # plus leger
    learning_rate=1e-3,
    beta=0.3,
    lambda_hsic=1.0,
    batch_size=64,        # batch plus petit = moins de calcul kernel
    n_epochs=10,
    sigma_hsic=2.0,
    random_state=42,
    verbose=True,
)
vae_eeg.fit(X_eeg_sub)
t_vae = time.perf_counter() - t0

# Inference sur un sous-ensemble pour la visu
S_vae_eeg = vae_eeg.transform(X_eeg[:N_INFER])
print(f"VAE-ICA : {t_vae:.2f}s  |  shape: {S_vae_eeg.shape}")

fig, axes = plt.subplots(n_components_eeg, 1, figsize=(14, 3 * n_components_eeg))
for i, ax in enumerate(axes):
    ax.plot(S_vae_eeg[:2000, i], lw=0.5)
    ax.set_title(f"VAE-ICA composante {i+1}")
    ax.set_xlabel("Echantillon")
plt.tight_layout()
plt.show()
""".splitlines(keepends=True)

# ── Cellule 37 : Comparaison correlation – utiliser N_INFER ──
src37 = ''.join(nb['cells'][37]['source'])
src37 = src37.replace(
    'np.corrcoef(X_eeg[:, :n_components_eeg].T)',
    'np.corrcoef(X_eeg[:N_INFER, :n_components_eeg].T)'
)
src37 = src37.replace(
    'np.corrcoef(S_fastica_eeg.T)',
    'np.corrcoef(S_fastica_eeg[:N_INFER].T)'
)
nb['cells'][37]['source'] = [src37]

with open('c:/Users/phadi/Documents/Development/Reduction_dimension_projet/notebooks/demo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Cellules 35 et 37 mises a jour avec succes.")

# Verification
with open('c:/Users/phadi/Documents/Development/Reduction_dimension_projet/notebooks/demo.ipynb', 'r', encoding='utf-8') as f:
    nb2 = json.load(f)
src35 = ''.join(nb2['cells'][35]['source'])
print("\n=== Cellule 35 (debut) ===")
print(src35[:300])
