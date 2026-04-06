import json

with open('c:/Users/phadi/Documents/Development/Reduction_dimension_projet/notebooks/demo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src19 = ''.join(nb['cells'][19]['source'])

# Ajouter les imports en tete de cellule si pas deja presents
imports = """from ica import InfomaxICA, SGDICA, AdamICA
from experiments.benchmark import make_sources
import time

"""

if 'from ica import' not in src19:
    nb['cells'][19]['source'] = [imports + src19]
    print("Imports ajoutes a la cellule 19.")
else:
    print("Imports deja presents.")

with open('c:/Users/phadi/Documents/Development/Reduction_dimension_projet/notebooks/demo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

# Verification
with open('c:/Users/phadi/Documents/Development/Reduction_dimension_projet/notebooks/demo.ipynb', 'r', encoding='utf-8') as f:
    nb2 = json.load(f)
print('--- Debut cellule 19 ---')
print(''.join(nb2['cells'][19]['source'])[:300])
