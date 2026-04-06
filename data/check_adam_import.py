import json

with open('c:/Users/phadi/Documents/Development/Reduction_dimension_projet/notebooks/demo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'AdamICA' in src or 'SGDICA' in src or 'InfomaxICA' in src:
        print(f'Cellule {i}: {src[:200]}')
        print()
