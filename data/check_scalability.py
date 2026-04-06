import json

with open('c:/Users/phadi/Documents/Development/Reduction_dimension_projet/notebooks/demo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'scalab' in src.lower():
        print(f'--- Cellule {i} (type={cell["cell_type"]}) ---')
        print(src[:1000])
        print()
