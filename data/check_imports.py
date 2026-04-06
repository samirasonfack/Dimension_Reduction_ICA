import json

with open('c:/Users/phadi/Documents/Development/Reduction_dimension_projet/notebooks/demo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Affiche les 5 premieres cellules de code
count = 0
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        print(f'--- Cellule code {i} ---')
        print(src[:600])
        print()
        count += 1
        if count >= 5:
            break
