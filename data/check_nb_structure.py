import json, sys

with open('c:/Users/phadi/Documents/Development/Reduction_dimension_projet/notebooks/demo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    ctype = cell['cell_type']
    preview = src[:120].replace('\n', ' ').encode('ascii', errors='replace').decode()
    print(f"[{i:02d}] {ctype:8s} | {preview}")
