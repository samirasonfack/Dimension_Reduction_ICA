import json, sys

with open('c:/Users/phadi/Documents/Development/Reduction_dimension_projet/notebooks/demo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][23]['source'])
sys.stdout.buffer.write(src.encode('utf-8'))
