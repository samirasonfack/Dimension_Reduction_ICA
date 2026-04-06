import json, re, sys

with open('c:/Users/phadi/Documents/Development/Reduction_dimension_projet/notebooks/demo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'VAEICA' in src:
        matches = re.findall(r'beta\s*=\s*[0-9.]+', src)
        if matches:
            sys.stdout.buffer.write(('Cellule ' + str(i) + ': ' + str(matches) + '\n').encode('utf-8'))
