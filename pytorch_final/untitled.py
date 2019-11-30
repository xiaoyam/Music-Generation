import IPython.nbformat.current as nbf
nb = nbf.read(open('a.py', 'r'), 'py')
nbf.write(nb, open('a.ipynb', 'w'), 'ipynb')