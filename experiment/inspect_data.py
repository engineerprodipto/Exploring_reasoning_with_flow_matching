import numpy as np
import json
from pathlib import Path
 
 
# ----- Sudoku -------
sudoku_train = Path("../repos/trm/TinyRecursiveModels/data/sudoku-extreme-1k/train")

x = np.load(sudoku_train/"all__inputs.npy")
y = np.load(sudoku_train/"all__labels.npy")

print("====== SUDOKU ======")
print("inputs shape:", x.shape, "dtype:", x.dtype)
print("labels shape:", y.shape, "dtype:", y.dtype)
print("inputs unique:", np.unique(x))
print("labels unique:", np.unique(y))

with open(sudoku_train/"dataset.json") as f:
    meta = json.load(f)
    print("dataset.json keys:", meta.keys())
    

# ------- Maze -------
maze_train = Path("../repos/trm/TinyRecursiveModels/data/maze-30x30-hard-1k/train")

xm = np.load(maze_train/"all__inputs.npy")
ym = np.load(maze_train/"all__labels.npy")  

print("====== MAZE ======")
print("inputs shape:", xm.shape, "dtype:", xm.dtype)
print("labels shape:", ym.shape, "dtype:", ym.dtype)
print("inputs unique:", np.unique(xm))
print("labels unique:", np.unique(ym))