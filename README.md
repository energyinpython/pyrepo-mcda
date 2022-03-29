# pyrepo-mcda

The Python 3 library for Multi-Criteria Decision Analysis.

## Installation

```
pip install pyrepo-mcda
```

## Usage

`pyrepo-mcda` can be used to rank alternatives after providing their performance values in two-dimensional decision matrix
with alternatives in rows and criteria in columns, and criteria weights and types in vectors. Here is example of using the TOPSIS
method:

```python
from pyrepo_mcda.mcda_methods import TOPSIS
from pyrepo_mcda import distance_metrics as dists
from pyrepo_mcda import normalizations as norms
from pyrepo_mcda.additions import rank_preferences

import numpy as np

matrix = np.array([[256, 8, 41, 1.6, 1.77, 7347.16],
[256, 8, 32, 1.0, 1.8, 6919.99],
[256, 8, 53, 1.6, 1.9, 8400],
[256, 8, 41, 1.0, 1.75, 6808.9],
[512, 8, 35, 1.6, 1.7, 8479.99],
[256, 4, 35, 1.6, 1.7, 7499.99]])

weights = np.array([0.405, 0.221, 0.134, 0.199, 0.007, 0.034])
types = np.array([1, 1, 1, 1, -1, -1])

topsis = TOPSIS(normalization_method=norms.vector_normalization, distance_metric=dists.euclidean)
pref = topsis(matrix, weights, types)
rank = rank_preferences(pref, reverse = True)
print(rank)
```

## License

`pyrepo-mcda` was created by Aleksandra Bączkiewicz. It is licensed under the terms of the MIT license.

## Documentation

Documentation of this library with instruction of installation and usage is 
provided [here](https://pyrepo-mcda.readthedocs.io/en/latest/)