# pyrepo-mcda

The Python 3 library for Multi-Criteria Decision Analysis.

## Installation

```
pip install pyrepo-mcda
```

## Usage

`pyrepo-mcda` can be used to rank alternatives after providing their performance values in the two-dimensional decision matrix `matrix`
with alternatives in rows and criteria in columns, and criteria weights `weights` and types `types` in vectors. 
All criteria weights must sum to 1. Criteria types are equal to 1 for profit criteria and -1 for cost criteria. The TOPSIS method returns a
vector with preference values `pref` assigned to alternatives. To rank alternatives according to TOPSIS preference values, we have to sort them
in descending order because, in the TOPSIS method, the best alternative has the highest preference value. The alternatives are ranked using 
the `rank_preferences` method provided in the `additions` module of the `pyrepo-mcda` package. Parameter `reverse = True` means that alternatives 
are sorted in descending order. Here is an example of using the TOPSIS method:

```python
import numpy as np
from pyrepo_mcda.mcda_methods import TOPSIS
from pyrepo_mcda import distance_metrics as dists
from pyrepo_mcda import normalizations as norms
from pyrepo_mcda.additions import rank_preferences

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