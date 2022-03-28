Usage
=====

.. _installation:

Installation
------------

To use pyrepo-mcda, first install it using pip:

.. code-block:: python

	pip install pyrepo-mcda

Importing methods from pyrepo_mcda package
-------------------------------------

Import MCDA methods from module `mcda_methods`:

.. code-block:: python

	from pyrepo_mcda.mcda_methods import CODAS, TOPSIS, WASPAS, VIKOR, SPOTIS, EDAS, MABAC, MULTIMOORA

Import weighting methods from module `weighting_methods`:

.. code-block:: python

	from pyrepo_mcda import weighting_methods as mcda_weights

Import normalization methods from module `normalizations`:

.. code-block:: python

	from pyrepo_mcda import normalizations as norms

Import correlation coefficient from module `correlations`:

.. code-block:: python

	from pyrepo_mcda import correlations as corrs

Import distance metrics from module `distance_metrics`:

.. code-block:: python

	from pyrepo_mcda import distance_metrics as dists

Import compromise rankings methods from module `compromise_rankings`:

.. code-block:: python

	from pyrepo_mcda import compromise_rankings as compromises

Import Sensitivity analysis method from module `sensitivity_analysis`:

.. code-block:: python

	from pyrepo_mcda.sensitivity_analysis import Sensitivity_analysis_weights

Import method for ranking alternatives according to prefernce values from module `additions`:

.. code-block:: python

	from pyrepo_mcda.additions import rank_preferences



Usage examples
----------------------

The TOPSIS method
___________________

Parameters
	matrix : ndarray
		Decision matrix with m alternatives in rows and n criteria in columns
	weights : ndarray
		Vector with criteria weights
	types : ndarray
		Vector with criteria types
		
Returns
	ndarray
		Vector with preference values of alternatives. Alternatives have to be ranked in descending order according to preference values.

.. code-block:: python

	import numpy as np
	from pyrepo_mcda.mcda_methods import TOPSIS
	from pyrepo_mcda import normalizations as norms
	from pyrepo_mcda import distance_metrics as dists
	from pyrepo_mcda.additions import rank_preferences

	# provide decision matrix in array numpy.darray
	
	matrix = np.array([[256, 8, 41, 1.6, 1.77, 7347.16],
	[256, 8, 32, 1.0, 1.8, 6919.99],
	[256, 8, 53, 1.6, 1.9, 8400],
	[256, 8, 41, 1.0, 1.75, 6808.9],
	[512, 8, 35, 1.6, 1.7, 8479.99],
	[256, 4, 35, 1.6, 1.7, 7499.99]])

	# provide criteria weights in array numpy.darray. All weights must sum to 1.
	
	weights = np.array([0.405, 0.221, 0.134, 0.199, 0.007, 0.034])

	# provide criteria types in array numpy.darray. Profit criteria are represented by 1 and cost criteria by -1.
	
	types = np.array([1, 1, 1, 1, -1, -1])

	# Create the TOPSIS method object providing normalization method and distance metric.
	
	topsis = TOPSIS(normalization_method = norms.minmax_normalization, distance_metric = dists.euclidean)

	# Calculate the TOPSIS preference values of alternatives
	
	pref = topsis(matrix, weights, types)

	# Generate ranking of alternatives by sorting alternatives descendingly according to the TOPSIS algorithm (reverse = True means sorting in descending order) according to preference values
	
	rank = rank_preferences(pref, reverse = True)

	print('Preference values: ', np.round(pref, 4))
	print('Ranking: ', rank)
	
Output

.. code-block:: console

	Preference values:  [0.4242 0.3217 0.4453 0.3353 0.8076 0.2971]
	Ranking:  [3 5 2 4 1 6]

	
	
The VIKOR method
__________________

Parameters
	matrix : ndarray
		Decision matrix with m alternatives in rows and n criteria in columns
	weights : ndarray
		Vector with criteria weights
	types : ndarray
		Vector with criteria types
		
Returns
	ndarray
		Vector with preference values of alternatives. Alternatives have to be ranked in ascending order according to preference values.

.. code-block:: python

	import numpy as np
	from pyrepo_mcda.mcda_methods import VIKOR
	from pyrepo_mcda.additions import rank_preferences

	# provide decision matrix in array numpy.darray
	matrix = np.array([[8, 7, 2, 1],
	[5, 3, 7, 5],
	[7, 5, 6, 4],
	[9, 9, 7, 3],
	[11, 10, 3, 7],
	[6, 9, 5, 4]])

	# provide criteria weights in array numpy.darray. All weights must sum to 1.
	weights = np.array([0.4, 0.3, 0.1, 0.2])

	# provide criteria types in array numpy.darray. Profit criteria are represented by 1 and cost criteria by -1.
	types = np.array([1, 1, 1, 1])

	# Create the VIKOR method object providing v parameter. The default v parameter is set to 0.5, so if you do not provide it, v will be equal to 0.5.
	vikor = VIKOR(v = 0.625)

	# Calculate the VIKOR preference values of alternatives
	pref = vikor(matrix, weights, types)

	# Generate ranking of alternatives by sorting alternatives ascendingly according to the VIKOR algorithm (reverse = False means sorting in ascending order) according to preference values
	rank = rank_preferences(pref, reverse = False)

	print('Preference values: ', np.round(pref, 4))
	print('Ranking: ', rank)
	
Output

.. code-block:: console

	Preference values:  [0.6399 1.     0.6929 0.2714 0.     0.6939]
	Ranking:  [3 6 4 2 1 5]
	

	
The SPOTIS method
__________________

Parameters
	matrix : ndarray
		Decision matrix with m alternatives in rows and n criteria in columns
	weights : ndarray
		Vector with criteria weights
	types : ndarray
		Vector with criteria types
		
Returns
	ndarray
		Vector with preference values of alternatives. Alternatives have to be ranked in ascending order according to preference values.

.. code-block:: python

	import numpy as np
	from pyrepo_mcda.mcda_methods import SPOTIS

	import numpy as np
	from pyrepo_mcda.mcda_methods import SPOTIS
	from pyrepo_mcda.additions import rank_preferences

	# provide decision matrix in array numpy.darray
	matrix = np.array([[15000, 4.3, 99, 42, 737],
		[15290, 5.0, 116, 42, 892],
		[15350, 5.0, 114, 45, 952],
		[15490, 5.3, 123, 45, 1120]])

	# provide criteria weights in array numpy.darray. All weights must sum to 1.
	weights = np.array([0.2941, 0.2353, 0.2353, 0.0588, 0.1765])

	# provide criteria types in array numpy.darray. Profit criteria are represented by 1 and cost criteria by -1.
	types = np.array([-1, -1, -1, 1, 1])

	# Determine minimum bounds of performance values for each criterion in decision matrix
	bounds_min = np.array([14000, 3, 80, 35, 650])

	# Determine maximum bounds of performance values for each criterion in decision matrix
	bounds_max = np.array([16000, 8, 140, 60, 1300])

	# Stack minimum and maximum bounds vertically using vstack. You will get a matrix that has two rows and a number of columns equal to the number of criteria
	bounds = np.vstack((bounds_min, bounds_max))

	# Create the SPOTIS method object
	spotis = SPOTIS()

	# Calculate the SPOTIS preference values of alternatives
	pref = spotis(matrix, weights, types, bounds)

	# Generate ranking of alternatives by sorting alternatives ascendingly according to the SPOTIS algorithm (reverse = False means sorting in ascending order) according to preference values
	rank = rank_preferences(pref, reverse = False)

	print('Preference values: ', np.round(pref, 4))
	print('Ranking: ', rank)
	
Output

.. code-block:: console

	Preference values:  [0.478  0.5781 0.5557 0.5801]
	Ranking:  [1 3 2 4]

	
	
The CODAS method
__________________

Parameters
	matrix : ndarray
		Decision matrix with m alternatives in rows and n criteria in columns
	weights : ndarray
		Vector with criteria weights
	types : ndarray
		Vector with criteria types
		
Returns
	ndarray
		Vector with preference values of alternatives. Alternatives have to be ranked in descending order according to preference values.

.. code-block:: python

	import numpy as np
	from pyrepo_mcda.mcda_methods import CODAS
	from pyrepo_mcda import normalizations as norms
	from pyrepo_mcda import distance_metrics as dists
	from pyrepo_mcda.additions import rank_preferences

	# provide decision matrix in array numpy.darray
	matrix = np.array([[45, 3600, 45, 0.9],
	[25, 3800, 60, 0.8],
	[23, 3100, 35, 0.9],
	[14, 3400, 50, 0.7],
	[15, 3300, 40, 0.8],
	[28, 3000, 30, 0.6]])

	# provide criteria weights in array numpy.darray. All weights must sum to 1.
	weights = np.array([0.2857, 0.3036, 0.2321, 0.1786])

	# provide criteria types in array numpy.darray. Profit criteria are represented by 1 and cost criteria by -1.
	types = np.array([1, -1, 1, 1])

	# Create the CODAS method object providing normalization method (in CODAS it is linear_normalization by default), distance metric, and tau parameter, which is equal to 0.02 default. tau must be in the range from 0.01 to 0.05.
	codas = CODAS(normalization_method = norms.linear_normalization, distance_metric = dists.euclidean, tau = 0.02)

	# Calculate the CODAS preference values of alternatives
	pref = codas(matrix, weights, types)

	# Generate ranking of alternatives by sorting alternatives descendingly according to the CODAS algorithm (reverse = True means sorting in descending order) according to preference values
	rank = rank_preferences(pref, reverse = True)

	print('Preference values: ', np.round(pref, 4))
	print('Ranking: ', rank)
	
Output

.. code-block:: console

	Preference values:  [ 1.3914  0.3411 -0.217  -0.5381 -0.7292 -0.2481]
	Ranking:  [1 2 3 5 6 4]

	
	
The WASPAS method
___________________

Parameters
	matrix : ndarray
		Decision matrix with m alternatives in rows and n criteria in columns
	weights : ndarray
		Vector with criteria weights
	types : ndarray
		Vector with criteria types
		
Returns
	ndarray
		Vector with preference values of alternatives. Alternatives have to be ranked in descending order according to preference values.

.. code-block:: python

	import numpy as np
	from pyrepo_mcda.mcda_methods import WASPAS
	from pyrepo_mcda import normalizations as norms
	from pyrepo_mcda.additions import rank_preferences

	# provide decision matrix in array numpy.darray
	matrix = np.array([[5000, 3, 3, 4, 3, 2],
	[680, 5, 3, 2, 2, 1],
	[2000, 3, 2, 3, 4, 3],
	[600, 4, 3, 1, 2, 2],
	[800, 2, 4, 3, 3, 4]])

	# provide criteria weights in array numpy.darray. All weights must sum to 1.
	weights = np.array([0.157, 0.249, 0.168, 0.121, 0.154, 0.151])

	# provide criteria types in array numpy.darray. Profit criteria are represented by 1 and cost criteria by -1.
	types = np.array([-1, 1, 1, 1, 1, 1])

	# Create the WASPAS method object providing normalization method (in WASAPS it is linear_normalization by default), and lambda parameter, which is equal to 0.5 default. tau must be in the range from 0 to 1.
	waspas = WASPAS(normalization_method=norms.linear_normalization, lambda_param=0.5)

	# Calculate the WASPAS preference values of alternatives
	pref = waspas(matrix, weights, types)

	# Generate ranking of alternatives by sorting alternatives descendingly according to the WASPAS algorithm (reverse = True means sorting in descending order) according to preference values
	rank = rank_preferences(pref, reverse = True)

	print('Preference values: ', np.round(pref, 4))
	print('Ranking: ', rank)
	
Output

.. code-block:: console

	Preference values:  [0.5622 0.6575 0.6192 0.6409 0.7228]
	Ranking:  [5 2 4 3 1]

	
	
The EDAS method
_________________

Parameters
	matrix : ndarray
		Decision matrix with m alternatives in rows and n criteria in columns
	weights : ndarray
		Vector with criteria weights
	types : ndarray
		Vector with criteria types
		
Returns
	ndarray
		Vector with preference values of alternatives. Alternatives have to be ranked in descending order according to preference values.

.. code-block:: python

	import numpy as np
	from pyrepo_mcda.mcda_methods import EDAS
	from pyrepo_mcda.additions import rank_preferences

	# provide decision matrix in array numpy.darray
	matrix = np.array([[256, 8, 41, 1.6, 1.77, 7347.16],
	[256, 8, 32, 1.0, 1.8, 6919.99],
	[256, 8, 53, 1.6, 1.9, 8400],
	[256, 8, 41, 1.0, 1.75, 6808.9],
	[512, 8, 35, 1.6, 1.7, 8479.99],
	[256, 4, 35, 1.6, 1.7, 7499.99]])

	# provide criteria weights in array numpy.darray. All weights must sum to 1.
	weights = np.array([0.405, 0.221, 0.134, 0.199, 0.007, 0.034])

	# provide criteria types in array numpy.darray. Profit criteria are represented by 1 and cost criteria by -1.
	types = np.array([1, 1, 1, 1, -1, -1])

	# Create the EDAS method object.
	edas = EDAS()

	# Calculate the EDAS preference values of alternatives
	pref = edas(matrix, weights, types)

	# Generate ranking of alternatives by sorting alternatives descendingly according to the EDAS algorithm (reverse = True means sorting in descending order) according to preference values
	rank = rank_preferences(pref, reverse = True)

	print('Preference values: ', np.round(pref, 4))
	print('Ranking: ', rank)
	
Output

.. code-block:: console

	Preference values:  [0.4141 0.13   0.4607 0.212  0.9443 0.043 ]
	Ranking:  [3 5 2 4 1 6]

	
	
The MABAC method
___________________

Parameters
	matrix : ndarray
		Decision matrix with m alternatives in rows and n criteria in columns
	weights : ndarray
		Vector with criteria weights
	types : ndarray
		Vector with criteria types
		
Returns
	ndarray
		Vector with preference values of alternatives. Alternatives have to be ranked in descending order according to preference values.

.. code-block:: python

	import numpy as np
	from pyrepo_mcda.mcda_methods import MABAC
	from pyrepo_mcda import normalizations as norms
	from pyrepo_mcda.additions import rank_preferences

	# provide decision matrix in array numpy.darray
	matrix = np.array([[2.937588, 2.762986, 3.233723, 2.881315, 3.015289, 3.313491],
	[2.978555, 3.012820, 2.929487, 3.096154, 3.012820, 3.593939],
	[3.286673, 3.464600, 3.746009, 3.715632, 3.703427, 4.133620],
	[3.322037, 3.098638, 3.262154, 3.147851, 3.206675, 3.798684],
	[3.354866, 3.270945, 3.221880, 3.213207, 3.670508, 3.785941],
	[2.796570, 2.983000, 2.744904, 2.692550, 2.787563, 2.878851],
	[2.846491, 2.729618, 2.789990, 2.955624, 3.123323, 3.646595],
	[3.253458, 3.208902, 3.678499, 3.580044, 3.505663, 3.954262],
	[2.580718, 2.906903, 3.176497, 3.073653, 3.264727, 3.681887],
	[2.789011, 3.000000, 3.101099, 3.139194, 2.985348, 3.139194],
	[3.418681, 3.261905, 3.187912, 3.052381, 3.266667, 3.695238]])

	# provide criteria weights in array numpy.darray. All weights must sum to 1.
	weights = np.array([0.171761, 0.105975, 0.191793, 0.168824, 0.161768, 0.199880])

	# provide criteria types in array numpy.darray. Profit criteria are represented by 1 and cost criteria by -1.
	types = np.array([1, 1, 1, 1, 1, 1])

	# Create the MABAC method object providing normalization method. In MABAC it is minmax_normalization by default.
	mabac = MABAC(normalization_method=norms.minmax_normalization)

	# Calculate the MABAC preference values of alternatives
	pref = mabac(matrix, weights, types)

	# Generate ranking of alternatives by sorting alternatives descendingly according to the MABAC algorithm (reverse = True means sorting in descending order) according to preference values
	rank = rank_preferences(pref, reverse = True)

	print('Preference values: ', np.round(pref, 4))
	print('Ranking: ', rank)
	
Output

.. code-block:: console

	Preference values:  [-0.1553 -0.0895  0.5054  0.1324  0.2469 -0.3868 -0.1794  0.3629 -0.0842
	 -0.1675  0.1399]
	Ranking:  [ 8  7  1  5  3 11 10  2  6  9  4]

	
	
The MULTIMOORA method
_______________________

Parameters
	matrix : ndarray
		Decision matrix with m alternatives in rows and n criteria in columns
	weights : ndarray
		Vector with criteria weights
	types : ndarray
		Vector with criteria types
		
Returns
	ndarray
		Vector with preference values of alternatives. Alternatives have to be ranked in descending order according to preference values.

.. code-block:: python

	import numpy as np
	from pyrepo_mcda.mcda_methods import MULTIMOORA
	from pyrepo_mcda.additions import rank_preferences
	from pyrepo_mcda import compromise_rankings as compromises

	# provide decision matrix in array numpy.darray
	matrix = np.array([[4, 3, 3, 4, 3, 2, 4],
	[3, 3, 4, 3, 5, 4, 4],
	[5, 4, 4, 5, 5, 5, 4]])

	# provide criteria weights in array numpy.darray. All weights must sum to 1.
	weights = np.array([0.215, 0.215, 0.159, 0.133, 0.102, 0.102, 0.073])

	# provide criteria types in array numpy.darray. Profit criteria are represented by 1 and cost criteria by -1.
	types = np.array([1, 1, 1, 1, 1, 1, 1])

	# Create the MULTIMOORA method object providing compromise_rank_method. In MULTIMOORA it is dominance_directed_graph by default.
	multimoora = MULTIMOORA(compromise_rank_method = compromises.dominance_directed_graph)

	# Calculate the MULTIMOORA ranking of alternatives
	rank = multimoora(matrix, weights, types)

	print('Ranking: ', rank)
	
Output

.. code-block:: console

	Ranking:  [3 2 1]
	

	
Methods for determining compromise rankings
_____________________________________________
	
The Copeland Method for compromise ranking

Parameters
	matrix : ndarray
		Matrix with rankings provided by different MCDA methods in particular columns.
		
Returns
	ndarray
		Vector with compromise ranking.

.. code-block:: python

	import numpy as np
	from pyrepo_mcda import compromise_rankings as compromises

	# Provide matrix with different rankings given by different MCDA methods in columns
	matrix = np.array([[7, 8, 7, 6, 7, 7],
	[4, 7, 5, 7, 5, 4],
	[8, 9, 8, 8, 9, 8],
	[1, 4, 1, 1, 1, 1],
	[2, 2, 2, 4, 3, 2],
	[3, 1, 4, 3, 2, 3],
	[10, 5, 10, 9, 8, 10],
	[6, 3, 6, 5, 4, 6],
	[9, 10, 9, 10, 10, 9],
	[5, 6, 3, 2, 6, 5]])
	
	# Calculate the compromise ranking using `copeland` method
	result = compromises.copeland(matrix)
	
	print('Copeland compromise ranking: ', result)
	
Output

.. code-block:: console

	Copeland compromise ranking:  [ 7  6  8  1  2  3  9  5 10  4]


	
The Dominance Directed Graph

Parameters
	matrix : ndarray
		Matrix with rankings provided by different MCDA methods in particular columns.
		
Returns
	ndarray
		Vector with compromise ranking.

.. code-block:: python

	import numpy as np
	from pyrepo_mcda import compromise_rankings as compromises

	# Provide matrix with different rankings given by different MCDA methods in columns
	matrix = np.array([[3, 2, 3],
	[2, 3, 2],
	[1, 1, 1]])
	
	# Calculate the compromise ranking using `dominance_directed_graph` method
	result = compromises.dominance_directed_graph(matrix)
	
	print('Dominance directed graph compromise ranking: ', result)
	
Output

.. code-block:: console

	Dominance directed graph compromise ranking:  [3 2 1]

	
	
The Rank Position compromise ranking method

Parameters
	matrix : ndarray
		Matrix with rankings provided by different MCDA methods in particular columns.
Returns
	ndarray
		Vector with compromise ranking.

.. code-block:: python

	import numpy as np
	from pyrepo_mcda import compromise_rankings as compromises

	# Provide matrix with different rankings given by different MCDA methods in columns
	matrix = np.array([[3, 2, 3],
	[2, 3, 2],
	[1, 1, 1]])
	
	# Calculate the compromise ranking using `rank_position_method` method
	result = compromises.rank_position_method(matrix)
	
	print('Rank position compromise ranking: ', result)
	
Output

.. code-block:: console

	Rank position compromise ranking:  [3 2 1]


	
The Improved Borda Rule compromise ranking method for MULTIMOORA

Parameters
	prefs : ndarray
		Matrix with preference values provided by different approaches of MULTIMOORA in particular columns.
	ranks : ndarray
		Matrix with rankings provided by different approaches of MULTIMOORA in particular columns.
Returns
	ndarray
		Vector with compromise ranking.

.. code-block:: python

	import numpy as np
	from pyrepo_mcda import compromise_rankings as compromises

	# Provide matrix with different preference values given by different MCDA methods in columns
	prefs = np.array([[4.94364901e-01, 4.56157867e-02, 3.85006756e-09],
	[5.26950959e-01, 6.08111832e-02, 9.62516889e-09],
	[6.77457681e-01, 0.00000000e+00, 4.45609671e-08]])

	# Provide matrix with different rankings given by different MCDA methods in columns
	ranks = np.array([[3, 2, 3],
	[2, 3, 2],
	[1, 1, 1]])

	# Calculate the compromise ranking using `improved_borda_rule` method
	result = compromises.improved_borda_rule(prefs, ranks)

	print('Improved Borda Rule compromise ranking: ', result)

Output

.. code-block:: console

	Improved Borda Rule compromise ranking:  [2 3 1]



Correlation coefficents
__________________________

Spearman correlation coefficient

Parameters
	R : ndarray
		First vector containing values
	Q : ndarray
		Second vector containing values
Returns
	float
		Value of correlation coefficient between two vectors

.. code-block:: python

	import numpy as np
	from pyrepo_mcda import correlations as corrs

	# Provide two vectors with rankings obtained with different MCDA methods
	R = np.array([1, 2, 3, 4, 5])
	Q = np.array([1, 3, 2, 4, 5])

	# Calculate the correlation using `spearman` coefficient
	coeff = corrs.spearman(R, Q)
	print('Spearman coeff: ', np.round(coeff, 4))
	
Output

.. code-block:: console

	Spearman coeff:  0.9

	
	
Weighted Spearman correlation coefficient

Parameters
	R : ndarray
		First vector containing values
	Q : ndarray
		Second vector containing values
Returns
	float
		Value of correlation coefficient between two vectors

.. code-block:: python

	import numpy as np
	from pyrepo_mcda import correlations as corrs

	# Provide two vectors with rankings obtained with different MCDA methods
	R = np.array([1, 2, 3, 4, 5])
	Q = np.array([1, 3, 2, 4, 5])

	# Calculate the correlation using `weighted_spearman` coefficient
	coeff = corrs.weighted_spearman(R, Q)
	print('Weighted Spearman coeff: ', np.round(coeff, 4))
	
Output

.. code-block:: console

	Weighted Spearman coeff:  0.8833
	
	
	
Similarity rank coefficient WS

Parameters
	R : ndarray
		First vector containing values
	Q : ndarray
		Second vector containing values
Returns
	float
		Value of similarity coefficient between two vectors

.. code-block:: python

	import numpy as np
	from pyrepo_mcda import correlations as corrs

	# Provide two vectors with rankings obtained with different MCDA methods
	R = np.array([1, 2, 3, 4, 5])
	Q = np.array([1, 3, 2, 4, 5])

	# Calculate the similarity using `WS_coeff` coefficient
	coeff = corrs.WS_coeff(R, Q)
	print('WS coeff: ', np.round(coeff, 4))
	
Output

.. code-block:: console

	WS coeff:  0.8542

	
	
Pearson correlation coefficient

Parameters
	R : ndarray
		First vector containing values
	Q : ndarray
		Second vector containing values
Returns
	float
		Value of correlation coefficient between two vectors

.. code-block:: python

	import numpy as np
	from pyrepo_mcda import correlations as corrs

	# Provide two vectors with rankings obtained with different MCDA methods
	R = np.array([1, 2, 3, 4, 5])
	Q = np.array([1, 3, 2, 4, 5])

	# Calculate the correlation using `pearson_coeff` coefficient
	coeff = corrs.pearson_coeff(R, Q)
	print('Pearson coeff: ', np.round(coeff, 4))
	
Output

.. code-block:: console

	Pearson coeff:  0.9
	
	
	
Methods for criteria weights determination
___________________________________________

Entropy weighting method

Parameters
	matrix : ndarray
		Decision matrix with performance values of m alternatives and n criteria
Returns
	ndarray
		vector of criteria weights
		
.. code-block:: python

	import numpy as np
	from pyrepo_mcda import weighting_methods as mcda_weights

	matrix = np.array([[30, 30, 38, 29],
	[19, 54, 86, 29],
	[19, 15, 85, 28.9],
	[68, 70, 60, 29]])
	
	weights = mcda_weights.entropy_weighting(matrix)
	
	print('Entropy weights: ', np.round(weights, 4))
	
Output

.. code-block:: console

	Entropy weights:  [0.463  0.3992 0.1378 0.    ]
	

CRITIC weighting method

Parameters
	matrix : ndarray
		Decision matrix with performance values of m alternatives and n criteria
Returns
	ndarray
		Vector of criteria weights
		
.. code-block:: python

	import numpy as np
	from pyrepo_mcda import weighting_methods as mcda_weights

	matrix = np.array([[5000, 3, 3, 4, 3, 2],
	[680, 5, 3, 2, 2, 1],
	[2000, 3, 2, 3, 4, 3],
	[600, 4, 3, 1, 2, 2],
	[800, 2, 4, 3, 3, 4]])
	
	weights = mcda_weights.critic_weighting(matrix)
	
	print('CRITIC weights: ', np.round(weights, 4))
	
Output

.. code-block:: console

	CRITIC weights:  [0.157  0.2495 0.1677 0.1211 0.1541 0.1506]


Standard deviation weighting method

Parameters
	matrix : ndarray
		Decision matrix with performance values of m alternatives and n criteria
Returns
	ndarray
		Vector of criteria weights
		
.. code-block:: python

	import numpy as np
	from pyrepo_mcda import weighting_methods as mcda_weights

	matrix = np.array([[0.619, 0.449, 0.447],
	[0.862, 0.466, 0.006],
	[0.458, 0.698, 0.771],
	[0.777, 0.631, 0.491],
	[0.567, 0.992, 0.968]])
	
	weights = mcda_weights.std_weighting(matrix)
	
	print('Standard deviation weights: ', np.round(weights, 4))
	
Output

.. code-block:: console

	Standard deviation weights:  [0.2173 0.2945 0.4882]
	
	
Distance metrics
_________________

Here are two examples of using distance metrics for Euclidean distance `euclidean` and Manhattan distance `manhattan`. Usage of other distance metrics
provided in module `distance metrics` is analogous.


Euclidean distance

Parameters
	A : ndarray
		First vector containing values
	B : ndarray
		Second vector containing values
Returns
	float
		distance value between two vectors

.. code-block:: python
	
	import numpy as np
	from pyrepo_mcda import distance_metrics as dists
	
	A = np.array([0.165, 0.113, 0.015, 0.019])
    B = np.array([0.227, 0.161, 0.053, 0.130])

    dist = dists.euclidean(A, B)
    print('Distance: ', np.round(dist, 4))
	
Output

.. code-block:: console

	Distance:  0.1411
	
	
Manhattan distance

Parameters
	A : ndarray
		First vector containing values
	B : ndarray
		Second vector containing values
Returns
	float
		distance value between two vectors

.. code-block:: python
	
	import numpy as np
	from pyrepo_mcda import distance_metrics as dists
	
	A = np.array([0.165, 0.113, 0.015, 0.019])
    B = np.array([0.227, 0.161, 0.053, 0.130])

    dist = dists.manhattan(A, B)
    print('Distance: ', np.round(dist, 4))
	
Output

.. code-block:: console

	Distance:  0.259
	
	
Normalization methods
______________________

Here is an example of vector normalization usage. Other normalizations provided in module `normalizations`, namely `minmax_normalization`, `max_normalization`,
`sum_normalization`, `linear_normalization`, `multimoora_normalization` are used in analogous way.


Vector normalization

Parameters
	matrix : ndarray
		Decision matrix with m alternatives in rows and n criteria in columns
	types : ndarray
		Criteria types. Profit criteria are represented by 1 and cost by -1.
Returns
	ndarray
		Normalized decision matrix

.. code-block:: python
	
	import numpy as np
	from pyrepo_mcda import normalizations as norms

	matrix = np.array([[8, 7, 2, 1],
    [5, 3, 7, 5],
    [7, 5, 6, 4],
    [9, 9, 7, 3],
    [11, 10, 3, 7],
    [6, 9, 5, 4]])

    types = np.array([1, 1, 1, 1])

    norm_matrix = norms.vector_normalization(matrix, types)
    print('Normalized matrix: ', np.round(norm_matrix, 4))
	
Output

.. code-block:: console
	
	Normalized matrix:  [[0.4126 0.3769 0.1525 0.0928]
	 [0.2579 0.1615 0.5337 0.4642]
	 [0.361  0.2692 0.4575 0.3714]
	 [0.4641 0.4845 0.5337 0.2785]
	 [0.5673 0.5384 0.2287 0.6499]
	 [0.3094 0.4845 0.3812 0.3714]]

	
Method for sensitivity analysis considering criteria weights modification
__________________________________________________________________________

sensitivity_analysis

Parameters
	matrix : ndarray
		Decision matrix with alternatives performances data. This matrix includes
		data on m alternatives in rows considering criteria in columns
	weights : ndarray
		Vector with criteria weights. All weights in this vector must sum to 1.
	types : ndarray
		Vector with criteria types. Types can be equal to 1 for profit criteria and -1
		for cost criteria.
	percentages : ndarray
		Vector with percentage values of given criteria weight modification.
	mcda_name : str
		Name of applied MCDA method
	j : int
		Index of column in decision matrix `matrix` that indicates for which criterion
		the weight is modified. 
		
Returns
	data_sens : DataFrame
        dataframe with rankings calculated for subsequent modifications of criterion j weight

.. code-block:: python

	import numpy as np
	from pyrepo_mcda.sensitivity_analysis import Sensitivity_analysis_weights
	
	import numpy as np
	from pyrepo_mcda.mcda_methods import CODAS

	# provide decision matrix in array numpy.darray
	matrix = np.array([[45, 3600, 45, 0.9],
	[25, 3800, 60, 0.8],
	[23, 3100, 35, 0.9],
	[14, 3400, 50, 0.7],
	[15, 3300, 40, 0.8],
	[28, 3000, 30, 0.6]])

	# provide criteria weights in array numpy.darray. All weights must sum to 1.
	weights = np.array([0.2857, 0.3036, 0.2321, 0.1786])
	
	# provide criteria types in array numpy.darray. Profit criteria are represented by 1 and cost criteria by -1.
	types = np.array([1, -1, 1, 1])
	
	# provide vector with percentage values of chosen criterion weight modification
	percentages = np.arange(0.05, 0.5, 0.1)
	
	#create the chosen MCDA object
    method = TOPSIS(normalization_method=norms.minmax_normalization, distance_metric=dists.euclidean)
	
	# provide index of j-th chosen criterion whose weight will be modified in sensitivity analysis, for example j = 1 for criterion in the second column
	j = 1
	
	# Create the Sensitivity_analysis_weights object
	sensitivity_analysis = Sensitivity_analysis_weights()

	# Generate DataFrame with rankings for different modification of weight of chosen criterion
	# Provide decision matrix `matrix`, vector with criteria weights `weights`, criteria types `types`, initialized object of chosen MCDA 
	# method `method`, index of chosen criterion whose weight will be modified and list with directions of weights value modification
	data_sens = sensitivity_analysis(matrix, weights, types, percentages, method, j, [1])
	
