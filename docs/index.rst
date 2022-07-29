Welcome to pyrepo-mcda documentation!
========================================

pyrepo-mcda is Python 3 library for Multi-Criteria Decision Analysis.
This library includes:

- MCDA methods:

	- ``TOPSIS``
	- ``CODAS``
	- ``MABAC``
	- ``MULTIMOORA``
	- ``MOORA``
	- ``VIKOR``
	- ``WASPAS``
	- ``EDAS``
	- ``SPOTIS``
	- ``AHP``
	- ``ARAS``
	- ``COPRAS``
	- ``CRADIS``
	- ``MARCOS``
	- ``PROMETHEE II``
	- ``PROSA C``
	- ``SAW``
	
- Distance metrics:

	- ``euclidean`` (Euclidean distance)
	- ``manhattan`` (Manhattan distance)
	- ``hausdorff`` (Hausdorff distance)
	- ``correlation`` (Correlation distance)
	- ``chebyshev`` (Chebyshev distance)
	- ``std_euclidean`` (Standardized Euclidean distance)
	- ``cosine`` (Cosine distance)
	- ``csm`` (Cosine similarity measure)
	- ``squared_euclidean`` (Squared Euclidean distance)
	- ``bray_curtis`` (Sorensen or Bray-Curtis distance)
	- ``canberra`` (Canberra distance)
	- ``lorentzian`` (Lorentzian distance)
	- ``jaccard`` (Jaccard distance)
	- ``dice`` (Dice distance)
	- ``bhattacharyya`` (Bhattacharyya distance)
	- ``hellinger`` (Hellinger distance)
	- ``matusita`` (Matusita distance)
	- ``squared_chord`` (Squared-chord distance)
	- ``pearson_chi_square`` (Pearson chi square distance)
	- ``squared_chi_square`` (Sqaured chi square distance)
	
- Correlation coefficients:

	- ``spearman`` (Spearman rank correlation coefficient)
	- ``weighted_spearman`` (Weighted Spearman rank correlation coefficient)
	- ``pearson_coeff`` (Pearson correlation coefficient)
	- ``WS_coeff`` (Similarity rank coefficient - WS coefficient)
	
- Methods for normalization of decision matrix:

	- ``linear_normalization`` (Linear normalization)
	- ``minmax_normalization`` (Minimum-Maximum normalization)
	- ``max_normalization`` (Maximum normalization)
	- ``sum_normalization`` (Sum normalization)
	- ``vector_normalization`` (Vector normalization)
	- ``multimoora_normalization`` (Normalization method dedicated for the MULTIMOORA method)
	
- Objective weighting methods for determining criteria weights required by Multi-Criteria Decision Analysis (MCDA) methods:

	- ``equal_weighting`` (Equal weighting method)
	- ``entropy_weighting`` (Entropy weighting method)
	- ``std_weighting`` (Standard deviation weighting method)
	- ``critic_weighting`` (CRITIC weighting method)
	- ``gini_weighting`` (Gini coefficient-based weighting method)
	- ``merec_weighting`` (MEREC weighting method)
	- ``stat_var_weighting`` (Statistical variance weighting method)
	- ``cilos_weighting`` (CILOS weighting method)
	- ``idocriw_weighting`` (IDOCRIW weighting method)
	- ``angle_weighting`` (Angle weighting method)
	- ``coeff_var_weighting`` (Coefficient of variation weighting method)
	
- Stochastic Multicriteria Acceptability Analysis Method - SMAA combined with VIKOR (``VIKOR_SMAA``)
	
- Methods for determination of compromise rankings based on several rankings obtained with different MCDA methods:

	- ``copeland`` (the Copeland method for compromise ranking)
	- ``dominance_directed_graph`` (Dominance Directed Graph for compromise ranking)
	- ``rank_position_method`` (Rank Position Method for compromise ranking)
	- ``improved_borda_rule`` (Improved Borda Rule method for compromise for MULTIMOORA method)
	
- Methods for sensitivity analysis:

	- ``Sensitivity_analysis_weights_percentages`` (Method for sensitivity analysis considering percentage modification of criteria weights)
	- ``Sensitivity_analysis_weights_values`` (Method for sensitivity analysis considering setting different values as chosen criterion weight)
	
- additions:

	- ``rank_preferences`` (Method for ordering alternatives according to their preference values obtained with MCDA methods)
	
Check out the :doc:`usage` section for further information, including
how to :ref:`installation` the project.

.. note::

   This project is under active development.

Contents
---------

.. toctree::
	:maxdepth: 2
	
	usage
	example
	autoapi/index
