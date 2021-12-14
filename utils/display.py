"""
methods for creating visual displays of data
"""

# __all__ = ...
__author__ = "Brunel Odegard"
__version__ = "0.0"


import math



def bin_count_from_n_data(n_data, factor=4, min_bins=25, max_bins=1000):
	n_bins = math.ceil(factor*math.sqrt(n_data))
	if min_bins is not None:
		n_bins = max([n_bins,min_bins])
	if max_bins is not None:
		n_bins = min([n_bins,max_bins])
	return n_bins