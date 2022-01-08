"""
data processing algorithms
"""

# __all__ = ...
__author__ = "Brunel Odegard"
__version__ = "0.0"



# # naive implementation of pulse analysis algorithm
# # used for testing and debugging
# # do not use for analysis

# def waveform_to_vmax_area(wvf, ped=150, nsigma=3):

# 	# assumes a [nEvents, nSamples] array
# 	n = wvf.shape[0]
# 	s = wvf.shape[1]

# 	pre  = wvf[:,:ped]
# 	post = wvf[:,ped:]

# 	pedestal = pre.mean(-1).reshape([n,1])
# 	pednoise = pre.std(-1).reshape([n,1])

# 	pulse = post * (post > (pedestal - nsigma*pednoise)).astype(int)

# 	vmax = pulse.max(-1)
# 	area = pulse.sum(-1)

# 	return vmax, area


