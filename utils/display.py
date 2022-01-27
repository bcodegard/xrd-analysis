"""
methods for creating visual displays of data
"""

# __all__ = ...
__author__ = "Brunel Odegard"
__version__ = "0.0"


import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec

def bin_count_from_n_data(n_data, factor=4, min_bins=25, max_bins=1000):
	n_bins = math.ceil(factor*math.sqrt(n_data))
	if min_bins is not None:
		n_bins = max([n_bins,min_bins])
	if max_bins is not None:
		n_bins = min([n_bins,max_bins])
	return n_bins



def display2d(
		xdata, ydata,
		xbins, ybins,
		xlabel=False, ylabel=False,
		density=False,

		cmap="inferno",
		norm="log",
		vmin=None,
		vmax=None,

		):

	counts, xe, ye = np.histogram2d(
		xdata,ydata,
		[np.array(xbins),np.array(ybins)],
		density=density,
	)

	gs = gridspec.GridSpec(2,2,width_ratios=[3,1],height_ratios=[1,3])

	# fig, ax = plt.subplots(2,2)

	if norm=="log":
		norm=LogNorm(vmin,vmax)

	ax2d = plt.subplot(gs[2])
	# image = ax2d.hist2d(
	# 	xdata,ydata,
	# 	[xbins, ybins],
	# 	cmap=cmap,
	# 	norm=norm,
	# )
	# image = ax2d.imshow(
	# 	counts.swapaxes(0,1),
	# 	cmap,
	# 	norm,
	# 	aspect="auto",
	# 	interpolation="none",
	# 	origin='lower',
	# 	extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
	# 	)
	image = ax2d.pcolor(
		xbins,
		ybins,
		counts.swapaxes(0,1),
		shading=None,
		cmap=cmap,
		norm=norm,
		)
	if xlabel:ax2d.set_xlabel(xlabel)
	if ylabel:ax2d.set_ylabel(ylabel)
	ax2d.set_xscale("log")
	ax2d.set_yscale("log")

	axpx = plt.subplot(gs[0])
	axpx.hist(xdata, xbins, density, histtype='step', align='mid', orientation='vertical', log=True, color='k')
	axpx.sharex(ax2d)
	axpx.set_xscale("log")

	axpy = plt.subplot(gs[3])
	axpy.hist(ydata, ybins, density, histtype='step', align='mid', orientation='horizontal', log=True, color='k')
	axpy.sharey(ax2d)
	axpy.set_yscale("log")

	plt.show()


