"""
methods for creating visual displays of data
"""

# __all__ = ...
__author__ = "Brunel Odegard"
__version__ = "0.0"


import math
import itertools
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

def bin_count_from_n_data(n_data, factor=4, min_bins=25, max_bins=1000):
	n_bins = math.ceil(factor*math.sqrt(n_data))
	if min_bins is not None:
		n_bins = max([n_bins,min_bins])
	if max_bins is not None:
		n_bins = min([n_bins,max_bins])
	return n_bins


# rows, columns by number of datasets
pair_layouts = {
	2: (1, 1),
	3: (1, 3),
	4: (2, 3),
	5: (2, 5),
	6: (3, 5),
}

def pairs2d(
		pData,
		pBins,
		pLog,
		pLabel,

		density=False,
		cmap="inferno",
		cbad=False,
		norm="log",
		vmin=None,
		vmax=None,

		show=False,
		):
	"""makes 2d colorplots + histograms for each pair of datasets"""

	# make figure, gridspec, etc.
	n_datasets = len(pData)
	nr, nc = pair_layouts.get(n_datasets)

	fig = plt.figure(figsize=(5*nc-1,5*nr-1))
	gs = fig.add_gridspec(
		nr*3-1, nc*3-1,
		width_ratios  = [6,2,2]*(nc-1) + [6,2],
		height_ratios = [2,6,2]*(nr-1) + [2,6],
		left=0.1, right=0.9,
		bottom=0.1, top=0.9,
		wspace=0.0, hspace=0.0,
	)

	# make each plot
	for it,(ix,iy) in enumerate(itertools.combinations(range(n_datasets), 2)):

		xData   = pData[ix]
		xBins   = pBins[ix]
		xLog    = pLog[ix]
		xLabel  = pLabel[ix]

		yData   = pData[iy]
		yBins   = pBins[iy]
		yLog    = pLog[iy]
		yLabel  = pLabel[iy]

		igx = 3*(it %  nc)
		igy = 3*(it // nc)

		display2d(
			xData, yData,
			xBins, yBins,
			xLog , yLog ,

			gs = [gs[igy+0,igx+0], gs[igy+1,igx+0], gs[igy+0,igx+1], gs[igy+1,igx+1]],
			fig = fig,
			
			xlabel = xLabel,
			ylabel = yLabel,
			density = density,

			cmap = cmap,
			cbad = cbad,
			norm = norm,
			vmin = vmin,
			vmax = vmax,
			)

	if show:
		plt.show()

def display2d(

		xdata, ydata,
		xbins, ybins,
		xlog , ylog ,
		
		# specify counts (calculate if not given)
		counts=False,

		# specify fig, gridspec to place plots onto
		# new gridspec will be created if not specified
		gs=False,
		fig=False,

		xlabel=False,
		ylabel=False,
		density=False,

		cmap="inferno",
		cbad=False,
		norm="log",
		vmin=None,
		vmax=None,

		show=False,

		):

	# calculate 2d bin counts if not given
	if not counts:
		counts, xe, ye = np.histogram2d(
			xdata,ydata,
			[np.array(xbins),np.array(ybins)],
			density=density,
		)

	# create grid if not specified
	if gs is False:
		fig = plt.figure(figsize=(8, 8))
		gs = fig.add_gridspec(
			2,2,
			width_ratios=[3,1],
			height_ratios=[1,3],
			left=0.1, right=0.9,
			bottom=0.1, top=0.9,
			wspace=0.0, hspace=0.0,
		)

	# log norm?
	if norm=="log":
		norm=LogNorm(vmin,vmax)

	# cmap, cmap bad color
	if type(cmap) is str:
		cmap = matplotlib.cm.get_cmap(cmap)
	if cbad:
		cmap.set_bad(cbad)

	# 2d plot goes in lower left
	ax2d = plt.subplot(gs[1])
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
	if xlog: ax2d.set_xscale("log")
	if ylog: ax2d.set_yscale("log")

	# color"bar" on upper right
	if fig is not False:
		axcb = plt.subplot(gs[2])
		fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap), cax=axcb, ax=ax2d)

	# x projection hist
	axpx = plt.subplot(gs[0])
	axpx.hist(xdata, xbins, density, histtype='step', align='mid', orientation='vertical', log=True, color='k')
	axpx.sharex(ax2d)
	if xlog:
		axpx.set_xscale("log")
	axpx.tick_params(axis="x", labelbottom=False)

	# y projection hist
	axpy = plt.subplot(gs[3])
	axpy.hist(ydata, ybins, density, histtype='step', align='mid', orientation='horizontal', log=True, color='k')
	axpy.sharey(ax2d)
	if ylog:
		axpy.set_yscale("log")
	axpy.tick_params(axis="y", labelleft=False)

	if show:
		plt.show()

	return fig, gs

