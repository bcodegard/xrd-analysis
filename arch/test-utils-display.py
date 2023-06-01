"""
script used for testing features during development
don't actually use this for anything else
"""

import numpy
import math
import sys
import numpy as np

import utils.display as display
import matplotlib.pyplot as plt

test_2d = True
if test_2d:
	
	dsize = 1000

	xdata = np.concatenate([
		np.random.exponential(scale=3, size=dsize),
		np.random.normal( 6,0.3,dsize),
		np.random.normal( 8,0.1,dsize),
		np.random.normal(17,3.0,dsize),
	])
	np.random.shuffle(xdata)

	signal = 5*(xdata[:dsize*3]-2.5)**2 + 400
	ydata = np.concatenate([
		np.random.normal(signal, 17*np.sqrt(signal)),
		np.random.exponential(750,dsize),
	])
	np.random.shuffle(ydata)

	zdata = np.concatenate([
		ydata[:dsize*2]/(1+np.random.poisson(2,size=dsize*2)*4),
		np.random.uniform(0,2*ydata.mean(),dsize*2),
	])

	nbins = 100
	xbins = np.logspace(math.log(max([xdata.min(),1]),10), math.log(xdata.max(),10), nbins)
	ybins = np.logspace(math.log(max([ydata.min(),1]),10), math.log(ydata.max(),10), nbins)
	zbins = np.logspace(math.log(max([zdata.min(),1]),10), math.log(zdata.max(),10), nbins)

	xz = xdata*zdata
	xzbins = np.logspace(math.log(max([xz.min(),1]),10), math.log(xz.max(),10), nbins)

	display.pairs2d(
		[xdata, ydata, zdata, xdata*zdata],
		[xbins, ybins, zbins, xzbins],
		[True , True , True , True],
		['x', 'y', 'z', 'xz'],
		)

	# fig=plt.figure(figsize=(16,8))
	# gs=fig.add_gridspec(
	# 	2,5,
	# 	width_ratios=[6,2,1,6,2],
	# 	height_ratios=[1,3],
	# 	left=0.1, right=0.9,
	# 	bottom=0.1, top=0.9,
	# 	wspace=0.0, hspace=0.0,
	# )

	# display.display2d(
	# 	xdata,ydata,
	# 	xbins,ybins,
	# 	True , True,

	# 	gs=[gs[0,0],gs[1,0],gs[0,1],gs[1,1]],
	# 	fig=fig,

	# 	xlabel="x label legibility",
	# 	ylabel="y label legibility",
	# 	density=False,

	# 	cmap="inferno",
	# 	cbad="k",

	# 	norm="log",

	# 	)

	# display.display2d(
	# 	xdata,zdata,
	# 	xbins,zbins,
	# 	True , True,

	# 	gs=[gs[0,3],gs[1,3],gs[0,4],gs[1,4]],
	# 	fig=fig,

	# 	xlabel="x label legibility",
	# 	ylabel="y label legibility",
	# 	density=False,

	# 	cmap="inferno",
	# 	cbad="k",

	# 	norm="log",

	# 	)

	plt.show()