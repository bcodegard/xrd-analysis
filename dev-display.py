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
	
	dsize = 100000

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

	nbins = 100
	xbins = np.logspace(math.log(max([xdata.min(),1]),10), math.log(xdata.max(),10), nbins)
	ybins = np.logspace(math.log(max([ydata.min(),1]),10), math.log(ydata.max(),10), nbins)

	# print(xbins)
	# print(ybins)

	display.display2d(
		xdata,ydata,
		xbins,ybins,
		"x","y",
		True,
		)
