"""
"""

__author__ = "Brunel Odegard"
__version__ = "0.0"


import sys
import os
import math
import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils.fileio  as fileio
import utils.display as display
import utils.model   as model


# greedy map
gmap = lambda fn,it:list(map(fn,it))


FIT_CSV_TYPELIST = [int, str, float, float, int, str, int, int, str, int, str]


if __name__ == '__main__':

	rs = 3455
	ts = 3455
	entries = fileio.load_csv("./data/xf/slices_{}_{}.csv".format(rs,ts), [str,str,float,float,float,float,int,float])

	run   = [_[0] for _ in entries]
	bf    = [_[1] for _ in entries]
	bf_lo = [_[2] for _ in entries]
	bf_hi = [_[3] for _ in entries]

	ti = [_[4] for _ in entries]
	tf = [_[5] for _ in entries]

	pars = np.array([_[ 7:10] for _ in entries])
	errs = np.array([_[10:13] for _ in entries])

	cols = 'rgb'
	labels = 'abc'
	centers = [0,1,0]
	for i in range(3):
		plt.subplot(3,1,i+1)
		plt.errorbar(ti, pars[:,i], errs[:,i], color=cols[i], ls='', marker='.', label=labels[i])
		plt.plot(ti,np.ones(len(ti))*centers[i],color='k',ls="--")
		plt.legend()
		plt.xlabel('time slice')
		plt.ylabel('parameter value')
	plt.suptitle("reference spectrum = {}\ntime slices of run {}".format(rs,ts))
	plt.show()
