"""
"""

__author__ = "Brunel Odegard"
__version__ = "0.0"


import os
import re
import sys
import math
import argparse
import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import scipy.optimize as opt

import utils.cli     as cli
import utils.data    as data
import utils.model   as model
import utils.fileio  as fileio
import utils.display as display



data_file = "../xrd-analysis/data/scint-experiment/root/Run{}.root"

runs_pre = {
	1: [4291, 4292, 4293, 4294, 4295, 4296, ],# 4107, ],
	2: [4291, 4292, 4293, 4294, 4295, 4296, ],# 4108, ],
	3: [4291, 4292, 4293, 4294, 4295, 4296, ],# 4109, ],
}

runs_post = {
	1: [4321, 4322, 4323, 4324, 4325, 4326, 4327],
	2: [4321, 4322, 4323, 4324, 4325, 4326, 4327],
	3: [4321, 4322, 4323, 4324, 4325, 4326, 4327],
}

branches_needed = {
	"area_3046_1", "area_3046_2", "area_3046_3", "area_3046_4",
	"vMax_3046_1", "vMax_3046_2", "vMax_3046_3", "vMax_3046_4",
}


COMPARE_RANGE = {
	1: [40000, 325000],
	2: [40000, 325000],
	3: [40000, 340000],	
}
NBINS = 500


def dict_concat(dicts):
	keys = dicts[0].keys()
	return {_:np.concatenate([__[_] for __ in dicts]) for _ in keys}


def procure_data(ch, diag=False):
	
	a_this = "area_3046_{}".format(ch)
	v_this = "vMax_3046_{}".format(ch)
	a_lyso = "area_3046_4"

	# procure raw data
	bm_pre = data.BranchManager(
		dict_concat([fileio.load_branches(data_file.format(run), branches_needed) for run in runs_pre[ch]]),
		export_copies=False,
		import_copies=False
	)

	# show test plots if diagnosing
	if diag:
		plt.hist(
			bm_pre.mask(data.cut(a_lyso,lo=15000), key_or_keys = a_this, apply_mask=False),
			bins=np.linspace(10000,450000,200),
			histtype='step',
			label='LYSO activity'
		)
		plt.hist(
			bm_pre.mask(data.cut(a_lyso,hi=15000), key_or_keys = a_this, apply_mask=False),
			bins=np.linspace(10000,450000,200),
			histtype='step',
			label='no LYSO activity'
		)
		plt.yscale('log')
		plt.show()
	
	# apply LYSO cut and vMax cut
	bm_pre.mask(data.cut(a_lyso, lo=15000), apply_mask=True)
	bm_pre.mask(data.cut(v_this, hi=950), apply_mask=True)


	# same for post-shipping data
	bm_post = data.BranchManager(
		dict_concat([fileio.load_branches(data_file.format(run), branches_needed) for run in runs_post[ch]]),
		export_copies=False,
		import_copies=False
	)
	bm_post.mask(data.cut(a_lyso, lo=15000), apply_mask=True)
	bm_post.mask(data.cut(v_this, hi=950), apply_mask=True)


	return bm_pre, bm_post



def adjuster(raw_data, area_adj_fn, edges, ydata, yerr_sq):

	def calc(args):

		# unpack
		adj_count = args[0]
		adj_area = args[1:]

		# multiply area values by adj_area
		# bin the data
		# multiply counts by adj_count
		shifted_counts = np.histogram(area_adj_fn(raw_data, *adj_area), edges)[0]
		adjusted_counts = shifted_counts * adj_count
		adjusted_counts_err = np.sqrt(shifted_counts) * adj_count

		return adjusted_counts, adjusted_counts_err

	def pulls(args):

		# # unpack
		# adj_area, adj_count = args

		# # multiply area values by adj_area
		# # bin the data
		# # multiply counts by adj_count
		# shifted_counts = np.histogram(raw_data * adj_area, edges)[0]
		# adjusted_counts = shifted_counts * adj_count
		# adjusted_counts_err = np.sqrt(shifted_counts) * adj_count

		adjusted_counts, adjusted_counts_err = calc(args)

		# compare to ydata
		resid = ydata - adjusted_counts
		resid_err = np.sqrt(yerr_sq + adjusted_counts_err**2)

		# return pulls
		return resid / resid_err

	return pulls, calc 



def main(ch, diag=False):

	a_this = "area_3046_{}".format(ch)
	v_this = "vMax_3046_{}".format(ch)
	a_lyso = "area_3046_4"

	# acquire data
	bm_pre, bm_post = procure_data(ch, diag)

	if diag:
		plt.hist(bm_pre[ a_this], bins=np.linspace(10000,450000,300), histtype='step', label='pre')
		plt.hist(bm_post[a_this], bins=np.linspace(10000,450000,300), histtype='step', label='post')
		plt.yscale('log')
		plt.legend()
		plt.show()

	# create bin array
	edges = np.linspace(*COMPARE_RANGE[ch], NBINS)
	mids = (edges[1:] + edges[:-1]) * 0.5

	# bin the pre-shipping data
	ydata = np.histogram(bm_pre[a_this], edges)[0]

	# compose fit function
	pulls, calc = adjuster(
		bm_post[a_this],
		# lambda raw,b,c:raw*b+c,
		lambda raw,b:raw*b,
		edges,
		ydata,
		ydata
	)


	# p0 = [1.75, 1.12, -1000]
	p0 = [1.0, 1.0]
	result = opt.least_squares(
		pulls,
		# xdata=None,
		# ydata=ydata,
		x0=p0,
		# sigma=np.sqrt(ydata),
		# absolute_sigma=True,
		method='lm',
		diff_step = 0.01 # [0.001, 0.001, 0.001]
	)


	print('\n')
	print(a_this)
	print(result)
	print(result.x)
	print(result.cost)


	# goodness of fit
	pulls_opt = pulls(result.x)
	print("\nsum of squares of pulls: {:.3f}".format((pulls_opt**2).sum()))
	print("number of data points: {}".format(pulls_opt.size))
	print("chi2/dof: {}".format((pulls_opt**2).sum() / pulls_opt.size))
	plt.hist(pulls_opt, bins=50)
	plt.show()

	# plot results
	plt.step(mids, ydata, color='k', where='mid', label='pre-ship')
	plt.step(mids, calc(p0)[0], color='g', where='mid', label='p0')
	plt.step(mids, calc(result.x)[0], color='b', where='mid', label='popt')
	plt.yscale('log')
	plt.legend()
	plt.xlabel(a_this)
	plt.title("ch {}".format(ch))
	plt.show()


	sys.exit(0)





if __name__ == "__main__":
	for channel in [1]:
		main(channel)
