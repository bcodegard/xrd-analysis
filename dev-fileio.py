"""
script used for testing features during development
don't actually use this for anything else
"""

import numpy
import random
import math
import numpy as np

import matplotlib.pyplot as plt

import utils.fileio as fileio


test_load_fit_results = False
if test_load_fit_results:
	file_load = "./data/fits/smono_test.csv"
	fits = fileio.load_fits(file_load)
	ex = fits[0]
	print(ex.ngaus, ex.gaus_names, ex.gaus_bounds)
	print(ex.nsmono, ex.smono_order, ex.smono_bounds)
	print(ex.ncuts, ex.cut_br, ex.cut_lo, ex.cut_hi)
	print(ex.chi2, ex.ndof)
	print(ex.background, ex.npars_bg)
	print(ex.popt_bg)
	print(ex.popt_gaus)
	print(ex.popt_smono)
	print(ex.perr_smono)

test_root = False
if test_root:
	file_load = "./temp/Run3130.root"

	# get the keys
	keys = fileio.get_keys(file_load)
	print('\n'.join(keys))

	# plot a random one that ends with 1
	key = random.choice([_ for _ in keys if _.endswith("1")])
	branch = fileio.load_branches(file_load, key)[key]
	plt.hist(branch, bins=250, density=True)
	plt.xlabel(key)
	plt.show()


test_csv = False
if test_csv:

	file_missing = "./temp/missing.csv"
	file_update = "./temp/update.csv"
	file_backup = "./temp/update_backup.csv"

	# load a missing csv
	mc = fileio.loadcsv(file_missing, [str, int, int, float])
	print(mc)

	# update a csv file
	fileio.updatecsv(
		file_update,
		[["a",1,2,3],["b",2,3,4]],
		[str,int],
		1,)
	fileio.updatecsv(
		file_update,
		[["a",4,2,3],["c",8,8,2]],
		[str,int],
		1,
		backup=file_backup)
	print(fileio.loadcsv(file_update, [str,int]))

	# save (overwite) a csv file
	fileio.savecsv(file_update, ["z",4,0,4])
