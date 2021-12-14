"""
script used for testing features during development
don't actually use this for anything else
"""

import sys
import os
import math
import numpy as np

import matplotlib.pyplot as plt

import utils.fileio as fileio
import utils.display as display

DEFAULT_FILE = "./data/root/scintillator/Run{}.root"
FIG_LOCATION = "./figs/{}.png"

argtypes = [str, str]
kwargtypes = {
	"bfLo":float, # low end of fit region
	"bfHi":float, # high end of fit region
	"bc"  :str,   # which branch to cut on
	"bcLo":float, # low end of cut range
	"bcHi":float, # high ene of cut range
	# "m"   :int, # id of model to apply
	# "cf"  :str, # calibration file to search
	# "ct"  :int, # whether cuts are on transformed (1) or raw (0) data
	"svf" :str, # save figure to this file
	"show":int, # whether to show results
	"bins":int, # number of bins to use
}
defaults = {
	"bfLo":None, # lower bound on bf = minumum datum
	"bfHi":None, # upper bound on bf = maximum datum
	"bc"  :None, # don't cut on any branch
	"bcLo":None, # lower cut on bc = minumum datum
	"bcHi":None, # upper cut on bc = maximum datum
	# "m"   :-1, # no model; don't transform
	# "cf"  :"", # no default calibration file; must be specified
	# "ct"  :1,  # yes, cuts specified on transformed data (if transforming)
	"svf" :"", # don't save a figure
	"show":1,  # yes, show results
	"bins":0,  # calculate bin count automatically
}

if __name__ == '__main__':

	# todo: verbosity flag and print stuff based on it
	
	# parse command line arguments and assign locals
	args, kwargs = fileio.parse_cla(sys.argv, argtypes, kwargtypes)
	file = args[0]
	bf   = args[1]
	bfLo = kwargs.get("bfLo", defaults.get("bfLo"))
	bfHi = kwargs.get("bfHi", defaults.get("bfHi"))
	bc   = kwargs.get("bc"  , defaults.get("bc"  ))
	bcLo = kwargs.get("bcLo", defaults.get("bcLo"))
	bcHi = kwargs.get("bcHi", defaults.get("bcHi"))
	# m    = kwargs.get("m"   , defaults.get("m"   ))
	# cf   = kwargs.get("cf"  , defaults.get("cf"  ))
	# ct   = kwargs.get("ct"  , defaults.get("ct"  ))
	svf  = kwargs.get("svf" , defaults.get("svf" ))
	show = kwargs.get("show", defaults.get("show"))
	bins = kwargs.get("bins", defaults.get("bins"))

	# default file location and template applied if file is int
	try:
		ifile = int(file)
		file = DEFAULT_FILE.format(ifile)
	except:
		pass

	# load needed branches
	branches_needed = {bf}
	if bc is not None:
		branches_needed |= {bc}
	branches = fileio.load_branches(file, branches_needed)
	raw_fit = branches[bf]
	if bc is not None:
		raw_cut = branches[bc]

	# transform if requested
	# todo: transform data if model != -1
	# todo: transform bounds if model != -1 and ct == 0
	data_fit = raw_fit
	if bc is not None:
		data_cut = raw_cut

	# calculate bound filters
	filters = []
	if bfLo is not None:
		filters.append(data_fit > bfLo)
	if bfHi is not None:
		filters.append(data_fit < bfHi)
	if bcLo is not None:
		filters.append(data_cut > bcLo)
	if bcHi is not None:
		filters.append(data_cut < bcHi)
	
	# apply bound filters if there are any
	if filters:
		filters = np.stack(filters, axis=0)
		ftr = np.all(filters, axis=0)
		data = data_fit[ftr]
		del ftr
	else:
		data = data_fit

	# clean up unused arrays, since we can move forward using solely data
	del branches
	del filters
	del raw_fit, data_fit
	if bc is not None:
		del raw_cut, data_cut

	# calculate number of bins to use
	if bins == 0:
		bins = display.bin_count_from_n_data(len(data))

	# compose figure
	# todo: better labels
	plt.hist(data, bins=bins)
	plt.xlabel(bf)
	plt.ylabel("counts")
	plt.title(file.rpartition(os.sep)[2])

	# show and/or save if specified
	if svf:
		plt.savefig(FIG_LOCATION.format(svf), dpi=120)
	if show:
		plt.show()
	else:
		plt.clf()
