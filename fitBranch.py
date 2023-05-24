"""
load a branch from a root file, cut, transform, and fit
optionally output fit results as csv and/or figure
"""

__author__ = "Brunel Odegard"
__version__ = "0.0"


import sys
import os
import math
import gzip
import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils.fileio  as fileio
import utils.display as display
import utils.model   as model
import utils.data    as data
import utils.cli     as cli
import utils.config  as config




# These are static globals that are not configurable.
# All static globlals that are configurable have been moved to
# configuration files.

# file extensions
EXT_ROOT = ".root"
EXT_CSV  = ".csv"
EXT_PNG  = ".png"
EXT_NPZ  = ".npz"
EXT_TXT  = ".txt"
EXT_GZ   = ".gz"
EXT_DATA_FILES = [
	EXT_ROOT,
	EXT_NPZ,
	EXT_TXT,
]

# list of parameters corresponding to the locations of peaks
PEAK_PARAMETERS = ["mu", "xpeak"]

# error message templates
ERR_NO_VALID_CALIBRATION = "No matching calibration entry found in file {} for id {} and branch {}"
ERR_NO_FIT_MODEL = "Fit model is empty. Need to specify at least one function (background, gaussian, etc."

# mu low,hi; sigma lo,hi; A lo,hi
GAUS_DEFAULTS  = ("",-np.inf,np.inf,0.0,np.inf,0.0,np.inf)
SMONO_DEFAULTS = (3,-np.inf,np.inf,0,np.inf)

# csv file type lists
FIT_CSV_TYPELIST = [int, str, float, float, int, str, int, int, str, int, str]
XF_CSV_TYPELIST  = [str, int, float]




# utility functions
# todo: some of these should be moved into a utils module since many scripts will need the functionality
def bin_count_from_ndata(ndata):
	nraw = math.ceil(BIN_COUNT_MULT * math.sqrt(ndata))
	return max([BIN_COUNT_MIN, nraw])

# TODO switch to using a fit_result class during whole process
#      and let the class, defined in fileio, handle data formatting.
def csv_format_gaus(gaus):
	"""flattens and formats gaussian names and bounds for CSV file"""
	# gaus_names = []
	# gaus_bounds = []
	gaus_flat = []
	for g in gaus:
		gaus_flat.append(g[0] if g[0] else "-")
		this_bounds = [g[5],g[6],g[1],g[2],g[3],g[4]]
		gaus_flat += this_bounds
	# return gaus_names, gaus_bounds
	return gaus_flat

def csv_format_smono(smono_bounds):
	"""flattens and formats suppressed monomial bounds for CSV file"""
	smono_flat = []
	for sm in smono_bounds:
		for bound in sm:
			smono_flat += bound
	return smono_flat

def csv_format_cuts(cuts):
	"""flattens and formats cuts for CSV file"""
	# cut_branches = []
	# cut_bounds = []
	cut_flat = []
	for c in cuts:
		cut_flat.append(c[0])
		cut_flat += c[1:3]
	# return cut_branches, cut_bounds
	return cut_flat

def edges_lin(xmin, xmax, nbins):
	return np.linspace(xmin, xmax, nbins+1)

def edges_log(xmin, xmax, nbins):
	return np.logspace(math.log(xmin,10), math.log(xmax,10), nbins+1)

def edges_symlog(xmin, xmax, nbins, l=1):
	slxmin = symlog(xmin, l)
	slxmax = symlog(xmax, l)
	y = np.linspace(slxmin, slxmax, nbins+1)
	return isymlog(y, l)

def symlog(x, l):

	isscalar = np.isscalar(x)
	x = np.atleast_1d(x)

	y = np.zeros(x.shape)
	
	ftr_pos = (x >  l)
	ftr_neg = (x < -l)
	ftr_lin = np.logical_not(np.logical_or(ftr_pos,ftr_neg))
	
	b=math.e/l
	y[ftr_pos] =  np.log( b*x[ftr_pos])
	y[ftr_neg] = -np.log(-b*x[ftr_neg])
	y[ftr_lin] = x[ftr_lin] * (b/math.e)
	
	if isscalar:
		return y[0]
	else:
		return y

def isymlog(y, l):
	
	isscalar = np.isscalar(y)
	y = np.atleast_1d(y)

	x = np.zeros(y.shape)

	ftr_pos = (y >  1)
	ftr_neg = (y < -1)
	ftr_lin = np.logical_not(np.logical_or(ftr_pos,ftr_neg))

	x[ftr_pos] =  np.exp( y[ftr_pos] - 1)
	x[ftr_neg] = -np.exp(-y[ftr_neg] - 1)
	x[ftr_lin] = y[ftr_lin]

	if isscalar:
		return x[0] * l
	else:
		return x * l


# analysis stages

# stage 0: args
# extract and process arguments into local variables
# for now, function just returns list of args
# TODO change to using instance of a class,
#      which would be passed through the whole routine
def extract_arguments(args):
	"""extract and prrocess command line arguments"""

	# run
	# try to interpret as integer -> numeric file in default location
	try:
		run_id = int(args["run"])
		run = DATA_FILE_NUMERIC.format(num = run_id)
	# not an integer
	except:
		run = args["run"]

		# get data file extension, ignoring .gz if file is compressed
		if run.endswith(EXT_GZ):
			ext = run.rpartition('.')[0].rpartition('.')[2]
		else:
			ext = run.rpartition('.')[2]

		# no extension -> add default extension
		# if not any(run.endswith(_) for _ in EXT_DATA_FILES+[EXT_GZ]):
		if not ('.{}'.format(ext) in EXT_DATA_FILES):
			run += args["default_data_extension"]

		# no sep -> default location based on file extension
		if os.sep not in run:
			# ext = run.rpartition('.')[2]
			run = os.sep.join([DATA_DIR_TYPES[ext], run])

		# extract file name and extension
		name = run.rpartition(os.sep)[2].partition('.')[0]
		# name, _, ext = run.rpartition(os.sep)[2].rpartition('.')

		# try to extract number from name
		try:
			if name.lower().startswith("run"):
				run_id = int(name[3:])
			else:
				run_id = -1
		except:
			run_id = -1
		# todo: better extraction of run#. Will fail if filename format changes.
		# could also switch to using full filename instead of ID in calibration entries.

	# fit
	fits = [data.split_with_defaults(_, [None,-np.inf,np.inf], [str,float,float]) for _ in args["fit"]]
	fit = fits[0]

	# --cut
	if args["cut"] is None:
		cuts = []
	else:
		cuts = data.split_with_defaults(args["cut"], [None,-np.inf,np.inf,"and"], [str,float,float,str])

	# --er
	event_range = args["event_range"]

	# --con
	# TODO change to specifying convolution per branch being used
	#      or have list of branches to convolve if specified
	convolve = args["convolve"]

	# --model, --model_explicit
	if not (args["model"] is None):
		model_id, model_cal_file = data.split_with_defaults(args["model"],[-1,None],[int,str])
	elif args["model_explicit"]:
		model_id = args["model_explicit"][0]
		model_cal_file = None
	else:
		model_id = -1
		model_cal_file = None

	# -r
	raw_bounds = args["raw_bounds"]

	# --bins
	nbins = args["nbins"]

	# --bg
	background = args["background"]

	# --g
	gaus = args["gaus"]

	# --s
	smono = args["smono"]

	# --rs
	ref_spec = args["rs"]
	
	# --xf
	ref_xf = args["xf"]

	# display args
	disp = args["disp"]
	ylim = args["ylim"]
	xlog = args["xlog"]
	ylog = args["ylog"]
	show = args["show"]
	label = args["label"]
	density = args["density"]
	fill = args["fill"]

	# output args
	file_out = args["out"]
	fig_out  = args["fig"]
	xf_out   = args["xfout"]
	verbosity = args["v"]

	# print raw and processed args if verbosity >= 2
	if verbosity >= 2:
		print("unprocessed args")
		for key,value in sorted(args.items(), key=lambda kv:kv[0]):
			print(key,value)
		print("\nprocessed args")
		print("run        : {}".format(run))
		print("run_id     : {}".format(run_id))
		print("fits       : {}".format(fits))
		print("fit        : {}".format(fit))
		print("cuts       : {}".format(cuts))
		print("event_range: {}".format(event_range))
		print("convolve   : {}".format(convolve))
		print("model      : {}".format([model_id, model_cal_file]))
		print("raw_bounds : {}".format(raw_bounds))
		print("nbins      : {}".format(nbins))
		print("background : {}".format(background))
		print("gaus       : {}".format(gaus))
		print("smono      : {}".format(smono))
		print("ref_spec   : {}".format(ref_spec))
		print("ref_xf     : {}".format(ref_xf))
		print("disp       : {}".format(disp))
		print("ylim       : {}".format(ylim))
		print("xlog       : {}".format(xlog))
		print("ylog       : {}".format(ylog))
		print("show       : {}".format(show))
		print("label      : {}".format(label))
		print("density    : {}".format(density))
		print("fill       : {}".format(fill))
		print("file_out   : {}".format(file_out))
		print("fig_out    : {}".format(fig_out))
		print("xf_out     : {}".format(xf_out))
		print("verbosity  : {}".format(verbosity))
		print("")

	return [
		[run,run_id,fits,fit,cuts,event_range,convolve,model_id,model_cal_file,raw_bounds,],
		[background,gaus,smono,nbins,ref_spec,ref_xf],
		[disp,ylim,xlog,ylog,show,label,density,fill,file_out,fig_out,xf_out],
		verbosity,
	]


# stage 1: data
# load and and process branches from root files
# get calibration and apply model
# calculate and apply filters
# finish with final data to analyse, and concise info for display and log
def procure_data(args,verbosity,run,run_id,fit,cuts,event_range,convolve,model_id,model_cal_file,raw_bounds):
	# load data
	branches_needed = {fit[0]}
	branches_needed |= {_[0] for _ in cuts}

	# branches not to be requested from root file
	# must have other defined behavior
	branches_special = {"entry","Entry"}


	# create file object to pass to loading function based on content type.
	# at this point, run is already the full file path; default file types
	# and directories and names have been applied.
	# 
	# if the file path ends with ".gz", its type will come from the
	# second-to-last extension present.
	if run.endswith(EXT_GZ):
		data_file = gzip.open(run,'rb')
		data_file_type = (run.rpartition('.')[0]).rpartition('.')[2]
	else:
		data_file = open(run,'rb')
		data_file_type = run.rpartition('.')[2]

	# load non-special branches from file
	if data_file_type == "root":
		branches = fileio.load_branches(data_file, branches_needed - branches_special)
	elif data_file_type == "npz":
		branches = {key:arr for key,arr in np.load(data_file).items() if key in branches_needed - branches_special}
	elif data_file_type == "txt":
		branches = fileio.load_rpi_txt(data_file, branches_needed - branches_special, args["trigger_window"])

	# print(data_file)
	# print(data_file_type)

	# close data file
	data_file.close()


	# add special branches
	if {"entry","Entry"} & branches_needed:
		for which in list({"entry","Entry"} & branches_needed):
			branches[which] = np.arange(list(branches.values())[0].shape[0])

	# convolve scalers
	kernel_size = convolve[0]
	if kernel_size:

		if len(convolve)>1:
			convolve_method = convolve[1]
		else:
			convolve_method = 0

		if convolve_method == 0:
			kernel = np.ones(kernel_size)/kernel_size
			for key in branches.keys():
				if key.startswith("scaler"):# and (fit[0]!=key):
					branches[key] = np.convolve(branches[key], kernel, mode='same')

		elif convolve_method == 1:
			for key in branches.keys():
				if key.startswith("scaler"):
					max_l = np.stack([np.roll(branches[key], _) for _ in range(-kernel_size,0)],axis=0).max(0)
					max_r = np.stack([np.roll(branches[key], _) for _ in range(1,kernel_size+1)],axis=0).max(0)
					branches[key] = np.maximum(branches[key], np.minimum(max_l, max_r))
					del max_l
					del max_r

		else:
			print("Warning: convolution method {} is not defined. No convolution has been applied.".format(convolve_method))
		
	if verbosity:
		print("loaded branches: {}".format(branches_needed))
		print("shapes: {}".format([_.shape for _ in branches.values()]))
		print("")

	# slice branches to specified event range before any processing is done
	if event_range:

		# assume slice is on axis 0
		n_events = branches[fit[0]].shape[0]

		# calculate initial and final indices
		ei = math.floor(n_events * event_range[0])
		ef = math.floor(n_events * event_range[1])

		if verbosity:
			print("{} events".format(n_events))
			print("lo {} -> {}".format(event_range[0], ei))
			print("hi {} -> {}".format(event_range[1], ef))
			print("cutting all branches to specified range")
			print("")
		
		branches = {key:arr[ei:ef] for key,arr in branches.items()}

	# if using model, get calibration for fit branch
	# if not raw_bounds, convert bounds from specified transformed values into raw values
	if model_id != -1:


		# create model to use
		this_model = model.models_by_id[model_id]()


		# model params specified explicitly by --me
		if model_cal_file is None:
			bc_fit = [
				args["model_explicit"][1],
				args["model_explicit"][2:2+this_model.npars],
				args["model_explicit"][2+this_model.npars:2+this_model.npars*2],
			]


		# model params in file specified by --m
		else:
			# load entries from model_cal_file
			calibration_entries = fileio.load_calibration(model_cal_file)
			print(model_cal_file)
			print(calibration_entries)

			# todo: get voltage from run. currently ignoring voltage.
			# get best calibration for fit branch
			ibc_fit,bc_fit = fileio.get_best_calibration(
				calibration_entries,
				# fit[0],
				None, # ignore branch
				None, # ignore voltage
				run_id,
				model_id,
				ard=True,
				)
			
			# error if no matching calibration found
			if ibc_fit == -1:
				raise ValueError(ERR_NO_VALID_CALIBRATION.format(model_cal_file,model_id,fit[0]))


		if verbosity:
			print("calibration found for fit branch")
			print("model {}".format(model_id))
			print("pct error: {}".format(bc_fit[0]))
			print("parameters: {}".format(bc_fit[1]))
			print("par errors: {}".format(bc_fit[2]))

		# convert bounds to raw values if not raw_bounds
		if not raw_bounds:

			# transform fit bounds
			if fit[1] != -np.inf:
				fit_lo_new = this_model.fn(fit[1], *bc_fit[1]) if this_model.val(fit[1], *bc_fit[1]) else 0.0
			else:
				fit_lo_new = -np.inf
			if fit[2] != np.inf:
				fit_hi_new = this_model.fn(fit[2], *bc_fit[1]) if this_model.val(fit[2], *bc_fit[1]) else 0.0
			else:
				fit_hi_new = np.inf
			if verbosity:
				print("bounds are raw, so transforming fit branch bounds")
				print("fit lo bound {} -> {}".format(fit[1], fit_lo_new))
				print("fit hi bound {} -> {}".format(fit[2], fit_hi_new))
				print("")
			fit[1] = fit_lo_new
			fit[2] = fit_hi_new

			# transform cut bounds
			for i,cut in enumerate(cuts):

				# get best calibration for cut branch
				ibc_cut,bc_cut = fileio.get_best_calibration(
					calibration_entries,
					None, # ignore branch
					None, # ignore voltage
					run_id,
					model_id,
					ard=True,
					)

				# error if no matching calibration found
				if ibc_cut == -1:
					raise ValueError(ERR_NO_VALID_CALIBRATION.format(model_cal_file,model_id,cut[0]))

				if verbosity:
					print("calibration found for cut branch {}".format(cut[0]))
					print("model {}".format(model_id))
					print("pct error: {}".format(bc_cut[0]))
					print("parameters: {}".format(bc_cut[1]))
					print("par errors: {}".format(bc_cut[2]))

				# apply model to cut bounds
				if cut[1] != -np.inf:
					cut_lo_new = this_model.fn(cut[1], *bc_cut[1]) if this_model.val(cut[1], *bc_cut[1]) else 0.0
				else:
					cut_lo_new = cut[1]
				if cut[2] != np.inf:
					cut_hi_new = this_model.fn(cut[2], *bc_cut[1]) if this_model.val(cut[2], *bc_cut[1]) else 0.0
				else:
					cut_hi_new = cut[2]
				if verbosity:
					print("bounds are not raw, so transforming cut branch bounds to raw values")
					print("cut lo bound {} -> {}".format(cut[1], cut_lo_new))
					print("cut hi bound {} -> {}".format(cut[2], cut_hi_new))
					print("")
				cut[1] = cut_lo_new
				cut[2] = cut_hi_new



	# list of filters to apply (with and logic)
	fit_data = branches[fit[0]]
	filters = []
	filters_or = []

	# fit branch validation
	if model_id != -1:
		filters.append(this_model.val(branches[fit[0]], *bc_fit[1]))
		if verbosity:
			print("{} valid: {} / {} pass".format(fit[0], filters[-1].sum(), filters[-1].shape[0]))

	# apply model to fit branch
	if model_id != -1:
		# print(fit_data[:10])
		fit_data = this_model.fn(fit_data, *bc_fit[1])
		branches[fit[0]] = fit_data
		# print(fit_data[:10])
		if verbosity:
			print("applied model to filtered fit data")
			print("")

	# calculate bound filters
	for i,cut in enumerate([fit] + cuts):

		# determine cut logic group
		if i == 0:
			logic = "and"
		else:
			logic = cut[3]

		filter_group = filters if logic == "and" else filters_or

		b,bLo,bHi = cut[:3]

		if i==0:
			br = fit_data
		else:
			br = branches[b]

		if bLo != -np.inf:
			filter_group.append(br >= bLo)
			print("{} >= {}: {} / {} pass".format(b,bLo,filter_group[-1].sum(), filter_group[-1].shape[0]))
		if bHi != np.inf:
			filter_group.append(br <= bHi)
			print("{} <= {}: {} / {} pass".format(b,bHi,filter_group[-1].sum(), filter_group[-1].shape[0]))

	# turn filters_or into one filter, append to filters
	if filters_or:
		ftr_or = np.any(np.stack(filters_or, axis=0), axis=0)
		filters.append(ftr_or)
		if verbosity:
			print("combined or filter (any): {} / {} pass".format(ftr_or.sum(), ftr_or.size))
			print("combined filter appended to list of AND filters")
	else:
		if verbosity:
			print("no filters with OR logic")

	# stack and compress filters; apply to get fit data
	if filters:
		ftr = np.all(np.stack(filters, axis=0), axis=0)
		fit_data = fit_data[ftr]
		if verbosity:
			if model_id == -1:
				print("combined filter: {} / {} pass".format(ftr.sum(), ftr.shape[0]))
			else:
				print("combined filter (including fit branch validation): {} / {} pass".format(ftr.sum(), ftr.shape[0]))
			print("applied filters to fit data")
			print("")
	else:
		if verbosity:
			print("no filters to apply")
			print("")
		# fit_data = branches[fit[0]]

	# clean up unneeded arrays
	if filters:
		del ftr
	del filters
	del branches

	# print("fit data mean")
	# print(fit_data.mean())
	# print("")

	# print("fit data segment means")
	# sl = 5000
	# sample_means = []
	# for i in range(fit_data.size // sl):
	# 	sample_means.append(fit_data[i*sl:(i+1)*sl].mean())
	# print(sample_means)
	# print("")

	# if specified, print statistical measures (moments) of fit_data
	if args["moments"]:
		print("moments of {} after all cuts are applied:".format(fit[0]))
		print("mean  : {}".format(fit_data.mean()))
		print("stdev : {}".format(fit_data.std()))
		print("min   : {}".format(np.min(fit_data)))
		print("max   : {}".format(np.max(fit_data)))
		print("percentiles")
		print(" 1% : {}".format(np.percentile(fit_data,  1)))
		print(" 5% : {}".format(np.percentile(fit_data,  5)))
		print("25% : {}".format(np.percentile(fit_data, 25)))
		print("50% : {}".format(np.percentile(fit_data, 50)))
		print("75% : {}".format(np.percentile(fit_data, 75)))
		print("95% : {}".format(np.percentile(fit_data, 95)))
		print("99% : {}".format(np.percentile(fit_data, 99)))
		print("")

	return fit_data


# stage 2: fit
# bin fit_data
# parse fit function components and compose fit function
# fit the fit function to binned fit_data
# acquire parameters' best fit and errors
def perform_fit(args,verbosity,fit_data,fit,vars_fit,xlog,density):

	# unpack vars
	background,gaus,smono,nbins,ref_spec,ref_xf = vars_fit

	# process reference spectrum arguments
	if ref_spec:
		fit_to_reference = True
		rs_file,_,rs_run = ref_spec.partition(',')
		if not (os.sep in rs_file):
			rs_file = RS_LOC.format(rs_file)
		rs_run = int(rs_run) if rs_run else None
	
		xf_order = int(ref_xf[0])
		XF_PARAM_GUESS = [0.0, 1.0, 0.0]
		for iv,value in enumerate(ref_xf[1:]):
			XF_PARAM_GUESS[2-xf_order+iv]=value
	else:
		fit_to_reference = False

	# make bin array
	# auto generate bin count if not specified
	if nbins == 0:
		nbins = bin_count_from_ndata(fit_data.shape[0])
		if verbosity:
			print("automatic bin count = {}".format(nbins))
			print("")
	fit_branch = fit[0]
	fit_bounds = [fit[1], fit[2]]
	edgeLo = fit_data.min() if fit_bounds[0] == -np.inf else fit_bounds[0]
	edgeHi = fit_data.max() if fit_bounds[1] ==  np.inf else fit_bounds[1]
	# if xlog argument supplied at least twice, bins are calculated in log space.
	# warning: fitting to log bins has not been confirmed to be accurate.
	if args["xsymlog"]:
		edges = edges_symlog(edgeLo, edgeHi, nbins, args["xsymlog"])
	elif xlog>1:
		edges = edges_log(edgeLo, edgeHi, nbins)
	else:
		edges = edges_lin(edgeLo, edgeHi, nbins)
	midpoints = 0.5 * (edges[1:] + edges[:-1])
	# TODO: better handling of density: should be display-only, not at data level.
	if density:
		print("WARNING: using density mode. Currently this only is supported for display. Fitting density data may lead to undefined behavior.")
	counts, edges = np.histogram(fit_data, edges, density=density)

	# scale histogram counts if specified. default value of scale is 1.0
	counts = args["scale"] * counts

	if ref_spec:
		
		# load reference spectrum and choose best entry
		print("WARNING: skipping branch identity check due to channel re-assignment. May cause issues.")
		# TODO better handling of branch identity. May need to remove this check entirely, but care needed to examine effects.
		spectra = [_ for _ in fileio.load_fits(rs_file)]# if _.fit_branch == fit_branch]
		if rs_run is not None:
			found_match = False
			for spectrum in spectra:
				if spectrum.run == rs_run:
					ref = spectrum
					found_match = True
					break
			if not found_match:
				raise ValueError("no spectrum found in file {} with branch {} and run {}".format(rs_file, fit[0], rs_run))
		else:
			ref = spectra[0]

		# compose fit model
		# relevant properties:
		# 	background
		# 	ngaus, gaus_names, gaus_bounds
		# 	nsmono, smono_order, smono_bounds

		# create components
		fit_model_components = []
		if "q" in ref.background:
			fit_model_components.append(model.quadratic())
		elif "l" in ref.background:
			fit_model_components.append(model.line())
		elif "c" in ref.background:
			fit_model_components.append(model.constant([[0,np.inf]]))
		if "e" in ref.background:
			fit_model_components.append(model.exponential())
		if "p" in ref.background:
			fit_model_components.append(model.powerlaw())
		for bounds in ref.gaus_bounds:
			fit_model_components.append(model.gaus(bounds))
		for bounds in ref.smono_bounds:
			fit_model_components.append(model.smono(bounds))

		# compose model
		if verbosity>1:
			print("fit model components and bounds, from reference spectrum")
			print("")
			for component in fit_model_components:
				print(component.arch.name)
				print(component.pnames)
				print(component.bounds)
				print("")
		if len(fit_model_components):
			fit_model = fit_model_components[0]
			for component in fit_model_components[1:]:
				fit_model = fit_model + component
		else:
			raise ValueError(ERR_NO_FIT_MODEL)

		# use model with reference spectrum parameters to guess a total magnitude factor
		sum_ref  = fit_model(midpoints, *ref.popt).sum()
		sum_data = counts.sum()
		relative_magnitude = sum_data / sum_ref

		# components on top of reference spectrum
		n_postref_params = 0
		# # add gaussians
		gaus_names = []
		for ig,g in enumerate(gaus):
			gaus_names.append(g[0])
			# re-arrange so as to have mu bounds specified first
			this_bounds = [[g[5],g[6]], [g[1],g[2]], [g[3],g[4]]]
			component = model.gaussian(this_bounds)
			fit_model_components.append(component)
			fit_model = fit_model + component
			n_postref_params += 3

		# # test model integrity
		# plt.plot(midpoints, fit_model(midpoints, *ref.popt), 'r-')
		# plt.plot(midpoints, counts, 'k.')
		# plt.show()

		# todo: base bounds on user input and/or x rescaling
		XF_PARAM_BOUNDS = [[-0.1, 0.1], [0.5, 2.0], [-np.inf, np.inf]]
		XF_PARAM_NAMES = ["a", "b", "c"]
		PNAMES_XF  = ["mu","xpeak"]
		# PNAMES_XF  = ["mu"]
		PNAMES_MAG = ["a", "b", "c", "a0", "a1", "a2",  "q"]
		def make_xf_lambda(a,a0,order):
			if order == 2:
				lam = lambda q:q[0]*(a**2)/a0 + q[1]*a + q[2]
				rfs = "({}*[0] + {}*[1] + [2])".format((a**2)/a0, a)
			elif order == 1:
				lam = lambda q:q[0]*a + q[1]
				rfs = "({}*[0] + [1])".format(a)
			elif order == 0:
				lam = lambda q:a + q[0]
				rfs = "({} + [0])".format(a)
			return lam,rfs

		# calculate number of paramters that aren't being transformed
		n_other_params = 0
		for ip,p in enumerate(fit_model.pnames):

			is_postref = ip >= (len(fit_model.pnames) - n_postref_params)

			# not free if transforming it
			if (p in PNAMES_XF) and not is_postref:
				continue

			# # still need to keep fixed parameters as parameters!
			# # not free if lo,hi bounds are identical
			# if fit_model.bounds[ip][0] == fit_model.bounds[ip][1]:
			# 	continue

			# all checks pass -> is free
			n_other_params += 1

		n_params = n_other_params + 1 + xf_order
		i_next_param = 1 + xf_order

		# initialize meta model properties
		xfp     = []
		xfp_rfs = []
		qbounds = XF_PARAM_BOUNDS[2-xf_order:]
		qnames  = XF_PARAM_NAMES[2-xf_order:]
		q0      = XF_PARAM_GUESS[2-xf_order:]

		# construct meta model properties
		for ip,p in enumerate(fit_model.pnames):

			is_postref = ip >= (len(fit_model.pnames) - n_postref_params)

			# location param -> xf via polynomial
			if (p in PNAMES_XF) and not is_postref:
				lam,rfs = make_xf_lambda(ref.popt[ip], 100000, xf_order)
				xfp.append(lam)
				xfp_rfs.append(rfs)

			# else -> assign to param in q if free, constant if fixed
			else:
				bounds = fit_model.bounds[ip]

				# fixed parameter
				if bounds[0] == bounds[1]:
					q0.append(bounds[0])

				# not fixed parameter
				else:

					# postref (not part of reference spectrum) -> guess = bounds midpoint
					if is_postref:
						q0.append(0.5 * (fit_model.bounds[ip][0] + fit_model.bounds[ip][1]))

					# not postref: take from reference spectrum, maybe with scaling
					else:
						# if p represents a magnitude parameter, apply scaling to q0 entry
						if p in PNAMES_MAG:
							q0.append(ref.popt[ip] * relative_magnitude)
						else:
							q0.append(ref.popt[ip])

				# # TEMP hacking order parameter
				# if p == "n":
				# 	bounds = [1.0,4.0]

				unitvec = np.zeros(n_params, dtype=bool)
				unitvec[i_next_param]=True
				xfp.append(unitvec)
				qbounds.append(bounds)
				qnames.append(p)
				xfp_rfs.append(None)
				i_next_param += 1

		# create meta model
		meta = model.metamodel(
			fit_model,
			xfp     = xfp,
			xfp_rfs = xfp_rfs,
			xfx     = False,
		)

		# display reference model and meta model parameter info
		if verbosity>1:
			print("reference model parameters (p)")
			line_template = "{:>8} | {:<32} | {:<32.4s} | {:<32.4s}"
			print(line_template.format("pname", "reference spectrum bounds", "xfp_rfs", "xfp"))
			print(line_template.format(" ", " ", " ", " "))
			for ip,p in enumerate(fit_model.pnames):
				print(line_template.format(str(p), str(fit_model.bounds[ip]), str(xfp_rfs[ip]), str(xfp[ip])))
			print("")
			print("meta model parameters (q)")
			print(line_template.format("pname", "bounds", "guess", ""))
			print(line_template.format(" ", " ", " ", " "))
			for iq,q in enumerate(qnames):
				print(line_template.format(str(q), str(qbounds[iq]), str(q0[iq]), ""))
			print("")

		# # test metamodel integrity
		# plt.plot(midpoints, fit_model(midpoints, *ref.popt), 'r-')
		# plt.plot(midpoints, meta(midpoints, *q0), 'b-')
		# plt.plot(midpoints, counts, 'k.')
		# plt.show()

		# fit binned data with meta -> (popt, perr, chi2, ndod, cov) for parameters *q of metamodel
		qopt, qerr, chi2, ndof, qcov = model.fit_hist_with_root(
			meta.fn,
			midpoints,
			counts,
			qbounds,
			q0,
			True
			)

		# value and covariance of transformation parameters
		# this is all we need in order to apply the same transformation to other measured peaks
		xf_cov = qcov[:xf_order+1,:xf_order+1]
		xf_par = qopt[:xf_order+1]

		# print results
		if verbosity:
			line_template = "{:>8} | {:>12} | {:>12} | {:>24} | {:>24}"
			print("")
			print("meta model performance: chi2/ndof = {}/{} = {}".format(
				round(chi2,DISPLAY_PRECISION),
				ndof,
				round(chi2/ndof,DISPLAY_PRECISION),
				))
			print("meta model parameters")
			print(line_template.format("par","lo bnd","hi bnd","popt","perr"))
			for iq,q in enumerate(qnames):
				print(line_template.format(
					q,
					round(qbounds[iq][0],DISPLAY_PRECISION),
					round(qbounds[iq][1],DISPLAY_PRECISION),
					round(qopt[iq],DISPLAY_PRECISION),
					round(qerr[iq],DISPLAY_PRECISION),
				))

			print("\ncovariance of transformation parameters {}".format(','.join(qnames[:xf_order+1])))
			print(xf_cov)
			if verbosity >= 2:
				print("\nfull covariance matrix")
				print(qcov)
			print("\nsquare root of diagonal of covariance matrix (should be equal to parameter errors)")
			print(np.sqrt(np.diag(qcov)))

		# # test result integrity
		# plt.plot(midpoints, fit_model(midpoints, *ref.popt), 'r-', label="reference")
		# plt.plot(midpoints, meta(midpoints, *q0)           , 'b-', label="q0")
		# plt.plot(midpoints, meta(midpoints, *qopt)         , 'g-', label="qopt")
		# plt.plot(midpoints, counts, 'k.')
		# plt.legend()
		# plt.show()

		# calculate popt from meta model qopt
		popt = meta.transform_parameters(qopt)
		perr = False
		pcov = False

		smono_bounds = ref.smono_bounds
		n_bg_parameters = ref.npars_bg

		# todo: calculate value and errors for each location parameter based on xf_order
		#       and potentially errors on all parameters in popt
		# 
		#       not needed now, since we only use this routine to match LYSO to LYSO
		#       to get an area transformation. All we need is a,b,c and their covariance
		# 
		#       will want to implement this later if we want to compare other spectra
		#       via this method, or to measure LYSO's peaks

	else:
		# list of components for fit model
		fit_model_components = []

		# add background components
		if "q" in background:
			fit_model_components.append(model.quadratic())
		elif "l" in background:
			fit_model_components.append(model.line())
		elif "c" in background:
			fit_model_components.append(model.constant([[0,np.inf]]))
		if "e" in background:
			fit_model_components.append(model.exponential())
		if "p" in background:
			fit_model_components.append(model.powerlaw())

		# store number of parameters used by background components
		n_bg_parameters = sum([_.npars for _ in fit_model_components])

		# add gaussians
		gaus_names = []
		for ig,g in enumerate(gaus):
			gaus_names.append(g[0])
			# re-arrange so as to have mu bounds specified first
			this_bounds = [[g[5],g[6]], [g[1],g[2]], [g[3],g[4]]]
			fit_model_components.append(model.gaussian(this_bounds))

		# add suppressed monomials
		smono_bounds = []
		for ism,sm in enumerate(smono):
			this_bounds = [[sm[1],sm[2]],[sm[3],sm[4]],[sm[0]-1,sm[0]+1]]
			smono_bounds.append(this_bounds)
			order = int(sm[0]) if int(sm[0]) == sm[0] else sm[0]
			fit_model_components.append(model.smono(this_bounds))

		# compose model
		if len(fit_model_components):
			fit_model = fit_model_components[0]
			for component in fit_model_components[1:]:
				fit_model = fit_model + component
		else:
			return [
				[nbins, edges, midpoints, counts],
				[None, 0, None],
				[None, None, None, None, None, None, None],
			]

		# fit the binned data
		popt, perr, chi2, ndof, pcov = fit_model.fit(midpoints, counts, need_cov = True)
		if verbosity:
			col_sep = "\t"
			line_template    = col_sep.join(["{}","{:>6}","{:>8f}","{:>8f}","{:>20.12f}","{:>20.12f}"])
			line_template_ex = col_sep.join(["{}","{:>6}","{:>8}" ,"{:>8}" ,"{:>20}"    ,"{:>20}"    ])
			print("")
			print("model performance: chi2/ndof = {}/{} = {}".format(
				float(chi2),
				ndof,
				float(chi2/ndof),
				))
			print("model parameters")
			print(line_template_ex.format("f","par","lo bnd","hi bnd","popt","perr"))
			ipar = 0
			for ic,component in enumerate(fit_model_components):
				for ip,pname in enumerate(component.pnames):
					print(line_template.format(
						ic,
						pname,
						float(component.bounds[ip][0]),
						float(component.bounds[ip][1]),
						float(popt[ipar]),
						float(perr[ipar]),
					))
					ipar += 1
			if verbosity >= 2:
				print("\nfull covariance matrix")
				print(pcov)
			print("\nsquare root of diagonal of covariance matrix (should be equal to parameter errors)")
			print(np.sqrt(np.diag(pcov)))
			print("\nintegrated area of gaussians")
			print("{} | {:<12} | {:<12} | {:<12}".format("i","c","sigma (bins)","integral = c*sigma*sqrt(2*pi)"))
			bin_width = edges[1]-edges[0]
			total_gaussian_integrals = 0
			
			for ig,g in enumerate(gaus):
				this_c        = popt[n_bg_parameters + 3*ig + 0]
				this_sigma    = popt[n_bg_parameters + 3*ig + 2] / bin_width
				this_integral = this_c * this_sigma * math.sqrt(math.pi*2)
				total_gaussian_integrals += this_integral
				print("{} | {:>12.2f} | {:>12.2f} | {:>12.2f}".format(ig, this_c, this_sigma, this_integral))
			print("sum of integrals of gaussians: {:.2f}".format(total_gaussian_integrals))

			print("\ngaussian component fit results:")
			for ig,g in enumerate(gaus):
				print("\t".join(map("{:.8f}".format,[
					popt[n_bg_parameters + 3*ig + 0],
					perr[n_bg_parameters + 3*ig + 0],
					popt[n_bg_parameters + 3*ig + 1],
					perr[n_bg_parameters + 3*ig + 1],
					popt[n_bg_parameters + 3*ig + 2],
					perr[n_bg_parameters + 3*ig + 2],
				])))
			print("")

			xf_cov = False
			xf_par = False


	return [
		[nbins, edges, midpoints, counts],
		[smono_bounds, n_bg_parameters, fit_model],
		[popt, perr, chi2, ndof, pcov, xf_cov, xf_par],
	]


# stage 3: display and output
# draw a figure if showing or saving it
# show and/or save the figure
# save info to csv if specified
# TODO split this functionality up into two parts
#      a function to display fit data and results,
#      and the save functionality handled by the class instance
def display_and_write(args, verbosity, vars_data,vars_fit,vars_display, fit_data, bin_data,model_data,fit_results, suspend_show, colors):
	
	run,run_id,fits,fit,cuts,event_range,convolve,model_id,model_cal_file,raw_bounds = vars_data
	background,gaus,smono,nbins,ref_spec,ref_xf                                      = vars_fit
	disp,ylim,xlog,ylog,show,label,density,fill,file_out,fig_out,xf_out                        = vars_display

	nbins, edges, midpoints, counts             = bin_data
	smono_bounds, n_bg_parameters, fit_model    = model_data
	popt, perr, chi2, ndof, cov, xf_cov, xf_par = fit_results

	no_fit = (popt is False) or (popt is None)
	has_fit = not no_fit
	
	label_suffix="{}".format(label) if label else ""
	if has_fit:
		label_suffix = ", {}".format(label_suffix)

	# if showing or saving the figure, we need to compose it
	if show or fig_out:

		# display data
		if "d" in disp:
			plt.step(midpoints, counts, where='mid', color=colors.get("d","k"), label="{}{}".format("data" if has_fit else "",label_suffix))
			if fill:
				plt.fill_between(midpoints,counts,step='mid',alpha=fill,color=colors.get("d","k"))

		# display root result
		if ("r" in disp) and has_fit:
			plt.plot(midpoints, fit_model(midpoints,*popt), "g-", label="fit{}".format(label_suffix))

		# display peaks
		if ("p" in disp) and has_fit:
			first_peak = True
			for ip,p in enumerate(fit_model.pnames):
				if p in PEAK_PARAMETERS:
					par = popt[ip]
					if not (perr is False):
						stat_err_str = "\xb1{}".format(round(perr[ip],DISPLAY_PRECISION))
					else:
						stat_err_str = ""
					if model_id == -1:
						label_center = "{}={}".format(
							p,
							round(par,DISPLAY_PRECISION),
							stat_err_str,
						)
					else:
						syst_err = par * bc_fit[6]
						label_center = "{}={}\xb1{}\xb1{}".format(
							p,
							round(par,DISPLAY_PRECISION),
							stat_err_str,
							round(syst_err,DISPLAY_PRECISION),
						)

					# display statistical errors
					if "e" in disp:
						label_error = "\xb1stat err" if first_peak else None
						plt.axvline(popt[ip]+perr[ip], ls='--', color=colors.get("pe","r"), label=label_error)
						plt.axvline(popt[ip]-perr[ip], ls='--', color=colors.get("pe","r"), )

					plt.axvline(popt[ip], ls='--', color=colors.get("p","darkred"), label=label_center)

					first_peak = False

		if ylim is not None:
			plt.ylim(top=ylim)

		if has_fit:
			rchi2 = (chi2/ndof) if ndof else -1
			plt.title(FIGURE_TITLE.format(
				run=run_id,
				branch=fit[0],
				ndof=ndof,
				chi2=round(chi2,DISPLAY_PRECISION) if chi2 else "undef",
				rchi2=round(rchi2,DISPLAY_PRECISION) if rchi2 else "undef",
			))
		else:
			plt.title(FIGURE_TITLE_NOFIT.format(
				run=run_id,
				branch=fit[0],
			))
			
		plt.legend()
		plt.xlabel(fit[0])
		plt.ylabel("counts")
		
		# apply axis scaling
		if args["xsymlog"]:
			plt.xscale("symlog",linthresh=args["xsymlog"], linscale=1/math.e)
		elif xlog:
			plt.xscale("log")

		if args["ysymlog"]:
			plt.yscale("symlog",linthresh=args["ysymlog"], linscale=1/math.e)
		elif ylog:
			plt.yscale("log")


		# set figure size	
		fig = plt.gcf()
		fig.set_size_inches(
			args.get("figure_width_inches" ),
			args.get("figure_height_inches"),
		)
		fig.set_dpi(args.get("figure_dpi"))


	# save the figure if specified
	if fig_out:
		
		# just filename: save in ./figs/
		if not (os.sep in fig_out):
			fig_file = FIG_LOC.format(fig_out)
		else:
			fig_file = fig_out


		# save the figure to an image file
		plt.savefig(fig_file, dpi = args["figure_dpi"] if args["figure_dpi"] else None)

	# show the figure if requested
	if show:
		if not suspend_show:
			plt.show()

	# output transformation parameters and their covariance
	# and necessary idenfitication information (fit branch, xf_order)
	# todo: output a bit more than that
	if ref_spec and xf_out:

		# process reference spectrum arguments
		# TODO this is duplicated; need to handle these variables better
		if ref_spec:
			fit_to_reference = True
			rs_file,_,rs_run = ref_spec.partition(',')
			if not (os.sep in rs_file):
				rs_file = RS_LOC.format(rs_file)
			rs_run = int(rs_run) if rs_run else None
		
			xf_order = int(ref_xf[0])
			XF_PARAM_GUESS = [0.0, 1.0, 0.0]
			for iv,value in enumerate(ref_xf[1:]):
				XF_PARAM_GUESS[2-xf_order+iv]=value
		else:
			fit_to_reference = False

		# just filename: save in ./data/xf/
		if not (os.sep in xf_out):
			csv_file = XF_LOC.format(xf_out)
		else:
			csv_file = xf_out

		# compose new entry
		this_contents = [
			fit[0],
			xf_order,
			*xf_par,
			*xf_cov.flatten(),
		]

		# update the specifed file with new entry
		fileio.update_csv(
			csv_file,
			this_contents,
			XF_CSV_TYPELIST,
			0,
			backup=None,
		)

	# save details to a csv file if requested
	if file_out:
		
		# just filename: save in ./data/fits/
		if not (os.sep in file_out):
			csv_file = RESULT_LOC.format(file_out)
		else:
			csv_file = file_out

		# todo: use fit_result class during whole script
		#       then just use this_contents = result.pack()
		# 
		#       if changing to have model classes handle results,
		#       then do contents = fit_model.pack()

		# compose new entry
		this_contents = [

			# fit data
			run_id, # numerical ID of run
			fit[0], # branch fit
			fit[1], # branch fit lo cut
			fit[2], # branch fit hi cut

			# model info
			model_id, # numerical ID of model
			model_cal_file if model_cal_file else "-", # calibration file used
			int(raw_bounds), # whether bounds are specified on untransformed data

			# fit routine input
			# todo: automatically pack components with archetype,parameters,static etc
			#       none of this case-by-case nonsense
			#       this can be handled better by the model class itself.
			nbins, # number of bins used (copies atuo bin count)
			background if background else "-", # background function
			len(gaus), # number of gaussians
			*csv_format_gaus(gaus), # name,*bounds per gaussian, flattened
			len(smono),
			*csv_format_smono(smono_bounds),
			
			# cut data
			len(cuts), # number of cuts
			*csv_format_cuts(cuts), # br,lo,hi per cut, flattened

			# results
			chi2,
			ndof,
			n_bg_parameters,
			*popt,
			*perr,

		]
		
		# update the specifed file with new entry
		fileio.update_csv(
			csv_file,
			this_contents,
			FIT_CSV_TYPELIST,
			0,
			backup=None,
		)




def main(args, suspend_show=False, colors={}):

	# stage 0: args
	vars_data, vars_fit, vars_display, verbosity = extract_arguments(args)
	run,run_id,fits,fit,cuts,event_range,convolve,model_id,model_cal_file,raw_bounds = vars_data
	background,gaus,smono,nbins,ref_spec,ref_xf                                      = vars_fit
	disp,ylim,xlog,ylog,show,label,density,fill,file_out,fig_out,xf_out                        = vars_display

	# >1 fit branches: 2d comparisons
	# TODO add fitting per branch and plotting fits within comparisons
	if len(fits) > 1:

		n_ds = len(fits)
		
		# need same cuts for all -> cut on all fit data ranges as well
		cuts_all = cuts + [_ + ['and'] for _ in fits]

		# fetch each datasets
		pData = [procure_data(args,verbosity,run,run_id,_,cuts_all,event_range,convolve,model_id,model_cal_file,raw_bounds) for _ in fits]
		pLo = [_.min() if fits[i][1] == -np.inf else fits[i][1] for i,_ in enumerate(pData)]
		pHi = [_.max() if fits[i][2] ==  np.inf else fits[i][2] for i,_ in enumerate(pData)]

		if not nbins:
			nbins = 50
		if args["xsymlog"]:
			pBins = [edges_symlog(min((pLo[i]),(pData[i][pData[i]>0]).min()), pHi[i], nbins, args["xsymlog"]) for i in range(n_ds)]
		elif xlog>1:
			pBins = [edges_log(min((pLo[i]),(pData[i][pData[i]>0]).min()), pHi[i], nbins) for i in range(n_ds)]
		else:
			pBins = [edges_lin(pLo[i], pHi[i], nbins) for i in range(n_ds)]

		gs = display.pairs2d(
			pData,
			pBins,
			[args["xsymlog"] if args["xsymlog"] else bool(xlog)]*n_ds,
			[_[0] for _ in fits],
			cmap="afmhot",
			cbad="grey",
			norm="log" if ylog else None,
		)

		# fit 2d gaussian(s) to 2d data. only supported for single comparison (2 fit variables.)
		if args["gaus2d"]:
			assert len(fits) == 2

			xdata, ydata = pData
			xbins, ybins = pBins
			xmids = (xbins[1:] + xbins[:-1])*0.5
			ymids = (ybins[1:] + ybins[:-1])*0.5

			counts, xedges, yedges = np.histogram2d(
				xdata,ydata,
				[np.array(xbins),np.array(ybins)],
			)

			x2d,y2d = np.meshgrid(xmids, ymids, indexing='ij')
			xyflat = np.stack([x2d.flatten(),y2d.flatten()],axis=0)
			countsflat = counts.flatten()

			fit_model = model.gaus2d(zip(args["gaus2d"][0][0::2],args["gaus2d"][0][1::2]))
			for bounds in args["gaus2d"][1:]:
				fit_model = fit_model + model.gaus2d(zip(bounds[0::2],bounds[1::2]))

			popt, perr, chi2, ndof = fit_model.fit(xyflat,countsflat)
			print(popt)
			print(perr)
			print(chi2)
			print(ndof)

			const,x0,y0,s1,s2,theta = popt
			x1,y1 = x0+s1*math.cos(theta), y0+s1*math.sin(theta)
			x2,y2 = x0-s2*math.sin(theta), y0+s2*math.cos(theta)

			# display center and axes
			plt.subplot(gs[1,0])
			plt.plot([x1,x0,x2],[y1,y0,y2],'b-')

			# projections
			copt = fit_model(xyflat, *popt).reshape((nbins,nbins))
			cxopt = copt.sum(axis=1)
			cyopt = copt.sum(axis=0)

			plt.subplot(gs[0,0])
			plt.plot(xmids, cxopt, color="r", ls='--', marker='.')
			plt.subplot(gs[1,1])
			plt.plot(cyopt, ymids, color="r", ls='--', marker='.')




		plt.suptitle("{}: {}\n{}".format(run, fits, cuts))

		# set figure size	
		fig = plt.gcf()
		fig.set_size_inches(
			args.get("figure_width_inches" ),
			args.get("figure_height_inches"),
		)
		fig.set_dpi(args.get("figure_dpi"))

		# save the figure if specified
		if fig_out:
			
			# just filename: save in ./figs/
			if not (os.sep in fig_out):
				fig_file = FIG_LOC.format(fig_out)
			else:
				fig_file = fig_out

			# save the figure to an image file
			plt.savefig(fig_file, dpi = args["figure_dpi"] if args["figure_dpi"] else None)

		if show:
			plt.show()

		if args.get("dump", False):
			# calculate 1d bin counts
			pCounts = [np.histogram(dat, pBins[idat])[0] for idat,dat in enumerate(pData)]
			dump(
				args.get("dump"),
				[_[0] for _ in fits],
				pBins,
				pCounts,
			)


	else:
		# stage 1: data
		fit_data = procure_data(args,verbosity,run,run_id,fit,cuts,event_range,convolve,model_id,model_cal_file,raw_bounds)

		# stage 2: fit
		bin_data, model_data, fit_results = perform_fit(args,verbosity,fit_data,fit,vars_fit,xlog,density)

		# stage 3: display and output
		display_and_write(args, verbosity, vars_data,vars_fit,vars_display, fit_data, bin_data,model_data,fit_results, suspend_show, colors)

		if args.get("dump", False):
			# No need to send most arguments, since we can just save sys.argv
			# We only need to send lists, per fit branch, of:
			#   branch
			#   bin edges
			#   bin counts
			# 
			# send as lists of length 1, so that the function works for both
			# single- and multiple-branch calls (1d and 2d plots)
			nbins, edges, midpoints, counts = bin_data
			dump(
				args.get("dump"),
				[fits[0][0]],
				[edges],
				[counts],
			)

def dump(file, branches, edge_arrays, count_arrays):

	# list of lines for CSV file
	lines = []

	# add line for command line arguments
	lines.append(sys.argv)

	# add lines for each branch
	for ibr,branch in enumerate(branches):

		# line describing branch
		lines.append(["begin", "1d", branch])

		# lines for histogram bins and counts
		for ic,count in enumerate(count_arrays[ibr]):
			lines.append([
				ic,
				edge_arrays[ibr][ic],
				edge_arrays[ibr][ic+1],
				count,
			])

	# determine file location
	if os.sep in file:
		filepath = file
	else:
		# strip redudant extension if present
		if file.endswith(".csv"):
			file = file[:-4]
		filepath = DUMP_LOC.format(file)

	fileio.save_csv(filepath, lines)




if __name__ == '__main__':

	# initial implementation of config handling
	# todo: system that catches when expected fields are missing instead
	# 
	#       of using cfg.get(value, default)
	#       Want to be able to warn when fields are missing, including
	#       when the default config has been updated to include new
	#       fields.
	# 
	#       In this case, we'd warn the user that fields are missing, and
	#       take their values from the default-config (without copying it?)
	# 
	#       This might be best handled by automatically comparing with the
	#       default config file even when it's not being copied. or by having
	#       each script communicate which keys it expects, or by implementing
	#       a class with a method for missing keys.

	# todo: process cgf stuff within main() call
	#       and do ?(config.deep_merge(args, cfg))? inside main
	#       
	#       that way, we can use the same priority structure for cli arguments
	#       as for configuration, granted that variables names and formats are consistent.
	# 
	#       some thought should be given to how to implement arguments that take
	#       multiple parameters for convenience, such as --figsize dpi w h
	#       We want to be able to specify them as such, rather than needing three
	#       separate arguments, but we also want the configuration file to have them
	#       separately, rather then as a list.
	# 
	#       maybe a custom action which writes each specified value to a different variable
	#       E.G. dest=["figure_dpi", "figure_w_inches", "figure_h_inches"]
	#       Actually, just extending the __call__ attribute of the actions defined in utils.cli
	#       to handle the case where dest is a list, and distribute values (setting or appending
	#       or changing, based on the action behavior.)
	# 
	#       It's also worth figuring out how to handle more complicated actions, and
	#       how to avoid overwriting stuff (E.G. specifying a cut in the config tile,
	#       but it gets ignored if any cuts are specified via arguments.)
	# 
	#       perhaps the merge could be done before processing arguments?
	#
	# 
	#       I basically do want to merge argparse's output with the config contents,
	#       but with a couple of caveats:
	#       1) Some arguments have specific actions when the key is present in both,
	#          rather than simply overwriting (if not dict) or merging (if dict.) These
	#          would include concatenating lists for MergeAction and similar.
	#       2) Some arguments should not accept input from config file, though this is
	#          pretty optional. It could get confusing if there are things like cuts or
	#          fits in the config file, so it may make sense to preclude those.


	# load config from files (returns a dict)
	cfg = config.load("fitBranch", "common")


	# filesystem locations
	# 
	# base location (the "data" folder, or equivalent)
	DATA_DIR_BASE = os.sep.join(cfg["data_directory"]["base"])
	#
	DATA_DIR_ROOT = os.sep.join(cfg["data_directory"]["scint_exp_root"]).format(base = DATA_DIR_BASE)
	DATA_DIR_NPZ  = os.sep.join(cfg["data_directory"]["scint_exp_np"  ]).format(base = DATA_DIR_BASE)
	DATA_DIR_TXT  = os.sep.join(cfg["data_directory"]["rpi_txt"       ]).format(base = DATA_DIR_BASE)
	DATA_DIR_TYPES = {
		"root":DATA_DIR_ROOT,
		"npz" :DATA_DIR_NPZ,
		"txt" :DATA_DIR_TXT,
	}
	DATA_DF = cfg["default_data_file"]
	DATA_DF_EXT = DATA_DF.rpartition('.')[0].rpartition('.')[2] if DATA_DF.endswith(EXT_GZ) else DATA_DF.rpartition(".")[2]
	DATA_FILE_NUMERIC = os.sep.join([DATA_DIR_TYPES[DATA_DF_EXT], DATA_DF])
	# 
	# was [".", "data", "calibration", "{}.csv"]
	CALIB_DIR  = os.sep.join(cfg["data_directory"]["calibration"]).format(base = DATA_DIR_BASE)
	CALIB_LOC  = os.sep.join([CALIB_DIR, "{}.csv"])
	# 
	# was [".", "data", "fits", "{}.csv"]
	RESULT_DIR = os.sep.join(cfg["data_directory"]["fits"]).format(base = DATA_DIR_BASE)
	RESULT_LOC = os.sep.join([RESULT_DIR, "{}.csv"])
	# 
	# was [".", "data", "fits", "{}.csv"]
	RS_DIR = os.sep.join(cfg["data_directory"]["fits"]).format(base = DATA_DIR_BASE)
	RS_LOC = os.sep.join([RS_DIR, "{}.csv"])
	# 
	# was [".", "data", "xf", "{}.csv"]
	XF_DIR = os.sep.join(cfg["data_directory"]["xf"]).format(base = DATA_DIR_BASE)
	XF_LOC = os.sep.join([XF_DIR, "{}.csv"])
	# 
	# was [".", "figs", "{}.png"]
	FIG_DIR = os.sep.join(cfg["figure_directory"]["base"])
	FIG_LOC = os.sep.join([FIG_DIR, "{}.png"])
	# 
	DUMP_DIR = os.sep.join(cfg["data_directory"]["spectra"]).format(base = DATA_DIR_BASE)
	DUMP_LOC = os.sep.join([DUMP_DIR, "{}.csv"])


	# multiplier and minumum for automatic bin count calculation
	BIN_COUNT_MULT = cfg["bin_count_mult"]
	BIN_COUNT_MIN  = cfg["bin_count_min" ]

	# how many places past decimal point to show for display purposes
	# does not affect precision of saved information
	DISPLAY_PRECISION = cfg["max_display_precision"]

	# display string templates
	FIGURE_TITLE       = cfg["figure_title"      ]
	FIGURE_TITLE_NOFIT = cfg["figure_title_nofit"]

	# default dpi and size for figures
	FIG_DPI      = cfg["figure_dpi"]
	FIG_W_INCHES = cfg["figure_width_inches"]
	FIG_H_INCHES = cfg["figure_height_inches"]

	# was ["k","m","b","r","c","y",'tab:brown','darkred','magenta']
	COLOR_SEQ = cfg["color_sequence"]

	# for raspberry pi digitizer output (.txt files)
	RPI_TRIG_MIN = cfg["rpi_trigger_inclusive_min"]
	RPI_TRIG_MAX = cfg["rpi_trigger_inclusive_max"]

	

	parser = argparse.ArgumentParser(
		description="fit a model to a binned branch, optionally transforming and cutting",
		)

	# version
	parser.add_argument("--version",action="version",version="%(prog)s {}".format(__version__))

	# data arguments
	parser.add_argument("run"  ,type=str,help="file location, name, or number")
	parser.add_argument("fit"  ,type=str,nargs="+",help="branch to fit: branch,min,max")

	parser.add_argument("--cut",type=str,action='append',help="branch to cut on: branch,min,max")
	parser.add_argument("--er" ,type=float,dest="event_range",nargs=2,help="use a subset of the dataset. --er start stop")
	parser.add_argument("-r"   ,action='store_true',dest="raw_bounds",help="if specified, specified bounds are raw")
	parser.add_argument("--con",type=int,default=[0],nargs="+",dest="convolve",help="scaler convolution (default none); kernel_size [, type]")
	parser.add_argument(
		"--scale", "--s",
		type=float,
		dest="scale",
		default=1.0,
		help="scale all bin counts by this amount",
	)

	parser.add_argument(
		"--trigger-window", "--tw",
		type=str,
		nargs="+",
		dest="trigger_window",
		action=cli.MergeAction,
		const=((int,int),(RPI_TRIG_MIN, RPI_TRIG_MAX)),
		default=(RPI_TRIG_MIN, RPI_TRIG_MAX),
		help="specify trigger window (inclusive) for rpi/digitizer data. --tw min max",
	)

	parser.add_argument("--m",type=str,dest="model",help="model to use: model_id,calibration_file")
	parser.add_argument(
		"--me",
		type=str,
		nargs="+",
		dest="model_explicit",
		action=cli.MergeAction,
		const=((int, float), (0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
		default=(),
		help="explicitly provide model info: ID, max dev, *params, *param errors",
	)

	# fitting arguments
	parser.add_argument("--bins",type=int,default=0,dest="nbins",help="number of bins to use")
	parser.add_argument("--bg"  ,type=str,default="",dest="background",help="background function: any combination of (p)ower (e)xp (c)onstant (l)ine (q)uadratic")
	parser.add_argument(
		"--gaus","--g",
		dest="gaus",
		type=str,
		nargs="+",
		default=[],
		action=cli.MergeAppendAction,
		const=((str,)+(float,)*6,GAUS_DEFAULTS),
		help="gaussian components: min_mu,max_mu (or) name=min_mu,max_mu",
	)
	parser.add_argument(
		"--smono","--sm",
		dest='smono',
		type=str,
		nargs="+",
		default=[],
		action=cli.MergeAppendAction,
		const=((int,)+(float,)*4,SMONO_DEFAULTS),
		help="suppressed monomial: order, min,max xpeak, min,max c, min,max k",
	)
	parser.add_argument(
		"--g2",
		type=str,
		nargs="+",
		dest="gaus2d",
		action=cli.MergeAppendAction,
		const=((float,),(0,np.inf,-np.inf,np.inf,-np.inf,np.inf,0,np.inf,0,np.inf,0,math.pi/2)),
		default=[],
		help="2d gaussian"
	)	

	# if --rs is not emtpy, fit to reference
	# get all fit info from reference spectrum
	# file may contain multiple entries. choose by:
	#     1) identical fit branch
	#     2) run number if specified, else first entry passing 1).
	parser.add_argument("--rs",type=str,default="",help="file containing reference spectrum, and optionally which run to use the spectrum of")
	parser.add_argument("--xf",type=float,default=2,nargs="+",help="order of area transformation, [0-2]. 0 = const. offset, 1 = linear, 2 = quadratic")

	# display arguments
	parser.add_argument("--d"   ,type=str,default="drp",dest='disp',help="display: any combinration of (d)ata (r)oot (p)eaks (e)rror")
	parser.add_argument("--ylim",type=float,help="upper y limit on plot")
	parser.add_argument("-x"    ,dest="xlog",default=0,action="count",help="sets x axis of figure to log scale")
	parser.add_argument("-y"    ,dest="ylog",action="store_true",help="sets y axis of figure to log scale")
	parser.add_argument("-s"    ,dest="show",action="store_false",help="don't show figure as pyplot window")
	parser.add_argument("-d"    ,dest="density",action="store_true",help="use density instead of counts for hist y axes")
	parser.add_argument("--label" ,type=str,default="",dest='label',help="custom label")
	parser.add_argument("--fill"  ,type=float,default=0.0,dest='fill',help="fill under histograms")

	parser.add_argument("--xs",dest="xsymlog",type=float,default=None,help="sets x axis of figure to symmetric log scale")
	parser.add_argument("--ys",dest="ysymlog",type=float,default=None,help="sets y axis of figure to symmetric log scale")

	parser.add_argument("-m",dest="moments",action="store_true",help="print statistical moments of datasets")

	# output arguments
	parser.add_argument("--dump" ,type=str,default="",help="dump details and data to file")
	parser.add_argument("--out"  ,type=str,default="",help="location to save fit results as csv file (appends if file exists)")
	parser.add_argument("--fig"  ,type=str,default="",help="location to save figure as png image (overwrites if file exists)")
	# FIGSIZE_DEFAULT = (FIG_DPI, FIG_W_INCHES, FIG_H_INCHES)
	parser.add_argument(
		"--figsize", "--fig-size", "--fs",
		dest=("figure_dpi", "figure_width_inches", "figure_height_inches"),
		type=str,
		nargs="+",
		action=cli.MultipleDestAction,
		const=(int,float,float),
		# default=FIGSIZE_DEFAULT,
		help="figure size: --fs dpi w_inches h_inches"
	)
	parser.add_argument("--xfout",type=str,default="",help="location to save transformation as csv file (appends if file exists)")
	parser.add_argument("-v"     ,action='count',default=0,help="verbosity")


	# split arguments into calls separated by any delimiter argument,
	# defined by cli.DEFAULT_DELIMITERS
	calls = cli.split_argument_sets(sys.argv[1:])

	# call main with each set of arguments in turn,
	# additionally communicating that each call is
	# part of a multi-call execution
	for i,call in enumerate(calls):
		this_args = vars(parser.parse_args(call))
		main(
			config.deep_merge(this_args, cfg),
			# todo: better behavior in main when part of multiple set call
			suspend_show = True,
			colors = {"d":COLOR_SEQ[i], "p":COLOR_SEQ[i]},
		)

	# show figure after all calls have been processed
	plt.show()

	# exit without error
	sys.exit(0)
