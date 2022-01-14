"""
load a branch from a root file, cut, transform, and fit
optionally output fit results as csv and/or figure
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




# filesystem locations
# todo: use os.sep to support multiple platforms
DATA_LOC   = "./data/root/scintillator/Run{}.root"
CALIB_LOC  = "./data/calibration/{}.csv"
FIG_LOC    = "./figs/{}.png"
RESULT_LOC = "./data/fits/{}.csv"
RS_LOC     = "./data/fits/{}.csv" # for now, reference spectra are not separate from general fit results.
XF_LOC     = "./data/xf/{}.csv"

# known file extensions
EXT_ROOT = ".root"
EXT_CSV  = ".csv"
EXT_PNG  = ".png"
EXT_NPZ  = ".npz"

# delimiters for argparse
AP_DELIMITER = ","
AN_DELIMITER = "="

# error message templates
ERR_NO_VALID_CALIBRATION = "No matching calibration entry found in file {} for id {} and branch {}"
ERR_NO_FIT_MODEL = "Fit model is empty. Need to specify at least one function (background, gaussian, etc."

# multiplier and minumum for aturomatic bin count calculation
BIN_COUNT_MULT = 4.0
BIN_COUNT_MIN = 50

# mu low,hi; sigma lo,hi; A lo,hi
GAUS_DEFAULTS  = [-np.inf,np.inf,0.0,np.inf,0.0,np.inf]
SMONO_DEFAULTS = [3,-np.inf,np.inf,0,np.inf]

# how many places past decimal point to show for display purposes
# does not affect precision of saved information
DISPLAY_PRECISION = 3

# list of parameters corresponding to the locations of peaks
PEAK_PARAMETERS = ["mu","xpeak"]

# display string templates
FIGURE_TITLE = "run {run}, {branch}, chi2/dof={chi2:.2f}\n{rfs}"

# csv file type lists
FIT_CSV_TYPELIST = [int, str, float, float, int, str, int, int, str, int, str]
XF_CSV_TYPELIST  = [str, int, float]




# utility functions
# todo: some of these should be moved into a utils module since many scripts will need the functionality
def split_with_defaults(s,defaults,types=None,delimiter=AP_DELIMITER,name_delimiter=None,allow_blank=False):
	"""split string by delimiter and cast to list with defaults and types"""

	# list/tuple of strings -> individual calls per entry
	if type(s) in [list, tuple]:
		return [split_with_defaults(_,defaults,types,delimiter,name_delimiter,allow_blank) for _ in s]

	# single string
	else:
		result = []
		
		if name_delimiter is not None:
			if name_delimiter in s:
				name,_,s = s.rpartition(name_delimiter)
			else:
				name=""

		parts = s.split(delimiter)
		for i,part in enumerate(parts):
			if part or allow_blank:
				if types:
					result.append(types[i](part))
				else:
					result.append(part)
			else:
				result.append(defaults[i])

		result = result + defaults[len(result):]

		if name_delimiter is not None:
			result = [name] + result

		return result

def bin_count_from_ndata(ndata):
	nraw = math.ceil(BIN_COUNT_MULT * math.sqrt(ndata))
	return max([BIN_COUNT_MIN, nraw])

def csv_format_gaus(gaus):
	"""flattens and formats gaussian names and bounds for CSV file"""
	# gaus_names = []
	# gaus_bounds = []
	gaus_flat = []
	for g in gaus:
		gaus_flat.append(g[0] if g[0] else "-")
		gaus_flat += g[1:]
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
		cut_flat += c[1:]
	# return cut_branches, cut_bounds
	return cut_flat




# main
def main(args, suspend_show=False, colors={}):


	# stage 0: args
	# extract and process arguments into local variables

	# run
	try:
		run_id = int(args["run"])
		run = DATA_LOC.format(run_id)
	except:
		run = args["run"]
		if not (run.endswith(EXT_ROOT) or run.endswith(EXT_NPZ)):
			run += EXT_ROOT
		which_ext = '.'+run.rpartition('.')[2]
		run_id = int(run.rpartition(os.sep)[2].partition(which_ext)[0][3:])
		# todo: better extraction of run#. Will fail if filename format changes.
		# could also switch to using full filename instead of ID in calibration entries.

	# fit
	fit = split_with_defaults(args["fit"], [None,-np.inf,np.inf], [str,float,float])

	# --cut
	if args["cut"] is None:
		cuts = []
	else:
		cuts = split_with_defaults(args["cut"], [None,-np.inf,np.inf], [str,float,float])

	# --er
	event_range = args["event_range"]

	# --model
	if args["model"] is None:
		model_id = -1
		model_cal_file = None
	else:
		model_id, model_cal_file = split_with_defaults(args["model"],[-1,None],[int,str])

	# -r
	raw_bounds = args["raw_bounds"]

	# --bins
	nbins = args["nbins"]

	# --rs
	rs_file,_,rs_run = args["rs"].partition(',')
	if not (os.sep in rs_file):
		rs_file = RS_LOC.format(rs_file)
	rs_run = int(rs_run) if rs_run else None

	# --xf
	xf_order = args["xf"]

	# display args
	display = args["display"]
	ylim = args["ylim"]
	xlog = args["xlog"]
	ylog = args["ylog"]
	show = args["show"]
	label = args["label"]
	label_suffix=", {}".format(label) if label else ""

	# output args
	file_out = args["out"]
	fig_out  = args["fig"]
	verbosity = args["v"]

	# print raw and processed args if verbosity >= 2
	if verbosity >= 2:
		print("unprocessed args")
		for key,value in args.items():
			print(key,value)
		print("\nprocessed args")
		print("run        : {}".format(run))
		print("fit        : {}".format(fit))
		print("cuts       : {}".format(cuts))
		print("event_range: {}".format(event_range))
		print("model      : {}".format([model_id, model_cal_file]))
		print("raw_bounds : {}".format(raw_bounds))
		print("nbins      : {}".format(nbins))
		print("rs_file    : {}".format(rs_file))
		print("rs_run     : {}".format(rs_run))
		print("xf_order   : {}".format(xf_order))
		print("display    : {}".format(display))
		print("ylim       : {}".format(ylim))
		print("xlog       : {}".format(xlog))
		print("ylog       : {}".format(ylog))
		print("show       : {}".format(show))
		print("file_out   : {}".format(file_out))
		print("fig_out    : {}".format(fig_out))
		print("verbosity  : {}".format(verbosity))
		print("")


	# stage 1: data
	# load and and process branches from root files
	# get calibration and apply model
	# calculate and apply filters
	# finish with final data to analyse, and concise info for display and log

	# load data
	branches_needed = {fit[0]}
	branches_needed |= {_[0] for _ in cuts}

	if run.endswith(EXT_ROOT):
		branches = fileio.load_branches(run, branches_needed)
	else:
		branches = {key:arr for key,arr in np.load(run).items() if key in branches_needed}

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

		# load entries from model_cal_file
		calibration_entries = fileio.load_calibration(model_cal_file)

		# todo: get voltage from run. currently ignoring voltage.
		# get best calibration for fit branch
		ibc_fit,bc_fit = fileio.get_best_calibration(
			calibration_entries,
			fit[0],
			None, # ignore voltage
			run_id,
			model_id,
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
					cut[0],
					None, # ignore voltage
					run_id,
					model_id,
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
	filters = []

	# fit branch validation
	if model_id != -1:
		filters.append(this_model.ival(branches[fit[0]], *bc_fit[1]))
		if verbosity:
			print("{} valid: {} / {} pass".format(fit[0], filters[-1].sum(), filters[-1].shape[0]))

	# calculate bound filters
	for i,cut in enumerate([fit] + cuts):
		b,bLo,bHi = cut
		if bLo != -np.inf:
			filters.append(branches[b] >= bLo)
			print("{} >= {}: {} / {} pass".format(b,bLo,filters[-1].sum(), filters[-1].shape[0]))
		if bHi != np.inf:
			filters.append(branches[b] <= bHi)
			print("{} <= {}: {} / {} pass".format(b,bHi,filters[-1].sum(), filters[-1].shape[0]))

	# stack and compress filters; apply to get fit data
	if filters:
		ftr = np.all(np.stack(filters, axis=0), axis=0)
		fit_data = branches[fit[0]][ftr]
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
		fit_data = branches[fit[0]]

	# apply model to fit branch
	if model_id != -1:
		fit_data = this_model.ifn(fit_data, *bc_fit[1])
		if verbosity:
			print("applied model to filtered fit data")
			print("")

	# clean up unneeded arrays
	if filters:
		del ftr
	del filters
	del branches


	# stage 2: fit
	# bin fit_data
	# parse fit function components and compose fit function
	# fit the fit function to binned fit_data
	# acquire parameters' best fit and errors

	# auto generate bin count if not specified
	if nbins == 0:
		nbins = bin_count_from_ndata(fit_data.shape[0])
		if verbosity:
			print("automatic bin count = {}".format(nbins))
			print("")

	# make bin array
	# no need to assess min/max values since fit_data already has cuts applied
	edges = np.linspace(fit_data.min(), fit_data.max(), nbins + 1)
	midpoints = 0.5 * (edges[1:] + edges[:-1])
	counts, edges = np.histogram(fit_data, edges)

	# load reference spectrum and choose best entry
	spectra = [_ for _ in fileio.load_fits(rs_file) if _.fit_branch == fit[0]]
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
		fit_model_components.append(model.constant())
	if "e" in ref.background:
		fit_model_components.append(model.exponential())
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

	# # test model integrity
	# plt.plot(midpoints, fit_model(midpoints, *ref.popt), 'r-')
	# plt.plot(midpoints, counts, 'k.')
	# plt.show()

	# todo: base bounds on user input and/or x rescaling
	XF_PARAM_BOUNDS = [[-0.1, 0.1], [0.5, 2.0], [-np.inf, np.inf]]
	XF_PARAM_NAMES = ["a", "b", "c"]
	XF_PARAM_GUESS = [0.0, 1.0, 0.0]
	PNAMES_XF  = ["mu","xpeak"]
	PNAMES_MAG = ["a", "b", "c", "a0", "a1", "a2",  "q"]
	def make_xf_lambda(a,order):
		if order == 2:
			lam = lambda q:q[0]*(a**2) + q[1]*a + q[2]
			rfs = "({}*[0] + {}*[1] + [2])".format(a**2, a)
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

		# not free if transforming it
		if p in PNAMES_XF:
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

		# location param -> xf via polynomial
		if p in PNAMES_XF:
			lam,rfs = make_xf_lambda(ref.popt[ip], xf_order)
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
				# if p represents a magnitude parameter, apply scaling to q0 entry
				if p in PNAMES_MAG:
					q0.append(ref.popt[ip] * relative_magnitude)
				else:
					q0.append(ref.popt[ip])

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
		line_template = "{:>8} | {:<32} | {:<32.32s} | {:<32.32s}"
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

		print("meta model root function string, full")
		print(meta.rfs(False))
		print("")

	# # test metamodel integrity
	# plt.plot(midpoints, fit_model(midpoints, *ref.popt), 'r-')
	# plt.plot(midpoints, meta(midpoints, *q0), 'b-')
	# plt.plot(midpoints, counts, 'k.')
	# plt.show()

	# fit binned data with meta -> (popt, perr, chi2, ndod, cov) for parameters *q of metamodel
	qopt, qerr, chi2, ndof, cov = model.fit_hist_with_root(
		meta.fn,
		midpoints,
		counts,
		qbounds,
		q0,
		meta.rfs(False),
		True
		)

	# value and covariance of transformation parameters
	# this is all we need in order to apply the same transformation to other measured peaks
	xf_cov = cov[:xf_order+1,:xf_order+1]
	xf_par = qopt[:xf_order+1]

	# print results
	if verbosity:
		line_template = "{:>8} | {:>10} | {:>10} | {:>12} | {:>12}"
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
			print(cov)
		print("\nsquare root of diagonal of covariance matrix (should be equal to parameter errors)")
		print(np.sqrt(np.diag(cov)))

	# # test result integrity
	# plt.plot(midpoints, fit_model(midpoints, *ref.popt), 'r-', label="reference")
	# plt.plot(midpoints, meta(midpoints, *q0)           , 'b-', label="q0")
	# plt.plot(midpoints, meta(midpoints, *qopt)         , 'g-', label="qopt")
	# plt.plot(midpoints, counts, 'k.')
	# plt.legend()
	# plt.show()

	# calculate popt from meta model qopt
	popt = meta.transform_parameters(qopt)

	# todo: calculate value and errors for each location parameter based on xf_order
	#       and potentially errors on all parameters in popt
	# 
	#       not needed now, since we only use this routine to match LYSO to LYSO
	#       to get an area transformation. All we need is a,b,c and their covariance
	# 
	#       will want to implement this later if we want to compare other spectra
	#       via this method, or to measure LYSO's peaks


	# stage 3: display and output
	# draw a figure if showing or saving it
	# show and/or save the figure
	# save info to csv if specified

	# if showing or saving the figure, we need to compose it
	if show or fig_out:
		
		# display data
		if "d" in display:
			plt.step(midpoints, counts, where='mid', color=colors.get("d","k"), label="data{}".format(label_suffix))

		# display root result
		if "r" in display:
			plt.plot(midpoints, meta(midpoints,*qopt), "g-", label="fit{}".format(label_suffix))

		# # todo: can work out error on modeled counts
		# #       using full covariance and some gross calculus
		# # display root result +- errors
		# if "e" in display:
		# 	...

		# display peaks
		if "p" in display:
			first_peak = True
			for ip,p in enumerate(fit_model.pnames):
				if p in PEAK_PARAMETERS:

					# par = popt[ip]
					# stat_err = perr[ip]
					# if model_id == -1:
					# 	label_center = "{}={}\xb1{}".format(
					# 		p,
					# 		round(par,DISPLAY_PRECISION),
					# 		round(stat_err,DISPLAY_PRECISION)
					# 	)
					# else:
					# 	syst_err = par * bc_fit[6]
					# 	label_center = "{}={}\xb1{}\xb1{}".format(
					# 		p,
					# 		round(par,DISPLAY_PRECISION),
					# 		round(stat_err,DISPLAY_PRECISION),
					# 		round(syst_err,DISPLAY_PRECISION),
					# 	)

					label_center = "{}={}".format(p,round(popt[ip],DISPLAY_PRECISION))

					# display widths as mu +- sigma
					if ("w" in display) and (p=="mu"):
						label_sigma = "\xb1width" if first_peak else None
						mu_plus_sigma  = popt[ip]+popt[ip+1]
						mu_minus_sigma = popt[ip]-popt[ip+1]
						plt.axvline(mu_plus_sigma , ls='--',     color=colors.get("pe","r"), label=label_sigma)
						plt.axvline(mu_minus_sigma, ls='--',     color=colors.get("pe","r"), )
						plt.axvspan(mu_minus_sigma,mu_plus_sigma,color=colors.get("pe","r"), alpha=0.1)

					# # display statistical errors
					# if "e" in display:
					# 	label_error = "\xb1stat err" if first_peak else None
					# 	plt.axvline(popt[ip]+perr[ip], ls='--', color=colors.get("pe","r"), label=label_error)
					# 	plt.axvline(popt[ip]-perr[ip], ls='--', color=colors.get("pe","r"), )

					plt.axvline(popt[ip], ls='--', color=colors.get("p","darkred"), label=label_center)

					first_peak = False

		if ylim is not None:
			plt.ylim(top=ylim)

		plt.legend()
		plt.xlabel(fit[0])
		plt.ylabel("counts")
		plt.title(FIGURE_TITLE.format(
			run=run_id,
			branch=fit[0],
			chi2=chi2/ndof,
			rfs=fit_model.rfs(),
		))

		# apply axis scaling
		if xlog:
			plt.xscale("log")
		if ylog:
			plt.yscale("log")
		
	# save the figure if specified
	if fig_out:
		
		# just filename: save in ./figs/
		if not (os.sep in fig_out):
			fig_file = FIG_LOC.format(fig_out)
		else:
			fig_file = fig_out

		# save the figure to an image file
		plt.savefig(fig_file)

	# show the figure if requested
	if show:
		if not suspend_show:
			plt.show()

	# output transformation parameters and their covariance
	# and necessary idenfitication information (fit branch, xf_order)
	# todo: output a bit more than that

	if file_out:

		# just filename: save in ./data/xf/
		if not (os.sep in file_out):
			csv_file = XF_LOC.format(file_out)
		else:
			csv_file = file_out

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


	# # save details to a csv file if requested
	# if file_out:
		
	# 	# just filename: save in ./data/fits/
	# 	if not (os.sep in file_out):
	# 		csv_file = RESULT_LOC.format(file_out)
	# 	else:
	# 		csv_file = file_out

	# 	# todo: use fit_result class during whole script
	# 	#       then just use this_contents = result.pack()
	# 	# 
	# 	#       if changing to have model classes handle results,
	# 	#       then do contents = fit_model.pack()

	# 	# compose new entry
	# 	this_contents = [

	# 		# fit data
	# 		run_id, # numerical ID of run
	# 		fit[0], # branch fit
	# 		fit[1], # branch fit lo cut
	# 		fit[2], # branch fit hi cut

	# 		# model info
	# 		model_id, # numerical ID of model
	# 		model_cal_file if model_cal_file else "-", # calibration file used
	# 		int(raw_bounds), # whether bounds are specified on untransformed data

	# 		# fit routine input
	# 		# todo: automatically pack components with archetype,parameters,static etc
	# 		#       none of this case-by-case nonsense
	# 		#       this can be handled better by the model class itself.
	# 		nbins, # number of bins used (copies atuo bin count)
	# 		background if background else "-", # background function
	# 		len(gaus), # number of gaussians
	# 		*csv_format_gaus(gaus), # name,*bounds per gaussian, flattened
	# 		len(smono),
	# 		*csv_format_smono(smono),
			
	# 		# cut data
	# 		len(cuts), # number of cuts
	# 		*csv_format_cuts(cuts), # br,lo,hi per cut, flattened

	# 		# results
	# 		chi2,
	# 		ndof,
	# 		n_bg_parameters,
	# 		*popt,
	# 		*perr,

	# 	]
		
	# 	# update the specifed file with new entry
	# 	fileio.update_csv(
	# 		csv_file,
	# 		this_contents,
	# 		FIT_CSV_TYPELIST,
	# 		0,
	# 		backup=None,
	# 	)




if __name__ == '__main__':

	parser = argparse.ArgumentParser(
		description="fit a model to a binned branch, optionally transforming and cutting",
		)

	# version
	parser.add_argument("--version",action="version",version="%(prog)s {}".format(__version__))

	# data arguments
	parser.add_argument("run"  ,type=str,help="file location, name, or number")
	parser.add_argument("fit"  ,type=str,help="branch to fit: branch,min,max")
	parser.add_argument("--cut",type=str,action='append',help="branch to cut on: branch,min,max")
	parser.add_argument("--er" ,type=float,dest="event_range",nargs=2,help="use a subset of the dataset. --er start stop")
	parser.add_argument("--m"  ,type=str,dest="model",help="model to use: model_id,calibration_file")
	parser.add_argument("-r"   ,action='store_true',dest="raw_bounds",help="if specified, specified bounds are raw")

	# fitting arguments
	parser.add_argument("--bins",type=int,default=0,dest="nbins",help="number of bins to use")

	# get all fit info from reference spectrum
	# file may contain multiple entries. choose by:
	#     1) identical fit branch
	#     2) run number if specified, else first entry passing 1).
	parser.add_argument("--rs",type=str,default="",help="file containing reference spectrum, and optionally which run to use the spectrum of")
	parser.add_argument("--xf",type=int,default=2 ,help="order of area transformation, [0-2]. 0 = const. offset, 1 = linear, 2 = quadratic")

	# display arguments
	parser.add_argument("--d",type=str,default="drp",dest='display',help="display: any combinration of (d)ata (r)oot (p)eaks (w)idths (e)rror")
	parser.add_argument("--ylim",type=float,help="upper y limit on plot")
	parser.add_argument("-x",dest="xlog",action="store_true",help="sets x axis of figure to log scale")
	parser.add_argument("-y",dest="ylog",action="store_true",help="sets y axis of figure to log scale")
	parser.add_argument("-s",dest="show",action="store_false",help="don't show figure as pyplot window")
	parser.add_argument("--l",type=str,default="",dest='label',help="custom label")

	# output arguments
	# parser.add_argument("--out",type=str,default="",help="location to save fit results as csv file (appends if file exists)")
	parser.add_argument("--out",type=str,default="",help="location to save transformation as csv file (appends if file exists)")
	parser.add_argument("--fig",type=str,default="",help="location to save figure as png image (overwrites if file exists)")
	parser.add_argument("-v",action='count',default=0,help="verbosity")

	ARG_MULTI = "AND"

	# indicates multiple argument sets -> multiple plots, put on same plot
	if ARG_MULTI in sys.argv:

		# todo better color handling
		# todo better label/legend handling
		# todo better axis labels and title handling
		# todo better display axis bounds handling

		# list of complete argument sets
		arg_sets = []
		
		# current argument set being constructed
		this_set = []

		# iterate through argv
		for a in sys.argv[1:]:

			# delimiter
			if a == ARG_MULTI:

				# add current set to list of complete sets
				arg_sets.append(this_set)

				# reset current set
				this_set = []

			# not delimiter
			else:
				# add to current set
				this_set.append(a)

		# add last set to list of complete sets
		arg_sets.append(this_set)


		COLOR_SEQ = "kmbrcy"
		
		# call main with each set of arguments in turn,
		# additionally communicating that each call is
		# part of a multi-call execution
		for i,arg_set in enumerate(arg_sets):
			this_args = vars(parser.parse_args(arg_set))
			main(
				this_args,
				suspend_show = True,
				colors = {"d":COLOR_SEQ[i], "p":COLOR_SEQ[i]},
			)

		# show figure with all calls' data
		plt.show()

	# single call
	else:
		args = vars(parser.parse_args())
		main(args)
