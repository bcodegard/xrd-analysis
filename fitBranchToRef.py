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
import itertools
import ROOT

import numpy as np
import matplotlib.pyplot as plt

import utils.fileio  as fileio
import utils.display as display
import utils.model   as model


GLOBAL_DOWNSCALE = 1e3


# filesystem locations
# todo: use os.sep to support multiple platforms
DATA_LOC   = "./data/root/scintillator/Run{}.root"
CALIB_LOC  = "./data/calibration/{}.csv"
FIG_LOC    = "./figs/{}.png"
RESULT_LOC = "./data/fits/{}.csv"

# known file extensions
EXT_ROOT = ".root"
EXT_CSV  = ".csv"
EXT_PNG  = ".png"

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
SMONO_DEFAULTS = [3,0.0,np.inf,-np.inf,np.inf]

# how many places past decimal point to show for display purposes
# does not affect precision of saved information
DISPLAY_PRECISION = 3

# list of parameters corresponding to the locations of peaks
PEAK_PARAMETERS = ["mu"]

# display string templates
FIGURE_TITLE = "run {run}, {branch}, chi2/dof={chi2:.2f}\n{rfs}"

# csv file type lists
FIT_CSV_TYPELIST = [int, str, float, float, int, str, int, int, str, int, str]




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

def csv_format_smono(smono):
	"""flattens and formats suppressed monomial bounds for CSV file"""
	smono_flat = []
	for sm in smono:
		smono_flat += sm
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
def main(args):


	# stage 0: args
	# extract and process arguments into local variables

	# run
	try:
		run_id = int(args["run"])
		run = DATA_LOC.format(run_id)
	except:
		run = args["run"]
		if not run.endswith(EXT_ROOT):
			run += EXT_ROOT
		run_id = int(run.rpartition(os.sep)[2].partition(EXT_ROOT)[0][3:])
		# todo: better extraction of run#. Will fail if filename format changes.
		# could also switch to using full filename instead of ID in calibration entries.

	# fit
	fit = split_with_defaults(args["fit"], [None,-np.inf,np.inf], [str,float,float])

	# --cut
	if args["cut"] is None:
		cuts = []
	else:
		cuts = split_with_defaults(args["cut"], [None,-np.inf,np.inf], [str,float,float])

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

	# --bg
	background = args["background"]

	# --g
	if args["gaus"] is None:
		gaus = []
	else:
		gaus = split_with_defaults(args["gaus"],GAUS_DEFAULTS,[float]*6,name_delimiter=AN_DELIMITER)

	# --s
	if args["smono"] is None:
		smono = []
	else:
		smono = split_with_defaults(args["smono"],SMONO_DEFAULTS,[float]+[float]*6)

	# display args
	display = args["display"]
	ylim = args["ylim"]
	xlog = args["xlog"]
	ylog = args["ylog"]
	show = args["show"]

	# output args
	file_out = args["out"]
	fig_out  = args["fig"]
	verbosity = args["v"]


	# --dr
	if args["dr"]:
		dr = True
		dr_lo, dr_hi = split_with_defaults(args["dr"],[0.0,1.0],[float,float])
	else:
		dr = False
		dr_lo, dr_hi = 0,1

	# xfout
	xfout = args["xfout"]

	# print raw and processed args if verbosity >= 2
	if verbosity >= 2:
		print("unprocessed args")
		for key,value in args.items():
			print(key,value)
		print("\nprocessed args")
		print("run        : {}".format(run))
		print("fit        : {}".format(fit))
		print("cuts       : {}".format(cuts))
		print("model      : {}".format([model_id, model_cal_file]))
		print("raw_bounds : {}".format(raw_bounds))
		print("nbins      : {}".format(nbins))
		print("background : {}".format(background))
		print("gaus       : {}".format(gaus))
		print("smono      : {}".format(smono))
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
	branches = fileio.load_branches(run, branches_needed)
	if verbosity:
		print("loaded branches: {}".format(branches_needed))
		print("shapes: {}".format([_.shape for _ in branches.values()]))
		print("")

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

	# apply downscale
	fit_data /= GLOBAL_DOWNSCALE

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
	# no need to assess min/max values since fit_data alreayd has cuts applied
	edges = np.linspace(fit_data.min(), fit_data.max(), nbins + 1)
	midpoints = 0.5 * (edges[1:] + edges[:-1])
	if dr:
		ir_lo = math.floor(dr_lo * (fit_data.shape[0] - 1))
		ir_hi = math.floor(dr_hi * (fit_data.shape[0] - 1))
		counts, edges = np.histogram(fit_data[ir_lo:ir_hi], edges)
	else:
		counts, edges = np.histogram(fit_data, edges)

	# list of components for fit model
	fit_model_components = []

	# add background components
	if "q" in background:
		fit_model_components.append(model.quadratic())
	elif "l" in background:
		fit_model_components.append(model.line())
	elif "c" in background:
		fit_model_components.append(model.constant())
	if "e" in background:
		fit_model_components.append(model.exponential())
	elif "E" in background:
		fit_model_components.append(model.exponential_local(static_parameters=[midpoints[0]]))

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
	for ism,sm in enumerate(smono):
		this_bounds = [[sm[1],sm[2]],[sm[3],sm[4]]]
		order = int(sm[0]) if int(sm[0]) == sm[0] else sm[0]
		fit_model_components.append(model.smono(this_bounds, [sm[0]]))

	# compose model
	if len(fit_model_components):
		fit_model = fit_model_components[0]
		for component in fit_model_components[1:]:
			fit_model = fit_model + component
	else:
		raise ValueError(ERR_NO_FIT_MODEL)

	# fit the binned data
	# popt, perr, chi2, ndof = fit_model.fit(midpoints, counts)

	hi_rs   = [np.inf,np.inf,100000.0,np.inf,np.inf,260000.0,np.inf,np.inf,375000.0,np.inf,np.inf,520000.0,np.inf,np.inf,640000.0,np.inf,np.inf,680000.0,np.inf,np.inf,np.inf]
	lo_rs   = [-np.inf,0.0,85000.0,0.0,0.0,225000.0,0.0,0.0,340000.0,0.0,0.0,440000.0,0.0,0.0,560000.0,0.0,0.0,650000.0,0.0,0.0,-np.inf]
	popt_rs = [1.681,86.803,93459.792,8536.231,49.358,243750.235,11386.371,65.788,357405.705,13452.823,22.471,474836.311,67533.321,16.68,608209.665,35331.345,20.651,666866.604,7990.784,154605.676,219.469]

	for ip,pname in enumerate(fit_model.pnames):
		if pname in ["mu", "sigma", "xpeak", ]:
			popt_rs[ip] /= GLOBAL_DOWNSCALE

	popt_rs_mu    = itertools.cycle([_ for i,_ in enumerate(popt_rs) if fit_model.pnames[i] == "mu"])
	popt_rs_xpeak = itertools.cycle([_ for i,_ in enumerate(popt_rs) if fit_model.pnames[i] == "xpeak"])
	print([_ for i,_ in enumerate(popt_rs) if fit_model.pnames[i] == "mu"])
	print([_ for i,_ in enumerate(popt_rs) if fit_model.pnames[i] == "xpeak"])

	xfo = 2
	if xfo == 2:
		TEMPLATE_XF = "[0] + [1]*{0} + [2]*{0}**2"
		xf_npars = 3
		xf_fn = lambda x,*c:c[0] + c[1]*x + c[2]*x**2
	elif xf0 == 1:
		TEMPLATE_XF = "[0] + [1]*{}"
		xf_npars = 2
		xf_fn = lambda x,*c:c[0] + c[1]*x
	TEMPLATE_GAUS  = "[{0}]*exp(-0.5*((x-({xf}))/[{1}])**2)"
	TEMPLATE_SMONO = "[{0}]*(({s[0]}*x/({xf}))**{s[0]})*exp(-{s[0]}*x/({xf}))"

	istart = xf_npars
	ipref  = 0
	scale_fit_components = []
	pguess = [0.0, 1.0, 0.0][:xf_npars]
	hi     = [ (1e5)/GLOBAL_DOWNSCALE, 2.0, 0.1][:xf_npars]
	lo     = [-(1e5)/GLOBAL_DOWNSCALE, 0.5,-0.1][:xf_npars]
	for ic,component in enumerate(fit_model_components):
		if component.arch is model.gaus:
			this_mu = next(popt_rs_mu)
			this_component = TEMPLATE_GAUS.format(
				*range(istart,istart+2),
				# xf=TEMPLATE_XF.format(this_mu,this_mu),
				xf=TEMPLATE_XF.format(this_mu),
				)
			pguess += [popt_rs[ipref], popt_rs[ipref+2]]
			hi     += [  hi_rs[ipref],   hi_rs[ipref+2]]
			lo     += [  lo_rs[ipref],   lo_rs[ipref+2]]
			istart += 2
		elif component.arch is model.smono:
			this_xpeak = next(popt_rs_xpeak)
			this_component = TEMPLATE_SMONO.format(
				istart,
				# xf=TEMPLATE_XF.format(this_xpeak,this_xpeak),
				xf=TEMPLATE_XF.format(this_xpeak),
				s=component.static_parameters,
				)
			pguess += [popt_rs[ipref+1]]
			hi     += [  hi_rs[ipref+1]]
			lo     += [  lo_rs[ipref+1]]
			istart += 1
		else:
			this_component = component.rfs(istart)
			pguess += popt_rs[ipref:ipref+component.npars]
			hi     +=   hi_rs[ipref:ipref+component.npars]
			lo     +=   lo_rs[ipref:ipref+component.npars]
			istart += component.npars
		ipref += component.npars
		print(this_component)
		scale_fit_components.append(this_component)
	print("")
	print(pguess)
	print(hi)
	print(lo)
	print("")
	scale_fit = " + ".join(scale_fit_components)

	print("\n".join(scale_fit_components))
	print("")
	print(scale_fit)
	print("")

	# root object initialization and fit
	xdata = midpoints
	ydata = counts
	rf = ROOT.TF1("multifit",scale_fit,0.0,1.0)
	ndata = xdata.shape[0]
	hist = ROOT.TH1F("hist", "data to fit", ndata, xdata[0], xdata[-1])
	for j in range(ndata):
		hist.SetBinContent(j+1, ydata[j])
	for i,_ in enumerate(pguess):
		rf.SetParameter(i,_)
	for ip in range(len(lo)): # set bounds on root parameters
		this_lo, this_hi = lo[ip], hi[ip]
		if this_lo == -np.inf:this_lo=-1e7 # 1e7 in place of inf
		if this_hi ==  np.inf:this_hi= 1e7 # 
		rf.SetParLimits(ip, this_lo, this_hi)
	hist.Fit(rf,"N")
	par = rf.GetParameters()
	err = rf.GetParErrors()
	popt_xf = [par[_] for _ in range(len(pguess))]
	perr_xf = [err[_] for _ in range(len(pguess))]
	chi2 = rf.GetChisquare()
	ndof = rf.GetNDF()

	# popt_xf = pguess
	# perr_xf = [-1]*len(pguess)
	# chi2 = 666
	# ndof = 666

	print(popt_xf)

	# translate to parameters for original model
	popt = []
	istart = xf_npars
	coeff = popt_xf[:xf_npars]
	for ic,component in enumerate(fit_model_components):
		if component.arch is model.gaus:
			this_rs_mu = next(popt_rs_mu)
			popt += [
				popt_xf[istart],
				xf_fn(this_rs_mu, *coeff),
				popt_xf[istart+1],
			]
			istart += 2
		elif component.arch is model.smono:
			this_rs_xpeak = next(popt_rs_xpeak)
			popt += [
				xf_fn(this_rs_xpeak, *coeff),
				popt_xf[istart],
			]
			istart += 1
		else:
			popt += popt_xf[istart:istart+component.npars]
			istart += component.npars

	# save xf deets
	if xfout:
		file = "./data/xf/{}.csv".format(xfout)
		this_entry = [
			run,
			*fit,
			dr_lo,
			dr_hi,
			xf_npars,
			*popt_xf[:xf_npars],
			*perr_xf[:xf_npars],
		]
		fileio.update_csv(file, this_entry, [str,str,float,float,float,float,int,float], 0)
	
	# display fit results
	if verbosity:
		line_template = "{}| {:>6}| {:>8}| {:>8}| {:>10}| {:>10}"
		print("")
		print("model performance: chi2/ndof = {}/{} = {}".format(
			round(chi2,DISPLAY_PRECISION),
			ndof,
			round(chi2/ndof,DISPLAY_PRECISION),
			))
		print("model parameters")
		print(line_template.format("f","par","lo bnd","hi bnd","popt","perr"))
		for ip,p in enumerate(popt_xf):
			print(line_template.format(
				0,
				ip,
				round(lo[ip],DISPLAY_PRECISION),
				round(hi[ip],DISPLAY_PRECISION),
				round(popt_xf[ip],DISPLAY_PRECISION),
				round(perr_xf[ip],DISPLAY_PRECISION),
			))


	# stage 3: display and output
	# draw a figure if showing or saving it
	# show and/or save the figure
	# save info to csv if specified

	# if showing or saving the figure, we need to compose it
	if show or fig_out:

		if xlog and ylog:
			plotter = plt.loglog
		elif xlog:
			plotter = plt.semilogx
		elif ylog:
			plotter = plt.semilogy
		else:
			plotter = plt.plot
		
		# display data
		if "d" in display:
			plotter(midpoints, counts, "k.", label="data")

		# display root result
		if "r" in display:
			plotter(midpoints, fit_model(midpoints,*popt), "g-", label="fit")

		# # display root result +- errors
		# if "e" in display:
		# 	...

		# display peaks
		if "p" in display:
			first_peak = True
			for ip,p in enumerate(fit_model.pnames):
				if p in PEAK_PARAMETERS:
					par = popt[ip]
					stat_err = -1.0 # perr[ip]

					if model_id == -1:
						label_center = "{}={}\xb1{}".format(
							p,
							round(par,DISPLAY_PRECISION),
							round(stat_err,DISPLAY_PRECISION)
						)
					else:
						syst_err = par * bc_fit[6]
						label_center = "{}={}\xb1{}\xb1{}".format(
							p,
							round(par,DISPLAY_PRECISION),
							round(stat_err,DISPLAY_PRECISION),
							round(syst_err,DISPLAY_PRECISION),
						)

					# display statistical errors
					if "e" in display:
						label_error = "\xb1stat err" if first_peak else None
						plt.axvline(popt[ip]+perr[ip], ls='--', color='r', label=label_error)
						plt.axvline(popt[ip]-perr[ip], ls='--', color='r', )

					plt.axvline(popt[ip], ls='--', color='darkred', label=label_center)

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
		plt.show()

	# save details to a csv file if requested
	if file_out:
		
		# just filename: save in ./data/fits/
		if not (os.sep in file_out):
			csv_file = RESULT_LOC.format(file_out)
		else:
			csv_file = file_out

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
			# todo- automatically pack components with archetype,parameters,static etc
			#       none of this case-by-case nonsense
			nbins, # number of bins used (copies atuo bin count)
			background if background else "-", # background function
			len(gaus), # number of gaussians
			*csv_format_gaus(gaus), # name,*bounds per gaussian, flattened
			len(smono),
			*csv_format_smono(smono),
			
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
	parser.add_argument("--m"  ,type=str,dest="model",help="model to use: model_id,calibration_file")
	parser.add_argument("-r"   ,action='store_true',dest="raw_bounds",help="if specified, specified bounds are raw")

	# fitting arguments
	parser.add_argument("--bins",type=int,default=0,dest="nbins",help="number of bins to use")
	parser.add_argument("--bg"  ,type=str,default="",dest="background",help="background function: any combination of (p)ower (e)xp (c)onstant (l)ine (q)uadratic")
	parser.add_argument("--g"   ,type=str,action='append',dest="gaus" ,help="gaussian components: min_mu,max_mu (or) name=min_mu,max_mu")
	parser.add_argument("--s"   ,type=str,action='append',dest='smono',help="suppressed monomial: order, min,max xpeak, min,max c, min,max k")

	# display arguments
	parser.add_argument("--d",type=str,default="drp",dest='display',help="display: any combinration of (d)ata (r)oot (p)eaks (e)rror")
	parser.add_argument("--ylim",type=float,help="upper y limit on plot")
	parser.add_argument("-x",dest="xlog",action="store_true",help="sets x axis of figure to log scale")
	parser.add_argument("-y",dest="ylog",action="store_true",help="sets y axis of figure to log scale")
	parser.add_argument("-s",dest="show",action="store_false",help="don't show figure as pyplot window")

	# output arguments
	parser.add_argument("--out",type=str,default="",help="location to save fit results as csv file (appends if file exists)")
	parser.add_argument("--fig",type=str,default="",help="location to save figure as png image (overwrites if file exists)")
	parser.add_argument("-v",action='count',default=0,help="verbosity")

	# reference spectrum related arguments
	parser.add_argument("--dr",type=str,default="")
	parser.add_argument("--xfout",type=str,default="")

	args = vars(parser.parse_args())
	main(args)
