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

	if run.endswith(EXT_ROOT):
		branches = fileio.load_branches(run, branches_needed)
	else:
		# print([_ for _ in np.load(run).keys()])
		branches = {key:arr for key,arr in np.load(run).items() if key in branches_needed}
		# print(branches.keys())

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
	# popt, perr, chi2, ndof = fit_model.fit(midpoints[80:], counts[80:])
	popt, perr, chi2, ndof = fit_model.fit(midpoints, counts)
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
		ipar = 0
		for ic,component in enumerate(fit_model_components):
			for ip,pname in enumerate(component.pnames):
				print(line_template.format(
					ic,
					pname,
					round(component.bounds[ip][0],DISPLAY_PRECISION),
					round(component.bounds[ip][1],DISPLAY_PRECISION),
					round(popt[ipar],DISPLAY_PRECISION),
					round(perr[ipar],DISPLAY_PRECISION),
				))
				ipar += 1


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
			plt.plot(midpoints, fit_model(midpoints,*popt), "g-", label="fit{}".format(label_suffix))

		# # display root result +- errors
		# if "e" in display:
		# 	...

		# display peaks
		if "p" in display:
			first_peak = True
			for ip,p in enumerate(fit_model.pnames):
				if p in PEAK_PARAMETERS:
					par = popt[ip]
					stat_err = perr[ip]

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
						plt.axvline(popt[ip]+perr[ip], ls='--', color=colors.get("pe","r"), label=label_error)
						plt.axvline(popt[ip]-perr[ip], ls='--', color=colors.get("pe","r"), )

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
	parser.add_argument("--l",type=str,default="",dest='label',help="custom label")

	# output arguments
	parser.add_argument("--out",type=str,default="",help="location to save fit results as csv file (appends if file exists)")
	parser.add_argument("--fig",type=str,default="",help="location to save figure as png image (overwrites if file exists)")
	parser.add_argument("-v",action='count',default=0,help="verbosity")

	ARG_MULTI = "AND"

	# indicates multiple argument sets -> multiple plots, put on same plot
	if ARG_MULTI in sys.argv:

		# todo better color handling
		# todo better label handling

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
