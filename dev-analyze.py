"""
load a branch from a root file, cut, transform, and fit
optionally output fit results as csv and/or figure
"""

__author__ = "Brunel Odegard"
__version__ = "0.1"


import os
import re
import sys
import math
import argparse
import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import utils.cli     as cli
import utils.data    as data
import utils.model   as model
import utils.fileio  as fileio
import utils.display as display
import utils.expression as expr




# list of branches which are constructed by the manager after loading
BRANCHES_CONSTRUCT = ['entry']

# todo: put this in a config file instead of code
ROOT_FILE = "../xrd-analysis/data/scint-experiment/root/Run{}.root"
FIG_FILE = "./figs/{}"
ARG_MULTI = ["AND","and"]

# shorthand for common branches, EG a1 -> area_xxxx_1
SHORTHAND = {
	"A{ch}":"(area_{board}_{ch}/1000)",
	"a{ch}":"area_{board}_{ch}",
	"w{ch}":"width_{board}_{ch}",
	"o{ch}":"offset_{board}_{ch}",
	"n{ch}":"noise_{board}_{ch}",
	"v{ch}":"vMax_{board}_{ch}",
	"t{ch}":"tMax_{board}_{ch}",
	"s{ch}":"scaler_{board}_{ch}",

	"T{ch}":"timestamp_{board}_{ch}",
	"T"    :"timestamp_{board}_1",
}

def replace_names(string, replacements):
	"""make replacements in string only replacing substrings which
	match a key in replacements surrounded by non-name characters,
	IE not alphanumeric or underscores. This ensures that only whole
	variable names are replaced, and not parts of variable names.
	"""
	if replacements:
		for pre, post in replacements.items():
			string = re.sub(
				r"(^|\W){pre}(?=$|\W)".format(pre=pre),
				"\\g<1>{post}".format(post=post),
				string,
			)	
	return string

iseq = lambda x,y,eps=1e-9:abs(x-y)<eps




class performer(object):
	"""Performs a single piece of specified analysis,
	within the supplied context of a routine."""

	def __init__(self, spec):
		self.spec = spec

	def perform(self, rtn):
		"""Perform the specified analysis in the context
		provided by the routine instance 'rtn' """
		...


class routine(object):
	"""Handles the performance of one or more pieces of analysis,
	managing continuity between them."""

	def __init__(self, n_specs):
		
		# The total number of specifications that will be requested.
		# This information is needed by performers, so that they can
		# know whether or not they are the last performer, etc.
		self.n_specs = n_specs

		# setup persistent objects, for continuity between
		# different performers
		self.setup()

	def setup(self):
		"""Create persistent objects for access by performers"""

		# list of completed performer instances
		self.performers = []

		# initialize sequential generators (for color choices, etc.)
		...

	def fulfill(self, spec):
		"""fulfill the supplied specifications by 
		performing the analysis specified by 'spec'"""

		# create performer instance with the supplied specifications
		perf = performer(spec)

		# perform the analysis, supplying self for context
		perf.perform(self)

		# after performer is finished, append it to the list
		# of completed performers
		self.performers.append(perf)









def procure_data(args):
	"""Load branches from specified file as needed to calculate
	all fit and cut expressions. Then apply cuts and binning, and
	return only the processed fit data."""


	# look up list of all branches in the specified root file
	# determine four-digit number of DRS board used
	# apply shorthand (a1 -> area_xxxx_1, etc.)
	root_file = args.run if os.sep in args.run else ROOT_FILE.format(args.run)
	branches_all = fileio.get_keys(root_file)
	# find all channels present by matching noise_*
	channels = {_.rpartition("_")[2] for _ in branches_all if _.startswith("noise_")}
	# any matching channels: fill replacements using templates in SHORTHAND
	if channels:
		test_branch = next(_ for _ in branches_all if _.startswith("noise_"))
		board = test_branch.partition('_')[2].partition('_')[0]

		replacements = {}
		for pre,post in SHORTHAND.items():
			if "{ch}" in pre:
				replacements.update(
					{pre.format(ch=_):post.format(board=board,ch=_) for _ in channels}
				)
			else:
				replacements.update(
					{pre:post.format(board=board,ch=next(_ for _ in channels))}
				)
	# no matching branches found: don't apply shorthand
	else:
		replacements = {}


	# set of branches needed to evaluate all fits, cuts, defs, and xfs
	branches_needed = set()

	# compile expressions and update branches_needed
	fn_fits = []
	for fit in args.fits:
		# skip fits where the first entry is None. This happens when
		# the positional argument is not specified, so handling this
		# case lets us supply all fits via --fit if desired.
		if fit[0] is None:
			continue
		fn = expr.check_and_compile(replace_names(fit[0], replacements))
		fn_fits.append(fn)
		branches_needed |= fn.kwargnames

	# copy at this point to capture branches needed for fit expressions
	branches_needed_fit = branches_needed.copy()
	
	fn_cuts = []
	for cut in args.cuts:
		fn = expr.check_and_compile(replace_names(cut[0], replacements))
		fn_cuts.append(fn)
		branches_needed |= fn.kwargnames

	# copy branches_needed at this point to capture which are needed
	# explicitly for fits and cuts
	branches_fit_and_cut = branches_needed.copy()
	
	fn_defs = []
	for def_ in args.defs:
		fn = expr.check_and_compile(replace_names(def_[1], replacements))
		fn_defs.append(fn)
		branches_needed |= fn.kwargnames
	
	fn_xfs = []
	for xf in args.xfs:
		raise Exception("xfs not implemented yet")
		fn = expr.check_and_compile(replace_names(xf[1], replacements))
		fn_xfs.append(fn)
		branches_needed |= fn.kwargnames


	# load branches from specified root file, allowing for missing
	# branches. missing branches must be generated by one of the
	# defs or xfs included.
	branches = fileio.load_branches(root_file, branches_needed - set(BRANCHES_CONSTRUCT))

	# initialize the branch manager instance with the resulting branches
	bm = data.BranchManager(branches, export_copies=False, import_copies=False)

	# construct branches if needed
	if "entry" in branches_needed:
		bm.bud(data.bud_entry)

	# apply scaler rectification
	if args.rectify_scalers:
		if any(_.startswith("scaler_") for _ in bm.keys):
			bm.bud(data.rectify_scaler(), overwrite=True)

	# apply timestamp fix and localization
	if any(_.startswith("timestamp_") for _ in bm.keys):
		bm.bud(data.fix_monotonic_timestamp(), overwrite=True)
		if args.localize_timestamps:
			bm.bud(data.localize_timestamp(), overwrite=True)


	# process defs and xfs to create new branches
	# todo: current implementation is slightly inefficient. defs and xfs
	# are evaluated before applying any cuts, resulting in excess
	# computation in the case where cuts do not depend on defs or xfs.
	# an implementation which applies each cut as soon as it is able to,
	# and prioritizes defs and xfs which enable cuts, would be faster.
	fn_defs_remain = [True for _ in fn_defs]
	fn_xfs_remain  = [True for _ in fn_xfs]
	n_remaining = len(fn_defs) + len(fn_xfs)
	while n_remaining:

		for i,remain in enumerate(fn_defs_remain):
			if remain and fn_defs[i].kwargnames.issubset(bm.keys):

				this_name = args.defs[i][0]
				this_fn = fn_defs[i]

				bm.bud(
					lambda man:{this_name:this_fn(**{_:man[_] for _ in this_fn.kwargnames})}
				)

				fn_defs_remain[i] = False

		# # xfs not implemented yet
		# for i,remain in enumerate(fn_xfs_remain):
		# 	if remain and fn_xfs[i].kwargnames.issubset(bm.keys):
		# 		bm.bud()
		# 		fn_xfs_remain[i] = False

		# if we have all branches needed for fits and cuts, there's
		# no need to keep evaluating defs and xfs
		if branches_fit_and_cut.issubset(bm.keys):
			break

		# check to see if progress has been made
		# if not, then it never will, and we have to exit
		n_remaining_now = sum(fn_defs_remain) + sum(fn_xfs_remain)
		if n_remaining_now == n_remaining:
			print("could not evaluate all definititions and transformations")
			print("missing one or more variables for completion")
			sys.exit(1)
		n_remaining = n_remaining_now


	# wrapper functions to capture loop variable values
	# if we don't use these, the overwritten value of fn and other
	# variables used in the loop will change, and the change will affect
	# the function calls to calculate masks
	def mask_bool(fn):
		mask = lambda man:fn(**{_:man[_] for _ in fn.kwargnames})
		return mask
	def mask_range(fn,lo,hi):
		mask = lambda man:data.inrange(fn(**{_:man[_] for _ in fn.kwargnames}),lo,hi)
		return mask
	
	# process cuts
	masks = []
	for icut,fn in enumerate(fn_cuts):
		this_cut = args.cuts[icut]

		# no bounds specified: boolean expression
		if (this_cut[1] is None) and (this_cut[2] is None):
			masks.append(mask_bool(fn))

		# at least one bound specified: lo<expression<hi
		else:
			masks.append(mask_range(fn,this_cut[1],this_cut[2]))

	# apply cuts
	if masks:
		data_fit_raw = bm.mask(
			data.mask_all(*masks),
			branches_needed_fit,
			apply_mask = False,
		)
	else:
		data_fit_raw = {_:bm[_] for _ in branches_needed_fit}


	# data_fit_raw are all the branches that show up in the expression
	# for at least one fit. to get the fit data, we have still have to
	# evaluate the expressions.
	fit_data = []
	for fn in fn_fits:
		fit_data.append(fn(**{_:data_fit_raw[_] for _ in fn.kwargnames}))

	# get counts and edges by binning data_fit_raw
	fit_counts = []
	fit_edges = []
	for i,fit in enumerate(args.fits):
		this_data = fit_data[i]

		# determine bin edges
		lo = this_data.min() if fit[1] in [None,-np.inf] else fit[1]
		hi = this_data.max() if fit[2] in [None, np.inf] else fit[2]
		if fit[3]:
			nbins = fit[3]
		else:
			this_ndata = data.inrange(this_data,lo,hi,True,True).sum()
			nbins = data.bin_count_from_ndata(this_ndata)
		if fit[4].startswith("li"):
			this_edges = data.edges_lin(lo,hi,nbins)
		elif fit[4].startswith("lo"):
			if lo<=0:
				lo = this_data[this_data>0].min()
			this_edges = data.edges_log(lo,hi,nbins)
		elif fit[4].startswith("s"):
			this_edges = data.edges_symlog(lo,hi,nbins)
		
		# calculate histogram counts and append
		this_counts, _ = np.histogram(this_data, this_edges)
		# scale by constant multiplier (1.0 by default)
		this_counts = this_counts * fit[6]
		fit_counts.append(this_counts)
		fit_edges.append(this_edges)

	return fit_counts, fit_edges


def model_counts(args, fit_counts, fit_edges):
	return None


def display_and_write(args, fit_counts, fit_edges, model_results_placeholder):
	return None


def main(args,iset=None,nsets=None):
	""""""

	# todo: verbosity argument that controls what information is printed
	#       better printing for args; detail based on verbosity
	# 
	# todo: class that holds all kept information which is not stored
	#       explicitly in args, and is passed to each analysis stage to
	#       be accessed and updated. This would allow more convenient
	#       handling and access for the information, as well as allow
	#       functionality such as packing data to a CSV format to be
	#       defined within the class.

	if args.verbosity >= 1:
		
		# placeholder diagnostic info: just print whole set of args
		klmax = max([len(_) for _ in vars(args).keys()])
		print("args")
		for i,kv in enumerate(vars(args).items()):
			print("{} : {}".format(str(kv[0]).ljust(klmax), kv[1]))
		print("")

	# load, process, cut, and bin data into form ready for fitting
	fit_counts, fit_edges = procure_data(args)

	# construct models and fit them to data
	model_results_placeholder = model_counts(args, fit_counts, fit_edges)

	# display and/or save information
	display_and_write(args, fit_counts, fit_edges, model_results_placeholder)


	# testing fit data results: just show plots with counts
	for i,fit in enumerate(args.fits):
		print("fit {:>12} - total counts {:>7}".format(fit[0], fit_counts[i].sum()))
		plt.step(
			(fit_edges[i][1:]+fit_edges[i][:-1])*0.5,
			fit_counts[i],
			where='mid',
			label=fit[5] if fit[5] else fit[0],
		)

	# hlines and vlines
	for loc,col,ls,label in args.vlines:
		plt.axvline(loc, color=col, ls=ls, label=label)
	for loc,col,ls,label in args.hlines:
		plt.axhline(loc, color=col, ls=ls, label=label)
	for lo,hi,fc,alph,ec,lw,label in args.vspans:
		facecolor = colors.to_rgb(fc)+(alph,)
		edgecolor = ec if ec is None else colors.to_rgb(ec)+(1,)
		plt.axvspan(
			lo,hi,
			facecolor=facecolor,
			edgecolor=edgecolor,
			linewidth=lw,
			label=label,
		)
	for lo,hi,fc,alph,ec,lw,label in args.hspans:
		facecolor = colors.to_rgb(fc)+(alph,)
		edgecolor = ec if ec is None else colors.to_rgb(ec)+(1,)
		plt.axhspan(
			lo,hi,
			facecolor=facecolor,
			edgecolor=edgecolor,
			linewidth=lw,
			label=label,
		)


	# decoration
	plt.legend()

	if args.ylog:
		plt.yscale("log")

	if args.title:
		plt.title(args.title)

	if args.xlabel:
		plt.xlabel(args.xlabel)
	if args.ylabel:
		# plt.ylabel("Number of Events")
		plt.ylabel(args.ylabel)

	# scaling
	if fit[4].startswith("lo"):
		plt.xscale("log")
	elif fit[4].startswith("s"):
		...
	
	# if (iset is None) or (iset == nsets-1):
	# 	plt.show()

	return fit_counts, fit_edges




if __name__ == '__main__':


	# how to handle CLI arguments and config file contents
	# coherently, simply, and robustly?
	# 
	# 1) compose 'args' from CLI arguments via argparse
	# 2) load 'cfg' from config file(s) via config module
	# 3) compose 'specificatoin' from 'args' and 'cfg' via a function of the routine class
	#        rtn.compose_specification(args, cfg)
	# 
	# the 'compose_specification' function would handle renaming, processing, merging, 
	# and resolving information that isn't handled by argparse actions.
	# 
	# this includes
	#     composing file paths from config file contents
	#     determining specification values based on priorities from different sources (values in one or more config file, values from one or more CLI arguments, ...)
	#     anything else needed
	# 
	# 
	# 
	# what this would look like, ignoring the 'and' argument
	# 
	# >>> args = parser.parse_args(sys.argv)
	# >>> cfg  = config.load_config(['analyze', 'common'])
	# >>> spec = compose_specs(args, cfg)
	# >>> 
	# >>> rtn = routine(spec)
	# >>> rtn.perform()
	# 
	# 
	# 
	# the 'and' argument could be handled by splitting and making lists
	# of 'args' and 'spec' for each, then instantiating and running routine
	# objects for each entry in spec.
	# 'spec' would additionally contain info on how many sets there are,
	# and which one this is.
	# 
	# Alternatively, the same routine instance could be asked to process each set
	# one after the other. This would have the advantage of natively handling the
	# how-many-and-which problem, as the routine could keep track on its own. This
	# would also allow the routine to keep its own generators for colors, etc.
	# This is the way to go.
	# 
	# >>> ...
	# >>> 
	# >>> specs = [compose_specs(_,cfg) for _ in args]
	# >>> rtn = routine()
	# >>> for spec in specs:
	# >>>     rtn.perform(spec)
	# >>> 
	# 
	# The routine object would set up generators and structures on init.
	# Then, separate instances of a 'performer' class would be created and
	# performed each time rtn.perform(spec) is called. These instances would
	# be given access to the routine's persistent objects, and the results
	# of the instances would be kept in memory. This would allow continuity
	# between separate specifications, and for performers to access the
	# results of previous performers. This would in turn permit comparisons
	# to be made between the spectra or fit results of different performers.
	# 
	# 


	parser = argparse.ArgumentParser(
		description="fit a model to a binned branch, optionally transforming and cutting",
	)

	# version
	parser.add_argument("--version",action="version",version="%(prog)s {}".format(__version__))

	# verbosity
	parser.add_argument("-v",action='count',dest="verbosity",default=0,help="verbosity")

	# data arguments
	parser.add_argument("run",type=str,help="file location, name, or number")

	# allow fit data specification to be supplied by positional argument
	# representing the first dataset to be fit, and further datasets
	# to be specified using --fit 
	callables_fit = (str, float, float, int, str, str, float)
	defaults_fit = (None, -np.inf, np.inf, 0, "lin", "", 1.0)
	parser.add_argument(
		"fits",
		type=str,
		nargs="*",
		action=cli.MergeAppendAction,
		const=(callables_fit, defaults_fit),
		default=[],
		help="expression min=-inf max=inf nbins=auto binning=lin(,log,symlog) scale=1.0",
	)
	parser.add_argument(
		"--fit","--f",
		type=str,
		nargs="+",
		dest="fits",
		action=cli.MergeAppendAction,
		const=(callables_fit, defaults_fit),
		default=[],
		help="same as fit, used to specify multiple while allowing positional specification for first",
	)
	
	# cuts
	parser.add_argument(
		"--cut","--c",
		type=str,
		nargs="+",
		dest="cuts",
		action=cli.MergeAppendAction,
		const=((str,float,float),(None,None,None)),
		default=[],
		help="cut on expression. logical_all applied if multiple cuts.",
	)

	# todo: implement this
	# 
	# first N arguments are expressions (dynamically determine based on syntax)
	# remaining arguments are vertices in the space of the specified expressions
	# points in the dataset pass the cut if they are inside the convex hull of
	# the set of supplied vertices.
	# 
	# maybe add option specifying whether to also draw region?
	# specify fill color and alpha, edge color and alpha
	# mostly useful when drawing multiple datasets to the same canvas.
	parser.add_argument(
		"--pulycut","--pc",
		type=str,
		nargs="+",
		dest="polycuts",
		action="append",
		default=[],
		help="(NYI) cut on points being in the convex hull of the supplied vertices",
	)
	
	# expression definitions
	parser.add_argument(
		"--def", "--d",
		type=str,
		nargs=2,
		dest="defs",
		action=cli.MergeAppendAction,
		const=((str,str),(None,None)),
		default=[],
		help="""define a new branch using an expression.
		Usage --def new_branch_name expression""",
	)

	# models
	# todo: implement this
	parser.add_argument(
		"--xf",
		type=str,
		nargs="+",
		dest="xfs",
		action=cli.MergeAppendAction,
		const=((str,str,str,int),(None,None,None,None)),
		default=[],
		help="""define a new branch using a transformation stored in a file.
		Usage --xf new_branch_name expression file model_id=any""",
	)

	# splits
	# todo: implement this
	parser.add_argument(
		"--split","--s",
		type=str,
		nargs="+",
		dest="split",
		action=cli.MergeAction,
		const=((str,float,float,float),("",None,None,1)),
		default=False,
		help="split on expression. --split expression start=min stop=max step=1",
	)

	# scaler rectification
	parser.add_argument(
		"--no-rectify-scalers","--nrs",
		dest="rectify_scalers",
		action="store_const",
		const=0,
		default=12,
		help="don't rectify scalers",
	)
	parser.add_argument(
		"--rectify-scalers","--rs",
		dest="rectify_scalers",
		type=int,
		default=12,
		help="rectify scalers. Usage --rs kernel_size=12",
	)

	# timestamp localization
	parser.add_argument(
		"--no-timestamp-local","--ntl",
		dest="localize_timestamps",
		action="store_false",
		help="don't localize timestamps",
	)


	# fitting arguments

	# arbitrary functions, specifing function name as first argument
	# and parameter specifications as subsequent arguments.
	# 
	# Since there is one specification per parameter, and different
	# functions have different numbers of parameters, we can't know 
	# how many defaults to add.
	# Instead, we'll have to pad the arguments with additional defaults
	# when parsing the entries.
	component_callables = (str, )
	component_defaults  = ("", )
	parser.add_argument(
		"--add-component", "--add", "--a",
		dest="components",
		type=str,
		nargs="+",
		action=cli.MergeAppendAction,
		const=(component_callables, component_defaults),
		help="""add a model component by name.
		--add fn_name par_1 par_2 ...""",
	)

	# specific common functions such as gausians
	# 
	# we do know how many parameters each one of these takes
	# but since the destination and format are identical to --add,
	# and we'll need to handle the parameter count to support --add,
	# there's no reason to include that information here. We get it
	# for free.
	# 
	# we do have to a bit of lambda-waving to get the resulting format
	# to agree with that of --add
	def merge_and_concat(left,callables,defaults):
		"""return a function which performs a merge using callables
		and defaults, then concatenates the result with left
		"""
		if type(left) in (tuple,set):
			left=list(left)
		elif type(left) is not list:
			left = [left]
		return lambda values:left+cli.merge(values,callables,defaults)
	parser.add_argument(
		"--gaus","--g",
		dest="components",
		type=str,
		nargs="*",
		action=cli.FunctionAppendAction,
		const=merge_and_concat("gaus",component_callables,component_defaults),
		help="""add a gaussian to the fit model. --g par_1 ... 
		Equivalent to --add gaus par_1 ...""",
	)

	# free parameter creation
	parser.add_argument(
		"--parameter", "--par", "--p",
		dest="free_parameters",
		type=str,
		nargs="+",
		action=cli.MergeAppendAction,
		const=((str, float), ("", -np.inf, np.inf, 0)),
		default=[],
		help="""add free parameters for use in other parameter expressions.
		--par name lo=-inf hi=inf guess=0""",
	)

	# reference spectrum
	# 
	# todo: how best to implement this feature, concerning:
	#       how should parameters be controlled?
	#       how should the user specify the method and its options?
	#       should the reference spectrum's parameters be accessible
	#       to other components of the model, and how?
	# 
	# given that the future approach for modeling spectra for XRD will
	# be more deliberately constructed, perhaps it's not worth
	# implementing this feature.


	# other analytic features
	parser.add_argument("--check-consistent","--cc",action="store_true",help="assess whether spectra are consistent with each other")


	# display arguments
	# 
	# todo: add format code support for title (E.G. {chi2} {dof} etc. get replaced by results of fit routine)
	# 
	parser.add_argument('-y',dest="ylog",action="store_true",help="y axis log scale")
	parser.add_argument("--title","--t",dest="title",type=str,default="",help="figure title")
	parser.add_argument("--xlabel","--xl",dest="xlabel",type=str,default="",help="x axis label")
	parser.add_argument("--ylabel","--yl",dest="ylabel",type=str,default="number of events",help="y axis label")
	parser.add_argument("--no-show","--ns",dest="show",action="store_false",help="don't show the figure")
	parser.add_argument(
		"--vline","--vl",
		dest="vlines",
		type=str,
		nargs="+",
		action=cli.MergeAppendAction,
		const=((float, str),(0.0, "k", "-", None)),
		default=[],
		help="add vlines. --vl location color=k linestyle=solid label=None",
	)
	parser.add_argument(
		"--hline","--hl",
		dest="hlines",
		type=str,
		nargs="+",
		action=cli.MergeAppendAction,
		const=((float, str),(0.0, "k", "-", None)),
		default=[],
		help="add hlines. --hl location color=k linestyle=solid label=None",
	)
	parser.add_argument(
		"--vspan","--vs",
		dest="vspans",
		type=str,
		nargs="+",
		action=cli.MergeAppendAction,
		const=((float, float, str, float, str, int, str),(0.0, 1.0, "k", 0.1, None, 1, None)),
		default=[],
		help="add vspans. --vl lo=0 hi=1 color=k alpha=0.1 edgecolor=None linewidth=1 label=None",
	)
	parser.add_argument(
		"--hspan","--hs",
		dest="hspans",
		type=str,
		nargs="+",
		action=cli.MergeAppendAction,
		const=((float, float, str, float, str, int, str),(0.0, 1.0, "k", 0.1, None, 1, None)),
		default=[],
		help="add hspans. --hl lo=0 hi=1 color=k alpha=0.1 edgecolor=None linewidth=1 label=None",
	)

	# todo: implement this (would be nice)
	#       specify equation using expressions, eg. 
	#           "v1/a1 == 10.3"
	#           "(v1-v2)**2 > 100"
	#       also specify edge and fill colors (incl. alphas), labels, etc.
	# 
	#       on plots, draw the solution to the equation.
	#       if it's an equality, draw the line/surface/contour
	#       if it's an inequality, shade the interior and mark the boundary
	# 
	# parser.add_argument("--draw", ...)


	# file output arguments

	parser.add_argument(
		"--save-fig","--svf","--sf",
		dest="save_fig",
		type=str,
		nargs="+",
		action=cli.MergeAction,
		const = ((str, int, str), ("", 100, "png")),
		default=False,
		help="save figure to filename"
	)

	# todo: implement this
	# 
	# first argument is file to save data to. should make config entry for
	# data folder and file {base}/cuts/{}.npz
	# 
	# the boolean array for a given cut is always saved.
	# 
	# second argument is whether to save the array containing the indices of all the
	# True values in the boolean array, for each cut included.
	#
	# the combined cut is always saved.
	# 
	# third argument is whether to save all the individual cuts in addition to the
	# combined cut.
	parser.add_argument(
		"--save-cuts", "--sc",
		dest="save_cuts",
		type=str,
		nargs="+",
		action=cli.MergeAction,
		const=((str,cli.as_bool,cli.as_bool), ("",True,False)),
		default="",
		help="save applied cuts, and arrays of which events pass them, to a file",
	)



	# TODO: create routine class, like used in dev-gain.py
	#       accumulate class instances while executing argument sets
	#       pass list of routines to display-and-save function
	#       routine class has functions which draw to / modify a figure

	# accumulate fit results for making comparisons
	check_counts = []
	check_edges  = []

	# dict of format strings to replace in figure title
	title_keys = {}
	
	# at least one call delimeter argument is present
	if any(_ in sys.argv for _ in ARG_MULTI):
		arg_sets = []
		current_set = []
		for arg in sys.argv[1:]:
			if arg in ARG_MULTI:
				arg_sets.append(current_set)
				current_set = []
			else:
				current_set.append(arg)
		arg_sets.append(current_set)
		nsets = len(arg_sets)

		for i,arg_set in enumerate(arg_sets):
			this_args = parser.parse_args(arg_set)
			this_args.fits = [_ for _ in this_args.fits if _[0] is not None]
			this_counts, this_edges = main(this_args, i, nsets)
			if this_args.check_consistent:
				check_counts += this_counts
				check_edges  += this_edges

		# copy last arg set as reference for plotting related args
		# todo: better handling of display args across sets
		args = this_args

	# no call delimeter argument is present
	else:
		args = parser.parse_args()
		# remove fit args with empty expression
		args.fits = [_ for _ in args.fits if _[0] is not None]
		counts, edges = main(args)
		if args.check_consistent:
			check_counts += counts
			check_edges  += edges

	# if requested by any sub-call, check whether each pair of spectra is consistent
	if check_counts:
		for ia,ib in itertools.combinations(range(len(check_counts)), 2):

			ca = check_counts[ia]
			cb = check_counts[ib]
			print("\ncomparison {},{}".format(ia,ib))

			# skip if different shapes
			if ca.shape != cb.shape:
				print("skip: different shapes {},{}".format(ca.shape,cb.shape))
				continue

			# skip if bins are different
			edge_diff_sum = (check_edges[ia]-check_edges[ib]).sum()
			if not iseq(0, edge_diff_sum):
				print("skip: different bin edges, diff sum = {}".format(edge_diff_sum))

			# check consistency
			chi2, ndof = data.chi2_identical_poisson(ca, cb)
			print("chi2 / ndof = {:.4f}/{} = {:.3f}".format(
				chi2,
				ndof,
				chi2/ndof,
			))

			title_keys['chi2'] = title_keys['chisq'] = chi2
			title_keys['dof']  = title_keys['ndof']  = ndof
			title_keys['c2d']  = chi2/ndof


	class ddict(dict):
		"""dict sublass which returns "{key}" for missing key"""
		def __missing__(self,key):
			return '{'+key+'}'
	fig = plt.gcf()
	title = fig.axes[0].get_title()
	plt.title(title.format_map(ddict(title_keys)))

	if args.save_fig:
		fname, dpi, fmt = args.save_fig
		if fname:
			if '.' not in fname:
				fname = '{}.{}'.format(fname, fmt)
			plt.savefig(FIG_FILE.format(fname), dpi=dpi, format=fmt)

	if args.show:
		plt.show()







