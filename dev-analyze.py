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

import xrd.core.cli        as cli
import xrd.core.fileio     as fileio
import xrd.core.config     as config
import xrd.core.expression as expr

import xrd.process.data    as data
import xrd.process.model   as model

import xrd.ui.display as display





# todo: put this in a config file instead of code
ROOT_FILE = "../xrd-analysis/data/scint-experiment/root/Run{}.root"
# ROOT_FILE = "/home/bode/Documents/drsProcessing/processed_wvf/Run{}.root"
FIG_FILE = "./figs/{}"
ARG_MULTI = ["AND","and"]


# if file argument is interpretable as an integer, this template will be used.
DFILE_NUMERIC = "../xrd-analysis/data/scint-experiment/root/Run{}.root"

# default location(s) in which to look for files
DIR_DATA_DEFAULT = "../xrd-analysis/data/scint-experiment/root"
DIR_DATA = {
	"root":"../xrd-analysis/data/scint-experiment/root",
	# "npz" :"/home/bode/Documents/python/xrd-scope-pulses/runs/npz-low-thresh",
	"npz" :"/home/bode/Documents/python/xrd-scope-pulses/dirpi/npz",
}




iseq = lambda x,y,eps=1e-9:abs(x-y)<eps




class ObjDict(dict):
	"""Provides recursive attribute-style access to dictionary items"""

	@classmethod
	def convert(cls, obj):
		if isinstance(obj, cls):
			return obj
		if isinstance(obj, dict):
			return cls({k:cls.convert(v) for k,v in obj.items()})
		if isinstance(obj, list):
			return [cls.convert(_) for _ in obj]
		if isinstance(obj, tuple):
			return tuple(cls.convert(_) for _ in obj)
		if isinstance(obj, set):
			return {cls.convert(_) for _ in obj}
		return obj

	@classmethod
	def _convert_dict(cls, obj):
		return {k:cls.convert(v) for k,v in obj.items()}

	def __init__(self, d):
		super(ObjDict, self).__init__(ObjDict._convert_dict(d))

	def __getattr__(self, attr):
		return self[attr]

	def __setattr__(self, attr, value):
		self[attr] = ObjDict.convert(value)

	def __repr__(self):
		return 'ObjDict({})'.format(super(ObjDict, self).__repr__())




def compose_parser():

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

	parser.add_argument(
		"--scatterplot", "--sp",
		dest="scatterplots",
		type=str,
		nargs="+",
		action=cli.MergeAppendAction,
		const=((str,str,str,str,str,str), ("","",".","",None,None)),
		default=[],
		help="--plot var1 var2 mkr=. ls= color=auto label=None"
	)
	parser.add_argument(
		"--mapplots","--mp","--heatmap","--hm",
		dest="mapplots",
		action="store_true",
		default=False,
		help="if specified, make 2d colorplots for each choice of 2 fits"
	)

	parser.add_argument(
		"--xticks", "--xt",
		dest="xticks",
		nargs = "+",
		type = float,
		default = None, 
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
		const = ((str, int, str, float, float), ("", 100, "png", 0.0, 0.0)),
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

	# print statistics on fit expressions
	parser.add_argument(
		"-m",
		dest="means",
		action="count",
		help="print statistics on fit data"
	)

	return parser	

def patch_args(args):
	
	# remove fits with no values to allow skipping positional arg
	args.fits = [_ for _ in args.fits if _[0] is not None]

def patch_cfg(cfg):
	...

def compose_specifications(args, cfg):
	spec = config.deep_merge(vars(args),cfg)
	return ObjDict(spec)




def repl_template(template, ctx=None):
	if ctx is None:
		ctx = {}
	def repl(match):
		# print(match, match.string, match.groups())
		return template.format(*match.groups(), **ctx)
	return repl

def apply_shorthand(string, shorthand, context, enforce_word_boundary=True):

	modified = string
	for pattern,template in shorthand.items():

		if enforce_word_boundary:
			pattern = r"\b" + pattern + r"\b"

		modified = re.sub(pattern, repl_template(template, context), modified)

	return modified





WARN_UNKNOWN_EXTENSION = "File may have unrecognized extension. Assuming default file type ({}) for file {}"
class Routine(object):

	# list of branches which are constructed by the manager after loading
	BRANCHES_CONSTRUCT = ['entry']
	
	def __init__(self, spc, irtn=1, nrtn=1, display=True):
		self.spc = spc

		self.irtn = irtn
		self.nrtn = nrtn
		self.is_last = (irtn == nrtn)

		self.display = display

	def perform(self):

		# procure the (unbinned) data needed for fitting and plotting
		self.procure_fit_data()

		# compose the specified figure from the fit data.
		# if 1d histogram(s), bin and plot each fit dataset.
		# if 2d heatmaps, bin and plot each combination of two fit datasets.
		# if 2d scatterplot, draw scatterplots.
		self.draw_data()

		# save and/or show if appropriate
		self.save_and_show()


	def procure_fit_data(self):

		# open data file interface
		self.load_dfile()

		# apply shorthand to data expressions
		self.apply_shorthand()

		# compile expressions for fits, scatterplots, cuts, and defs
		self.compile_expressions()

		# load the branches that need to come from the data file
		self.load_branches_needed()

		# construct special branches
		self.construct_branches()

		# apply adjustments, E.G. scaler rectification
		self.apply_branch_adjustments()

		# create defined branches and discard intermediate branches
		self.evaluate_defs()

		# apply cuts and discard branches not in fit or plot expressions
		self.apply_cuts()

		# evaluate the expressions for fit data
		self.calculate_fit_data()

	def load_dfile(self):
		dfile = self.spc.run

		# if the os.sep character is in the supplied argument, use it as-is.
		if os.sep in self.spc.run:
			pass

		# if dfile is interpretable as integer, use the numeric file template
		elif dfile.isdigit():
			dfile = DFILE_NUMERIC.format(dfile)

		# otherwise, try to interpret it based on extension
		else:
			
			# if there's no recognized file extension, use the default
			if not any((dfile.endswith(_) for _ in self.spc.supported_file_extensions)):

				# if there's a period in the filename, but no recognized extension, warn
				if "." in dfile:
					raise Warning(WARN_UNKNOWN_EXTENSION.format(
						self.spc.default_data_extension,
						dfile
					))

				dfile = "{}{}".format(
					dfile,
					self.spc.default_data_extension
				)

			# use the default directory (since there's no os.sep character
			# in the argument, we need to make it a full path.)
			ext = dfile.rpartition(".")[2]
			dfile = os.sep.join([
				DIR_DATA.get(ext, DIR_DATA_DEFAULT),
				dfile
			])

		ext = dfile.rpartition(".")[2]
		self.dfi = fileio.load_dfile(dfile)

	def apply_shorthand(self):
		
		if self.dfi.ftype == "root":
			context = {
				"drs"  : self.dfi.drsnum(),
				"chan" : self.dfi.channels()[0],
			}

		if self.dfi.ftype == "npz":
			context = {}

		shorthand = self.spc.shorthand.by_file_type[self.dfi.ftype]

		# apply shorthand to fit variables
		for fit in self.spc.fits:
			fit[0] = apply_shorthand(
				fit[0],
				shorthand,
				context,
				self.spc.shorthand.enforce_word_boundary,
			)

		# apply shorthand to scatterplots
		for plot in self.spc.scatterplots:
			plot[0] = apply_shorthand(
				plot[0],
				shorthand,
				context,
				self.spc.shorthand.enforce_word_boundary,
			)
			plot[1] = apply_shorthand(
				plot[1],
				shorthand,
				context,
				self.spc.shorthand.enforce_word_boundary,
			)

		# apply shorthand to cuts
		for cut in self.spc.cuts:
			cut[0] = apply_shorthand(
				cut[0],
				shorthand,
				context,
				self.spc.shorthand.enforce_word_boundary,
			)

		# apply shorthand to defs
		for def_ in self.spc.defs:
			def_[1] = apply_shorthand(
				def_[1],
				shorthand,
				context,
				self.spc.shorthand.enforce_word_boundary,
			)
		
	def compile_expressions(self):
		
		# set of branches needed to evaluate all fits, cuts, and defs
		self.branches_needed = set()

		# compile fit expressions
		self.fn_fits = []
		for fit in self.spc.fits:
			fn = expr.check_and_compile(fit[0])
			self.fn_fits.append(fn)
			self.branches_needed |= fn.kwargnames

		# scatterplots don't currently support expressions, but we still need to
		# know what variables are needed. these can be base branches, or defs.
		for plot in self.spc.scatterplots:
			self.branches_needed |= {plot[0], plot[1]}

		# snapshot of branches needed for plotting and fitting
		self.branches_needed_fit = self.branches_needed.copy()

		# compile expressions for cuts, and update branches_needed
		self.fn_cuts = []
		for cut in self.spc.cuts:
			fn = expr.check_and_compile(cut[0])
			self.fn_cuts.append(fn)
			self.branches_needed |= fn.kwargnames

		# snapshot of branches needed for fitting and/or cutting
		self.branches_needed_cut = self.branches_needed.copy()

		# compile expressions for defs, and update branches_needed.
		# also track the branches that are defined, so that they can
		# be excluded from the set of branches to load from the file.
		self.fn_defs = []
		self.branches_def = set()
		for def_ in self.spc.defs:
			fn = expr.check_and_compile(def_[1])
			self.fn_defs.append(fn)
			self.branches_def |= {def_[0]}
			self.branches_needed |= fn.kwargnames

	def load_branches_needed(self):
		# load branches from specified file. branches not constructed or
		# defined must be contained in the data file.
		branches = self.dfi.load_branches(
			self.branches_needed-(set(self.BRANCHES_CONSTRUCT)|self.branches_def)
		)

		# initialize the branch manager instance with the resulting branches
		self.bm = data.BranchManager(branches, export_copies=False, import_copies=False)

		for branch in self.bm.keys:
			print(branch, self.bm[branch].shape)

		# for key in self.bm.keys:
		# 	print(key, self.bm[key].dtype, self.bm[key].shape)

	def construct_branches(self):
		# construct special branches if needed
		if "entry" in self.branches_needed:
			self.bm.bud(data.bud_entry)

	def apply_branch_adjustments(self):
		# apply scaler rectification
		if self.spc.rectify_scalers:
			if any(_.startswith("scaler_") for _ in self.bm.keys):
				self.bm.bud(data.rectify_scaler(), overwrite=True)

		# apply timestamp fix and localization
		if any(_.startswith("timestamp_") for _ in self.bm.keys):
			self.bm.bud(data.fix_monotonic_timestamp(), overwrite=True)
			if self.spc.localize_timestamps:
				self.bm.bud(data.localize_timestamp(), overwrite=True)

	def evaluate_defs(self):

		# process defs to create new branches
		# todo: current implementation is slightly inefficient. defs
		# are evaluated before applying any cuts, resulting in excess
		# computation in the case where cuts do not depend on defs.
		# an implementation which applies each cut as soon as it is able to,
		# and prioritizes defs which enable cuts, would be faster.
		# it would also get rid of "divide by zero encountered in true_divice" errors.
		fn_defs_remain = [True for _ in self.fn_defs]
		n_remaining = len(self.fn_defs)
		n_round = 0
		while n_remaining:

			n_round += 1
			if self.spc.verbosity >= 2:
				print("branch construction, round {}; {} remain".format(n_round, n_remaining))

			for i,remain in enumerate(fn_defs_remain):

				if remain and self.fn_defs[i].kwargnames.issubset(self.bm.keys):

					if self.spc.verbosity >= 2:
						print("can    construct branch {}/{} this round".format(i+1,n_remaining))

					this_name = self.spc.defs[i][0]
					this_fn = self.fn_defs[i]

					self.bm.bud(
						lambda man:{this_name:this_fn(**{_:man[_] for _ in this_fn.kwargnames})}
					)
					if self.spc.verbosity >= 2:
						print("created branch {} with shape {}".format(this_name, self.bm[this_name].shape))

					fn_defs_remain[i] = False

				elif (self.spc.verbosity >= 2) and remain:
					print("cannot construct branch {}/{} this round".format(i+1,n_remaining))

			# if we have all branches needed for fits and cuts, there's
			# no need to keep evaluating defs
			if self.branches_needed_cut.issubset(self.bm.keys):
				print("successfully constructed all branches")
				break

			# check to see if progress has been made
			# if not, then it never will, and we have to exit
			n_remaining_now = sum(fn_defs_remain)
			if n_remaining_now == n_remaining:
				print("could not evaluate all definititions and transformations")
				print("missing one or more variables for completion")
				print("missing: {}".format(self.branches_needed_cut - set(self.bm.keys)))
				sys.exit(1)

			n_remaining = n_remaining_now

		# discard all branches not needed for fits or cuts
		self.bm.prune(set(self.bm.keys) - self.branches_needed_cut)

	def apply_cuts(self):

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
		for icut,fn in enumerate(self.fn_cuts):
			this_cut = self.spc.cuts[icut]

			# no bounds specified: boolean expression
			if (this_cut[1] is None) and (this_cut[2] is None):
				masks.append(mask_bool(fn))

			# at least one bound specified: lo<expression<hi
			else:
				masks.append(mask_range(fn,this_cut[1],this_cut[2]))

		# apply combined mask
		# todo: this applies the mask to branches that are needed to calculate
		# the mask, but not afterward. could save some computation by first
		# generating the mask, then discarding unneeded branches, then appling
		# the mask.
		if masks:
			self.bm.mask(data.mask_all(*masks), apply_mask=True)
		
		# discard all branches not needed for fits
		self.bm.prune((self.bm.keys) - self.branches_needed_fit)

	def calculate_fit_data(self):

		# we now have all the branches that show up in the expression
		# for at least one fit. to get the fit data, we have still have to
		# evaluate the expressions.
		self.fit_data = []
		for fn in self.fn_fits:
			self.fit_data.append(fn(**{_:self.bm[_] for _ in fn.kwargnames}))
			if self.spc.verbosity >= 1:
				print("calculated fit data with shape {}".format(self.fit_data[-1].shape))

		# print statistics if requested
		if self.spc.means:
			for ifit,fit in enumerate(self.spc.fits):
				print("")
				print(fit[0])
				print("mean: {}".format(self.fit_data[ifit].mean()))
			print("")


	def draw_data(self):

		# 2d scatterplots if requested
		if self.spc.scatterplots:
			
			for plot in self.spc.scatterplots:
				plt.plot(
					self.bm[plot[0]],
					self.bm[plot[1]],
					marker    = plot[2],
					linestyle = plot[3],
					color     = plot[4],
					label     = plot[5],
				)
				plt.xlabel(plot[0])
				plt.ylabel(plot[1])
			
			# decorate scatterplots
			self.decorate_plots()

			return


		# calculate bin edges
		self.fit_edges = []
		self.fit_centers = []
		for i,fit in enumerate(self.spc.fits):
			this_data = self.fit_data[i]

			# calculate bounds if not given
			lo = this_data[~np.isnan(this_data)].min() if fit[1] in [None,-np.inf] else fit[1]
			hi = this_data[~np.isnan(this_data)].max() if fit[2] in [None, np.inf] else fit[2]

			# calculate nbins if not given
			if fit[3]:
				nbins = fit[3]
			else:
				this_ndata = data.inrange(this_data,lo,hi,True,True).sum()
				nbins = data.bin_count_from_ndata(this_ndata)

			print(i,lo,hi,nbins)
			
			# generate edges based on bounds, nbins, and binning type
			if fit[4].startswith("li"):
				this_edges = data.edges_lin(lo,hi,nbins)
			elif fit[4].startswith("lo"):
				if lo<=0:
					lo = this_data[this_data>0].min()
				this_edges = data.edges_log(lo,hi,nbins)
			elif fit[4].startswith("s"):
				this_edges = data.edges_symlog(lo,hi,nbins)
			
			self.fit_edges.append(this_edges)
			self.fit_centers.append((this_edges[1:] + this_edges[:-1])*0.5)

		# make 2d mapplots if specified
		if self.spc.mapplots:
			gs = display.pairs2d(
				self.fit_data,
				self.fit_edges,
				[_[4].startswith("lo")  for _ in self.spc.fits],
				[_[5] if _[5] else _[0] for _ in self.spc.fits],
				cmap="afmhot",
				cbad="grey",
				norm="log" if self.spc.ylog else None,
			)

			if self.spc.title:
				plt.suptitle(self.spc.title)

			return

		# make and plot 1d binning of fit data
		self.fit_counts = []
		for i,fit in enumerate(self.spc.fits):
			this_data    = self.fit_data[i]
			this_edges   = self.fit_edges[i]
			this_centers = self.fit_centers[i]

			# calculate histogram counts and append
			this_counts, _ = np.histogram(this_data, this_edges)
			
			# scale by constant multiplier (1.0 by default)
			this_counts = this_counts * fit[6]
			self.fit_counts.append(this_counts)

			# plot histograms
			print("fit {:>12} - total counts {:>7}".format(fit[0], self.fit_counts[i].sum()))
			plt.step(
				this_centers,
				self.fit_counts[i],
				where='mid',
				label=fit[5] if fit[5] else fit[0],
			)

		# decorate histogram plots
		self.decorate_plots()

	def decorate_plots(self):
			
		# hlines and vlines
		for loc,col,ls,label in self.spc.vlines:
			plt.axvline(loc, color=col, ls=ls, label=label)
		for loc,col,ls,label in self.spc.hlines:
			plt.axhline(loc, color=col, ls=ls, label=label)

		# hspans and vspans
		for lo,hi,fc,alph,ec,lw,label in self.spc.vspans:
			facecolor = colors.to_rgb(fc)+(alph,)
			edgecolor = ec if ec is None else colors.to_rgb(ec)+(1,)
			plt.axvspan(
				lo,hi,
				facecolor=facecolor,
				edgecolor=edgecolor,
				linewidth=lw,
				label=label,
			)
		for lo,hi,fc,alph,ec,lw,label in self.spc.hspans:
			facecolor = colors.to_rgb(fc)+(alph,)
			edgecolor = ec if ec is None else colors.to_rgb(ec)+(1,)
			plt.axhspan(
				lo,hi,
				facecolor=facecolor,
				edgecolor=edgecolor,
				linewidth=lw,
				label=label,
			)

		# legeng, labels, title
		plt.legend()
		if self.spc.title:
			plt.title(self.spc.title)
		if self.spc.xlabel:
			plt.xlabel(self.spc.xlabel)
		if self.spc.ylabel:
			plt.ylabel(self.spc.ylabel)

		# ticks
		if self.spc.xticks:
			plt.xticks(self.spc.xticks)

		# scaling
		if self.spc.ylog:
			plt.yscale("log")
		if self.spc.fits and self.spc.fits[-1][4].startswith("lo"):
			plt.xscale("log")
		elif self.spc.fits and self.spc.fits[-1][4].startswith("s"):
			...


	def save_and_show(self):

		# don't show or save unless this is the last routine
		if not self.is_last:
			return

		# don't show or save if display is False
		if not self.display:
			return

		if self.spc.save_fig:
			self.save_fig()

		if self.spc.show:
			plt.show()

	def save_fig(self):
		
		fname, dpi, fmt, wi, hi = self.spc.save_fig
		
		fig = plt.gcf()
		
		# figure dimensions
		if args.save_fig[4]:
			fig.set_figheight(hi)
		if args.save_fig[3]:
			fig.set_figwidth(wi)

		# save figure
		if fname:
			if '.' not in fname:
				fname = '{}.{}'.format(fname, fmt)
			plt.savefig(FIG_FILE.format(fname), dpi=dpi, format=fmt)









if __name__ == '__main__':

	cfg = config.load("common")
	patch_cfg(cfg)

	parser = compose_parser()

	# split arguments into calls separated by any delimiter argument,
	# defined by cli.DEFAULT_DELIMITERS
	calls = cli.split_argument_sets(sys.argv[1:])

	# accumulate fit results for making comparisons
	check_counts = []
	check_edges  = []

	# dict of format strings to replace in figure title
	title_keys = {}


	# process each set in order
	nsets = len(calls)
	for icall,call in enumerate(calls):

		# parse this set of args
		this_args = parser.parse_args(call)
		patch_args(this_args)

		# discard empty fits
		this_args.fits = [_ for _ in this_args.fits if _[0] is not None]

		# merge args and cfg to create specifications
		this_spec = compose_specifications(this_args, cfg)

		# create Routine instance and perform routine
		rtn = Routine(this_spec, icall+1, nsets, False)
		rtn.perform()

		if this_spec.check_consistent:
			check_counts += rtn.fit_counts
			check_edges  += rtn.fit_edges


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


	# get the last Routine instance to save and/or show the figure
	rtn.display = True
	rtn.save_and_show()

	# exit without error
	sys.exit(0)
