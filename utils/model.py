"""
classes, functions, and data for models used in analysis
"""

# __all__ = ...
__author__ = "Brunel Odegard"
__version__ = "0.0"

import math
import itertools
import numpy as np

import scipy.optimize as opt
import ROOT

one_over_e = 1/math.e
NO_BOUNDS = [-np.inf, np.inf]




def fit_graph_with_root(
		model,
		xdata,
		ydata,
		xerr,
		yerr,
		):
	"""	performs a root fit on xdata,ydata with specified errors"""

	# get bounds into format expected
	lo = [_[0] for _ in model.bounds]
	hi = [_[1] for _ in model.bounds]

	# scipy fit for parameter guess input to root fit
	# no need for errors, as this is just used as a guess by root
	p0 = model.guess(xdata, ydata)
	popt, pcov = opt.curve_fit(model, xdata, ydata, p0, bounds=[lo,hi])

	# initialize TGraphErrors
	n_points = len(xdata)
	graph = ROOT.TGraphErrors(n_points, xdata, ydata, xerr, yerr)

	# initialize and fit TF1
	rf = ROOT.TF1("multifit",model.rfs(),0.0,1.0)
	rf.SetParameters(*popt)
	graph.Fit(rf, "N")

	# extract results
	par = rf.GetParameters()
	err = rf.GetParErrors()
	popt_root = [par[_] for _ in range(model.npars)]
	perr_root = [err[_] for _ in range(model.npars)]
	chi2 = rf.GetChisquare()
	ndof = rf.GetNDF()
	
	# return results
	return np.array(popt_root), np.array(perr_root), chi2, ndof

def fit_hist_with_root(
		model,
		xdata,
		ydata,
		):
	"""	performs a root fit on xdata,ydata assuming poisson errors"""
	
	# get bounds into format expected
	lo = [_[0] for _ in model.bounds]
	hi = [_[1] for _ in model.bounds]

	# scipy fit for parameter guess input to root fit
	p0 = model.guess(xdata, ydata)
	popt, pcov = opt.curve_fit(model, xdata, ydata, p0, bounds=[lo,hi])

	# root object initialization and fit
	rf = ROOT.TF1("multifit",model.rfs(),0.0,1.0)
	ndata = xdata.shape[0]
	hist = ROOT.TH1F("hist", "data to fit", ndata, xdata[0], xdata[-1])
	for j in range(ndata):
		hist.SetBinContent(j+1, ydata[j])
	rf.SetParameters(popt)
	for ip in range(len(lo)): # set bounds on root parameters
		this_lo, this_hi = lo[ip], hi[ip]
		if this_lo == -np.inf:this_lo=-1e7 # 1e7 in place of inf
		if this_hi ==  np.inf:this_hi= 1e7 # 
		rf.SetParLimits(ip, this_lo, this_hi)
	hist.Fit(rf,"N")
	par = rf.GetParameters()
	err = rf.GetParErrors()
	popt_root = [par[_] for _ in range(model.npars)]
	perr_root = [err[_] for _ in range(model.npars)]
	chi2 = rf.GetChisquare()
	ndof = rf.GetNDF()
	
	return np.array(popt_root), np.array(perr_root), chi2, ndof



class function_archetype(object):
	"""
	instances define a type of function; for example, gaussian or line
	instances can be called to create a model_single object
	"""

	def __init__(
			self,

			name,
			formula,
			pnames,

			model_function,
			i_model_function = None,

			parameter_guess_function = None,
			
			input_validation_function = None,
			i_input_validation_function = None,
			
			root_function_template = None,
			i_root_function_template = None,

			):

		# name of formula. display only.
		self.name = name

		# string describing forumla. display only.
		self.formula = formula

		# list of parameter names. display only.
		self.pnames = pnames

		# number of parameters
		self.npars = len(self.pnames)
		
		self.model_function   = model_function
		self.i_model_function = i_model_function

		# parameter guess function f(x,y) -> p0
		# same function for inverse
		self.parameter_guess_function   = parameter_guess_function

		# input validation functions (None -> assume all valid)
		self.input_validation_function   = input_validation_function
		self.i_input_validation_function = i_input_validation_function

		self.root_function_template   = root_function_template
		self.i_root_function_template = i_root_function_template

	def __call__(self, bounds=None, static_parameters=[]):
		if bounds is None:
			bounds = [NO_BOUNDS]*self.npars
		else:
			bounds = [_ if _ is not None else NO_BOUNDS for _ in bounds]
		return model_single(self, bounds, static_parameters)




ERR_NO_I = "function {} has no inverse function defined"
ERR_NO_GUESS  = "function {} has no parameter guess function defined"
ERR_NO_VAL  = "function {} has no input validation function defined"
ERR_NO_IVAL = "function {} has no inverse input validation function defined"
ERR_NO_RFS  = "function {} has no root function template defined"
ERR_NO_IRFS = "function {} has no inverse root function template defined"
class model_single(object):
	""""""

	def __init__(self, arch, bounds=None, static_parameters=[]):
		self.arch       = arch
		self.bounds     = bounds

		# dev
		self.static_parameters = static_parameters

		# properties copied from arch
		self.name    = arch.name
		self.formula = arch.formula
		self.pnames  = arch.pnames
		self.npars   = arch.npars

	def __call__(self, x, *parameters):
		if self.static_parameters:
			return self.fn(x,*parameters[:self.npars],*self.static_parameters)
		else:
			return self.fn(x,*parameters[:self.npars])

	def __add__(self, other):
		if type(other) is model_single:
			return model_multiple([self, other])
		elif type(other) is model_multiple:
			return model_multiple([self, *other.components])

	def fn(self, x, *parameters):
		return self.arch.model_function(x,*parameters[:self.npars],*self.static_parameters)
	def ifn(self, y, *parameters):
		return self.arch.i_model_function(y,*parameters[:self.npars],*self.static_parameters)

	def val(self, x, *parameters):
		return self.arch.input_validation_function(x,*parameters[:self.npars],*self.static_parameters)
	def ival(self, y, *parameters):
		return self.arch.i_input_validation_function(y,*parameters[:self.npars],*self.static_parameters)

	def guess(self, x, y):
		if self.static_parameters:
			return self.arch.parameter_guess_function(x,y,self.bounds,self.static_parameters)
		else:
			return self.arch.parameter_guess_function(x,y,self.bounds)

	def rfs(self, istart=0):
		if istart is not None:
			return self.arch.root_function_template.format(*range(istart, istart+self.npars),s=self.static_parameters)
		else:
			return self.arch.root_function_template
	def irfs(self, istart=0):
		if istart is not None:
			return self.arch.i_root_function_template.format(*range(istart, istart+self.npars),s=self.static_parameters)
		else:
			return self.arch.i_root_function_template

	def fit(self, *args, **kwargs):
		return fit_hist_with_root(self, *args, **kwargs)

	def fit_with_errors(self, *args, **kwargs):
		return fit_graph_with_root(self, *args, **kwargs)


ERR_I_MULTI = "model_multiple does not support inverse operations"
class model_multiple(object):
	""""""

	def __init__(self,components):

		self.arch = None
		self.bounds_c = [_.bounds for _ in components]
		self.bounds = sum(self.bounds_c, [])

		self.components = components
		
		self.pnames_c = [_.pnames for _ in components]
		self.npars_c  = [_.npars  for _ in components]
		self.pf_indices = [0] + list(itertools.accumulate(self.npars_c))
		self.pnames = sum(self.pnames_c, [])
		self.npars  = len(self.pnames)

	def __call__(self,x,*parameters):
		return self.fn(x,*parameters)

	def __add__(self,other):
		if type(other) is model_single:
			return model_multiple([*self.components,other])
		elif type(other) is model_multiple:
			return model_multiple([*self.components,*other.components])

	def fn(self,x,*parameters):
		ans = self.components[0].fn(x,*parameters[0:self.pf_indices[1]])
		for ic,c in enumerate(self.components[1:]):
			ans += c.fn(x,*parameters[self.pf_indices[ic+1]:self.pf_indices[ic+2]])
		return ans
	def ifn(self,y,*parameters):
		raise ValueError(ERR_I_MULTI)

	def val(self,x,*parameters):
		all_valid = None
		first = True
		for ic,c in enumerate(self.components):
			if c.arch.input_validation_function is not None:
				i1 = self.pf_indices[ic]
				i2 = self.pf_indices[ic+1]
				if first:
					all_valid = c.val(x,*parameters[i1:i2])
				else:
					all_valid = np.logical_and(all_valid, c.val(x,*parameters[i1:i2]))
		return all_valid
	def ival(self,y,*parameters):
		raise ValueError(ERR_I_MULTI)

	def guess(self,x,y):
		p0 = []
		for ic,c in enumerate(self.components):
			p0 += c.guess(x,y)
		return p0

	def rfs(self,istart=0):
		pieces = []
		if istart is not None:
			for ic,c in enumerate(self.components):
				pieces.append(c.rfs(istart))
				istart += c.npars
		else:
			istart = 0
			for ic,c in enumerate(self.components):
				formatters = ["{"+str(_)+"}" for _ in range(istart,istart+c.npars)]
				pieces.append(c.rfs(None).format(*formatters))
				istart += c.npars
		return ' + '.join(pieces)
	def irfs(self,istart=0):
		raise ValueError(ERR_I_MULTI)

	def fit(self, *args, **kwargs):
		return fit_hist_with_root(self, *args, **kwargs)

	def fit_with_errors(self, *args, **kwargs):
		return fit_graph_with_root(self, *args, **kwargs)



# standard archetypes
# TODO add bounds consideration for guess functions

def always_true(z,*pars):
	if type(z) is np.ndarray:
		return np.ones(z.shape,dtpye=bool)
	else:
		return True

def exp_guess(x,y,bounds):
	dx = x[-1] - x[0]
	ry = y[-1] / y[0]
	if ry <= 0: # invalid -> just assume it's 1/e
		ry = one_over_e
	k = math.log(ry) / dx
	b = y[0] * ry ** (-x[0] / dx)
	return [b,k]
exponential = function_archetype(
	"exponential", # name
	"b*exp(k*x)",  # formula
	["b","k"],     # parameters
	lambda x,b,k:b*np.exp(k*x), # model function
	lambda y,b,k:np.log(y/b)/k, # inverse model function
	exp_guess, # parameter guess function
	None,                 # input validation function
	lambda y,b,k:(y/b)>0, # inverse input validation function
	"[{0}]*exp([{1}]*x)", # root function template
	"log(x/[{0}])/[{1}]", # inverse root function template
)
exp = exponential

# def linfrac_guess(x,y,bounds):
# 	return 1.0,0.0,0.0,1.0
# linfrac = function_archetype(
# 	"linear_fractional",
# 	"(a*x+b)/(c*x+d)",
# 	["a","b","c","d"],
# 	lambda x,a,b,c,d:(a*x+b)/(c*x+d),
# 	None,
# 	linfrac_guess,
# 	lambda x,a,b,c,d:c*x != d,
# 	None,
# 	"(x*[{0}]+[{1}])/(x*[{2}]+[{3}])",
# 	None,
# )

def suppressed_monomial_guess(x,y,bounds,static_parameters):
	n = static_parameters[0]
	if bounds[0][1] is np.inf:
		i_xpeak = np.argmax(y)
	else:
		xpeak = sum(bounds[0])/2
		i_xpeak = np.searchsorted(x, xpeak)
	xpeak = x[i_xpeak]
	c = y[i_xpeak]/n
	return xpeak,c
def mf_smono(x,xpeak,c,n):
	quantity = n*(x/xpeak)
	return c*(quantity**n)*np.exp(-quantity)
suppressed_monomial = function_archetype(
	"suppressed_monomial",         # name
	"c * (n*x/xpeak)**n * exp(-n*x/xpeak)", # formula
	["xpeak","c"], # dynamic parameters
	mf_smono, # python model function
	None,     # inverse python model function
	suppressed_monomial_guess, # parameter guess function
	lambda x,xpeak,c,n:x>0, # input validation function
	None,                                     # inverse input validation function
	"[{1}]*(({s[0]}*x/[{0}])**{s[0]})*exp(-{s[0]}*x/[{0}])", # root function templte
	None,                                              # inverse root function template
)
smono = suppressed_monomial

# exponential with static parameter x0
def exponential_local_guess(x,y,bounds,static_parameters):
	x0 = static_parameters[0]
	k = x0 * (math.log(y[0]) - math.log(y[-1])) / (x[0] - x[-1])
	b = y[0] * math.exp(-k*(x[0]/x0-1))
	return [b,k]
exponential_local = function_archetype(
	"local_exponential",
	"b*exp(k*(x/x0-1))",
	["b","k"],
	lambda x,b,k,x0:b*np.exp(k*(x/x0-1)),
	lambda y,b,k,x0:(np.log(y/b)/k + 1)*x0,
	exponential_local_guess,
	None,
	lambda y,b,k,x0:(y/b)>0,
	"[{0}]*exp([{1}]*(x/{s[0]}-1))",
	"(log(x/[{0}])/[{1}]+1)*{s[0]}",
	)
expl = exponential_local

constant = function_archetype(
	"constant",
	"a0",
	["a0"],
	lambda x,c:x*0+c,
	None,
	lambda x,y,bounds:[y.mean()],
	None,
	None,
	"[{}]",
	None,
)
poly0 = constant
mono0 = constant

def line_guess(x,y,bounds):
	dx = x[-1] - x[0]
	dy = y[-1] - y[0]
	a1 = dy/dx
	a0 = y[0] - a1*x[0]
	return [a1,a0]
line = function_archetype(
	"line",
	"a0 + a1*x",
	["a1", "a0"],
	lambda x,a1,a0:a0+x*a1,
	lambda y,a1,a0:(y-a0)/a1,
	line_guess,
	None,
	None,
	"[{1}] + [{0}]*x",
	"(x-[{1}])/[{0}]",
)
poly1 = line

def quadratic_guess(x,y,bounds):
	dx = x[-1] - x[0]
	dy = y[-1] - y[0]
	a1 = dy/dx
	a0 = y[0] - a1*x[0]
	return [0.0,a1,a0]
quadratic = function_archetype(
	"quadratic",
	"a0 + a1*x + a2*x**2",
	["a2","a1","a0"],
	lambda x,a2,a1,a0:a0+x*a1+(x**2)*a2,
	lambda y,a2,a1,a0:(-a1 + np.sqrt(a1**2 - 4*a2*(a0-y)))/(2*a2),
	quadratic_guess,
	always_true,
	lambda y,a0,a1,a2:(y*a2) > (a0*a2-(a1**2)/4.0),
	"[{2}] + [{1}]*x + [{0}]*x**2",
	"(-[{1}] + sqrt([{1}]**2 - 4*[{0}]*([{2}]-x)))/(2*[{0}])",
)
poly2 = quadratic
quad = quadratic

def powerlaw_guess(x,y,bounds):
	return [y.max() / x.max(), 1.0]
powerlaw = function_archetype(
	"powerlaw",
	"b*x^m",
	["b","m"],
	lambda x,b,m:np.power(x,m)*b,
	lambda y,b,m:np.power(y/b, 1/m),
	powerlaw_guess,
	lambda x,b,m:x > 0,
	lambda y,b,m:y/b > 0,
	"[{0}]*(x**[{1}])",
	"(x/[{0}])**(1/[{1}])",
)

def powerlaw_plus_constant_guess(x,y,bounds):
	return [y.max() / x.max(), 1.0, 0.0]
powerlaw_plus_constant = function_archetype(
	"powerlaw+constant",
	"b*x^m+c",
	["b","m","c"],
	lambda x,b,m,c:np.power(x,m)*b+c,
	lambda y,b,m,c:np.power((y-c)/b, 1/m),
	powerlaw_guess,
	lambda x,b,m,c:x > 0,
	lambda y,b,m,c:(y-c)/b > 0,
	"[{0}]*(x**[{1}])+[{2}]",
	"((x-[{2}])/[{0}])**(1/[{1}])",
)
powc = powerlaw_plus_constant

def gaussian_guess(x,y,bounds):
	mu_bounds = bounds[1]
	y_bound = y[np.logical_and(x > mu_bounds[0], x < mu_bounds[1])]
	x_bound = x[np.logical_and(x > mu_bounds[0], x < mu_bounds[1])]
	# constant = highest value within bounds
	c_guess = y_bound.max()
	# mu = location of highest value
	mu_guess = x_bound[np.argmax(y_bound)]
	# sigma = half the interval
	sigma_guess = 0.5 * (min([mu_bounds[1],x.max()]) - max([mu_bounds[0],x.min()]))
	return [c_guess, mu_guess, sigma_guess]
gaussian = function_archetype(
	"gaussian",
	"c*exp(-(x-mu)**2/(2*sigma**2))",
	["c","mu","sigma"],
	lambda x, c, mu, sigma:c*np.exp(-0.5*((x-mu)/sigma)**2),
	None,
	gaussian_guess,
	None,
	None,
	"gaus({0})",
	None,
)
gaus=gaussian
norm=gaussian
normal=gaussian




# for referring to models via short strings
# used to define pieces of multi-fits via CLI arguments
shorthand = {
	"p":powerlaw,
	"e":exponential,
	"E":exponential_local,
	"c":constant,
	"l":line,
	"q":quadratic,
	"g":gaussian,
	# "x":gaussian,
	"s":suppressed_monomial,
}

# for referring to models via numerical ID
# used to specify models in calibration
models_by_id = {
	0:powerlaw_plus_constant,
	1:quadratic,
}
