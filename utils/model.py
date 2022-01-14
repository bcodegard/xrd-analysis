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
		func,
		xdata,
		ydata,
		xerr,
		yerr,
		bounds,
		p0,
		rfs,
		need_cov=False,
		):
	"""	performs a root fit on xdata,ydata with specified errors"""

	# get bounds into format expected
	lo = [_[0] for _ in bounds]
	hi = [_[1] for _ in bounds]

	# scipy fit for parameter guess input to root fit
	# no need for errors, as this is just used as a guess by root
	popt, pcov = opt.curve_fit(func, xdata, ydata, p0, bounds=[lo,hi])

	# initialize TGraphErrors
	n_points = len(xdata)
	graph = ROOT.TGraphErrors(n_points, xdata, ydata, xerr, yerr)

	# initialize and fit TF1
	rf = ROOT.TF1("multifit",rfs,0.0,1.0)
	rf.SetParameters(*popt)
	fitResult = graph.Fit(rf,"NS" if need_cov else "N")

	# extract results
	par = rf.GetParameters()
	err = rf.GetParErrors()
	popt_root = [par[_] for _ in range(len(p0))]
	perr_root = [err[_] for _ in range(len(p0))]
	chi2 = rf.GetChisquare()
	ndof = rf.GetNDF()

	results = (np.array(popt_root), np.array(perr_root), chi2, ndof)

	if need_cov:
		cov = fitResult.GetCovarianceMatrix()
		nr = cov.GetNrows()
		nc = cov.GetNcols()

		cov_mtx_ptr = cov.GetMatrixArray()

		cov_arr = np.array([cov_mtx_ptr[_] for _ in range(nr*nc)]).reshape(nr,nc)

		results = results + (cov_arr, )

	return results

def fit_hist_with_root(
		func,
		xdata,
		ydata,
		bounds,
		p0,
		rfs,
		need_cov=False,
		skip_p0_improvement=False,
		):
	"""performs a root fit on xdata,ydata assuming poisson errors"""

	# get bounds into format expected
	lo = [_[0] for _ in bounds]
	hi = [_[1] for _ in bounds]

	if skip_p0_improvement:
		popt = np.array(p0)

	else:
		# determined fixed parameters and exclude from fit
		is_fixed = [l==h for l,h in bounds]
		if any(is_fixed):
			# for now, just use p0 in this case
			# TODO use scipy fit in case of fixed parameters instead of defaulting to p0
			# can make lambda which meshes dynamic and fixed parameters nicely
			popt = np.array(p0)

		else:
			# scipy fit for parameter guess input to root fit
			popt, pcov = opt.curve_fit(func, xdata, ydata, p0, bounds=[lo,hi])

	# root object initialization and fit
	rf = ROOT.TF1("multifit",rfs,0.0,1.0)
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
	fitResult = hist.Fit(rf,"NS" if need_cov else "N")
	par = rf.GetParameters()
	err = rf.GetParErrors()
	popt_root = [par[_] for _ in range(len(p0))]
	perr_root = [err[_] for _ in range(len(p0))]
	chi2 = rf.GetChisquare()
	ndof = rf.GetNDF()

	results = (np.array(popt_root), np.array(perr_root), chi2, ndof)

	if need_cov:
		cov = fitResult.GetCovarianceMatrix()
		nr = cov.GetNrows()
		nc = cov.GetNcols()

		cov_mtx_ptr = cov.GetMatrixArray()

		cov_arr = np.array([cov_mtx_ptr[_] for _ in range(nr*nc)]).reshape(nr,nc)

		results = results + (cov_arr, )

	return results




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

	def __call__(self, bounds=None):
		if bounds is None:
			bounds = [NO_BOUNDS]*self.npars
		else:
			bounds = [_ if _ is not None else NO_BOUNDS for _ in bounds]
		return model_single(self, bounds)




ERR_NO_I = "function {} has no inverse function defined"
ERR_NO_GUESS  = "function {} has no parameter guess function defined"
ERR_NO_VAL  = "function {} has no input validation function defined"
ERR_NO_IVAL = "function {} has no inverse input validation function defined"
ERR_NO_RFS  = "function {} has no root function template defined"
ERR_NO_IRFS = "function {} has no inverse root function template defined"
class model_single(object):
	""""""

	# TODO model_single is just model_multiple with one component
	# Can use one class instead of two?

	def __init__(self, arch, bounds=None):
		self.arch       = arch
		self.bounds     = bounds

		self.is_fixed = np.array([l==h for l,h in self.bounds], dtype=bool)
		self.is_free  = np.logical_not(self.is_fixed)
		self.fixed_parameters = np.array([l for l,h in self.bounds if l==h])

		# properties copied from arch
		self.name    = arch.name
		self.formula = arch.formula
		self.pnames  = arch.pnames
		self.npars   = arch.npars

	def __call__(self, x, *parameters):
		return self.fn(x,*parameters[:self.npars])

	def __add__(self, other):
		if type(other) is model_single:
			return model_multiple([self, other])
		elif type(other) is model_multiple:
			return model_multiple([self, *other.components])

	def fn(self, x, *parameters):
		# if len(parameters) < self.npars, mesh given parameters with fixed parameters
		if len(parameters) < self.npars:
			pars_new = np.zeros(self.npars)
			pars_new[self.is_free ] = np.array(parameters)
			pars_new[self.is_fixed] = self.fixed_parameters
			parameters = pars_new

		return self.arch.model_function(x,*parameters[:self.npars])
	def ifn(self, y, *parameters):
		return self.arch.i_model_function(y,*parameters[:self.npars])

	def val(self, x, *parameters):
		return self.arch.input_validation_function(x,*parameters[:self.npars])
	def ival(self, y, *parameters):
		return self.arch.i_input_validation_function(y,*parameters[:self.npars])

	def guess(self, x, y):
		return self.arch.parameter_guess_function(x,y,self.bounds)

	def rfs_custom(self, x=None, p=None, needs_parens=True, inv=False):
		template = self.arch.i_root_function_template if inv else self.arch.root_function_template
		post_format = "({})" if needs_parens else "{}"

		# case: x is None
		if x is None:
			x = "x"
		
		# case: p is None -> [0, npars)
		if p is None:
			p = range(self.npars)

		# case: type(p) is int -> [p, p+npars)
		if type(p) is int:
			p = range(p, p+self.npars)

		p_formatted = []
		for par in p:
			if type(par) is int:
				p_raw = "[{}]".format(par)
			else:
				p_raw = par
			p_formatted.append(post_format.format(p_raw))

		return template.format(x=post_format.format(x), p=p_formatted)

	def irfs_custom(self, x=None, p=None, needs_parens=True):
		return self.rfs_custom(x, p, needs_parens, True)

	def rfs(self, istart=0, inv=False):
		template = self.arch.i_root_function_template if inv else self.arch.root_function_template
		if istart is not None:
			return template.format(x="x", p=["[{}]".format(_) for _ in range(istart,istart+self.npars)])
		else:
			return template.format(x="x", p=["[{{{}}}]".format(_) for _ in range(self.npars)])

	def irfs(self, istart=0):
		return self.rfs(istart, True)

	def fit(self, xdata, ydata, need_cov=False):
		return fit_hist_with_root(self.fn, xdata, ydata, self.bounds, self.guess(xdata,ydata), self.rfs(), need_cov)

	def fit_with_errors(self, xdata, ydata, xerr, yerr, need_cov=False):
		return fit_graph_with_root(self.fn, xdata, ydata, xerr, yerr, self.bounds, self.guess(xdata,ydata), self.rfs(), need_cov)


ERR_I_MULTI = "model_multiple does not support inverse operations"
class model_multiple(object):
	""""""

	def __init__(self,components):

		self.arch = None
		self.bounds_c = [_.bounds for _ in components]
		self.bounds = sum(self.bounds_c, [])

		self.is_fixed = np.array([l==h for l,h in self.bounds], dtype=bool)
		self.is_free  = np.logical_not(self.is_fixed)
		self.fixed_parameters = np.array([l for l,h in self.bounds if l==h])

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
		# if len(parameters) < self.npars, mesh given parameters with fixed parameters
		if len(parameters) < self.npars:
			pars_new = np.zeros(self.npars)
			pars_new[self.is_free ] = np.array(parameters)
			pars_new[self.is_fixed] = self.fixed_parameters
			parameters = pars_new

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

	def rfs_custom(self, x, p, needs_parens=True):
		pieces = []
		istart = 0
		for ic,c in enumerate(self.components):
			pieces.append(c.rfs_custom("{x}",["{{p[{}]}}".format(_) for _ in range(istart,istart+c.npars)],False))
			istart += c.npars
		template = " + ".join(pieces)

		post_format = "({})" if needs_parens else "{}"

		# case: x is None
		if x is None:
			x = "x"
		
		# case: p is None -> [0, npars)
		if p is None:
			p = range(self.npars)

		# case: type(p) is int -> [p, p+npars)
		if type(p) is int:
			p = range(p, p+self.npars)

		p_formatted = []
		for par in p:
			if type(par) is int:
				p_raw = "[{}]".format(par)
			else:
				p_raw = par
			p_formatted.append(post_format.format(p_raw))

		return template.format(x=post_format.format(x), p=p_formatted)

	def irfs_custom(self, x, p):
		raise ValueError(ERR_I_MULTI)

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

	def fit(self, xdata, ydata, need_cov=False):
		return fit_hist_with_root(self.fn, xdata, ydata, self.bounds, self.guess(xdata,ydata), self.rfs(), need_cov)

	def fit_with_errors(self, xdata, ydata, xerr, yerr, need_cov=False):
		return fit_graph_with_root(self.fn, xdata, ydata, xerr, yerr, self.bounds, self.guess(xdata,ydata), self.rfs(), need_cov)




class metamodel(object):
	""""""

	def __init__(self, m, xfp, xfp_rfs=None, xfx=False):
		
		# reference to model being transformed
		# might not need to actually store this reference
		self.m = m

		# transformations of parameters of m
		# one token per parameter that model m takes
		self.xfp = xfp

		# list of root function strings describing
		# the dependence of each parameter p of model m
		# on the parameters q of the metamodel.
		# 
		# entries are only accessed for user-defined functions
		# and are auto-generated for all other cases.
		# 
		# can be None if no custom functions are being used
		self.xfp_rfs = xfp_rfs

		# constant linear transformation of independent variable (x)
		# used to constrain magnitude of numbers in fit routine
		# xprime = xfx[0] + x*xfx[1], or xprime=x if xfx is False
		self.xfx = xfx

	def __call__(self, x, *q):
		return self.fn(x, *q)

	def rfs(self,needs_parens=True):
		"""generate root function string"""

		# transforming x is done before fitting since the transformation of x is constant
		x = None

		# entries in p correspond to parameters of m
		# but [0] etc. refer to components of q, not p.
		p = []

		for i,token in enumerate(self.xfp):

			# case: number -> str(token)
			if type(token) in (int, float):
				value = str(token)

			# case: array a -> a dot "[0], [1], ... "
			elif type(token) is np.ndarray:
				value = "+".join(["[{}]".format(i) if val==1 else "[{}]*{}".format(i,val) for i,val in enumerate(token) if val])
				if not value:
					value = "0"

			# case: function -> use value in xfp_rfs
			else:
				value = self.xfp_rfs[i]

			p.append(value)

		# return "+".join(p)
		return self.m.rfs_custom(x=x,p=p,needs_parens=needs_parens)


	def transform_parameters(self, q):
		"""calculate parameters p for model m given parameters q"""

		p = []
		for token in self.xfp:

			# case: number -> literal constant value
			if type(token) in (int, float):
				value = token

			# # todo: this is a bad way of doing this. disabled for now.
			# #       can just use array case with basis vector for now.
			# # case: str "n" -> equal to parameter at index n in q
			# elif type(token) is str:
			# 	value = q[int(token)]

			# case: array a -> a dot q
			elif type(token) is np.ndarray:
				value = np.dot(token, q)

			# case: function f -> f(q)
			else:
				value = token(q)

			p.append(value)

		return p

	def fn(self, x, *q, is_unprimed=True):
		
		# apply linear transformation
		# x is constant, and therefore so is xprime, for a given fit routine.
		# where to do this so that it only needs to happen once during fit routine?
		# maybe calc at start of fit routine and keep it
		# pass as argument to this function or have flag for whether to use stored value
		if is_unprimed and self.xfx:
			x = self.xfx[0] + x*self.xfx[1]

		p = self.transform_parameters(q)

		return self.m(x, *p)









# standard archetypes
# TODO add bounds consideration for guess functions

def always_true(z,*pars):
	if type(z) is np.ndarray:
		return np.ones(z.shape).astype(bool)
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
	name    = "exponential",
	formula = "b*exp(k*x)",
	pnames  = ["b","k"],
	model_function   = lambda x,b,k:b*np.exp(k*x),
	i_model_function = lambda y,b,k:np.log(y/b)/k,
	parameter_guess_function    = exp_guess,
	input_validation_function   = None,
	i_input_validation_function = lambda y,b,k:(y/b)>0,
	root_function_template      = "{p[0]}*exp({p[1]}*{x})",
	i_root_function_template    = "log({x}/{p[0]})/{p[1]}",
	# root_function_template      = "[{0}]*exp([{1}]*x)",
	# i_root_function_template    = "log(x/[{0}])/[{1}]",
)
exp = exponential


def sqrt_guess(x,y,bounds):
	q = y[-1]/((x[-1])**0.5)
	return q,0.0
sqrt = function_archetype(
	name    = "square_root",
	formula = "q*sqrt(r+x)",
	pnames  = ["q","r"],
	model_function   = lambda x,q,r:q*np.sqrt(r+x),
	i_model_function = lambda y,q,r:(y/q)**2 - r,
	parameter_guess_function    = sqrt_guess,
	input_validation_function   = lambda x,q,r:(r+x)>0,
	i_input_validation_function = lambda y,q,r:y>0,
	root_function_template      = "{p[0]}*sqrt({p[1]}+{x})",
	i_root_function_template    = "(({x}/{p[0]})**2 - {p[1]})",
	# i_root_function_template    = "[{0}]*sqrt([{1}]+x/{s[0]})",
	# i_input_validation_function = "{s[0]}*((x/[{0}])**2 - [{1}])",
)


def suppressed_monomial_guess(x,y,bounds):
	print(bounds)
	n=sum(bounds[2])/2
	if bounds[0][1] is np.inf:
		i_xpeak = np.argmax(y)
	else:
		xpeak = sum(bounds[0])/2
		i_xpeak = np.searchsorted(x, xpeak)
	xpeak = x[i_xpeak]
	c = y[i_xpeak]/n
	return xpeak,c,n
def mf_smono(x,xpeak,c,n):
	quantity = n*(x/xpeak)
	return c*(quantity**n)*np.exp(-quantity)
suppressed_monomial = function_archetype(
	name    = "suppressed_monomial",
	formula = "c * (n*x/xpeak)**n * exp(-n*x/xpeak)",
	pnames  = ["xpeak","c","n"],
	model_function   = mf_smono,
	i_model_function = None,
	parameter_guess_function    = suppressed_monomial_guess,
	input_validation_function   = lambda x,xpeak,c,n:x>0,
	i_input_validation_function = None,
	root_function_template      = "{p[1]}*(({p[2]}*{x}/{p[0]})**{p[2]})*exp(-{p[2]}*{x}/{p[0]})",
	i_root_function_template    = None,
)
smono = suppressed_monomial


constant = function_archetype(
	name    = "constant",
	formula = "a0",
	pnames  = ["a0"],
	model_function   = lambda x,c:x*0+c,
	i_model_function = None,
	parameter_guess_function    = lambda x,y,bounds:[y.mean()],
	input_validation_function   = None,
	i_input_validation_function = None,
	root_function_template      = "{p[0]}",
	i_root_function_template    = None,
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
	name    = "line",
	formula = "a0 + a1*x",
	pnames  = ["a1", "a0"],
	model_function   = lambda x,a1,a0:a0+x*a1,
	i_model_function = lambda y,a1,a0:(y-a0)/a1,
	parameter_guess_function    = line_guess,
	input_validation_function   = None,
	i_input_validation_function = None,
	root_function_template      = "{p[1]} + {p[0]}*{x}",
	i_root_function_template    = "({x}-{p[1]})/{p[0]}",
	# i_root_function_template    = "[{1}] + [{0}]*x",
	# i_input_validation_function = "(x-[{1}])/[{0}]",
)
poly1 = line

def quadratic_guess(x,y,bounds):
	dx = x[-1] - x[0]
	dy = y[-1] - y[0]
	a1 = dy/dx
	a0 = y[0] - a1*x[0]
	return [0.0,a1,a0]
quadratic = function_archetype(
	name    = "quadratic",
	formula = "a0 + a1*x + a2*x**2",
	pnames  = ["a2","a1","a0"],
	model_function   = lambda x,a2,a1,a0:a0+x*a1+(x**2)*a2,
	i_model_function = lambda y,a2,a1,a0:(-a1 + np.sqrt(a1**2 - 4*a2*(a0-y)))/(2*a2),
	parameter_guess_function    = quadratic_guess,
	input_validation_function   = always_true,
	i_input_validation_function = lambda y,a0,a1,a2:(y*a2) > (a0*a2-(a1**2)/4.0),
	root_function_template      = "{p[2]} + {p[1]}*{x} + {p[0]}*{x}**2",
	i_root_function_template    = "(-{p[1]} + sqrt({p[1]}**2 - 4*{p[0]}*({p[2]}-{x})))/(2*{p[0]})",
	# root_function_template      = "[{2}] + [{1}]*x + [{0}]*x**2",
	# i_root_function_template    = "(-[{1}] + sqrt([{1}]**2 - 4*[{0}]*([{2}]-x)))/(2*[{0}])",
)
poly2 = quadratic
quad = quadratic

def powerlaw_guess(x,y,bounds):
	return [y.max() / x.max(), 1.0]
powerlaw = function_archetype(
	name    = "powerlaw",
	formula = "b*x^m",
	pnames  = ["b","m"],
	model_function   = lambda x,b,m:np.power(x,m)*b,
	i_model_function = lambda y,b,m:np.power(y/b, 1/m),
	parameter_guess_function    = powerlaw_guess,
	input_validation_function   = lambda x,b,m:x > 0,
	i_input_validation_function = lambda y,b,m:y/b > 0,
	root_function_template      = "{p[0]}*({x}**{p[1]})",
	i_root_function_template    = "({x}/{p[0]})**(1/{p[1]})",
	# root_function_template      = "[{0}]*(x**[{1}])",
	# i_root_function_template    = "(x/[{0}])**(1/[{1}])",
)

def powerlaw_plus_constant_guess(x,y,bounds):
	return [y.max() / x.max(), 1.0, 0.0]
powerlaw_plus_constant = function_archetype(
	name    = "powerlaw+constant",
	formula = "b*x^m+c",
	pnames  = ["b","m","c"],
	model_function   = lambda x,b,m,c:np.power(x,m)*b+c,
	i_model_function = lambda y,b,m,c:np.power((y-c)/b, 1/m),
	parameter_guess_function    = powerlaw_guess,
	input_validation_function   = lambda x,b,m,c:x > 0,
	i_input_validation_function = lambda y,b,m,c:(y-c)/b > 0,
	root_function_template      = "{p[0]}*({x}**{p[1]})+{p[2]}",
	i_root_function_template    = "(({x}-{p[2]})/{p[0]})**(1/{p[1]})",
	# root_function_template      = "[{0}]*(x**[{1}])+[{2}]",
	# i_root_function_template    = "((x-[{2}])/[{0}])**(1/[{1}])",
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
	name    = "gaussian",
	formula = "c*exp(-0.5*((x-mu)/sigma)**2)",
	pnames  = ["c","mu","sigma"],
	model_function   = lambda x, c, mu, sigma:c*np.exp(-0.5*((x-mu)/sigma)**2),
	i_model_function = None,
	parameter_guess_function    = gaussian_guess,
	input_validation_function   = None,
	i_input_validation_function = None,
	root_function_template      = "{p[0]}*exp(-0.5*(({x}-{p[1]})/{p[2]})**2)",
	i_root_function_template    = None,
)
gaus=gaussian
norm=gaussian
normal=gaussian




# for referring to models via short strings
# used to define pieces of multi-fits via CLI arguments
shorthand = {
	"p":powerlaw,
	"e":exponential,
	# "E":exponential_local,
	"c":constant,
	"l":line,
	"q":quadratic,
	"g":gaussian,
	"s":suppressed_monomial,
}

# for referring to models via numerical ID
# used to specify models in calibration
models_by_id = {
	0:powerlaw_plus_constant,
	1:quadratic,
}
