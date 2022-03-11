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
import scipy.odr as odr

one_over_e = 1/math.e
NO_BOUNDS = [-np.inf, np.inf]




def fit_graph(func,xdata,ydata,xerr,yerr,bounds,p0,need_cov=False):
	"""performs a fit on xdata,ydata with xerr,yerr errors
	assumes no covariance in x or y errors"""

	# get bounds into format expected
	lo = [_[0] for _ in bounds]
	hi = [_[1] for _ in bounds]

	# enforce p0 in bounds
	for ip,p in enumerate(p0):
		if p < lo[ip]:p0[ip]=lo[ip]
		if p > hi[ip]:p0[ip]=hi[ip]

	is_fixed = [l==h for l,h in bounds]

	odr_data  = odr.RealData(xdata, ydata, xerr, yerr)
	odr_model = odr.Model(lambda beta,x:func(x,*beta))
	odr_inst  = odr.ODR(odr_data, odr_model, p0, maxit=500)#, None, is_fixed, np.zeros(xdata.shape))

	odr_inst.run()
	odr_out = odr_inst.output

	# print("{:<18}: {}".format("beta", odr_out.beta))
	# print("{:<18}: {}".format("sd_beta", odr_out.sd_beta))
	# print("{:<18}: {}".format("cov_beta", odr_out.cov_beta))
	# # print("{:<18}: {}".format("delta", odr_out.delta))
	# # print("{:<18}: {}".format("eps", odr_out.eps))
	# # print("{:<18}: {}".format("xplus", odr_out.xplus))
	# # print("{:<18}: {}".format("y", odr_out.y))
	# print("{:<18}: {}".format("res_var", odr_out.res_var))
	# print("{:<18}: {}".format("sum_square", odr_out.sum_square))
	# print("{:<18}: {}".format("sum_square_delta", odr_out.sum_square_delta))
	# print("{:<18}: {}".format("sum_square_eps", odr_out.sum_square_eps))
	# print("{:<18}: {}".format("inv_condnum", odr_out.inv_condnum))
	# print("{:<18}: {}".format("rel_error", odr_out.rel_error))
	# # print("{:<18}: {}".format("work", odr_out.work))
	# print("{:<18}: {}".format("work_ind", odr_out.work_ind))
	# print("{:<18}: {}".format("info", odr_out.info))
	# print("{:<18}: {}".format("stopreason", odr_out.stopreason))

	results = (odr_out.beta, np.sqrt(np.diag(odr_out.cov_beta)), odr_out.res_var, xdata.size - len(p0), )
	if need_cov:
		results = results + (odr_out.cov_beta, )
	return results

def fit_hist(func,xdata,ydata,bounds,p0,need_cov=False,yerr=None):
	"""performs a fit on xdata,ydata assuming poisson errors on y and no errors on x"""

	# get bounds into format expected
	lo = [_[0] for _ in bounds]
	hi = [_[1] for _ in bounds]

	# enforce p0 in bounds
	for ip,p in enumerate(p0):
		if p < lo[ip]:p0[ip]=lo[ip]
		if p > hi[ip]:p0[ip]=hi[ip]

	is_fixed = [l==h for l,h in bounds]
	assert not any(is_fixed)
	# unsupported right now

	# assume hist -> poisson errors on y
	if yerr is None:
		yerr = np.sqrt(ydata)
	popt, pcov = opt.curve_fit(func, xdata, ydata, p0, yerr, True, bounds=[lo,hi])
	
	perr  = np.sqrt(np.diag(pcov))
	chisq = ((func(xdata,*popt)-ydata)/yerr)**2
	ndof  = xdata.size - len(popt)

	results = (popt, perr, chisq.sum(), ndof)
	if need_cov:
		results = results + (pcov, )

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

			jacobian = None,

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

		# jacobian for y=f(x;*p) on variables {x, *p}
		self.jacobian = jacobian

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

	def fit(self, xdata, ydata, need_cov=False, p0=None, yerr=None, ):
		if p0 is None:
			p0 = self.guess(xdata, ydata)
		return fit_hist(self.fn, xdata, ydata, self.bounds, p0, need_cov, yerr)

	def fit_with_errors(self, xdata, ydata, xerr, yerr, need_cov=False):
		return fit_graph(self.fn, xdata, ydata, xerr, yerr, self.bounds, self.guess(xdata,ydata), need_cov)

	def jacobian(self, x, *parameters):
		if len(parameters) < self.npars:
			pars_new = np.zeros(self.npars)
			pars_new[self.is_free ] = np.array(parameters)
			pars_new[self.is_fixed] = self.fixed_parameters
			parameters = pars_new
		return self.arch.jacobian(x,*parameters)


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

	def fit(self, xdata, ydata, need_cov=False, p0=None):
		if p0 is None:
			p0 = self.guess(xdata, ydata)
		return fit_hist(self.fn, xdata, ydata, self.bounds, p0, need_cov)

	def fit_with_errors(self, xdata, ydata, xerr, yerr, need_cov=False):
		return fit_graph(self.fn, xdata, ydata, xerr, yerr, self.bounds, self.guess(xdata,ydata), need_cov)




class metamodel(object):
	""""""

	def __init__(self, m, xfp, xfx=False):
		
		# reference to model being transformed
		# might not need to actually store this reference
		self.m = m

		# transformations of parameters of m
		# one token per parameter that model m takes
		self.xfp = xfp

		# constant linear transformation of independent variable (x)
		# used to constrain magnitude of numbers in fit routine
		# xprime = xfx[0] + x*xfx[1], or xprime=x if xfx is False
		self.xfx = xfx

	def __call__(self, x, *q):
		return self.fn(x, *q)

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
	jacobian = lambda x,c:np.array([0,1]),
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
	jacobian = lambda x,a1,a0:np.array([a1,x,1]),
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
	jacobian = lambda x,a2,a1,a0:np.array([a1+2*x*a2,x**2,x,1]),
)
poly2 = quadratic
quad = quadratic

quadratic_inverse = function_archetype(
	name    = "quadratic inverse",
	formula = "(-b + sqrt(b**2 - 4*a*(c-x)))/(2*a)",
	pnames  = ["a","b","c"],
	model_function   = lambda y,a2,a1,a0:(-a1 + np.sqrt(a1**2 - 4*a2*(a0-y)))/(2*a2),
	i_model_function = lambda x,a2,a1,a0:a0+x*a1+(x**2)*a2,
	parameter_guess_function    = lambda x,y,bounds:quadratic_guess(y,x,bounds),
	input_validation_function   = lambda y,a0,a1,a2:(y*a2) > (a0*a2-(a1**2)/4.0),
	i_input_validation_function = always_true,
)
iquad = quadratic_inverse

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
	0:quadratic,
	1:powerlaw_plus_constant,
}
