""""""

__author__ = "Brunel Odegard"
__version__ = "0.0"


import sys
import os
import math
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines  as lines
import scipy.optimize    as opt

import utils.fileio  as fileio
import utils.display as display
import utils.model   as model
import utils.data    as data




# math constants
ROOT_8_LOG_2 = (8 * math.log(2)) ** 0.5
ONE_OVER_ROOT_TAU = 1 / (2 * math.pi)

# convenience functions
# 
# comparisons for floats
iseq = lambda f1,f2,eps=1e-9:abs(f1-f2)<eps
# convenient way to discard values which are zero or negative
positive = lambda ar:ar[ar>0]




def transform_y(f_x_y, t, dt_dyprime, *t_params):
	"""Transform f_x_y(x,y) into f_x_yprime(x,yprime)
	t is the inverse transformation: from yprime to y
	y = t(yprime, *t_params)
	dt_dyprime is its derivative with respect to yprime"""
	def f_x_yprime(x,yprime,*par):
		return f_x_y(x, t(yprime,*t_params), *par) * dt_dyprime(yprime,*t_params)

def gaus_spread(E_to_mu_sigma):
	"""Returns a function of (E,A,*par) -> rho_A(E)

	An event with energy E will produce a distribution of observed A
	This distribution is modeled here as a gaussian with some center and
	width defined by the energy.

	The function returned will calculate the area (A) density for the given energy.
	Integrating it for all A and all other values fixed should give 1.

	E_to_mu_sigma is a function which takes E, and optionally some parameters,
	and returns a value for mu and sigma. These parameters should be passed to the
	function returned, as they will be passed along to calculate mu and sigma.

	For example: mu = gamma*E, sigma = r*gamma*e
	this would give linear A:E relation, and constant fractional width."""
	def gaus(x_E,y_A,*par):
		mu,sigma = E_to_mu_sigma(x_E,*par)
		return ONE_OVER_ROOT_TAU * np.exp(-0.5*((y_A - mu)/sigma)**2) / sigma

def transformed_gaus_spread(E_to_mu_sigma, t, dt_dAprime):
	"""Conventienly compose and return a transformed gaussian distributor.
	returns f(E, A') where f(E,A) is gaussian in A with mu,sigma functions of E
	and t(A') = A
	"""
	f_E_A = gaus_spread(E_to_mu_sigma)
	f_E_Aprime = transform_y(f_E_A, t, dt_dAprime)
	return f_E_Aprime



class binned_projector(object):
	"""Approximates a projection integral from (x,y)->(y)
	f(y) = integral over x of rho(x) * g(x,y)
	f(y) ~ sum over xi of N(xi) g(xi,y)"""

	# 2d arrays of (bin) midpoints
	# x = axis 0
	# y = axis 1
	# xMids[i,j] = xi
	# yMids[i,j] = yj
	xMids = None
	yMids = None

	# sizes of x and y axes
	xRes = None
	yRes = None

	# density|weight spectrum rho(x)
	# data can be ones, integers, or any number
	# xSpec[i,j] = weight for bin xi
	xSpec = None

	def __init__(self, func, xMids=None, yMids=None, xSpec=None, xEdges=None, yEdges=None, xData=None):
		
		# function f(x,y,*p)
		self._func = func

		# don't copy, so that if multiple instances of the class are
		# needed, you can pass the same arrays to each and not duplicate
		self._xMids = xMids
		self._yMids = yMids
		self._xSpec = xSpec

		# use the rest of the kwargs to generate any that are None
		# as well as broadcast from 1d to 2d
		# also calculate and store some properties for convenience
		self._setup(xSpec, xEdges, yEdges, xData)

	def _setup(self, xSpec, xEdges, yEdges, xData):
		"""generates xMids yMids and xSpec if they are None or 1d"""

		# generate 1d xMids if it's None
		if self._xMids is None:
			self._xMids = 0.5*(xEdges[1:]+xEdges[:-1])
		self._xRes = self._xMids.shape[0]

		# generate 1d yMids if it's None
		if self._yMids is None:
			self._yMids = 0.5*(yEdges[1:]+yEdges[:-1])
		self._yRes = self._yMids.shape[0] if self._yMids.ndim == 1 else self._yMids.shape[1]

		# broadcast yMids to 2d if it's 1d
		if self._yMids.ndim == 1:
			self._yMids = np.broadcast_to(
				self._yMids.reshape([1,self._yRes]),
				[self._xRes,self._yRes],
			)

		# broadcast xMids to 2d if it's 1d
		if self._xMids.ndim == 1:
			self._xMids = np.broadcast_to(
				self._xMids.reshape([self._xRes,1]),
				[self._xRes,self._yRes],
			)

		# calculate 1d xSpec from counts if it's None
		if xSpec is None:
			self._xSpec, _ = np.histogram(xData, xEdges)

		# broadcast xSpec to 2d if it's 1d
		if self._xSpec.ndim == 1:
			self._xSpec = np.broadcast_to(
				self._xSpec.reshape([self._xRes,1]),
				[self._xRes,self._yRes],
			)

	def set_function(self,func):
		self._func = func


	def __call__(self, *parameters):
		return (self._func(self._xMids, self._yMids, *parameters) * self._xSpec).sum(0)




class param_holder(object):
	"""basically just an attribute holder"""

	RESERVED_ATTRIBUTES = [
		
		# global attributes
		"RESERVED_ATTRIBUTES",

		# unprotected method names
		"setup",
		"set_param_attr",
		"set_varied",
		"set_result",
		
		# information about parameter order
		"fixed" ,
		"varied",
		"v_names",
		"f_names",
		"v_index",
		"f_index",

		# ordered parameter values and covariance
		"v_values",
		"f_values",
		"v_opt" ,
		"v_cov" ,
		"v_err" ,

		# fit results
		"chi2"  ,
		"ndof"  ,
		"rchi2" ,
	]
	
	def __init__(self, params, fixed=None, v_names=None, f_names=None):
		self.setup(params, fixed, v_names, f_names)

	def setup(self, params, fixed, v_names, f_names):
		
		# set attributes and values
		for k,v in params.items():
			self.set_param_attr(k,v)

		# same dict of {"name":value} as used by parametrizer
		self.fixed = fixed

		# ordered lists of param names
		# so that the ordered information in opt, cov, etc. can be used
		self.v_names = v_names
		self.f_names = f_names

		if self.v_names is not None:
			self.v_index  = {n:i for i,n in enumerate(self.v_names)}
			self.v_values = [params[_] for _ in self.v_names]
		if self.f_names is not None:
			self.f_index = {n:i for i,n in enumerate(self.f_names)}
			self.f_values = [params[_] for _ in self.f_names]

	def set_param_attr(self,key,value):
		"""deny parameter names found in RESERVED_ATTRIBUTES"""
		if (key in self.RESERVED_ATTRIBUTES) or (key.startswith("_")):
			raise ValueError("illegal parameter name {}".format(key))
		else:
			self.__setattr__(key,value)

	def set_varied(self, v_values, v_cov=None):
		"""Set the values and optionally covariance of varied parameters"""
		# v_values should be a list, and have the same order
		# todo: support dict -> list
		#       can't do that for v_cov though, so maybe not worth it.
		self.v_values = self.v_opt = v_values
		self.v_cov = v_cov
		self.v_err = np.sqrt(np.diag(self.v_cov))

	def set_result(self, chi2, ndof):
		self.chi2 = chi2
		self.ndof = ndof
		self.rchi2 = chi2/ndof




class parametrizer(object):
	"""docstring for parametrizer"""

	def __init__(self, params=None):
		""""""

		# setup internals
		self.clear()

		# support initializing with some parameters
		# equivalent to initializing empty then adding them afterward
		if params is not None:
			self.add_parameters(params)

	def clear(self):
		""""""
		self._names = []
		self._guess = {}

	def add_parameters(self, params):
		""""""

		# string -> no guess, single parameter
		if type(params) is str:
			self._names.append(params)

		# dict -> {"name":default_guess_value or None}
		elif type(params) is dict:
			for name,guess in sorted(params.items()):
				self._names.append(name)
				if guess is not None:
					self._guess[name] = guess

		# set or other iterable -> no guesses
		# sort iterable before adding.
		else:
			for name in sorted(params):
				self._names.append(name)


	def _compose_params(self, pk, fixed, embellish=False):
		params = {}
		k = 0
		for name in self._names:
			# fixed parameter specified by "name":value
			if name in fixed:
				value = fixed[name]
			# varied parameter
			else:
				value = pk[k]
				k += 1
			params[name] = value
		if embellish:
			return param_holder(params, fixed, self._get_names_varied(fixed), self._get_names_fixed(fixed))
		else:
			return param_holder(params)

	def _get_names_varied(self,fixed):
		"""get list of names that are not in fixed"""
		return [_ for _ in self._names if _ not in fixed]

	def _get_names_fixed(self,fixed):
		"""get list of names that are in fixed"""
		return [_ for _ in self._names if _ in fixed]		

	def _wrap_for_curve_fit(self, f, f_args=[], f_kwargs={}, fixed={}):
		"""wrap function f(p, *args, **kwargs) into form where the
		parameters are specified sequentially, for use in optimization."""
		def wrapped(xdata, *pk):
			# compose object with attributes from pk
			p = self._compose_params(pk, fixed)
			return f(xdata, p, *f_args, **f_kwargs)
		return wrapped

	def _wrap_for_approx_fprime(self, f, f_args=[], f_kwargs={}, fixed={}):
		def wrapped(pk):
			p = self._compose_params(pk, fixed)
			return f(p,*f_args,**f_kwargs)
		return wrapped

	def _get_p0_varied_list(self, names_varied):
		return [self._guess.get(_,0.0) for _ in names_varied]

	def get_p0(self, fixed=None):
		...

	def fit_independent_poisson(self):
		...
		# todo: poisson log-likelihood minimizer
		#       this will be much more able to account for low statistics in the y data
		#       but requires a bit more work to get covariances out of.
		#       
		#       for now, just try to keep bin sizes big enough (>10 ok, >20 ideal)


	def curve_fit(self, xdata, ydata, yerr, f, f_args=[], f_kwargs={}, fixed=None, bounds=None, p0=None):
		"""fits f(p, *f_args, **f_kwargs) = ydata, using given yerr as sigma"""

		# ensure type(fixed) is dict
		# None -> empty dict
		if fixed is None:
			fixed = {}

		# convert function to form f([p1, p2, ..., pm], *args, **kwargs)
		# m is the number of varied parameters, which is the total number of
		# parameters registered in self._names, minus the number of
		# fixed parameters.
		f_wrapped = self._wrap_for_curve_fit(f, f_args, f_kwargs, fixed)

		# get list of parameter names which are not fixed
		names_varied = self._get_names_varied(fixed)


		# Compose p0. Resulting object is list.
		# 
		# If p0 is given, it should be a dict of {"name":guess}.
		# Since you don't know the order of the parameters, as an
		# iterable is inappropriate
		if p0 is None:
			p0 = self._get_p0_varied_list(names_varied)

		else:
			p0 = [p0.get(_,self._guess.get(_,0.0)) for _ in names_varied]

		# Compose bounds. Resulting object is [[lo, ...], [hi, ...]]
		# 
		# if given, bounds should be dict of {"name":[lo,hi]}
		# same reasoning as for p0
		NO_BOUNDS = (-np.inf, np.inf)
		if bounds is None:
			bounds = NO_BOUNDS
			# pass
		else:
			bounds = [
				[bounds.get(_,NO_BOUNDS)[0] for _ in names_varied],
				[bounds.get(_,NO_BOUNDS)[1] for _ in names_varied],
			]

		v_opt, v_cov = opt.curve_fit(
			f = f_wrapped,
			xdata = xdata,
			ydata = ydata,
			p0 = p0,
			sigma = yerr,
			absolute_sigma = True,
		)
		param_result = self._compose_params(v_opt, fixed, True)
		param_result.set_varied(v_opt, v_cov)

		y_opt = f_wrapped(xdata, *v_opt)
		y_resid = ydata - y_opt
		y_pull = y_resid / yerr

		chi2  = (y_pull**2).sum()
		ndof  = ydata.size - v_opt.size
		rchi2 = chi2/ndof

		param_result.set_result(chi2, ndof)

		return param_result


	def scalar_num_error_p_only(self, param, f, f_args=[], f_kwargs={}):
		"""Calculate error on the scalar quantity
		f(param, *f_args, **f_kwargs)
		assuming that the only source of error is the covariance of
		the parameters in param."""

		# wrap f(param_holder param, *f_args, **f_kwargs)
		# into f(pk, *f_args, **f_kwargs)
		f_wrapped = self._wrap_for_approx_fprime(f, f_args, f_kwargs, param.fixed)

		# calculate numerical jacobian of f with respect to the varied
		# paramters at param.v_opt
		# todo: make param_holder class have v_values, v_cov, etc.
		#       while being agnostic as to whether they correspond to
		#       the result of an optimization routine.
		f_jac = opt.approx_fprime(param.v_values, f_wrapped, 1.5e-8)
		
		# calculate sigma squared for f using J*sigma*J^t
		f_err_sq = np.matmul(np.matmul(f_jac, param.v_cov), np.transpose(f_jac))

		# return square root of sigma squared
		return f_err_sq ** 0.5

	def vector_num_error_p_only(self, ):
		...

	def scalar_num_error(self, ):
		...

	def vector_num_error(self, ):
		...







if __name__ == "__main__":

	print("testing fit routine")

	def gaus_normalized(x, mu, sigma):
		return ONE_OVER_ROOT_TAU * np.exp(-0.5 * ((x - mu) / sigma)**2) / sigma

	def ftest(p, x):
		# print(p.a, p.b, p.r1, p.r2)
		# return (p.a / (1 + (x - p.r1)**2)) + (p.b / (1 + (x - p.r2)**2))
		return p.bg + p.n1 * gaus_normalized(x, p.mu1, p.sigma1) + p.n2 * gaus_normalized(x, p.mu2, p.sigma2)

	def ftest_model(x, p):
		ymodel = ftest(p,x)
		return ymodel


	# true parameter values and a holder instance for them
	# vtrue = {"a":12,"b":18,"r1":4.25,"r2":13.50}
	vtrue = {"bg":120, "n1":240, "mu1":9, "sigma1":2, "n2":810, "mu2":16, "sigma2":1}
	ptrue = param_holder(vtrue, v_names = sorted(vtrue.keys()), f_names = [])

	# unfair: guess starts at true values
	par = parametrizer(vtrue)

	# make test data
	xtest  = np.linspace(0,20,500)
	ytrue = ftest_model(xtest, ptrue)
	ytest  = np.random.poisson(ytrue)

	# remove zero bins
	min_count = 1
	err_sqrt  = 0
	err_const = 0
	ftr_nz = ytest >= min_count
	xfit = xtest[ftr_nz]
	yfit = ytest[ftr_nz]
	
	# fit
	yerr = np.sqrt(yfit + err_sqrt) + err_const
	popt = par.curve_fit(
		xdata = xfit,
		ydata = yfit,
		yerr  = yerr,
		f = ftest_model,
	)

	pft = lambda fmt,ents,sep=' ':sep.join([fmt.format(_) for _ in ents])
	fmt_par  = '{:>12.6f}'
	fmt_name = '{:>12}'

	print("\npopt")
	print(popt)
	print(popt.f_names)
	print(pft(fmt_name, popt.v_names ))
	print(pft(fmt_name, ptrue.v_values))
	print(pft(fmt_par , popt.v_values))
	print(pft(fmt_par , popt.v_err   ))

	print("\ngoodness of fit")
	print("chi2 / ndof = {:.1f} / {} = {:.4f}".format(popt.chi2, popt.ndof, popt.rchi2))

	print("\ncalculating error on modeled counts")
	ym_opt  = ftest(popt, xtest)
	ym_err  = np.array([par.scalar_num_error_p_only(popt, ftest, f_args = [_]) for _ in xtest])
	ym_pull = (ytrue - ym_opt) / ym_err

	ychi2 = (ym_pull[ftr_nz] ** 2).sum()
	yndof = ftr_nz.sum()
	print("modeled counts vs. truth")
	print("chi2 / ndof = {:.1f} / {} = {:.4f}".format(ychi2, yndof, ychi2/yndof))

	print('\nplotting results')
	plt.step(xtest, ytest, 'k', where='mid', label='data')
	plt.fill_between(xtest, ytest, 0, color='k', alpha=0.1, step='mid')
	plt.plot(xtest, ytrue, 'r-', label='truth')

	plt.plot(xtest, ym_opt, 'g-', label='optimal model')
	plt.fill_between(xtest, ym_opt-ym_err, ym_opt+ym_err, step=None, color='g', alpha=0.25)

	plt.legend()
	plt.show()

	sys.exit(0)
