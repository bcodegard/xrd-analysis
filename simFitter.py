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
	def __init__(self, params):
		for k,v in params.items():
			self.__setattr__(k,v)

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


	def _compose_params(self, pk, fixed):
		params = {}
		k = 0
		for name in self._names:
			# fixed parameter specified by "name":value
			if name in fixed:
				value = fixed[name]
			# free parameter
			else:
				value = pk[k]
				k += 1
			params[name] = value
		return param_holder(params)

	def _get_free_names(self,fixed):
		"""get list of names that are not in fixed"""
		return [_ for _ in enumerate(self._names) if _ not in fixed]

	def _wrap(self, f, fixed):
		"""wrap function f(p, *args, **kwargs) into for where the
		parameters are specified sequentially, for use in optimization."""

		def wrapped(pk, *args, **kwargs):

			# compose object with attributes from pk
			p = self._compose_params(pk, fixed)

			return f(p, *args, **kwargs)

		return wrapped

	def minimize(self, f, f_args=[], f_kwargs={}, fixed=None, bounds=None, p0=None):
		""""""

		# ensure type(fixed) is dict
		# None -> empty dict
		if fixed is None:
			fixed = {}

		# convert function to form f([p1, p2, ..., pm], *args, **kwargs)
		# m is the number of free parameters, which is the total number of
		# parameters registered in self._names, minus the number of
		# fixed parameters.
		f_wrapped = self._wrap(f, fixed)

		# get list of parameter names which are not fixed
		names_free = self._get_free_names(fixed)


		# Compose p0. Resulting object is list.
		# 
		# If p0 is given, it should be a dict of {"name":guess}.
		# Since you don't know the order of the parameters, as an
		# iterable is inappropriate
		if p0 is None:
			p0 = [self._guess.get(_,0.0) for _ in names_free]
		else:
			p0 = [p0.get(_,self._guess.get(_,0.0)) for _ in names_free]

		# Compose bounds. Resulting object is [[lo, ...], [hi, ...]]
		# 
		# if given, bounds should be dict of {"name":[lo,hi]}
		# same reasoning as for p0
		NO_BOUNDS = (-np.inf, np.inf)
		if bounds is None:
			bounds = NO_BOUNDS
		else:
			bounds = [
				[bounds.get(_,NO_BOUNDS)[0] for _ in names_free],
				[bounds.get(_,NO_BOUNDS)[1] for _ in names_free],
			]

		result = opt.least_squares(
			f_wrapped, 
			p0,
			bounds = bounds,
			args=f_args,
			kwargs=f_kwargs,
		)

		# calculate stuff based on assuming that pulls are unit norm
		chi2  = (result.fun ** 2).sum()
		ndof  = result.fun.size - result.x.size
		rchi2 = chi2/ndof
		hess  = np.matmul(np.transpose(result.jac), result.jac)
		cov   = np.linalg.inv(hess) * rchi2

		result.chi2  = chi2
		result.ndof  = ndof
		result.rchi2 = rchi2
		result.hess  = hess
		result.xcov  = cov

		return self._compose_params(result.x, fixed), names_free, result




		






# def f(p, *args, **kwargs):

# 	# p is an object with attributes for each parameter
# 	# some may be fixed, some may be bound, some may be free
# 	# parametrizer class takes care of order and all that jass

# 	# *args and **kwargs contain other variables not being optimized for
# 	# for instance, if including some sort of "x data" equivalient, it would
# 	# need to go in args or kwargs.

# 	# do stuff with p.attributes
# 	p.gamma
# 	p.rho_p
# 	p.res_s_a
# 	p.res_s_b
# 	p.am241_n_bg
# 	p.am241_n_p1
# 	p.am241_n_p2

# 	# return pulls = residuals / uncertainties
# 	# since the least_squares method assumes equal (unit) weight on each
# 	return pulls

def ftest(p, xtest, ytest):

	ftr_pos = ytest>0
	x = xtest[ftr_pos]
	y = ytest[ftr_pos]
	yerr = np.sqrt(y)

	ymodel = (x + p.r1) * (x + p.r2) * p.c
	pulls = (y - ymodel) / yerr

	return pulls




par = parametrizer({"r1":1,"r2":1,"c":10})

xtest  = np.linspace(0,20,500)
ytest  = np.random.poisson((xtest+2)*(xtest+17)*25)
f_args = [xtest, ytest]
popt, order, result = par.minimize(ftest, f_args = f_args)

print("\nresults")
print(popt)
print(order)
print(result.x)
print(result.cost)
print(result.optimality)

print("\nextra results")
print(result.chi2)
print(result.ndof)
print(result.rchi2)
print(np.sqrt(np.diag(result.xcov)))
print(result.xcov)

print("\ncalculating chi2, etc.")
pulls = ftest(popt, *f_args)
chi2 = (pulls**2).sum()
ndof = np.count_nonzero(ytest) - result.x.size
print("chi2: {:.2f}".format(chi2))
print("ndof: {}".format(ndof))
print("chi2/ndof: {:.4f}".format(chi2/ndof))


sys.exit(0)


if __name__ == "__main__":

	print("testing classes")

	res=300

	xEdges = np.linspace(0,1,res+1)
	xMids = 0.5*(xEdges[1:] + xEdges[:-1])
	yEdges = np.linspace(0,10,res+1)
	xSpec  = np.ones(res)

	bp = binned_projector(
		lambda x,y,a,b:b/(1+(a*x - y)**2),
		xEdges=xEdges,
		yEdges=yEdges,
		xSpec=xSpec,
	)

	plt.step(
		xMids,
		bp(5,res),
		'b',
		where='mid',
	)
	plt.show()
