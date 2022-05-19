""""""

__author__ = "Brunel Odegard"
__version__ = "0.1"


import math
import numpy as np
import scipy.optimize    as opt




# math constants
ROOT_8_LOG_2 = (8 * math.log(2)) ** 0.5
ONE_OVER_ROOT_TAU = 1 / (2 * math.pi)

# convenience functions
# 
# comparisons for floats
iseq = lambda f1,f2,eps=1e-9:abs(f1-f2)<eps
# convenient way to discard values which are zero or negative
positive = lambda ar:ar[ar>0]




def transform_y(f_x_y, xf_and_dxf_dyprime):
	"""Transform f_x_y(x,y) into f_x_yprime(x,yprime)
	xf is the inverse transformation: from yprime to y
	y = xf(yprime, *xf_params)
	dxf_dyprime is its derivative with respect to yprime"""
	def f_x_yprime(x,yprime,p):
		xf, dxf_dyprime = xf_and_dxf_dyprime(yprime, p)
		return f_x_y(x, xf, p) * dxf_dyprime
	return f_x_yprime

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
	def gaus(x_E,y_A,pm):
		mu,sigma = E_to_mu_sigma(x_E,pm)
		return ONE_OVER_ROOT_TAU * np.exp(-0.5*((y_A - mu)/sigma)**2) / sigma
	return gaus

def transformed_gaus_spread(E_to_mu_sigma, xf_and_dxf):
	"""Conventienly compose and return a transformed gaussian distributor.
	returns f(E, A') where f(E,A) is gaussian in A with mu,sigma functions of E
	and t(A') = A
	"""
	f_E_A = gaus_spread(E_to_mu_sigma)
	f_E_Aprime = transform_y(f_E_A, xf_and_dxf)
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

	def __init__(self, func, xMids=None, xWidth=None, yMids=None, yWidth=None, xSpec=None, xEdges=None, yEdges=None, xData=None):
		
		# function f(x,y,*p)
		self._func = func

		# don't copy, so that if multiple instances of the class are
		# needed, you can pass the same arrays to each and not duplicate
		self._xMids  = xMids
		self._xWidth = xWidth
		self._yMids  = yMids
		self._yWidth = yWidth
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
			self._xWidth = xEdges[1:] - xEdges[:-1]
		self._xRes = self._xMids.shape[0]

		# generate 1d yMids if it's None
		if self._yMids is None:
			self._yMids = 0.5*(yEdges[1:]+yEdges[:-1])
			self._yWidth = yEdges[1:] - yEdges[:-1]
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

		# broadcast xWidth to 2d if it's 1d
		if self._xWidth.ndim == 1:
			self._xWidth = self._xWidth.reshape([self._xRes,1])
		# broadcast yWidth to 2d if it's 1d
		if self._yWidth.ndim == 1:
			self._yWidth = self._yWidth.reshape([1, self._yRes])

		self._xWidth *= self._xWidth.shape[0] / np.sum(self._xWidth[:,0])

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


	def __call__(self, parameters, xSpec=None):
		if xSpec is None:
			xSpec=self._xSpec
		if xSpec.ndim == 1:
			xSpec = xSpec.reshape([self._xRes,1])
		return (self._yWidth * self._func(self._xMids, self._yMids, parameters) * (xSpec)).sum(0)
		# return (self._func(self._xMids, self._yMids, parameters) * xSpec).sum(0)




class param_manager(object):
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

	chi2=None
	ndof=None
	rchi2=None
	
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
			raise ValueError("{} is not a parameter".format(key))
		else:
			self.__setattr__(key,value)

	def get_param_attr(self,key):
		if (key in self.RESERVED_ATTRIBUTES) or (key.startswith("_")):
			raise ValueError("{} is not a parameter".format(key))
		else:
			return self.__getattribute__(key)

	def __getitem__(self,key):
		return self.get_param_attr(key)

	def __setitem__(self,key,value):
		return self.set_param_attr(key,value)

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
			return param_manager(params, fixed, self._get_names_varied(fixed), self._get_names_fixed(fixed))
		else:
			return param_manager(params)

	def _get_names_varied(self,fixed):
		"""get list of names that are not in fixed"""
		return [_ for _ in self._names if _ not in fixed]

	def _get_names_fixed(self,fixed):
		"""get list of names that are in fixed"""
		return [_ for _ in self._names if _ in fixed]		

	def _wrap_for_curve_fit(self, f, f_valid, f_args=[], f_kwargs={}, fixed={}):
		"""wrap function f(p, *args, **kwargs) into form where the
		parameters are specified sequentially, for use in optimization."""
		def wrapped(xdata, *pk):
			# compose object with attributes from pk
			p = self._compose_params(pk, fixed)
			return f(xdata, p, *f_args, **f_kwargs)[f_valid]
		return wrapped

	def _wrap_for_approx_fprime(self, f, f_args=[], f_kwargs={}, fixed={}):
		def wrapped(pk):
			p = self._compose_params(pk, fixed)
			return f(p,*f_args,**f_kwargs)
		return wrapped

	def _get_p0_varied_list(self, names_varied):
		return [self._guess.get(_,0.0) for _ in names_varied]

	def get_p0(self, fixed={}):
		return self._compose_params(self._get_p0_varied_list(self._get_names_varied(fixed)),fixed,embellish=True)

	def fit_independent_poisson(self):
		...
		# todo: poisson log-likelihood minimizer
		#       this will be much more able to account for low statistics in the y data
		#       but requires a bit more work to get covariances out of.
		#       
		#       for now, just try to keep bin sizes big enough (>10 ok, >20 ideal)


	def curve_fit(self, xdata, ydata, yerr, f, f_args=[], f_kwargs={}, fixed=None, bounds=None, p0=None, xerr=None):
		"""fits f(xdata, p, *f_args, **f_kwargs) = ydata, using given yerr as sigma"""

		# ensure type(fixed) is dict
		# None -> empty dict
		if fixed is None:
			fixed = {}

		# create filter for positive (valid) yerr
		# and apply it to y data
		f_valid = (yerr > 0)
		ydata = ydata[f_valid]
		yerr  = yerr[ f_valid]
		# don't apply to xdata as it doesn't have same shape
		# if type(xdata) is np.ndarray:
		# 	xdata = xdata[f_valid]

		# convert function to form f([p1, p2, ..., pm], *args, **kwargs)
		# m is the number of varied parameters, which is the total number of
		# parameters registered in self._names, minus the number of
		# fixed parameters.
		f_wrapped = self._wrap_for_curve_fit(f, f_valid, f_args, f_kwargs, fixed)

		# get list of parameter names which are not fixed
		names_varied = self._get_names_varied(fixed)

		# Compose p0. Resulting object is list.
		# 
		# If p0 is given, it should be a dict of {"name":guess}.
		# Any values not given in p0 will be supplied internally.
		# 
		# Since you don't know the order of the parameters, as an
		# iterable is inappropriate.
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
		else:
			bounds = [
				[bounds.get(_,NO_BOUNDS)[0] for _ in names_varied],
				[bounds.get(_,NO_BOUNDS)[1] for _ in names_varied],
			]

		# perform the optimization
		v_opt, v_cov = opt.curve_fit(
			f      = f_wrapped,
			xdata  = xdata,
			ydata  = ydata,
			sigma  = yerr,
			absolute_sigma = True,
			p0     = p0,
			bounds = bounds,
		)
		param_result = self._compose_params(v_opt, fixed, True)
		param_result.set_varied(v_opt, v_cov)

		y_opt = f_wrapped(xdata, *v_opt)
		if xdata is None:
			y_opt_err = self.vector_num_error_p_only(param_result, f, xdata, f_args, f_kwargs)
		else:
			y_opt_err = self.vector_num_error_p_xdiag(param_result, f, xdata, xerr, f_args, f_kwargs)

		y_resid = ydata - y_opt
		y_resid_err = np.sqrt(yerr**2 + y_opt_err**2)
		y_pull = y_resid / y_resid_err

		chi2  = (y_pull**2).sum()
		ndof  = ydata.size - v_opt.size
		rchi2 = chi2/ndof

		param_result.set_result(chi2, ndof)

		# todo: class for fit results
		return param_result, y_opt, y_opt_err


	def vector_df_dp(self, param, f, xdata=None, f_args=[], f_kwargs={}, eps=1e-4, rel_eps=True):
		"""approximate the derivative of f with respect to parameters p"""
		f_p = f(xdata, param, *f_args, **f_kwargs)
		df_dp = []

		# print('param properties')
		# print(param.v_names)
		# print(param.v_index)
		# print(param.v_values)

		for ip,p in enumerate(param.v_names):
			# remember initial value so we can return it after varying
			p_initial = param[p]
			# determine amount to vary parameter
			this_eps = eps.get(p, 1e-6) if type(eps) is dict else eps
			delta_p = param[p] * this_eps if rel_eps else this_eps
			
			# calculate f with plus and minus delta_p
			param[p] = p_initial + delta_p
			f_p_plus_dp = f(xdata, param, *f_args, **f_kwargs)
			# param[p] = p_initial - delta_p
			# f_p_minus_dp = f(xdata, param, *f_args, **f_kwargs)
			
			# return to initial value
			param[p] = p_initial
			
			# calculate df/dp and append
			# delta_f = (f_p_plus_dp - f_p_minus_dp) / 2.0
			delta_f = (f_p_plus_dp - f_p)
			df_dp.append(delta_f / delta_p)

			# print("{:>2} {:>12} : {:>12.4f} (+ {:>8.4f}) -> {}".format(ip, p, param[p], delta_p, list(df_dp[-1][:4])))
			
		return np.stack(df_dp, axis=1)

	def vector_df_dx(self, param, f, xdata, f_args=[], f_kwargs={}, eps=1e-4, rel_eps=True):
		"""approximate the derivative of f with respect to xdata"""
		f_x = f(xdata, param, *f_args, **f_kwargs)
		df_dx = []

		for i,xi in enumerate(xdata):

			xi_initial = xdata[i]
			delta_xi = eps * xdata[i] if (rel_eps and xdata[i]) else eps

			# if delta_xi > 0:
			xdata[i] = xi_initial + delta_xi
			f_x_plus_dx = f(xdata, param, *f_args, **f_kwargs)
			xdata[i] = xi_initial
			df_dx.append((f_x_plus_dx - f_x) / delta_xi)

			# else:
				# df_dx.append(np.zeros(f_x.shape))

		return np.stack(df_dx, axis=1)


	def vector_num_error_p_only(self, param, f, xdata=None, f_args=[], f_kwargs={}, eps=1e-4, rel_eps=True):
		"""calculate approximate numerical error on vector valued
		function f(xdata, param, *f_args, **f_kwargs) with respect to
		param, at the given value of param and its covariance."""

		# calculate jacobian of f with respect to param
		f_jac = self.vector_df_dp(param, f, xdata, f_args, f_kwargs, eps, rel_eps)

		# calculate covariance of f using J * COV * J^t
		f_cov = np.matmul(np.matmul(f_jac, param.v_cov), np.transpose(f_jac))

		# print("shapes for jac, cov, jac*cov*jac^t")
		# print(f_jac.shape)
		# print(param.v_cov.shape)
		# print(f_err_sq.shape)
		
		# return error as sqrt(diag(cov))
		return np.sqrt(np.diag(f_cov))


	def vector_num_error_p_xdiag(self, param, f, xdata, xerr=None, f_args=[], f_kwargs={}, eps=1e-4, rel_eps=True):
		"""calculate approximate numerical error on vector valued
		function f(xdata, param, *f_args, **f_kwargs) with respect to
		param and xdata.
		Covariance for param is assumed to be supplied as param.v_cov.
		Covariance for xdata is assumed to be diagonal, and if supplied,
		should be supplied as a 1d array, xerr = sqrt(diag(xvoc)).
		If not supplied, is calculated as sqrt(xdata.)"""
		
		# calculate error contribution from param
		# f_err_param = self.vector_num_error_p_only(param, f, xdata, f_args, f_kwargs, eps, rel_eps)
		f_jac_p = self.vector_df_dp(param, f, xdata, f_args, f_kwargs, eps, rel_eps)
		f_err_p_squared = np.diag(np.matmul(np.matmul(f_jac_p, param.v_cov), np.transpose(f_jac_p)))

		# calculate xerr if not specified
		if xerr is None:
			xerr = np.sqrt(xdata)

		# calculate error contribution from xdata
		f_jac_x = self.vector_df_dx(param, f, xdata, f_args, f_kwargs, eps, rel_eps)
		f_err_x_squared = np.diag(np.matmul(f_jac_x * xerr[None,:], np.transpose(f_jac_x)))

		# return error as sum in quadrature
		# print("err_p_sq", list(f_err_p_squared[:20]))
		# print("err_x_sq", list(f_err_x_squared[:20]))
		return np.sqrt(f_err_p_squared + f_err_x_squared)


	def scalar_num_error_p_only(self, param, f, f_args=[], f_kwargs={}):
		"""Calculate error on the scalar quantity
		f(param, *f_args, **f_kwargs)
		assuming that the only source of error is the covariance of
		the parameters in param."""

		# wrap f(param_manager param, *f_args, **f_kwargs)
		# into f(pk, *f_args, **f_kwargs)
		f_wrapped = self._wrap_for_approx_fprime(f, f_args, f_kwargs, param.fixed)

		# calculate numerical jacobian of f with respect to the varied
		# paramters at param.v_opt
		# todo: make param_manager class have v_values, v_cov, etc.
		#       while being agnostic as to whether they correspond to
		#       the result of an optimization routine.
		f_jac = opt.approx_fprime(param.v_values, f_wrapped, 1.5e-8)
		
		# calculate sigma squared for f using J*sigma*J^t
		f_err_sq = np.matmul(np.matmul(f_jac, param.v_cov), np.transpose(f_jac))

		# return square root of sigma squared
		return f_err_sq ** 0.5







if __name__ == "__main__":
	import matplotlib.pyplot as plt

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
	ptrue = param_manager(vtrue, v_names = sorted(vtrue.keys()), f_names = [])

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
