"""
script used for testing features during development
don't actually use this for anything else
"""

import numpy
import math
import sys
import numpy as np

import utils.model as model
import matplotlib.pyplot as plt

test_fitting = False
if test_fitting:

	test_model = model.line() + model.gaus([None,[5,7],None])
	true_par = [30,270,250,6,1]

	xdata = np.linspace(0,10,250)
	ytrue = test_model(xdata, *true_par)
	# print(ytrue.min(), ytrue.max(), list(ytrue))
	ydata = np.random.poisson(ytrue)

	popt, perr, chi2, ndof = test_model.fit(xdata, ydata)
	print(popt)
	print(test_model.pnames)
	
	plt.plot(xdata,ydata,'k.')
	plt.plot(xdata,test_model(xdata,*popt),'r-')
	plt.title("chisq/dof = {}/{} = {}".format(round(chi2,3), ndof, round(chi2/ndof,3)))
	plt.show()

	print("test_fitting finished without errors")

test_odr_fitting = True
if test_odr_fitting:

	test_model = model.line() + model.exponential()

	# true_par = np.array([8.0, -12.5])#, 0.3, 2.25])
	true_par = np.array([8.0, -2.5, 41.0, -1.00])
	# true_x   = np.array([2.0, 2.25, 2.325, 2.6, 3.4, 3.6, 3.75, 3.76, 4.1, 4.8, 5.8, 7.5])
	true_x = np.random.uniform(0.0, 9.0, 10)
	true_y   = test_model(true_x, *true_par)

	# gaussian errors with sigma propto sqrt(size)
	co_x = 0.05
	co_y = 0.03
	obs_x = true_x + np.random.normal(0.0, np.ones(true_x.shape) * co_x)
	obs_y = true_y + np.random.normal(0.0, np.sqrt(true_y) * co_y)
	est_xerr = np.ones(obs_x.shape) * co_x
	est_yerr = np.sqrt(obs_y) * co_y

	xaxis = np.linspace(0.0, 9.0, 500)
	plt.errorbar(obs_x, obs_y, est_yerr, est_xerr, 'r.', label='observation')

	plt.plot(xaxis, test_model(xaxis, *true_par), color='k', ls='-', marker='')
	plt.plot(true_x, true_y, color='k', ls='', marker='.', label='truth')
	
	# bounds = [[0,20], [-50,50]]#, [0,1], [1,4]]
	bounds = [[0,20], [-50,50], [0,100], [-1,-1]]
	popt, perr, rc2, ndof = model.fit_graph(test_model, obs_x, obs_y, est_xerr, est_yerr, bounds, true_par)

	print(popt)
	print(perr)
	print(rc2)
	print(ndof)

	plt.plot(xaxis, test_model(xaxis, *popt), color='darkred', ls='--', marker='', label='odr result')
	plt.title("\nchi2/ndof = {:.2f}/{} = {:.2f}".format(rc2*ndof, ndof, rc2))

	plt.legend()
	plt.show()




test_multiple_functions = False
if test_multiple_functions:

	e = model.exponential()
	p = model.powc()
	c = model.constant()
	l = model.line()
	q = model.quad()

	ec  = e + c
	el  = e + l
	eec = e + e + c
	qe  = q + e
	pe  = p + e

	print("root function string calls")
	print(eec.rfs())
	print(eec.rfs(3))
	print(eec.rfs(None))
	print(qe.rfs(2))
	print(pe.rfs(4))

	print("\nfunction calls")
	print( ec(5, *[3,0,6]))
	print( ec(1, *[20,1,50]))
	print( el(2, *[1,1,5,10]))
	print(eec(3, *[1,1,4,0.2,5]))
	print( qe(5, *[10, 1, 1, 3, 1/5.]))
	print( pe(2, *[2,3.5,1,4,1]))

	print("\nguess calls")
	test_x = np.linspace(3,4,100)
	test_y = test_x * 0.2 + 3
	print(ec.guess(test_x,test_y))
	print(el.guess(test_x,test_y))
	print(qe.guess(test_x,test_y))
	print(pe.guess(test_x,test_y))

	print("\nvalidation calls")
	test_x_pos  = np.linspace(1,2,5)
	test_x_neg  = np.linspace(-1,-2,5)
	print(qe.val(test_x_pos, *[1,1,1,-1,-1]))
	print(qe.val(test_x_pos, *[1,1,1,-1, 1]))
	print(qe.val(test_x_neg, *[1,1,1, 1,-1]))
	print(qe.val(test_x_neg, *[1,1,1, 1, 1]))
	print(pe.val(test_x_pos, *[1, 2,1,1,1]))
	print(pe.val(test_x_pos, *[1,-2,1,1,1]))
	print(pe.val(test_x_neg, *[1, 2,1,1,1]))
	print(pe.val(test_x_neg, *[1,-2,1,1,1]))

	print("test_multiple_functions finished without errors")

test_individual_functions = False
if test_individual_functions:


	e = model.exponential()
	p = model.powc()
	c = model.constant()
	l = model.line()
	q = model.quad()

	sqrt  = model.sqrt()
	smono = model.smono()
	# eloc  = model.expl()
	power = model.powerlaw()
	gaus  = model.gaus()

	ms_all = [e, sqrt, smono, c, l, q, power, p, gaus]
	ms_inv = [e, sqrt, l, q, power, p]


	print("function calls")
	print(e(5 , *[1, 1/5.0]))
	print(e(1 , *[2, 1.0]))
	print(c(24, *[5]))
	print(c(19, *[666]))
	print(l(0 , *[5, 9]))
	print(l(5 , *[10, 10]))

	print("\ninverse function calls")
	print(e.ifn(math.e ** 2, *[1, 1]))
	print(e.ifn(10         , *[4, 0.5]))
	print(l.ifn(4          , *[1, 1]))
	print(l.ifn(10         , *[1, 0.001]))

	print("\nguess calls")
	test_x = np.linspace(3,4,100)
	test_y = test_x * 7 + 1
	for m in [e,l,c]:
		print(e.guess(test_x, test_y))
		print(l.guess(test_x, test_y))
		print(c.guess(test_x, test_y))

	print("\nvalidation and inverse calls")
	test_y_pos  = np.linspace(1,2,5)
	test_y_neg  = np.linspace(-1,-2,5)
	print(e.ival(test_y_pos, *[3, -0.4]))
	print(e.ival(test_y_neg, *[5, -0.2]))
	print(e.ival(test_y_pos, *[-2, 0.4]))
	print(e.ival(test_y_neg, *[-1, -0.2]))

	print("\nroot function template and inverse requests")
	for m in ms_all:
		print("\nrfs, {}".format(m.name))
		print(m.rfs())
		print(m.rfs(4))
		print(m.rfs(None))
	for m in ms_inv:
		print("\nirfs, {}".format(m.name))
		print(m.irfs())
		print(m.irfs(4))
		print(m.irfs(None))

	print("\ncustom root function formatting")
	for m in ms_all:
		print("\nrfs_custom, {}".format(m.name))
		print(m.rfs_custom())
		print(m.rfs_custom(x="x/[0] - 1", p=1))
		print(m.rfs_custom(x="x/5.0 - 1", p=[1]+[3.5]*(m.npars-1)))

	for m in ms_inv:
		print("\nirfs_custom, {}".format(m.name))
		print(m.irfs_custom())
		print(m.irfs_custom(x="x/[0] - 1", p=1))
		print(m.irfs_custom(x="x/[0] - 1", p=[3.5]*m.npars))

	print("test_individual_functions finished without errors")

test_metamodel = False
if test_metamodel:

	e = model.exponential()
	l = model.line()
	g = model.gaus()

	multi = l+g

	meta = model.metamodel(
		multi,
		xfp=[5, np.array([-1,0]), 5, np.array([0,1]), 8],
		xfp_rfs=[],
		xfx=None,
		)

	print(meta.rfs())
	print(meta(1, 2, 3))

print("all enabled tests finished without errors")
sys.exit(0)
