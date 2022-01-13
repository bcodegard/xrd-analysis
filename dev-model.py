"""
script used for testing features during development
don't actually use this for anything else
"""

import numpy
import math
import numpy as np

import utils.model as model
import matplotlib.pyplot as plt

test_fitting = False
if test_fitting:

	test_model = model.line() + model.gaus([None,[5,7],None])
	true_par = [3000,-30,250,6,1]

	xdata = np.linspace(0,10,250)
	ydata = np.random.poisson(test_model(xdata, *true_par))

	popt, perr, chi2, ndof = test_model.fit(xdata, ydata)
	print(popt)
	print(test_model.pnames)
	
	plt.plot(xdata,ydata,'k.')
	plt.plot(xdata,test_model(xdata,*popt),'r-')
	plt.title("chisq/dof = {}/{} = {}".format(round(chi2,3), ndof, round(chi2/ndof,3)))
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
	print(ec(5, [3,0,6]))
	print(ec(1, [20,1,50]))
	print(el(2, [1,1,5,10]))
	print(eec(3, [1,1,4,0.2,5]))
	print(qe(5, [10, 1, 1, 3, 1/5.]))
	print(pe(2, [2,3.5,1,4,1]))

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
	print(qe.val(test_x_pos, [1,1,1,-1,-1]))
	print(qe.val(test_x_pos, [1,1,1,-1, 1]))
	print(qe.val(test_x_neg, [1,1,1, 1,-1]))
	print(qe.val(test_x_neg, [1,1,1, 1, 1]))
	print(pe.val(test_x_pos, [1, 2,1,1,1]))
	print(pe.val(test_x_pos, [1,-2,1,1,1]))
	print(pe.val(test_x_neg, [1, 2,1,1,1]))
	print(pe.val(test_x_neg, [1,-2,1,1,1]))

test_individual_functions = True
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

