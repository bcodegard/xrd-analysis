"""
"""

__author__ = "Brunel Odegard"
__version__ = "0.0"


import sys
import os
import math
import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils.fileio  as fileio
import utils.display as display
import utils.model   as model


# greedy map
gmap = lambda fn,it:list(map(fn,it))


FIT_CSV_TYPELIST = [int, str, float, float, int, str, int, int, str, int, str]


if __name__ == '__main__':

	fits = fileio.load_fits("./data/fits/area_xf_src.csv") + fileio.load_fits("./data/fits/lyso_area_xf.csv")
	# example = fits[0]
	# print(example.run)
	# print(example.popt)
	# for key in dir(example):
	# 	print("{:12} = {}".format(key, example.__getattribute__(key)))
	# sys.exit(0)

	runs_nov22 = {3427,3428,3429,3430,3431,3432,3433,3434,3435,3436,3437,3438,3439,3440,3441,3442,3443}
	runs_nov23 = {3444,3445,3446,3447,3448,3449,3450,3451,3452,3453,3454,3455,3456,3457,3458,3459}

	require_id_for_assessing = False

	run1 = 3443
	run2 = 3455
	ch1 = "area_2988_1"
	ch2 = "area_2988_2"

	fits_ch1 = [_ for _ in fits if _.fit_branch == ch1]
	fits_ch2 = [_ for _ in fits if _.fit_branch == ch2]

	# peak IDs to exclude from area correction fit
	# exclude_from_cal = ["l2","l5","l6"]
	exclude_from_cal = []

	# peak to exclude and then assess
	# ['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'x200', 'x203', 'x500', 'x501', 'x601']
	test_peak_id = 601
	test_peak = "x{}".format(test_peak_id) if test_peak_id else ""
	test_peak_x = None
	exclude_from_cal.append(test_peak)

	# todo: switch to using fileio.load_fits
	entries = fileio.load_csv("./data/fits/lyso_area_xf.csv",FIT_CSV_TYPELIST) + fileio.load_csv("./data/fits/area_xf_src.csv",FIT_CSV_TYPELIST)
	
	# extract info from csv entries
	run    = [_[0] for _ in entries]
	branch = [_[1] for _ in entries]

	n_gaus = [_[9] for _ in entries]
	gb_start = 10
	gb_stop  = [10+_*7 for _ in n_gaus]
	gaus_names = [_[gb_start:gb_stop[i]:7] for i,_ in enumerate(entries)]

	chi2  = [float(_[gb_stop[i]+1]) for i,_ in enumerate(entries)]
	ndof  = [float(_[gb_stop[i]+2]) for i,_ in enumerate(entries)]
	n_bgp = [int(  _[gb_stop[i]+3]) for i,_ in enumerate(entries)]
	n_p = [_+3*n_gaus[i] for i,_ in enumerate(n_bgp)]

	p_start = [gb_stop[i]+4   for i,_ in enumerate(n_p)]
	p_stop  = [gb_stop[i]+4+_ for i,_ in enumerate(n_p)]

	bgp   = [_[p_start[i]         :p_start[i]+n_bgp[i]] for i,_ in enumerate(entries)]
	gausp = [_[p_start[i]+n_bgp[i]:p_stop[i]          ] for i,_ in enumerate(entries)]

	bge   = [_[p_stop[i]:p_stop[i]+n_bgp[i]] for i,_ in enumerate(entries)]
	gause = [_[p_stop[i]+n_bgp[i]:] for i,_ in enumerate(entries)]

	gaus_mu     = [gmap(float,_[1::3]) for _ in gausp]
	gaus_mu_err = [gmap(float,_[1::3]) for _ in gause]

	# iterate through entries
	for ibr,br_use in enumerate(sorted(list(set(branch)))):

		x     = {}
		x_err = {}
		y     = {}
		y_err = {}

		ids_found = set()

		for ie,br in enumerate(branch):
			if br != br_use:
				continue

			for ig,g in enumerate(gaus_names[ie]):

				# ignore entries whose ID is in the exclude list
				if g in exclude_from_cal:
					continue

				# ignore non-identified "-"
				if g == "-":
					continue

				# add to set of ids found
				ids_found |= {g}

				if run[ie] in runs_nov22:
					x[g]     = gaus_mu[ie][ig]
					x_err[g] = gaus_mu_err[ie][ig]

				elif run[ie] in runs_nov23:
					y[g]     = gaus_mu[ie][ig]
					y_err[g] = gaus_mu_err[ie][ig]

		seq_ids_use = sorted(ids_found)
		print(seq_ids_use)
		x     = np.array([x    [_] for _ in seq_ids_use])
		x_err = np.array([x_err[_] for _ in seq_ids_use])
		y     = np.array([y    [_] for _ in seq_ids_use])
		y_err = np.array([y_err[_] for _ in seq_ids_use])
		
		xaxis = np.linspace(min(x), max(x), 500)

		plt.plot(xaxis,xaxis*0,ls="--",marker="",color="k",label="y=x")
		plt.errorbar(x, (y-x)/x, y_err/x, x_err, ls="", marker='.', color='g', label="peaks")
		# plt.plot(xaxis,xaxis,ls="--",marker="",color="k",label="y=x")
		# plt.errorbar(x, y, y_err, x_err, ls="", marker='.', color='g', label="peaks")
		
		mod = model.poly2()
		popt, perr, chi2, ndof = mod.fit_with_errors(x,y,x_err,y_err)

		r3 = 3**0.5
		center = (mod(xaxis,*(popt        ))-xaxis)/xaxis
		high   = (mod(xaxis,*(popt+perr/r3))-xaxis)/xaxis
		low    = (mod(xaxis,*(popt-perr/r3))-xaxis)/xaxis
		# center = (mod(xaxis,*(popt        )))
		# high   = (mod(xaxis,*(popt+perr/r3)))
		# low    = (mod(xaxis,*(popt-perr/r3)))

		plt.plot(xaxis, center, "r-", label="best fit")
		plt.plot(xaxis, high  , color="darkred", ls="--", label="\xb1 err")
		plt.plot(xaxis, low   , color="darkred", ls="--")

		plt.legend()
		plt.xlabel("run{}".format(run1))
		plt.ylabel("(run{} - run{})/run{}".format(run2,run1,run1))
		# plt.ylabel("run {}".format(run2))
		plt.title("area transformation, {}\nquadratic, chisq/dof = {:.2f}/{:.2f} = {:.3f}".format(
			br_use,
			chi2,
			ndof,
			chi2/ndof
		))
		plt.show()

		# use fits to do stuff...
		fits_use = fits_ch1 if br_use == ch1 else fits_ch2

		mu_x     = []
		mu_err_x = []
		mu_y     = []
		mu_err_y = []

		for f in fits_use:

			mu     = f.popt_gaus[1::3]
			mu_err = f.perr_gaus[1::3]

			# if in x data, search for peak
			if f.run in runs_nov22:
				for ign,gn in enumerate(f.gaus_names):
					if gn == test_peak:
						test_peak_x = mu[ign]

			if require_id_for_assessing:
				has_id = [_!="-" for _ in f.gaus_names]
				mu     = [_ for i,_ in enumerate(mu    ) if has_id[i]]
				mu_err = [_ for i,_ in enumerate(mu_err) if has_id[i]]
			
			if f.run in runs_nov22:
				mu_x     += mu
				mu_err_x += mu_err

			elif f.run in runs_nov23:
				mu_y     += mu
				mu_err_y += mu_err

		mu_x     = np.array(mu_x    )
		mu_err_x = np.array(mu_err_x)
		mu_y     = np.array(mu_y    )
		mu_err_y = np.array(mu_err_y)

		mu_y_to_x = mod.ifn(mu_y, *popt)
		mu_x_to_y = mod.fn( mu_x, *popt)

		# print(mu_err_x)
		# print(mu_x)
		# print(mu_y_to_x)
		# print("")
		# print(mu_err_y)
		# print(mu_y)
		# print(mu_x_to_y)
		# print("")

		xaxis = np.linspace(mu_x.min(), mu_x.max(), 500)

		plt.plot(xaxis, xaxis*0.0, ls="--", marker="", color="k", label="y=x")

		if test_peak:
			plt.axvline(test_peak_x, color='r', ls=':', label="test peak")

		plt.errorbar(
			mu_x,
			(mu_y-mu_x)/mu_x,
			mu_err_y/mu_x,
			mu_err_x,
			ls="", marker=".", color="darkred", label="Nov 23"
		)
		plt.errorbar(
			mu_x,
			(mu_y_to_x-mu_x)/mu_x,
			mu_err_y/mu_x,
			mu_err_x,
			ls="", marker=".", color="lime", label="nov 23 -> nov 22"
		)
		plt.errorbar(
			mu_x,
			(mu_y-mu_x_to_y)/mu_x,
			mu_err_y/mu_x,
			mu_err_x,
			ls="", marker=".", color="c", label="nov 23 <- nov 22"
		)

		if test_peak:
			sources = {
				0:"None",
				1:"Cs137",
				2:"Ba133",
				3:"Na22",
				4:"Mn54",
				5:"Cd109",
				6:"Co57",
				7:"Co60",
				8:"Pb210",
				9:"Am241",
			}
			title="test peak id={} ({})".format(test_peak_id, sources[test_peak_id//100])
		else:
			title="effect of area transformation"

		plt.xlabel("Nov 22")
		plt.ylabel("Nov 23")
		plt.title("{}\n{}".format(title,br_use))
		plt.legend()
		plt.show()



