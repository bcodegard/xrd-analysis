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

class peak(object):
	"""attribute holder for peak information"""
	def __init__(self, **kwargs):
		for key,value in kwargs.items():
			self.__setattr__(key,value)
		if "sf" in kwargs.keys():
			self.e_err = 0.5 * (10**(-self.sf))
		else:
			self.e_err = 0.0

		if 'name' not in kwargs.keys():
			self.name = "{} {}".format(sources[self.src], self.e)

		if 'desc' not in kwargs.keys():
			self.desc = ""
peaks = {
	# -1: unidentified
	-1: peak(i=-1,src=0,e=0,),
	
	# 000 - 099 : no associated source
	
	# 100 - 199 : cs137
	100: peak(i=100, e=32.19  , sf=2, src=1, name="137Cs Ka", desc="Ka conversion line for Cs137"),
	101: peak(i=101, e=661.657, sf=2, src=1, name="137Cs gamma", desc=""),

	# 200 - 299 : Ba133
	200:peak(i = 200, e =  30.8510, sf=4, src=2),
	201:peak(i = 201, e =  53.1622, sf=4, src=2),
	202:peak(i = 202, e =  79.6142, sf=4, src=2),
	203:peak(i = 203, e =  80.9979, sf=4, src=2),
	204:peak(i = 204, e = 160.6121, sf=4, src=2),
	205:peak(i = 205, e = 223.237 , sf=3, src=2),
	206:peak(i = 206, e = 276.3992, sf=4, src=2),
	207:peak(i = 207, e = 302.8512, sf=4, src=2),
	208:peak(i = 208, e = 356.0134, sf=4, src=2),
	209:peak(i = 209, e = 383.8491, sf=4, src=2),

	# 300 - 399 : Na22
	300:peak(i=300, e=511.0, sf=1, src=3),

	# 400 - 499 : Mn54
	400:peak(i=400, e=239.   , sf=0, src=4),
	401:peak(i=401, e=834.848, sf=3, src=4),

	# 500 - 599 : Cd109
	500:peak(i=500, e= 22.1 , sf=1, src=5),
	501:peak(i=501, e= 88.04, sf=2, src=5),

	# 600 - 699 : Co57
	600:peak(i=600, e=  14.41300, sf=5, src=6),
	601:peak(i=601, e= 122.0614 , sf=4, src=6),
	602:peak(i=602, e= 136.4743 , sf=4, src=6),
	603:peak(i=603, e= 230.29   , sf=2, src=6),
	604:peak(i=604, e= 339.54   , sf=2, src=6),
	605:peak(i=605, e= 352.36   , sf=2, src=6),
	606:peak(i=606, e= 366.75   , sf=2, src=6),
	607:peak(i=607, e= 569.92   , sf=2, src=6),
	608:peak(i=608, e= 692.03   , sf=2, src=6),
	609:peak(i=609, e= 706.40   , sf=2, src=6),

	# 700 - 799 : Co60
	700:peak(i=700, e=346.93, sf=2, src=7),
	701:peak(i=701, e=826.06, sf=2, src=7),

	# 800 - 899 : Pb210
	800:peak(i=800, e=46.539, sf=3, src=8),

	# 900 - 999 : Am241
	900:peak(i=900, e=59.5412, sf=4, src=9),
	901:peak(i=901, e=26.3448, sf=4, src=9),
}


def main(

		test_spectra,
		branch,

		energy_model,
		ec_results,

		area_xf,

		exclude_source_peaks=False,
		require_id=False,

		colors=False,
		labels={},
		xlabel = False,
		ylabels = [],
		xax = False,

		plot_area = False,
		plot_area_t1 = False,
		plot_energy = False,
		plot_area_xf = False,

		show=True,

		):

	# test peaks
	area, _, area_err, _, peak_id = get_peaks(test_spectra, branch, exclude_source_peaks, require_id)

	if area_xf is None:
		area_t1     = area
		area_err_t1 = area_err

	else:

		# convert area, area_err from t2 (test peak dataset) to t1 (source dataset) using area xf
		assert branch == area_xf.branch

		if area_xf.order == 0:
			area_xf_model = model.poly1(bounds=[[1,1],[-np.inf,np.inf]])
		elif area_xf.order == 1:
			area_xf_model = model.poly1()
		elif area_xf.order == 2:
			area_xf_model = model.poly2()

		# assume order == 1 for now
		# invert parameters and covariance
		# need to update models to handle jacobians, inverses, error calculations, etc.
		print("")
		b = area_xf.opt[0]
		c = area_xf.opt[1]
		opt_inv = np.array([1/b, -c/b])
		print(b,c)
		print(opt_inv)
		jinv = np.array([[-1/(b**2), 0],[c/(b**2), -1/b]])
		print(jinv)
		cov_inv = np.matmul(jinv, np.matmul(area_xf.cov, jinv.swapaxes(0,1)))
		print(cov_inv)

		# replace area_xf opt and cov with inverse operation's values
		area_xf.opt = opt_inv
		area_xf.cov = cov_inv


		print("")
		print(area_xf.order)
		print(area_xf.opt)
		print(area_xf.cov)
		print("")

		area_t1     = []
		area_err_t1 = []
		line_template = "{:>4} | {:>10.2f} | {:>10.2f} | {:>10.2f} | {:>10.2f} | {}"
		print("a2 -> a1")
		for i,this_area in enumerate(area):
			this_area_err = area_err[i]

			# best value for A(t1) for this peak
			this_area_t1 = area_xf_model(this_area, *area_xf.opt)

			# jacobian for this peak
			this_jac = area_xf_model.jacobian(this_area, *area_xf.opt)

			# j[0] is for A(t2)
			# No covariance between A(t2) and the optimal xf paramters
			v00 = (this_area_err**2) * (this_jac[0]**2)

			# j[1:] is for parameters *q of area xf
			if area_xf.order == 0:
				xf_jac = this_jac[2:]
			else:
				xf_jac = this_jac[1:]
			vii = np.dot(xf_jac, np.matmul(area_xf.cov, xf_jac))

			# print(this_area_err, this_jac[0], v00**0.5, vii**0.5)
			this_area_err_t1 = (v00 + vii) ** 0.5

			area_t1.append(this_area_t1)
			area_err_t1.append(this_area_err_t1)

			print(line_template.format(peak_id[i],this_area,this_area_err,this_area_t1,this_area_err_t1,this_jac))
		print("")

	area = np.array(area)
	area_err = np.array(area_err)
	area_t1 = np.array(area_t1)
	area_err_t1 = np.array(area_err_t1)
	# if plot_area_xf:
	# 	plt.errorbar(area, area_t1-area, area_err_t1, area_err, ls="", color="k", marker=".")
	# 	plt.xlabel("area before xf")
	# 	plt.ylabel("delta from xf")
	# 	plt.show()

	# if plot_area_t1:
	# 	plt.errorbar(range(len(area_t1)), area_t1, area_err_t1, color='k', ls='', marker='.')
	# 	plt.xlabel("peak, entry number")
	# 	plt.ylabel("transformed peak area (pVs)")
	# 	plt.show()

	# convert a(t2), aerr(t2) to E, Eerr

	energy     = []
	energy_err = []

	popt_ec, perr_ec, chi2_ec, ndof_ec, cov_ec, maxdev_ec = ec_results
	print("a1 -> E")

	line_template = "{:>4} | {:>9.2f} \xb1 {:<7.2f} | {:>9.2f} \xb1 {:<7.2f} | {:>6.2f} \xb1 {:.2f} \xb1 {:<4.2f} | {}"
	for i,this_a1 in enumerate(area_t1):
		this_a1_err = area_err_t1[i]

		# best value for E for this peak
		this_energy = energy_model(this_a1, *popt_ec)

		# jacobian
		this_jac = energy_model.jacobian(this_a1, *popt_ec)

		v00 = (this_a1_err**2) * (this_jac[0]**2)

		xf_jac = this_jac[1:]
		vii = np.dot(xf_jac, np.matmul(cov_ec, xf_jac))

		this_energy_err = (v00 + vii)**0.5

		energy.append(this_energy)
		energy_err.append(this_energy_err)

		this_energy_stat_err = this_energy * maxdev_ec

		print(line_template.format(peak_id[i],area[i],area_err[i],this_a1,this_a1_err,this_energy,this_energy_err,this_energy_stat_err,this_jac))

	energy     = np.array(energy    )
	energy_err = np.array(energy_err)

	# if plot_energy:
	# 	plt.errorbar(range(len(energy)), energy, energy_err, color='k', ls='', marker='.')
	# 	plt.xlabel("peak, entry number")
	# 	plt.ylabel("peak energy (KeV)")
	# 	plt.show()

	ax1 = plt.subplot(1,3,1)
	ax2 = plt.subplot(1,3,2, sharex=ax1, sharey=ax1)
	ax3 = plt.subplot(1,3,3, sharex=ax2, )

	for i,(data, err, handle) in enumerate([(area, area_err, "area (pVs)"), (area_t1, area_err_t1, "transformed area (pVs)"), (energy, energy_err, "energy (KeV)")]):
		plt.subplot(1,3,i+1)
		if plot_area:
			colors_done = set()
			if type(colors) in (list,tuple):
				for ic,col in enumerate(colors):
					plt.errorbar(ic, data[ic], err[ic], color=col, ls='', marker='.', label=labels[len(colors_done)] if not (col in colors_done) else "")
					plt.xlabel("peak, entry number")
					plt.ylabel("peak {}".format(handle))
					colors_done |= {col}
			else:
				if colors:
					col = colors
					label = labels
				else:
					col = 'k'
					label = None

				if not xax:
					xax = range(len(data))
				plt.errorbar(xax, data, err, color=col, ls='', marker='.', label=label)
				plt.xlabel(xlabel if xlabel else "entry number")
				plt.ylabel((ylabels[i] if ylabels else "{}").format(handle))
	
	if show:
		plt.legend()
		plt.show()

	print("")
	print("Those are the real, final, statistical errors on E!")
	print("")

	energy_err_total = np.sqrt(energy_err**2 + (maxdev_ec * energy)**2)
	print(energy_err)
	print(energy_err_total)




def get_peaks(spectra, branch, exclude_source_peaks=False, require_id=True):
	area   = []
	energy = []
	area_err   = []
	energy_err = []
	is_incl = []
	peak_id = []

	if exclude_source_peaks is False:
		exclude_source_peaks = []

	for spec in spectra:

		if spec.fit_branch != branch:
			continue

		for ig,g in enumerate(spec.gaus_names):

			if require_id:
				if (g=="-"):
					continue
				if g.startswith("e"):
					continue
				if not g.startswith("x"):
					continue

			area.append(    spec.popt_gaus[1 + 3*ig])
			area_err.append(spec.perr_gaus[1 + 3*ig])

			# if g == "-":
			if (not g[1:].isnumeric()) or (g.startswith('e')):
				# this_peak_id = -1
				this_peak_id = g
				energy.append(0.0)
				energy_err.append(0.0)

			else:
				this_peak_id = int(g[1:])
				energy.append(    peaks[this_peak_id].e)
				energy_err.append(peaks[this_peak_id].e_err)

			is_incl.append(this_peak_id not in exclude_source_peaks)
			peak_id.append(this_peak_id)

	area_incl       = np.array([_ for i,_ in enumerate(area      ) if is_incl[i]])
	energy_incl     = np.array([_ for i,_ in enumerate(energy    ) if is_incl[i]])
	area_err_incl   = np.array([_ for i,_ in enumerate(area_err  ) if is_incl[i]])
	energy_err_incl = np.array([_ for i,_ in enumerate(energy_err) if is_incl[i]])

	result = (area_incl, energy_incl, area_err_incl, energy_err_incl)
	if True:
		result = result + (peak_id,)

	return result

def calibrate_energy(energy_model,spectra,branch,exclude_source_peaks=False,show_calibrations=False):
	area, energy, area_err, energy_err, peak_id = get_peaks(spectra, branch, exclude_source_peaks)

	popt, perr, chi2, ndof, cov = energy_model.fit_with_errors(
		area,
		energy,
		area_err,
		energy_err,
		True
		)

	# print((energy - energy_model(area, *popt)))
	maxdev = abs((energy - energy_model(area, *popt)) / energy).max()

	if show_calibrations:

		print(popt)
		print(perr)
		print(chi2, ndof, chi2/ndof)
		print(cov)

		result_string = "{}={:>.1e}\xb1{:>.1e}"
		xaxis = np.linspace(min(area), max(area), 500)

		plt.errorbar(
			area,
			energy,
			energy_err,
			area_err,
			color="olive",
			marker=".",
			ls="",
			label="data",
		)

		plt.plot(xaxis, energy_model.fn(xaxis,*popt), 'k-', label="model")
		plt.xlabel("Area (pVs) for {}".format(branch))
		plt.ylabel("Energy (KeV)")
		plt.title("y = {}; chi2/dof={}\n{}".format(
			energy_model.formula,
			round(chi2/ndof, 4),
			", ".join([result_string.format(energy_model.pnames[ip], p, perr[ip]) for ip,p in enumerate(popt)])
		))
		plt.legend()
		plt.show()

		# relative plot
		ec  = energy_model(area, *popt)
		elo = energy_model(area-area_err, *popt)
		ehi = energy_model(area+area_err, *popt)
		# elo = energy_model(area, *(popt-perr))
		# ehi = energy_model(area, *(popt+perr))
		plt.errorbar(energy,(ec-energy)/energy,(ehi-ec)/energy,color="darkgreen",marker=".",ls="",label="best\xb1error")
		# plt.plot(energy,ec  - energy,color="darkgreen",marker=".",ls="",label="best fit")
		# plt.plot(energy,elo - energy,color="g",marker=".",ls="",label="\xb1 error")
		# plt.plot(energy,ehi - energy,color="g",marker=".",ls="")
		plt.axhline(0,color='k')
		plt.axhline( 0.01,color='b',label='\xb11%')
		plt.axhline(-0.01,color='b')
		plt.axhline( 0.02,color='c',label='\xb12%')
		plt.axhline(-0.02,color='c')

		plt.xlabel("theoretical peak energy (KeV)")
		plt.ylabel("(model - theory)/theory")
		# plt.plot(xaxis, energy_model.fn(xaxis,*popt), 'k-', label="model")
		# plt.xlabel("Area (pVs) for {}".format(branch))
		# plt.ylabel("Energy (KeV)")
		plt.title("y = {}; chi2/dof={}\n{}".format(
			energy_model.formula,
			round(chi2/ndof, 4),
			", ".join([result_string.format(energy_model.pnames[ip], p, perr[ip]) for ip,p in enumerate(popt)])
		))
		plt.legend()
		plt.show()


	return popt, perr, chi2, ndof, cov, maxdev


if __name__ == '__main__':


	fit_direct = True

	if fit_direct:
		# file_src_spec  = './data/fits/src_nov22.csv'
		# file_test_spec = './data/fits/pb_ka_nolyso_3443.csv'
		file_src_spec  = './data/fits/nov11_src_ch2_NaI1_la.csv'
		# file_test_spec = './data/fits/nov11_ch1_IK.csv'
		file_test_spec = './data/fits/rs_lyso_nov11_sat.csv'
		branch="area_2988_2"
		exclude_source_peaks=[
			-1,  # unidentified
			602, # subdominant
			101,300,401,400, # high energy 
			# 200, # 31KeV Ba133
			500, # 22KeV Cd109
			100, # 32KeV Cs137
		]
		src_spec  = fileio.load_fits(file_src_spec)
		energy_model = model.quad()
		ec_results = calibrate_energy(
			energy_model,
			src_spec,
			branch,
			exclude_source_peaks=exclude_source_peaks,
			show_calibrations=False,
		)

		colors_per_run = ["tab:brown","darkred","darkviolet","k","b","g","r","c","m","y"]
		test_spec = fileio.load_fits(file_test_spec)
		# print([_.popt_gaus for _ in test_spec])
		test_colors = colors_per_run[:len(test_spec)]
		labels = ["cs", "cd", "ba", "co", "am"]
		main(
			test_spec,
			branch,
			energy_model,
			ec_results,
			None,
			require_id=False,
			colors=test_colors,
			labels=labels,
			plot_area=False,
			plot_area_t1=False,
			plot_energy=False,
			plot_area_xf=False,
			# show=False,
		)


	plop = False
	if plop:

		colors_per_run = ["tab:brown","darkred","darkviolet","k","b","g","r","c","m","y"]

		bnl_runs_test = [3695,3696,3698,3700,3701,3704,3705,3706]
		bnl_run_xf = 3695

		file_src_spec  = './data/fits/src_nov22.csv'
		file_test_spec = './data/fits/beam_{}.csv'#.format(bnl_run_test)
		# file_test_spec = './data/fits/peaks_nolyso_{}.csv'.format(bnl_run_test)
		# file_test_spec = './data/fits/beam_peak_{}.csv'.format(bnl_run_test)
		# file_test_spec = './data/fits/peaks_pbka_jan19_{}.csv'.format(bnl_run_test)

		ch = 1
		branch="area_2988_{}".format(ch)

		# file_xf = './data/xf/xf_lyso_req_{}_to_3443.csv'.format(bnl_run_xf)
		file_xf = './data/xf/xf_{}_to_3443.csv'.format(bnl_run_xf)
		xf = fileio.load_xf(file_xf)[0]#[ch-1]

		exclude_source_peaks=[
			-1,  # unidentified
			602, # subdominant
			101,300,401, # 
			400, # >200 KeV 
			100, # Cs137 32KeV outlier
		]

		src_spec  = fileio.load_fits(file_src_spec)
		test_specs = [fileio.load_fits(file_test_spec.format(_)) for _ in bnl_runs_test]
		test_colors = sum([ [colors_per_run[i]]*len(spec) for i,spec in enumerate(test_specs) ],[])

		energy_model = model.quad()
		ec_results = calibrate_energy(
			energy_model,
			src_spec,
			branch,
			exclude_source_peaks=exclude_source_peaks,
			show_calibrations=False,
		)

		main(
			sum(test_specs,[]),
			branch,
			energy_model,
			ec_results,
			xf,
			colors=test_colors,
			labels=bnl_runs_test,
			plot_area=True,
			plot_area_t1=True,
			plot_energy=True,
			plot_area_xf=False,
		)


	fit_transformed = False
	if fit_transformed:

		# bnl_run_test = 3705 # 3549
		# bnl_run_xf   = 3757 # 3537
		# file_test_spec = './data/fits/peaks_nolyso_{}.csv'.format(bnl_run_test)
		# file_test_spec = './data/fits/beam_peak_{}.csv'.format(bnl_run_test)
		# file_test_spec = './data/fits/peaks_pbka_jan19_{}.csv'.format(bnl_run_test)
		# file_test_spec = './data/fits/beam_{}.csv'.format(bnl_run_test)

		# bnl_run_xf   = 3695 # 3537
		# file_src_spec  = './data/fits/src_nov22.csv'
		# file_test_spec = './data/fits/beam_j20_4g.csv'

		# # file_xf = './data/xf/xf_lyso_req_{}_to_3443.csv'.format(bnl_run_xf)
		# file_xf = './data/xf/xf_{}_to_3443_trunc.csv'.format(bnl_run_xf)
		# xf = fileio.load_xf(file_xf)[0]#[ch-1]


		# # xf to run 3443
		# file_src_spec    = './data/fits/src_nov22.csv'
		# file_test_spec   = './data/fits/beam_j27_2g.csv'
		# file_xf_template = './data/xf/xf_{}_to_3443_unsat.csv'
		# runs_xf = [3757, 3766, 3767]
		# xfs = [fileio.load_xf(file_xf_template.format(run))[0] for run in runs_xf]
		# native_calib = False
		# channel_e_cal = "area_2988_1"

		
		# # xf to run 3356
		# file_src_spec    = './data/fits/nov11_src_ch2_NaI1_la.csv'
		# file_test_spec   = './data/fits/beam_j27_2g.csv'
		# file_xf_template = './data/xf/xf_{}_to_3356_sat.csv'
		# runs_xf = [3757, 3766, 3767]
		# xfs = [fileio.load_xf(file_xf_template.format(run))[0] for run in runs_xf]
		# native_calib = False
		# channel_e_cal = "area_2988_2"


		# xf to run 3356
		file_src_spec    = './data/fits/nov11_src_ch2_NaI1_la.csv'
		file_test_spec   = './data/fits/beam_j27_2g.csv'
		# file_test_spec   = './data/fits/rs_lyso_nov11_sat.csv'
		channel_e_cal = "area_2988_2"
		# file_xf = "./data/xf/xf_jan27_to_nov11_unsat_v2.csv"
		file_xf = "./data/xf/xf_jan27_to_nov11_sat.csv"

		native_calib = False
		# calibrate from same run
		if native_calib:

			runs_xf = [_ for _ in range(3753, 3767+1)]
			events_per_run = [7505,9311,2206,4572,13668,3954,3203,1121,2509,2493,2455,1312,3994,8618,13083]
			xfs = fileio.load_xf(file_xf)

			# some runs have poor stats for this
			exclude_runs = [3760]
			if exclude_runs:
				included = [not (_ in exclude_runs) for _ in runs_xf]
				runs_xf        = [_ for i,_ in enumerate(runs_xf       ) if included[i]]
				events_per_run = [_ for i,_ in enumerate(events_per_run) if included[i]]
				xfs            = [_ for i,_ in enumerate(xfs           ) if included[i]]

			thr=4000
			run_colors = ['r' if _<thr else 'k' for _ in events_per_run]
			below_thr = [_<thr for _ in events_per_run]

		else:
			# calibrate to a few specifc runs		
			runs_xf_all = [_ for _ in range(3753, 3767+1)]

			runs_xf = [3757, 3766, 3767]
			included = [_ in runs_xf for _ in runs_xf_all]
			xfs = [_ for i,_ in enumerate(fileio.load_xf(file_xf)) if included[i]]

			# native_calib = False


		exclude_source_peaks=[
			-1,  # unidentified
			602, # subdominant
			101,300,401,400, # high energy 
			# 200, # 31KeV Ba133
			# 500, # 22KeV Cd109
			# 100, # 32KeV Cs137
		]
		src_spec  = fileio.load_fits(file_src_spec)
		energy_model = model.quad()
		ec_results = calibrate_energy(
			energy_model,
			src_spec,
			channel_e_cal,
			exclude_source_peaks=exclude_source_peaks,
			show_calibrations=False,
		)



		ch = 1
		branch="area_2988_{}".format(ch)
		test_spec = fileio.load_fits(file_test_spec)
		if native_calib:
			colors_done = []
			for i,xf in enumerate(xfs):
				col = run_colors[i]
				if not (col in colors_done):
					label = "lyso event count {} {}".format("<" if below_thr[i] else ">", thr)
					colors_done.append(col)
				else:
					label=None
				main(
					test_spec[i:i+1],
					branch,
					energy_model,
					ec_results,
					xf,
					plot_area=True,
					plot_area_xf=False,
					show=False,
					colors=run_colors[i],
					labels=label,
					xlabel="run number",
					xax = [runs_xf[i]]*test_spec[i].ngaus,
				)
				plt.suptitle("LYSO calibration on same run")
		else:
			colseq=["darkred","g","b"]
			for i,bnl_run_xf in enumerate(runs_xf):
				# file_xf = file_xf_template.format(bnl_run_xf)
				xf = xfs[i]
				main(
					test_spec,
					branch,
					energy_model,
					ec_results,
					xf,
					plot_area=True,
					plot_area_xf=False,
					show=False,
					colors=colseq[i],
					labels="LYSO cal on run {}".format(bnl_run_xf),
					xlabel="run number",
					xax = sum([[_.run]*_.ngaus for _ in test_spec],[]),
				)
		plt.legend()
		plt.gcf().set_size_inches(16,6)#,dpi=200)
		plt.show()





	# file_xf = './data/xf/xf_22to23_s5.csv'
	# xfs = fileio.load_xf(file_xf)

	# main(
	# 	test_spec,
	# 	branch,
	# 	energy_model,
	# 	ec_results,
	# 	xfs[4],
	# )