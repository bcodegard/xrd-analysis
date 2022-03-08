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
		plot=True,

		custom_input = None,

		):

	# test peaks
	if test_spectra:
		area, _, area_err, _, peak_id = get_peaks(test_spectra, branch, exclude_source_peaks, require_id)
	else:
		area, area_err, peak_id = custom_input

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

	if plot:

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

	return peak_id, area, area_err, area_t1, area_err_t1, energy, energy_err, energy_err_total, this_jac




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

		print("chi2, ndof, maxdev, popt ({n}), perr ({n}), cov ({n2})".format(n=energy_model.npars,n2=energy_model.npars**2))
		print(
			' '.join(map(str,[
				chi2,
				ndof,
				maxdev,
				','.join(map(str,list(popt))),
				','.join(map(str,list(perr))),
				','.join(map(str,list(cov.flatten()))),
			]))
		)

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

	compare_peaks_across_datasets = False
	if compare_peaks_across_datasets:


		# bias trial data 
		
		# voltages = [900,950,1000,1050]
		# datasets = [fileio.load_fits('./data/fits/bias_f16_esc_{}v.csv'.format(_)) for _ in voltages]
		# branch="area_3046_1"
		
		voltages = [900,950,1000]
		datasets = [fileio.load_fits('./data/fits/bias_f16_vmax_{}v.csv'.format(_)) for _ in voltages]
		branch="vMax_3046_1"

		exclude_source_peaks={
			-1,  # unidentified
			602, # subdominant
			101,300,401,400, # high energy 
			# 200, # 31KeV Ba133
			# 500, # 22KeV Cd109
			# 100, # 32KeV Cs137
		}


		exclude_source_peaks_calib={
			-1,  # unidentified
			602, # subdominant
			101,300,401,400, # high energy 
			901, # Am241 26.5 KeV
			# 200, # 31KeV Ba133
			500, # 22KeV Cd109
			100, # 32KeV Cs137
		}
		energy_model = model.quad()
		calibrations = []
		transformed_mu = []
		transformed_sigma = []
		for ds in datasets:
			ec_results = calibrate_energy(
				energy_model,
				ds,
				branch,
				exclude_source_peaks=exclude_source_peaks_calib,
				show_calibrations=True,
			)
			calibrations.append(ec_results)

			all_mu     = sum([_.popt_gaus[1::3]  for _ in ds],[])
			all_mu_err = sum([_.perr_gaus[1::3]  for _ in ds],[])
			all_id     = sum([_.gaus_names for _ in ds],[])
			# peak_id, area, area_err, area_t1, area_err_t1, energy, energy_err, energy_err_total, this_jac = main(
			transformed_mu.append(main(
				False,
				branch,
				energy_model,
				ec_results,
				area_xf = None,
				exclude_source_peaks=False,
				require_id=True,
				show=False,
				plot=False,
				custom_input=[all_mu,all_mu_err,all_id]
				))

			# transformed_mu.append([peak_id, area, area_err, area_t1, area_err_t1, energy, energy_err, energy_err_total, this_jac])
			# print(peak_id, area, area_err, area_t1, area_err_t1, energy, energy_err, energy_err_total, this_jac)
			# print("")
			# print(peak_id)
			# print(energy)
			# print(energy_err)
			# print(energy_err.shape)
			# print(sum([_.ngaus for _ in ds]))
			# print(len(ds))
			# print("")

			all_sigma     = sum([_.popt_gaus[2::3]  for _ in ds],[])
			all_sigma_err = sum([_.perr_gaus[2::3]  for _ in ds],[])
			all_id        = sum([_.gaus_names for _ in ds],[])
			# print(all_id)
			transformed_sigma.append(main(
				False,
				branch,
				energy_model,
				ec_results,
				area_xf = None,
				exclude_source_peaks=False,
				require_id=True,
				show=False,
				plot=False,
				custom_input=[all_sigma, all_sigma_err, all_id],
				))
			# print(transformed_sigma[-1][0])


		results = [{} for _ in datasets]

		for i,spec in enumerate(datasets):
			
			for fit in spec:

				if fit.fit_branch != branch:
					continue

				this_ids   = fit.gaus_names
				# print(this_ids)
				# print(fit.popt_gaus)
				# this_peaks = fit.popt_gaus[1::3]

				for iid,this_id in enumerate(this_ids):
					# this_id = this_ids[ip]

					if this_id == '-':
						this_id = -1
					elif this_id.startswith('x'):
						this_id = int(this_id[1:])
					else:
						pass

					if this_id in results[i].keys():
						results[i][this_id].append([fit.popt_gaus[iid*3:iid*3+3], fit.perr_gaus[iid*3:iid*3+3]])
					else:
						results[i][this_id]=[[fit.popt_gaus[iid*3:iid*3+3], fit.perr_gaus[iid*3:iid*3+3]]]

			# print(results[i])

			for iid,this_id in enumerate(transformed_mu[i][0]):
				if this_id == '-':
						this_id = -1
				elif this_id.startswith('x'):
					this_id = int(this_id[1:])
				else:
					pass
				results[i][this_id][0].append([transformed_mu[i][5][iid],transformed_mu[i][6][iid],transformed_mu[i][7][iid]])
			
			for iid,this_id in enumerate(transformed_sigma[i][0]):
				if this_id == '-':
						this_id = -1
				elif this_id.startswith('x'):
					this_id = int(this_id[1:])
				else:
					pass
				results[i][this_id][0].append([transformed_sigma[i][5][iid],transformed_sigma[i][6][iid],transformed_sigma[i][7][iid]])
				# print(iid, this_id, transformed_mu[i][5][iid])
				# print(iid, this_id, transformed_mu[i][6][iid])
				# print(iid, this_id, transformed_mu[i][7][iid])

		peak_ids_present = set()
		for res in results:
			peak_ids_present |= res.keys()
		peak_ids_use = peak_ids_present - exclude_source_peaks

		colors = ["tab:brown","darkred","darkviolet","k","b","g","r","c","m","y"]

		for ii,i in enumerate(sorted(peak_ids_use,key=str)):

			if type(i) is int:
				this_label = "{} {}KeV".format(sources[i//100], round(peaks[i].e,1))
			else:
				j = int(i[1:])
				this_label = "{} {}KeV, escape".format(sources[j//100], round(peaks[j].e,1))

			this_v  = []
			this_mu = []
			this_mu_err = []
			this_sigma = []
			this_sigma_err = []

			for ires,res in enumerate(results):
				if res.get(i,False):
					entry = res.get(i)[0]
					# print(i, ires, entry)
					this_v.append(voltages[ires])
					# this_mu.append(        entry[0][1] )
					# this_sigma.append(     entry[0][2] )
					# this_mu_err.append(    entry[1][1] )
					# this_sigma_err.append( entry[1][2] )
					this_mu.append(        entry[2][0] )
					this_mu_err.append(    entry[2][1] )
					this_sigma.append(     entry[3][0] )
					this_sigma_err.append( entry[3][1] )

			this_v = np.array(this_v)
			this_mu = np.array(this_mu)
			this_sigma = np.array(this_sigma)
			this_mu_err = np.array(this_mu_err)
			this_sigma_err = np.array(this_sigma_err)

			plt.subplot(231)
			plt.errorbar(this_v, this_sigma, this_sigma_err, marker='.', ls=':', color=colors[ii], label=this_label)
			if ii==len(peak_ids_use)-1:
				plt.xlabel('bias voltage')
				plt.ylabel('peak widths')
				plt.legend()

			plt.subplot(234)
			plt.plot(this_v, this_sigma_err / this_sigma, marker='.', ls=':', color=colors[ii], label=this_label)
			if ii==len(peak_ids_use)-1:
				plt.xlabel('bias voltage')
				plt.ylabel('(width error) / width')
				# plt.legend()

			plt.subplot(232)
			plt.errorbar(this_v, this_mu, this_mu_err, marker='.', ls=':', color=colors[ii], label=this_label)
			if ii==len(peak_ids_use)-1:
				plt.xlabel('bias voltage')
				plt.ylabel('peak locations')
				# plt.legend()

			plt.subplot(235)
			plt.plot(this_v, this_mu_err / this_mu, marker='.', ls=':', color=colors[ii], label=this_label)
			if ii==len(peak_ids_use)-1:
				plt.xlabel('bias voltage')
				plt.ylabel('(mu error) / mu')
				# plt.legend()

			plt.subplot(233)
			this_sigma_over_sqrt_mu = this_sigma / np.sqrt(this_mu)
			this_sigma_over_sqrt_mu_err = this_sigma_over_sqrt_mu*np.sqrt((this_sigma_err/this_sigma)**2 + 0.25*(this_mu_err/this_mu)**2)
			plt.errorbar(this_v, this_sigma_over_sqrt_mu, this_sigma_over_sqrt_mu_err, marker='.', ls=':', color=colors[ii], label=this_label)
			if ii==len(peak_ids_use)-1:
				plt.xlabel('bias voltage')
				plt.ylabel('sigma / sqrt(mu)')
				# plt.legend()

			plt.subplot(236)
			this_sigma_over_mu = this_sigma / this_mu
			this_sigma_over_mu_err = this_sigma_over_mu*np.sqrt((this_sigma_err/this_sigma)**2 + (this_mu_err/this_mu)**2)
			plt.errorbar(this_v, this_sigma_over_mu, this_sigma_over_mu_err, marker='.', ls=':', color=colors[ii], label=this_label)
			if ii==len(peak_ids_use)-1:
				plt.xlabel('bias voltage')
				plt.ylabel('sigma / mu')
				# plt.legend()

		plt.suptitle("after converting to energy")
		# plt.tight_layout()
		plt.show()



	plot_peak_area = True
	if plot_peak_area:
		desc = ''

		# # cleaned spectrum, 1k slices
		# # sample_means = [42307.925564547404, 43185.868079086824, 42149.87690423032, 42276.53165913641, 41907.383671114774, 41223.240197089646, 40180.71892661169, 41168.844417424756, 41762.957075320504, 41314.478167832625, 42380.96606887341, 41696.58895129995, 42256.120783118145, 41593.36248119576, 40119.95761861602, 41002.897818257894, 41063.23052671674, 42411.20401576112, 41663.40664748301, 41094.516340429494, 41099.72797022124, 40323.86097413755, 41827.39105971151, 41130.580769472974, 42778.2757168929, 41223.17366424735, 41274.63821782462, 41319.49474734745, 40895.907733804524, 41445.51363613402, 42023.65374762734, 41799.03867972003, 40723.58911156323, 42210.06891203852, 39674.613070644, 41051.05873757255, 41441.4379746499, 41428.178431030574, 42002.70528271656, 40448.46509940285, 41557.592271966365, 41597.66054098482, 40494.56641929153, 40906.50564143757, 41190.647970051345, 42651.31759465513, 40428.328694653515, 41888.55849143689, 41481.32332393382, 42931.69900801772, 42866.683532376126, 41418.76950400706, 40763.836135428115, 39820.79760572058, 41140.48499179257, 41012.95100673316, 42455.01809605412, 41526.57932773156, 41465.808058519375, 41707.5531169887, 41488.61827669368, 40249.981254437356, 41678.200362735915, 40319.98254390541, 40877.848036285446, 41203.20030603551, 41470.300205984815, 40476.34904074073, 40636.98759806256, 40592.179043548276, 41465.44309619908, 40399.25619492133, 42168.769162762685, 41369.331803405286, 39278.11372905575, 40056.17488038634, 41464.28292692895, 41274.25888645481, 41236.50338792458, 41365.50117339949, 41329.92918649128, 41666.87758309178, 41687.501309605796, 40893.84113992547, 40099.86036957112, 41551.64662764652, 40909.94807274489, 41612.719438083026, 40447.72373558045, 42001.502700480836, 42349.402273190746, 41201.38056669016, 41230.477641802805, 41711.6932412664, 41696.28989273004, 41217.969276194905, 40768.37367230241, 40972.652309239966]

		# # cleaned spectrum, 10k slices
		# # sample_means = [41747.78246623949, 41528.22512517516, 41331.856448979415, 41280.280904696556, 41512.81999564288, 41417.84813753514, 40899.36466684297, 41007.76352414383, 41220.15501632211]

		# # raw spectrum, 1k slices
		# # sample_means = [48227.37685552413, 47716.97672933499, 49346.88638433052, 46457.23456986871, 47981.37109670778, 47276.76404637294, 47958.94769988271, 47636.512109028714, 47393.47797074725, 47801.353168775044, 47445.15864568239, 46589.11812933314, 48693.31921726758, 47154.61556002179, 47229.208857056125, 48142.13678016931, 48177.780295025324, 47399.68844774111, 47762.70294172201, 48146.83176601114, 47415.18956568876, 46998.56363783259, 45458.32322658765, 47683.84747004244, 46981.500501418835, 47436.57310046719, 46437.00544313701, 48315.96799060381, 48674.20549668639, 46969.46111737457, 48068.54746890606, 46346.74150561593, 48550.76015008882, 47182.39405500002, 45971.135947035706, 47173.515233795304, 46990.805060193125, 47349.86490653327, 47525.34433722948, 47387.3728203946, 48179.92403800456, 47727.01801003678, 46799.497292859, 47960.525824086326, 47249.811094103214, 47044.898808163045, 46313.954827606634, 47090.57533436434, 47658.14490602352, 48149.84326310355, 46889.27139662196, 47908.05120296441, 46798.86816644497, 47611.32123060538, 47379.129654010685, 46024.57564399281, 46556.749885502686, 47616.03899711096, 46545.713547209845, 48392.968171603934, 47362.1444837957, 47723.8116496523, 47879.946036472094, 46567.910376227046, 46279.38349565474, 48413.78523571348, 47274.37363620351, 46196.20583619474, 46867.70231796613, 46769.23258268549, 46079.238394779226, 48993.686691899326, 48406.354548801944, 49158.082772714304, 47322.382811511474, 47441.72793958886, 48234.59166518023, 47001.77356984703, 47812.036088712455, 48358.992838852035, 47234.15410597952, 47416.10442327019, 48416.27202964014, 46636.84943009968, 46604.77335231295, 46482.12675235072, 46213.435007713124, 47482.1781095714, 47099.36329251215, 48227.58553791983, 46542.92859414604, 47108.32961256426, 48606.099805178135, 47255.307127303364, 47596.1445011844, 46217.73745504728, 46954.3864570393, 47848.088432354896, 45997.92117367095, 47630.607090876314, 47384.47736941719, 45391.27470906615, 47691.47864780232, 45477.295636602656, 46448.248737839116, 48717.87192022937, 46877.71763123977, 46378.12415322569, 47040.81916138135, 47333.49528618847, 47323.270874364796, 48234.987447203435, 47674.888630744994, 48611.92788285387, 46688.82442658461, 48268.668545258784, 48954.95969826637, 48102.723135001586, 45974.026241101754, 45989.44385241332, 45895.60818603457, 47061.08453121372, 48844.95427607286, 47410.284600336956, 46826.668940305906, 46659.17086203146, 46923.15999976226, 47324.26744695833, 48219.02487416506, 47455.05641517505, 46868.22190337622, 47805.5622472361, 47709.42709310206, 47208.085348975, 46818.4295755631, 46576.98324114363, 48531.095246198565, 47967.81133375859, 47032.56969827253, 47786.218928442955, 47771.40290440473, 46690.27112016189, 47395.94400957182, 49112.415804460186, 48331.358991495144, 47145.27641046657, 46744.36007312532, 47942.217326809136, 46849.42148620375, 47011.275542593125, 47891.55054213129, 47732.440215720424, 47469.952075397116, 46288.26020407792, 47030.4231111006, 48096.56055044455, 48667.92975670962, 45581.60752034152]

		# # raw spectrum, 5k slices
		# sample_means = [47945.96912715322, 47613.41099896133, 47422.2840818722, 47925.82804613378, 46907.48488031405, 47566.64262965379, 47223.915825329306, 47285.380471629156, 47583.35525181798, 47251.48342785221, 47317.32833012948, 47027.209249084044, 47162.63920836037, 47104.259921752666, 47991.949043941255, 47769.82442043612, 47261.6306682605, 47100.93774001345, 47421.76192807524, 46929.74812179775, 46478.55502014549, 47269.605630452934, 47706.779852350344, 47457.96429440836, 47207.7201067928, 47316.135919618435, 47281.94523365049, 47578.93568956325, 47860.27856601875, 47138.51016783957, 47282.52522968547]

		# # LYSOdataset, NaI, area>1000, 5k
		# sl = 5000
		# sample_means = [114659.97710450015, 117958.3118495924, 117337.50085412068, 116844.48271265891, 117862.97416547425, 118060.35084214983, 115154.90468073287, 116386.59162036052, 116767.11043363552, 115773.77261546788, 117520.551109439, 116749.83615320845, 117083.55134766905, 118043.11065284384, 117467.63346523083, 115768.49951715869, 118827.4753033564, 116307.69723626354, 116644.55532080961, 114717.40473676304, 117253.23206331769, 115416.54423847914, 117081.38134660701, 117673.72574732447, 116847.1947067768, 117200.98187759462, 117899.38323378959, 118994.46672301502, 116982.85066819825, 116462.08999811017, 118744.15022331744, 117256.567749437, 115786.04566851057, 116865.33791835875, 118488.0351239043, 116607.0721890452, 118358.36942544083, 117252.79959265667, 118031.96446966138, 116356.6916908456, 119650.44887433939, 116682.93190512195, 116227.06545548086, 116124.63544868784, 116814.18122791519, 116782.55069019932, 117403.24749405977, 117291.74564644671, 117041.59305666511, 118556.61971048234, 118676.05680029311, 117324.20802715127, 117568.14290326273, 117478.46681512342, 116114.62287336645, 116529.43619181146, 116889.17676497082, 118161.10816851168]

		# # LYSOdataset, NaI, area>1000, 25k
		# sl = 25000
		# sample_means = [116932.64933726925, 116428.54603846933, 117372.93654567824, 116453.12642287025, 116854.41562050102, 117507.95450014152, 117428.02733670562, 117321.37947352993, 117099.85258230906, 117415.15131957064, 117432.2994838394]



		# # burst data, no time info yet
		# # 4029, whole dataset -> inconsistent
		# sl = 25000
		# sample_means = [69347.24210241126, 69856.00736646888, 68504.67785741012, 69556.06838194327, 69842.57362030122, 69246.9312006017, 69873.66934428959, 69271.49284538545, 68542.3935735913, 69815.61364862436, 68917.48612030242, 69199.85423425357, 69019.82891350047, 69803.48795871263, 69724.4679876508, 68945.21519414, 69062.87327135069, 69716.20383152767, 69361.4483374788, 69493.92213425766, 68561.69356301437, 68781.59953401922, 69206.97969765325, 68872.97642262999]
		# # 4029, 31KeV peak, 3k slices
		# sl = 3000
		# sample_means = [21567.272719337234, 21557.646607601226, 21555.43412971368, 21595.553259112436, 21613.486725072336, 21478.002898988852, 21401.597270788458, 21586.51671980459, 21567.875023066037, 21538.969125309737, 21513.469140961428, 21458.219101052517, 21507.19025373434, 21504.584523781792, 21598.174610303122, 21479.494483614322, 21528.49768052009, 21543.300534185535, 21563.760431867926, 21552.515030981634, 21430.99650475816, 21486.79984072321, 21484.580312905062, 21503.735829008663, 21510.436569104037, 21418.974833794928, 21408.83140381059, 21477.597317563508, 21511.644418308646, 21398.69014445887, 21420.86410733596, 21429.921014383657, 21540.994219565855, 21481.32699179555, 21472.686799327377, 21393.09087561847, 21441.885420790728, 21448.029440924834, 21422.84324681718, 21487.13968401071, 21353.412834357765, 21417.02761925828, 21395.383006949938, 21418.634901872803, 21433.231367649165, 21379.931456780214, 21391.883952764118, 21441.147869467135, 21389.881987938657, 21381.600590972615, 21383.333681314405, 21383.129971772727, 21328.2424721824, 21379.555588160685, 21405.462336523367, 21340.815320316742, 21374.343779889892, 21369.963070093167, 21377.510949645137, 21351.421316253556, 21415.701378276583, 21334.31192354065, 21296.92520055402, 21465.07781079666, 21411.999994221318, 21389.56027598123, 21339.312946591952, 21335.6391230451, 21414.800850254294, 21363.166875332332, 21375.733645320423, 21337.758343238536, 21212.50037306076, 21297.386208842177, 21228.706029228364, 21253.679405698924, 21225.951159079672, 21342.477451001654, 21272.18732554815, 21270.500505274154, 21323.88828985539, 21320.323083267805, 21297.81360082131]
		# # 4039, main + sub peak, 5k slices
		# sl = 5000
		# sample_means = [35206.055179460374, 35154.009359962205, 35397.50852530439, 35160.7089787005, 35157.33831502512, 35226.85293700236, 35139.811230178035, 35198.73107500641, 35034.5211226383, 35027.22037662097, 34991.37648117374, 35181.45206039481, 35113.140466863726, 35050.05512787247, 35124.9504005423, 35249.398826065815, 35408.34840977987, 35422.90183655665, 35463.98011502906, 35168.85053043811, 35058.11914214199, 34951.23860560597, 35045.67280382043, 35276.13539331289, 34970.07170008343, 35056.89229301272, 35324.305677982324, 35317.24602926791, 35104.454155759326, 35462.550207479064, 35162.02828239924, 35063.63803775045, 35646.36417255549, 35143.88263760303, 35276.882698144706, 35152.21085724234, 35367.61668709979, 35349.93184128336, 35454.28358343755, 35281.232768492344, 35321.369365437764, 35313.120407579554, 35144.649228413815, 35192.36567578272, 34859.67497503644, 35308.14346252513, 35308.55363196014, 35184.28384085974, 35256.4522361192, 35235.76617271576, 35267.25749348697, 35118.332845836754, 35081.02033649571, 35306.304244509, 35441.27102023559, 35267.053701450684, 35552.604875322315, 35199.89434084711, 35195.29816073121, 35272.43886557157, 35118.16021927891, 35262.80259462498, 35391.52125704066, 35554.23742545105, 35093.134009841015, 35366.59433847655, 35166.772896518094, 35228.82631807378, 35342.24686621619, 35098.06037268791, 35347.67909307746, 35345.74287565654, 35151.62063617661, 35135.541764455586, 35388.458373893816, 35408.820224781324, 35193.783456229714, 35203.42684017295, 35348.83932352605, 35315.185659478775, 35065.820137453105, 35291.82436288023, 35446.68699695979, 35054.048153778895, 35228.237018134096, 35394.0148703342, 35474.95267953729, 35177.346287145185, 35267.19085951893, 35184.8352641416, 35345.52070673217, 35169.95909897855, 35206.342975737694, 35290.60732526541, 35303.057112396076, 35342.703607018426, 35279.77228662361, 35097.04471919642, 35269.99517262163, 35320.312412362895, 35417.24463515051, 35001.07411466104, 35075.9855022695]
		
		# # # 4039, main peak only, 5k slices
		# # sl = 5000
		# # sample_means = [40171.970715172036, 40214.29103414742, 40165.745106474336, 40225.86257984087, 40176.5514115876, 40182.53957676059, 40181.437963902266, 40155.87908732553, 40211.24077208438, 40119.488104830474, 40121.0034716091, 40218.722670512725, 40119.35435161573, 40264.02283166102, 40216.916301388206, 40169.09379637144, 40144.29359144275, 40170.05043374776, 40113.21666481559, 40183.20209129078, 40104.899401795265, 40313.353558282906, 40268.11627138025, 40207.562165620846, 40231.29265179872, 40197.48866873846, 40221.754341436994, 40179.64707862431, 40165.98101129506, 40214.78327336823, 40134.38949452423, 40236.28525048242, 40208.401008985566, 40144.08816484213, 40198.28797014565, 40223.08935239493, 40217.902454710595, 40221.72051489762, 40212.53613184645, 40336.04753281711, 40146.3058119265, 40269.40575935977, 40184.64759550176, 40279.708470733465, 40185.59384973747, 40284.6575192666, 40218.62698291507, 40230.7576428047, 40222.122786295244, 40262.26590908148, 40303.691942948804, 40215.43587508958, 40264.0908451428, 40258.980726955204, 40292.75952375319, 40221.988286012725, 40205.50495921512, 40227.334831205764, 40232.547133318316, 40370.05421917796, 40255.44406963547, 40247.16682843851, 40223.207191879424, 40201.97315270038, 40214.78635968009, 40198.540296205625, 40269.29366050917, 40156.93832241579, 40265.34282605668, 40264.8063712352, 40182.703137132274, 40237.99685218357, 40248.08964685476, 40268.05378200897, 40193.51970537853, 40240.361292310205, 40221.367624265105, 40312.711391036704, 40275.46631733233]
		# # desc = "4039 (Am241), mean of events in main peak"
		# # # 4039, main peak only, 20k slices
		# # sl = 20000
		# # sample_means = [40194.467358908674, 40174.102009894, 40167.61375475917, 40192.346820259096, 40152.69069532423, 40223.482849269814, 40207.54568514962, 40187.859757417486, 40193.46662409206, 40247.05165856795, 40220.01690938036, 40229.90899868096, 40250.87912835378, 40259.45484546598, 40258.860285729286, 40231.94781066345, 40209.88965970266, 40237.71229665193, 40237.50610663811]
		# # desc = "run 4039 (Am241), mean of events in main peak"

		# plt.plot(range(len(sample_means)), sample_means, 'k.')
		# plt.xlabel("slice")
		# plt.ylabel("mean area")
		# plt.title("sample means, {}k events/slice{}".format(sl//1000,'\n{}'.format(desc) if desc else ''))
		# plt.show()
		# sys.exit(0)



		# run = 3927
		# sl = 10000
		# adj = 'peak1'

		run = 4039
		sl  = 50000
		adj = 'mainpeak'

		description = "area of 55KeV LYSO peak"
		fit_file = "./data/fits/drift_{}_{}_{}k.csv".format(run,adj,sl//1000)
		specs = fileio.load_fits(fit_file)

		colors = ["k","b","darkred","g","tab:brown","c","r","m","y"]
		all_mu     = np.array([])
		all_mu_err = np.array([])
		for i,sp in enumerate(specs):

			mu     = sp.popt_gaus[1::3]
			mu_err = sp.perr_gaus[1::3]
			all_mu     = np.concatenate([all_mu    , mu    ])
			all_mu_err = np.concatenate([all_mu_err, mu_err])
			# plt.errorbar([i]*sp.ngaus, mu, mu_err, ls='', marker='.', color='k')
		
		print("all_mu     mean {:<10.2f} rms {:<10.2f}".format(np.mean(all_mu    ), np.std(all_mu    )))
		print("all_mu_err mean {:<10.2f} rms {:<10.2f}".format(np.mean(all_mu_err), np.std(all_mu_err)))

		models = [model.poly0(), model.poly1(), model.poly2()]

		xax = np.arange(len(specs))
		results = [_.fit_with_errors(xax.astype(float), all_mu, xax*0.0, all_mu_err) for _ in models]
		# r0 = poly0.fit_with_errors(xax.astype(float), all_mu, xax*0.0, all_mu_err, )
		# r1 = poly1.fit_with_errors(xax.astype(float), all_mu, xax*0.0, all_mu_err, )

		print("fit models and results")
		for ires, res in enumerate(results):
			mod = models[ires]
			print("{:<12}: {}".format(mod.name, res))
			label="{:4.4}: {:.1f}/{} = {:.3f}".format(mod.name, res[2], res[3], res[2]/res[3])
			plt.plot(xax, mod(xax, *res[0]), colors[ires+1], marker='', ls='--', label=label)
		# print("constant: {}".format(r0))
		# print("line    : {}".format(r1))
		print("")

		for j in range(sp.ngaus):
			plt.errorbar(xax, all_mu[j::sp.ngaus], all_mu_err[j::sp.ngaus], ls='', marker='.', color=colors[j])
		plt.xlabel("slice")
		plt.ylabel("peaks (mu)")
		plt.title("Run {}, {}k events / slice\n{}".format(run, sl//1000, description))
		plt.legend()
		plt.ylim(40200,40900)
		plt.show()


	fit_direct = False

	if fit_direct:


		# Nov 11 UCSB data
		# # file_src_spec  = './data/fits/src_nov22.csv'
		# # file_test_spec = './data/fits/pb_ka_nolyso_3443.csv'
		# file_src_spec  = './data/fits/nov11_src_ch2_NaI1_la.csv'
		# # file_test_spec = './data/fits/nov11_ch1_IK.csv'
		# file_test_spec = './data/fits/rs_lyso_nov11_sat.csv'
		# branch="area_2988_2"
		# exclude_source_peaks=[
		# 	-1,  # unidentified
		# 	602, # subdominant
		# 	101,300,401,400, # high energy 
		# 	# 200, # 31KeV Ba133
		# 	500, # 22KeV Cd109
		# 	100, # 32KeV Cs137
		# ]

		# # bias trial data 
		# # voltages = [900,950,1000,1050]
		# voltages = [900,950,1000]
		# file_src_spec  = ['./data/fits/bias_f16_vmax_{}v.csv'.format(_) for _ in voltages]
		# file_test_spec = ['./data/fits/bias_f16_vmax_{}v.csv'.format(_) for _ in voltages]
		# branch="vMax_3046_1"
		# exclude_source_peaks=[
		# 	-1,  # unidentified
		# 	602, # subdominant
		# 	101,300,401,400, # high energy 
		# 	901, # Am241 26.5KeV
		# 	# 200, # 31KeV Ba133
		# 	500, # 22KeV Cd109
		# 	100, # 32KeV Cs137
		# ]

		# SmallNaI 1 and 2, 900V
		setups = [[1,'l'],[1,'h'],[2,'l'],[2,'h']]
		branches=["area_3046_3","area_3046_4","area_3046_3","area_3046_4"]
		file_src_spec  = ['./data/fits/snai{}_{}g_src.csv'.format(*_) for _ in setups]
		file_test_spec = ['./data/fits/snai{}_{}g_src.csv'.format(*_) for _ in setups]
		exclude_source_peaks=[
			-1,  # unidentified
			602, # subdominant
			101,300,401,400, # high energy 
			901, # Am241 26.5KeV
			# 200, # 31KeV Ba133
			500, # 22KeV Cd109
			100, # 32KeV Cs137
		]

		

		if type(file_src_spec) is str:
			file_src_spec = [file_src_spec]
			file_test_spec = [file_test_spec]

		for ifs, fs in enumerate(file_src_spec):
			ft = file_test_spec[ifs]

			src_spec  = fileio.load_fits(fs)
			energy_model = model.quad()
			ec_results = calibrate_energy(
				energy_model,
				src_spec,
				branches[ifs],
				exclude_source_peaks=exclude_source_peaks,
				show_calibrations=True,
			)

			colors_per_run = ["tab:brown","darkred","darkviolet","k","b","g","r","c","m","y"]
			test_spec = fileio.load_fits(ft)
			# print([_.popt_gaus for _ in test_spec])
			test_colors = colors_per_run[:len(test_spec)]
			labels = ["cs", "cd", "ba", "co", "am"]
			main(
				test_spec,
				branches[ifs],
				energy_model,
				ec_results,
				None,
				require_id=False,
				colors=test_colors,
				labels=labels,
				plot_area=True,
				plot_area_t1=False,
				plot_energy=True,
				plot_area_xf=False,
				show=True,
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