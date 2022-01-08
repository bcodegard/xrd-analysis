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

file_template = "./data/fits/mst_{}{}.csv"

scint_types = {
	2053:{"area_2988_1":"LYSO", "area_2988_2":"NaI1"},
	2403:{"area_2988_1":"NaI1", "area_2988_2":"LYSO"},
	2621:{"area_2988_1":"NaI1"},
	2629:{"area_2988_1":"NaI1"},
	2633:{"area_2988_1":"NaI1"},
	2637:{"area_2988_1":"NaI1"},
	2641:{"area_2988_1":"NaI1"},
	3420:{"area_2988_1":"NaI1","area_2988_2":"LYSOBar1"},
	3427:{"area_2988_1":"NaI1","area_2988_2":"NaI2","area_2988_3":"LYSObar1","area_2988_4":"LYSObar2"},
}

voltages = {
	2053:770,
	2403:860,
	2621:860,
	2629:860,
	2633:860,
	2637:860,
	2641:860,
	3420:900,
	3427:900,
}

COLORS_DEFAULT = {
	"NaI1":{
		770:"lightcoral",
		860:"firebrick",
		900:"maroon",
	},
	"NaI2":{
		770:"",
		860:"gold",
		900:"goldenrod",
	},
	"LYSO":{
		770:"cyan",
		860:"darkcyan",
		900:"darkslategrey",
	},
	"LYSOBar1":{
		770:"",
		860:"blue",
		900:"darkblue",
	},
	"LYSOBar2":{
		770:"",
		860:"blue",
		900:"darkblue",
	},
}

COLORS_ALTERNATE = {
	"NaI1":{
		770:"springgreen",
		860:"forestgreen",
		900:"darkgreen",
	},
	"NaI2":{
		770:"plum",
		860:"magenta",
		900:"purple",
	},
}

datasets = sorted(scint_types.keys())

def main(

		require_id = True,
		calibrate = False,
		plot_separately = False,
		fit_mu_vs_sigma = True,
		show_calibrations = False,
		show_excluded_peaks = False,

		do_fit=False,

		ni_suffix_template = "_1mv_ni{}",

		ni = 5,

		exclude_source_peaks = [-1,602, 101,300,400,401],

		ds_whitelist = [3427],
		ds_blacklist = [],

		colors = COLORS_DEFAULT,

		suffix=None,

		show=True,

		):

	mu        = np.array([])
	mu_err    = np.array([])
	sigma     = np.array([])
	sigma_err = np.array([])
	peak_id   = np.array([])
	voltage   = np.array([])
	scint     = np.array([])

	if do_fit:
		fitmodel = model.sqrt(static_parameters=[100.0])

	# require_id = True
	# calibrate = False
	# plot_separately = False
	# fit_mu_vs_sigma = True
	# show_calibrations = False
	# show_excluded_peaks = False

	# ni = 5

	# exclude_source_peaks = [-1,602, 101,300,400,401]

	# ds_whitelist = [3427]
	# ds_blacklist = []

	included = lambda n:(n not in ds_blacklist) and ((n in ds_whitelist) or (not ds_whitelist))

	ni_suffix = "" if ni is None else ni_suffix_template.format(ni)
	files = [file_template.format(_,ni_suffix) for _ in datasets if included(_)]
	ds_included = [_ for _ in datasets if included(_)]

	if not files:
		raise ValueError("all datasets excluded; nothing to do")

	fits = [fileio.load_fits(_) for _ in files]

	for ifit,fit in enumerate(fits):
		this_ds = ds_included[ifit]

		this_mu        = []
		this_mu_err    = []
		this_sigma     = []
		this_sigma_err = []
		this_peak_id   = []
		this_voltage   = []
		this_scint     = []
		this_channel   = []

		for entry in fit:
			for ig,g in enumerate(entry.gaus_names):

				if require_id:
					if g=="-":
						continue
					else:
						this_peak_id.append(int(g[1:]))
				else:
					if g=="-":
						this_peak_id.append(-1)
					else:
						this_peak_id.append(int(g[1:]))

				this_mu.append(        entry.popt_gaus[ig*3 + 1] )
				this_mu_err.append(    entry.perr_gaus[ig*3 + 1] )
				this_sigma.append(     entry.popt_gaus[ig*3 + 2] )
				this_sigma_err.append( entry.perr_gaus[ig*3 + 2] )
				this_voltage.append(voltages[this_ds])
				this_scint.append(scint_types[this_ds][entry.fit_branch])
				this_channel.append(entry.fit_branch)

		# convert to arrays
		this_mu        = np.array(this_mu       )
		this_mu_err    = np.array(this_mu_err   )
		this_sigma     = np.array(this_sigma    )
		this_sigma_err = np.array(this_sigma_err)
		this_peak_id   = np.array(this_peak_id  )
		this_voltage   = np.array(this_voltage  )
		this_scint     = np.array(this_scint    )
		this_channel   = np.array(this_channel  )

		if calibrate:
			mod = model.quad()
			energy = np.array([peaks[_].e if _ in peaks.keys() else 0 for _ in range(1000)])
			energy_err = np.array([peaks[_].e_err if _ in peaks.keys() else 0 for _ in range(1000)])

			for channel in set(this_channel):

				# filters
				ftr_ch = this_channel == channel
				ftr_id = np.logical_not(np.isin(this_peak_id, exclude_source_peaks))

				# which to use for calibrating this channel
				ftr_cal = np.logical_and(ftr_ch, ftr_id)

				if ftr_cal.sum() == 0:
					continue

				# calibrate on mu vs. peak energy
				popt, perr, chi2, ndof = mod.fit_with_errors(
					energy[this_peak_id[ftr_cal]],
					this_mu[ftr_cal],
					energy_err[this_peak_id[ftr_cal]],
					this_mu_err[ftr_cal],
				)

				if show_calibrations:
					plt.errorbar(
						this_mu[ftr_cal],
						energy[this_peak_id[ftr_cal]],
						energy_err[this_peak_id[ftr_cal]],
						this_mu_err[ftr_cal],
						color="olive",
						marker=".",
						ls="",
						label="data"
					)
					xaxis = np.linspace(min(this_mu[ftr_cal]), max(this_mu[ftr_cal]), 500)
					plt.plot(xaxis, mod.ifn(xaxis,*popt), 'k-', label="model")
					plt.xlabel("{}: mu (area, pVs)".format(channel))
					plt.ylabel("energy (KeV)")
					plt.title("set {}, A = a0 + a1*E + a2*E**2, chisq/dof={}\na0={:>2.2}\xb1{:>2.2} a1={:>2.2}\xb1{:>2.2} x2={:>2.2}\xb1{:>2.2}".format(
						this_ds,
						round(chi2/ndof,3),
						popt[2], perr[2],
						popt[1], perr[1],
						popt[0], perr[0],
					))
					plt.legend()
					if show:
						plt.show()

				this_mu_ps = mod.ifn(this_mu[ftr_ch] + this_sigma[ftr_ch], *popt)
				this_mu_ps_pse = mod.ifn(this_mu[ftr_ch] + this_sigma[ftr_ch] + this_sigma_err[ftr_ch], *popt)
				this_mu_pme = mod.ifn(this_mu[ftr_ch] + this_mu_err[ftr_ch], *popt)
				# this_mu_ms = mod.ifn(this_mu[ftr_ch] - this_sigma[ftr_ch], *popt)

				this_mu[ftr_ch]     = mod.ifn(this_mu[ftr_ch], *popt)
				this_mu_err[ftr_ch] = this_mu_pme - this_mu[ftr_ch]

				this_sigma[ftr_ch] = (this_mu_ps - this_mu[ftr_ch])
				this_sigma_err[ftr_ch] = (this_mu_ps_pse - this_mu_ps)

		# concatenate arrays from this iteration to main arrays
		mu        = np.concatenate([mu       , this_mu],0)
		mu_err    = np.concatenate([mu_err   , this_mu_err],0)
		sigma     = np.concatenate([sigma    , this_sigma],0)
		sigma_err = np.concatenate([sigma_err, this_sigma_err],0)
		peak_id   = np.concatenate([peak_id  , this_peak_id],0)
		voltage   = np.concatenate([voltage  , this_voltage],0)
		scint     = np.concatenate([scint    , this_scint],0)

	valid = np.isfinite(mu) & np.isfinite(mu_err) & np.isfinite(sigma) & np.isfinite(sigma_err)
	enthr = mu>10.0
	useid = np.logical_not(np.isin(peak_id, exclude_source_peaks))

	# fig_object = plt.figure()

	for this_scint in sorted(set(scint)):
		for this_voltage in sorted(set(voltage)):
			match = [(scint[i]==this_scint) and (voltage[i]==this_voltage) for i in range(len(scint))]
			ftr_use  = (match & valid & enthr & useid)
			if show_excluded_peaks:
				ftr_show = (match & valid & enthr)
			else:
				ftr_show = ftr_use
			if any(match):
				this_mu        = mu       [ftr_use] #[_ for i,_ in enumerate(mu       ) if ftr_use[i]]
				this_mu_err    = mu_err   [ftr_use] #[_ for i,_ in enumerate(mu_err   ) if ftr_use[i]]
				this_sigma     = sigma    [ftr_use] #[_ for i,_ in enumerate(sigma    ) if ftr_use[i]]
				this_sigma_err = sigma_err[ftr_use] #[_ for i,_ in enumerate(sigma_err) if ftr_use[i]]

				if type(colors) is dict:
					col = colors.get(this_scint, {})
					if type(col) is dict:
						col = col.get(this_voltage,'k')
				else:
					col = colors

				plt.errorbar(
					mu[ftr_show],
					sigma[ftr_show],
					sigma_err[ftr_show],
					mu_err[ftr_show],
					ls="",
					marker=".",
					color=col,#colors.get(this_scint, {}).get(this_voltage,'k'),
					label="{}, {}v{}".format(this_scint, this_voltage, "" if suffix is None else ", {}".format(suffix)),
				)

				if do_fit:
					popt, pcov, chi2, ndof = fitmodel.fit_with_errors(
						mu[ftr_show],
						sigma[ftr_show],
						mu_err[ftr_show],
						sigma_err[ftr_show],
					)

					xaxis = np.linspace(min(mu[ftr_show]), max(mu[ftr_show]), 250)
					plt.plot(xaxis, fitmodel(xaxis,*popt), ls="--", marker="", color=col, )

				if plot_separately:	
					if fit_mu_vs_sigma:
						mod_sigma = model.sqrt(static_parameters=[max(mu)])
						# print(this_mu)
						# print(this_sigma)
						# print(this_mu_err)
						# print(this_sigma_err)
						popt,perr,chi2,ndof = mod_sigma.fit_with_errors(
							np.array(this_mu),
							np.array(this_sigma),
							np.array(this_mu_err),
							np.array(this_sigma_err),
						)
						xaxis = np.logspace(
							math.log(max([min(this_mu),1]),10),
							math.log(max(this_mu),10),
							500)
						plt.plot(xaxis, mod_sigma(xaxis,*popt),'k--',label="best fit")
					plt.xlabel("mu ({})".format("energy, KeV" if calibrate else "area, pVs"))
					plt.ylabel("sigma ({})".format("energy, KeV" if calibrate else "area, pVs"))
					plt.xscale("log")
					plt.yscale("log")
					plt.title("{} peaks, {}, y=q*sqrt(r+x/x0)\nq={:>2.2}\xb1{:>2.2} r={:>2.2e}\xb1{:>2.2e} x0={:>2.2}".format(
						"primary source" if require_id else "all",
						"energy (KeV)" if calibrate else "area (pVs)",
						popt[0],
						perr[0],
						popt[1],
						perr[1],
						max(mu),
					))
					plt.legend()
					if show:
						plt.show()
	

	if not plot_separately:
		# eye_c = 17.0
		# eye_r = 1.8
		eye_c = 0.3 * 2.4/1.4 * 1.15
		eye_r = 1.2
		xaxis = np.logspace(max([1,math.log(min(mu[ftr_show]),10)]),math.log(max(mu[ftr_show]),10),500)
		plt.plot(xaxis,np.sqrt(xaxis)*eye_c,'k-',label="eyeballed sqrt(x)")
		plt.plot(xaxis,np.sqrt(xaxis)*eye_c*eye_r,'k--')
		plt.plot(xaxis,np.sqrt(xaxis)*eye_c/eye_r,'k--')
		plt.xlabel("mu ({})".format("energy, KeV" if calibrate else "area, pVs"))
		plt.ylabel("sigma ({})".format("energy, KeV" if calibrate else "area, pVs"))
		plt.xscale("log")
		plt.yscale("log")
		plt.title("{} peaks, {}\nmu vs. sigma".format(
			"primary source" if require_id else "all",
			"energy (KeV)" if calibrate else "area (pVs)",
		))
		plt.legend()

		# fileio.dump_figure(fig_object, "./figs/pickled/fig3.pickle")

		if show:
			plt.show()

if __name__ == '__main__':

	cseq = [
		{"NaI1":"darkgreen"  , "NaI2":"purple"},
		{"NaI1":"forestgreen", "NaI2":"magenta"},
		{"NaI1":"springgreen", "NaI2":"plum"},
	]


	calibrate = True

	# amounts = [0,5,25]
	amounts = [None]

	for i,amount in enumerate(amounts):
		main(
			ni=amount,
			show=False,

			do_fit=False,

			calibrate=calibrate,

			# ni_suffix_template = "_1mv_ni{}",
			ni_suffix_template = "_ni{}",

			# exclude_source_peaks = [-1,602, 101,300,400,401,  200],
			# require_id = False,
			# show_excluded_peaks = True,

			colors=cseq[i],
			suffix=None if amount is None else "+{}mV noise".format(amount),
		)

	plt.show()



