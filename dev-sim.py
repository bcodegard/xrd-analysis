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
import matplotlib.lines  as lines
import scipy.optimize    as opt

import utils.fileio  as fileio
import utils.display as display
import utils.model   as model
import utils.data    as data




CH = [1,2,3]#,4]
W_TO_S = (8 * math.log(2)) ** 0.5

# specify expermental data
DIR_EXP_DATA = '../xrd-analysis/data/root/scintillator'
EXP_RUNS = {
	"Am241":4291,
	"Ba133":4293,
	"Cd109":4292,
	"Co57" :4294,
	"Mn54" :0,
	"Na22" :0,
}
EXP_FILE = "Run{}.root"
EXP_FILES = {s:os.sep.join([DIR_EXP_DATA, EXP_FILE.format(r)]) for s,r in EXP_RUNS.items() if r}

BRANCHES_A = {"area_3046_{}".format(n) for n in CH}

# specify simulation data
DIR_SIM_DATA = '../xrd-analysis/data/root/simulation'
SIM_SOURCES = ["Am241", "Ba133", "Cd109", "Co57", "Mn54", "Na22"]
SIM_FILE = "{}.root"
SIM_FILES = {_:os.sep.join([DIR_SIM_DATA, SIM_FILE.format(_)]) for _ in SIM_SOURCES}

BRANCHES_E = {"Edep_MeV_Si{}".format(n) for n in CH}

# energy for plotting vlines
peak_e = {
	"Am241": [ 59.54        ],
	"Ba133": [ 30.85,  81.00],
	"Cd109": [ 22.1 ,  88.04],
	"Co57" : [122.06, 136.47],
	"Mn54" : [834.85        ],
	"Na22" : [511.0         ],
}



def sample_energy_to_area(e, gamma, res_s, rho_p):
	""""""
	area = np.zeros(e.shape)

	ftr_hit  = (e>0)
	e_hit = e[ftr_hit]

	# center of area distribution is linear function of energy
	mu = e_hit*gamma

	# resolution
	# res_s = res_s
	rp_squared = rho_p / e_hit
	res = np.sqrt(res_s**2 + rp_squared)

	# width is resolution * mu
	sigma = mu * res

	# sample from the distribution
	area[ftr_hit] = np.random.normal(mu, sigma)

	return area

def sampler(fn, *params):
	def xf(x):
		return fn(x, *params)
	return xf




def main():

	# create bud functions for converting MeV to KeV
	buds_mev_to_kev = [
		data.bud_function(
			"Edep_MeV_Si{}".format(n),
			"e{}".format(n),
			lambda e:e*1000
		) for n in CH
	]

	# create mask function for requiring that at least one scintillator
	# has energy deposited
	mask_any_activity = data.mask_any(*[
		data.cut(
			'Edep_MeV_Si{}'.format(n),
			0
		) for n in CH
	])

	# sampling conversion parameters
	# just guesses for now to see if things look right
	gamma = {
		1: 65835.2 / 60.0,
		2: 67874.3 / 60.0,
		3: 70020.9 / 60.0,
		4: 33384.1 / 60.0,
	}
	rp_eref = 122  # some reference energy
	rp_rref = 0.04 # pmt resolution contr. at the reference energy
	rho_p = rp_eref * rp_rref**2 # calculate rho_p from these quantities
	res_s = 0.06 / W_TO_S # 6%, converted from FWHM to sigma

	buds_sample_xf = [
		data.bud_function(
			"e{}".format(n),
			"a{}".format(n),
			sampler(
				sample_energy_to_area,
				gamma[n],
				res_s,
				rho_p,
			),
		) for n in CH
	]

	# load branches and create branch managers for simulation data
	bms_sim = {}
	for src in SIM_SOURCES:
		
		# load branches corresponding to energy deposits in each scint
		branches = fileio.load_branches(SIM_FILES[src], BRANCHES_E)

		# create and store branch manager
		this_bm = data.BranchManager(branches)
		bms_sim[src] = this_bm

		# discard events with no energy deposit in any scintillator
		print("{}, n events, n events with any energy deposit".format(src))
		print(this_bm['Edep_MeV_Si1'].shape)
		this_bm.mask(mask_any_activity, apply_mask = True)
		print(this_bm['Edep_MeV_Si1'].shape)
		print("")

		# make new branches to convert MeV to KeV for convenience
		this_bm.bud(buds_mev_to_kev)

		# make new branches using area sampling
		this_bm.bud(buds_sample_xf)

	# load branches and create branch managers for experimental data
	bms_exp = {}
	for src,file in EXP_FILES.items():
		branches = fileio.load_branches(file, BRANCHES_A)
		this_bm = data.BranchManager(branches)
		bms_exp[src] = this_bm


	# test view of some data
	# show_sources = ["Ba133"]
	# show_sources = bms_sim.keys()
	show_sources = []
	for src in show_sources:
		# plt.subplot(121)
		plt.hist(bms_sim[src]["e1"], histtype='step', bins=400, label="ch 1")
		plt.hist(bms_sim[src]["e2"], histtype='step', bins=400, label="ch 2")
		plt.hist(bms_sim[src]["e3"], histtype='step', bins=400, label="ch 3")
		plt.hist(bms_sim[src]["e4"], histtype='step', bins=400, label="ch 4")
		plt.yscale('log')
		plt.xlabel('simulated energy deposits (KeV)')
		plt.ylabel('counts')
		plt.title(src)
		plt.legend()
		# plt.savefig("./figs/sim_{}_edep.png".format(src.lower()))
		plt.show()

		# plt.subplot(122)
		plt.hist(bms_sim[src]["a1"], histtype='step', bins=400, label="ch 1")
		plt.hist(bms_sim[src]["a2"], histtype='step', bins=400, label="ch 2")
		plt.hist(bms_sim[src]["a3"], histtype='step', bins=400, label="ch 3")
		plt.hist(bms_sim[src]["a4"], histtype='step', bins=400, label="ch 4")
		plt.yscale('log')
		plt.xlabel('sampled area distribution (pVs)')
		plt.ylabel('counts')
		plt.title(src)
		plt.legend()
		plt.show()



	# define area ranges per source
	# area_ranges = {
	# 	"Am241":[1000,   90000],
	# 	"Ba133":[5000,  450000],
	# 	"Cd109":[2000,  115000],
	# 	"Co57" :[2000,  175000],
	# 	"Mn54" :[2000, 1080000],
	# 	"Na22" :[2000,  650000],
	# }
	area_ranges = {
		"Am241":[15000, 80000],
		"Ba133":[20000,110000],
		"Cd109":[15000,115000],
		"Co57" :[25000,175000],
		# "Mn54" :[],
		# "Na22" :[],
	}

	# set of sources with both simulation and experimental data
	src_comp = set(bms_sim.keys()) & set(bms_exp.keys())

	# make and plot comparisons
	for src in src_comp:

		# figure and axes for plotting results
		ncols=len(CH)
		nrows=1
		fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(ncols*8-1,nrows*8-1))
		ch_ax = {i+1:ax[i] for i in range(ncols)}
		
		nbins = 500
		xlog  = False
		ylog  = False

		for ch in CH:

			print("{}, channel {}".format(src, ch))

			# assign data to locals
			this_sim_br = "a{}".format(ch)
			this_exp_br = "area_3046_{}".format(ch)
			this_sim = bms_sim[src][this_sim_br]
			this_exp = bms_exp[src][this_exp_br]
			print("sim events: {}".format(this_sim.size))
			print("exp events: {}".format(this_exp.size))
			print("")

			# determine number of events to take from experimental dataset
			this_lo, this_hi = area_ranges[src]
			ftr_sim_range = (this_sim > this_lo) & (this_sim < this_hi)
			ftr_exp_range = (this_exp > this_lo) & (this_exp < this_hi)
			this_sim = this_sim[ftr_sim_range]
			this_exp = this_exp[ftr_exp_range]
			n_sim = this_sim.size
			n_exp = this_exp.size
			scale = n_sim / n_exp
			print("sim events in range: {}".format(n_sim))
			print("exp events in range: {}".format(n_exp))
			print("ratio: {:.4f}".format(scale))

			# scale down larger dataset by randomly choosing number of events
			# equal to the number of events in the smaller dataset
			if scale > 1:
				print("sim has more data")
				this_sim_scaled = np.random.choice(this_sim, n_exp, replace=False)
				this_exp_scaled = this_exp
			else:
				print("exp has more data")
				this_sim_scaled = this_sim
				this_exp_scaled = np.random.choice(this_exp, n_sim, replace=False)
			print("")

			# calculate bins
			if xlog:
				this_bins = data.edges_log(this_lo, this_hi, nbins)
			else:
				this_bins = data.edges_lin(this_lo, this_hi, nbins)

			# plot scaled histograms
			this_ax = ch_ax[ch]
			this_ax.hist(this_sim_scaled, bins=this_bins, histtype='step', label="sim")
			this_ax.hist(this_exp_scaled, bins=this_bins, histtype='step', label="exp")

			# plot peak energy vlines
			peak_ls = ['solid', 'dashed', 'dotted']
			for i,pe in enumerate(peak_e[src]):
				pa = gamma[ch] * pe
				this_ax.axvline(pa, color='k', ls=peak_ls[i], label="{:.1f} KeV".format(pe))

			this_ax.set_xscale("log" if xlog else "linear")
			this_ax.set_yscale("log" if xlog else "linear")
			this_ax.set_xlabel("area (pVs)")
			this_ax.set_ylabel("counts")
			this_ax.set_title("channel {}".format(ch))
			this_ax.legend()

		plt.suptitle(src)
		plt.savefig("./figs/sim_vs_exp_{}.png".format(src.lower()))
		plt.show()
		print("\n")

	








if __name__ == '__main__':
	main()
