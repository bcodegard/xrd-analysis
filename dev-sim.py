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
CH_ALL = [1,2,3,4]
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

BRANCHES_A = {"area_3046_{}".format(n) for n in CH_ALL}

# specify background data
DIR_BG_DATA = '../xrd-analysis/data/root/scintillator'
BG_RUNS = [4225, 4226]
BG_FILE = "Run{}.root"
BG_FILES = [os.sep.join([DIR_EXP_DATA, EXP_FILE.format(r)]) for r in BG_RUNS]

# specify simulation data
DIR_SIM_DATA = '../xrd-analysis/data/root/simulation'
SIM_SOURCES = ["Am241", "Ba133", "Cd109", "Co57", "Mn54", "Na22"]
SIM_FILE = "{}.root"
SIM_FILES = {_:os.sep.join([DIR_SIM_DATA, SIM_FILE.format(_)]) for _ in SIM_SOURCES}

BRANCHES_E = {"Edep_MeV_Si{}".format(n) for n in CH}

# energy for plotting vlines
peak_e = {
	"Am241": [ 26.34,  59.54],
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
		3: 35789.0 / 30.0,
		4: 33384.1 / 60.0,
	}

	# # initial eyeballing, same for all channels
	# rp_eref = 122  # some reference energy
	# rp_rref = 0.04 # pmt resolution contr. at the reference energy
	# rho_p = rp_eref * rp_rref**2 # calculate rho_p from these quantities
	# res_s = 0.06 / W_TO_S # 6%, converted from FWHM to sigma

	# per channel, refined a bit
	it = 4
	res_s   = {1:0.060, 2:0.080, 3:0.075}
	rp_rref = {1:0.035, 2:0.034, 3:0.038}
	rp_eref = {1:122  , 2:122  , 3:122  }
	rho_p = {c:(rp_eref[c] * rp_rref[c]**2) for c in CH}

	buds_sample_xf = [
		data.bud_function(
			"e{}".format(n),
			"a{}".format(n),
			sampler(
				sample_energy_to_area,
				gamma[n],
				res_s[n],
				rho_p[n],
			),
		) for n in CH
	]

	# comparison range for source spectra vs. sim+bg
	def_area_range = [10000,200000]
	# src_area_range = {}
	src_area_range = {
		"Am241":{1:[31000,70000], 2:[32000,72000], 3:[33000,75000]},
		"Ba133":{1:[10000,70000], 2:[10000,70000], 3:[10000,70000]},
		"Cd109":{1:[10000,70000], 2:[10000,70000], 3:[10000,70000]},
		"Co57" :{1:[10000,80000], 2:[10000,82000], 3:[10000,85000]},
		# "Mn54" :[],
		# "Na22" :[],
	}

	# range of spectra where only background is present
	bg_area_range = {
		"Am241":[100000, 500000],
		"Ba133":[460000, 620000],
		"Cd109":[120000, 600000],
		"Co57" :[190000, 600000],
	}

	save = True
	rs = 3
	fname = "./figs/mix_comp_{src}_rs{rs}.png"
	# fname = "./figs/mix_comp_{src}_it{it}.png"

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

		# apply lyso veto at 14000 pVs
		this_bm.mask(data.cut("area_3046_4",hi=14000.0), apply_mask=True)

	# load branches and create branch manager for experimental background spectra
	bg_branch_sets = []
	for file in BG_FILES:
		branches = fileio.load_branches(file, BRANCHES_A)
		bg_branch_sets.append(branches)
	combined_branches = {key:np.concatenate([_[key] for _ in bg_branch_sets]) for key in BRANCHES_A}
	bm_bg = data.BranchManager(combined_branches)
	bm_bg.mask(data.cut("area_3046_4",hi=14000.0), apply_mask=True)


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


	# set of sources with both simulation and experimental data
	src_comp = sorted(set(bms_sim.keys()) & set(bms_exp.keys()))

	# make and plot comparisons
	all_rchi2 = []
	all_ident = []
	for src in src_comp:

		# figure and axes for plotting results
		ncols=len(CH)
		nrows=1
		fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(ncols*8-1,nrows*8-1))
		ch_ax = {i+1:ax[i] for i in range(ncols)}

		nbins = 200
		xlog  = False
		ylog  = False

		for ch in CH:
			print("{}, channel {}".format(src, ch))


			# assign data to locals
			this_sim_br = "a{}".format(ch)
			this_exp_br = "area_3046_{}".format(ch)
			this_sim = bms_sim[src][this_sim_br]
			this_exp = bms_exp[src][this_exp_br]
			this_bg  = bm_bg[this_exp_br]
			print("sim events: {}".format(this_sim.size))
			print("exp events: {}".format(this_exp.size))
			print("bg  events: {}".format(this_bg.size ))
			print("")


			# calculate background contributions in experimental source data
			bg_lo, bg_hi = bg_area_range[src]
			ns_in_b = ((this_exp > bg_lo) & (this_exp < bg_hi)).sum()
			nb_in_b = ((this_bg  > bg_lo) & (this_bg  < bg_hi)).sum()
			src_lo, src_hi = src_area_range.get(src,{}).get(ch,def_area_range)
			ns_in_s = ((this_exp > src_lo) & (this_exp < src_hi)).sum()
			nb_in_s = ((this_bg  > src_lo) & (this_bg  < src_hi)).sum()
			print("src spectrum events in bg  sample range: {}".format(ns_in_b))
			print("bg  spectrum events in bg  sample range: {}".format(nb_in_b))
			print("src spectrum events in src sample range: {}".format(ns_in_s))
			print("bg  spectrum events in src sample range: {}".format(nb_in_s))

			rsb_in_b = ns_in_b / nb_in_b
			rsb_in_s = ns_in_s / nb_in_s
			print("ns / nb from bg  sample range: {:.4f}".format(rsb_in_b))
			print("ns / nb from src sample range: {:.4f}".format(rsb_in_s))

			pbg = rsb_in_b / rsb_in_s
			print("est. frac of src dataset in [{},{}] expected to be from bg: {:.4f}".format(src_lo,src_hi,pbg))
			print("")

			# print("set pbg to zero to remove bg contribution temporarily")
			# pbg=0
			# print("")


			# determine number of events to take from each dataset
			ftr_sim_range = (this_sim > src_lo) & (this_sim < src_hi)
			ftr_exp_range = (this_exp > src_lo) & (this_exp < src_hi)
			ftr_bg_range  = (this_bg  > src_lo) & (this_bg  < src_hi)
			this_sim = this_sim[ ftr_sim_range ]
			this_exp = this_exp[ ftr_exp_range ]
			this_bg  = this_bg[  ftr_bg_range  ]
			n_sim = this_sim.size
			n_exp = this_exp.size
			n_bg  = this_bg.size
			print("sim evts in test range: {}".format(n_sim))
			print("bg  evts in test range: {}".format(n_bg ))
			print("exp evts in test range: {}".format(n_exp))
			
			nmax_mix_from_sim = math.floor(n_sim / (1 - pbg)) if (1-pbg) else np.inf
			nmax_mix_from_bg  = math.floor(n_bg  / (    pbg)) if pbg else np.inf
			nmax_mix = min([nmax_mix_from_sim,nmax_mix_from_bg])
			print("max # evts in mixed sim+bg from sim: {}".format(nmax_mix_from_sim))
			print("max # evts in mixed sim+bg from bg : {}".format(nmax_mix_from_bg ))

			print("max # evts in mixed sim+bg : {}".format(nmax_mix))
			print("exp events in range        : {}".format(n_exp))
			scale = nmax_mix / n_exp
			print("ratio: {:.4f}".format(scale))
			print("")

			# take smaller number between sim+bg mix and exp as number for comparison
			ncomp = min([nmax_mix, n_exp])
			print("comparing samples of size {}".format(ncomp))


			# mix sim and bg datasets
			nmix_bg  = math.floor(ncomp*pbg)
			nmix_sim = ncomp - nmix_bg
			this_sim_scaled = np.concatenate([
				np.random.choice(this_sim, nmix_sim, replace=False),
				np.random.choice(this_bg , nmix_bg , replace=False),
			])
			print("mixed {} sim events and {} background events".format(nmix_sim, nmix_bg))

			# downsample exp dataset
			if n_exp > ncomp:
				this_exp_scaled = np.random.choice(this_exp, ncomp, replace=False)
				print("chose {} events from experimental source dataset".format(ncomp))
			else:
				this_exp_scaled = this_exp
				print("used entire experimental source dataset")


			# calculate bins
			if xlog:
				this_bins = data.edges_log(src_lo, src_hi, nbins)
			else:
				this_bins = data.edges_lin(src_lo, src_hi, nbins)

			# plot scaled histograms
			this_ax = ch_ax[ch]
			sim_counts, _, _ = this_ax.hist(this_sim_scaled, bins=this_bins, histtype='step', label="sim")
			exp_counts, _, _ = this_ax.hist(this_exp_scaled, bins=this_bins, histtype='step', label="exp")

			chi2,ndof = data.chi2_identical_poisson(sim_counts, exp_counts)
			print("chi2/dof = {:.2f}/{} = {:.4f}".format(chi2,ndof,chi2/ndof))
			all_rchi2.append(chi2/ndof)
			all_ident.append((src,ch))

			# plot peak energy vlines
			peak_ls = ['solid', 'dashed', 'dotted']
			for i,pe in enumerate(peak_e[src]):
				pa = gamma[ch] * pe
				this_ax.axvline(pa, color='k', ls=peak_ls[i], label="{:.1f} KeV".format(pe))

			this_ax.set_xscale("log" if xlog else "linear")
			this_ax.set_yscale("log" if xlog else "linear")
			this_ax.set_xlabel("area (pVs)")
			this_ax.set_ylabel("counts")
			this_ax.set_title("channel {}, p(bg) = {:.4f}\nchi2/dof = {:.1f}/{} = {:.4f}".format(ch,pbg,chi2,ndof,chi2/ndof))
			this_ax.legend()

			print("\n")

		plt.suptitle("{}, try {}, res_s = {}, rp_rref = {}".format(src, it, [res_s[i] for i in CH], [rp_rref[i] for i in CH]))
		if save:
			plt.savefig(fname.format(src=src.lower(),it=it,rs=rs))
		plt.show()
		print("\n")

	print("all chi2/dof")
	for i,rchi2 in enumerate(all_rchi2):
		ident = all_ident[i]
		print("{ident[0]:<6} - {ident[1]:<6} - {}".format(rchi2,ident=ident))
	print("")
	








if __name__ == '__main__':
	main()
