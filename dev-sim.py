"""
"""

__author__ = "Brunel Odegard"
__version__ = "0.0"


import sys
import time
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




ROOT_8_LOG_2 = (8 * math.log(2)) ** 0.5
ONE_OVER_ROOT_TAU = 1 / (2 * math.pi)

# comparisons for floats
iseq = lambda f1,f2,eps=1e-9:abs(f1-f2)<eps

# convenient way to discard values which are zero or negative
positive = lambda ar:ar[ar>0]




# class properties(np.ndarray):
# 	"""hold parameters and allow access by integer indexing
# 	as well as propety accessing"""

# 	def __init__(self, obj, names, np_kwargs={}):
# 		super(properties, self).__init__(obj, **np_kwargs)
	
# 	def __getattribute__(self, attr):
# 		"""return value of named element if attr is name of element
# 		else return super result"""

# 		if attr in self.attr_names:
# 			return self[attr]

# 		else:
# 			return super(properties,self).__getattribute__(attr)

# 	def __slice__(self, ):
# 		...




n_squish_pars = 2

# exponential thingy
squish_p0 = [0.1, 160.0]
k = 2
sat_crx     = lambda A1,b,a:A1*(1+b*np.exp(-(a*1000/A1)**k))
sat_crx_jac = lambda A1,b,a:1+(1-k*(a*1000/A1)**k)*b*np.exp(-(a*1000/A1)**k)

# # one over exponential thingy
# squish_p0 = [0.1, 160]
# k=2
# sat_crx     = lambda A1,b,a:A1 / (1 - b*np.exp(-(a*1000/A1)**k))
# sat_crx_jac = lambda A1,b,a:(1 - b*np.exp(-(a*1000/A1)**k)*(k*(a/A1)**k - 1)) / ((1 - b*np.exp(-(a*1000/A1)**k))**2)

# # # power law thingy
# squish_p0 = [0.01, 160]
# k = -2
# sat_crx     = lambda A1,b,a:A1*(1 + b*(a*1000/A1)**k)
# sat_crx_jac = lambda A1,b,a:1 + (1-k)*b*(a*1000/A1)**k

# # tanh
# squish_p0 = [1.0, 60.0]
# sat_crx     = lambda A1,b,a:b*(a*1000)*np.sinh(A1/(a*1000))
# sat_crx_jac = lambda A1,b,a:b*(a*1000)*np.cosh(A1/(a*1000))/A1

# 

class transformer(object):


	def __init__(self, en, en_edges=None, ar_edges=None):

		self.en       = en
		self.en_edges = en_edges
		self.ar_edges = ar_edges

		if (en_edges is not None) and (ar_edges is not None):
			self.setup()

	def __call__(self, ar_mids, *params, en_counts=None):

		# use internal values or broadcast supplied values
		if en_counts is None:
			dd_en_counts = self.dd_en_counts
		else:
			dd_en_counts = en_counts[None,:]

		if ar_mids is None:
			dd_ar_mids = self.dd_ar_mids
		else:
			dd_ar_mids = ar_mids[:,None]
			
		# unpack parameters
		C, gamma, *res_s, rho_p, ps0, ps1 = params

		# calculate pieces
		dd_mu = self.dd_en_mids * gamma
		dd_sigma = self.res(self.dd_en_mids, *params) * dd_mu
		# 
		# calculate inverse transformed area midpoints
		dd_ar_mids_it = sat_crx(dd_ar_mids, ps0, ps1)
		jacobian_it   = sat_crx_jac(dd_ar_mids, ps0, ps1)
		# 
		mult = C * ONE_OVER_ROOT_TAU * (dd_en_counts / dd_sigma) * jacobian_it
		expo = np.exp((-0.5) * ((dd_ar_mids_it - dd_mu) / (dd_sigma))**2)

		dd_ar_counts = mult*expo

		# sum along energy axis to get area counts
		ar_counts = dd_ar_counts.sum(1)
		return ar_counts

	def res(self, en, *params):
		
		# unpack
		C, gamma, *res_s, rho_p, ps0, ps1 = params

		rs0 = res_s[0]
		rs1 = res_s[1]

		# PMT resolution
		res_p_squared = rho_p / en

		# scintillator resolution
		# res_s_squared = rs0**2
		res_s_squared = rs0 - rs1*en

		return np.sqrt(abs(res_s_squared) + abs(res_p_squared))

	def setup(self, en_edges=None, ar_edges=None):
		"""performs binning and calculates static objects"""
		
		# assume new edges if specified
		if en_edges is not None:
			self.en_edges = en_edges
		if ar_edges is not None:
			self.ar_edges = ar_edges

		# if edges were specified at some point
		if (self.en_edges is not None) and (self.ar_edges is not None):
			self._bin()

		# edges not specified here or in init
		else:
			raise ValueError("need values for ar_edges and en_edges to perform setup")

	def _bin(self):
		"""calculates and stores bin midpoints and counts"""

		# energy midpoints and counts
		self.en_counts, en_edges = np.histogram(self.en, self.en_edges)
		if not iseq(0, (en_edges-self.en_edges).sum()):
			raise Warning("given and returned bin edges do not match!")
		self.en_mids = 0.5*(en_edges[1:] + en_edges[:-1])
		self.en_res = self.en_mids.size

		# area midpoints
		self.ar_mids = 0.5*(self.ar_edges[1:] + self.ar_edges[:-1])
		self.ar_res = self.ar_mids.size

		# 2d arrays
		# axis 0 = area
		# axis 1 = energy
		self.dd_en_counts = np.zeros([self.ar_res, self.en_res])
		self.dd_en_counts[:,:] = self.en_counts[None,:]
		self.dd_en_mids = np.zeros([self.ar_res, self.en_res])
		self.dd_en_mids[:,:] = self.en_mids[None,:]
		self.dd_ar_mids = np.zeros([self.ar_res, self.en_res])
		self.dd_ar_mids[:,:] = self.ar_mids[:,None]

class transformer_mixer(transformer):
	"""transformer subclass which also mixes in a background spectrum"""
	def __init__(self, en, en_edges, bg, ar_edges):
		self.bg = bg
		super(transformer_mixer, self).__init__(en, en_edges, ar_edges)
		# if ar_edges is not None:
		# 	self.setup_background()

	def setup(self,*args,**kwargs):
		super(transformer_mixer,self).setup(*args,**kwargs)
		if self.ar_edges is not None:
			self.setup_background()

	def setup_background(self):
		if self.ar_edges is not None:
			self._bin_background()
		else:
			raise ValueError("need ar_edges to perform setup_background")

	def _bin_background(self):
		self.bg_counts, bg_edges = np.histogram(self.bg, self.ar_edges)
		if not iseq(0, (bg_edges-self.ar_edges).sum()):
			raise Warning("given and returned bin values do not match!")

	def __call__(self, ar_mids, *params, en_counts=None, bg_counts=None, incl_src=True, incl_bg=True):
		"""calculate area"""

		# unpack parameters
		C, D, gamma, *res_s, rho_p, ps0, ps1 = params

		# calculate simulated source contribution
		p_src   = [C, gamma, *res_s, rho_p, ps0, ps1]
		res_src = super(transformer_mixer,self).__call__(ar_mids, *p_src, en_counts=en_counts)

		# calculate background contribution
		res_bg = D * (bg_counts if bg_counts is not None else self.bg_counts)

		# return sum of source and background contributions
		if incl_src and incl_bg:
			return res_src + res_bg
		elif incl_src:
			return res_src
		elif incl_bg:
			return res_bg
		else:
			return 0



class routine(object):

	verbosity = 2

	ch_comp = [1,2,3]
	ch_load = [1,2,3,4]

	# specify expermental source data
	dir_exp_data = '../xrd-analysis/data/root/scintillator'
	exp_runs = {
		"Am241":4291,
		"Ba133":4293,
		"Cd109":4292,
		"Co57" :4294,
		"Mn54" :0,
		"Na22" :0,
	}
	exp_file = "Run{}.root"

	# specify background data
	dir_bg_data = '../xrd-analysis/data/root/scintillator'
	bg_runs = [4225, 4226]
	bg_file = "Run{}.root"

	# specify simulation data
	dir_sim_data = '../xrd-analysis/data/root/simulation'
	# sim_sources = ["Am241", "Ba133", "Cd109", "Co57", "Mn54", "Na22"]
	sim_sources = ["Am241","Ba133","Cd109","Co57"]
	# sim_sources = ["Am241","Cd109","Ba133"]
	sim_file = "{}.root"
	



	# ranges capturing the entire range of each source's spectrum
	src_area_range_full = {
		"Am241": [10000,   85000],
		"Ba133": [ 5000,  520000],
		"Cd109": [10000,  130000],
		"Co57" : [ 5000,  200000],
		"Mn54" : [10000, 1200000],
		"Na22" : [10000,  800000],
	}
	src_energy_range_full = {
		"Am241": [0,  60],
		"Ba133": [0, 380],
		"Cd109": [0,  90],
		"Co57" : [0, 140],
		"Mn54" : [0, 900],
		"Na22" : [0, 520],
	}

	# comparison range for source spectra vs. sim+bg
	src_range_default = [10000,1000000]
	
	# # restrictive ranges, for excluding PMT saturation and low-tail of Am241
	# src_area_range = {
	# 	"Am241":{1:[31000,70000], 2:[32000,72000], 3:[33000,75000]},
	# 	"Ba133":{1:[10000,55000], 2:[10000,55000], 3:[10000,55000]},
	# 	"Cd109":{1:[10000,60000], 2:[10000,60000], 3:[10000,60000]},
	# 	"Co57" :{1:[12000,65000], 2:[12000,65000], 3:[12000,65000]},
	# 	"Mn54" :[],
	# 	"Na22" :[],
	# }

	# # less restrictive, but still mostly excluding PMT saturation
	# src_area_range = {
	# 	"Am241":{1:[11000,80000], 2:[11000,80000], 3:[11000,80000]},
	# 	"Ba133":{1:[11000,80000], 2:[11000,80000], 3:[11000,80000]},
	# 	"Cd109":{1:[11000,80000], 2:[11000,80000], 3:[11000,80000]},
	# 	"Co57" :{1:[11000,80000], 2:[11000,80000], 3:[11000,80000]},
	# 	"Mn54" :[],
	# 	"Na22" :[],
	# }

	# including pmt saturation range
	src_area_range = {
		"Am241":{1:[11000, 90000], 2:[11000, 90000], 3:[11000, 90000]},
		"Ba133":{1:[11000,600000], 2:[11000,600000], 3:[11000,600000]},
		"Cd109":{1:[11000,130000], 2:[11000,130000], 3:[11000,130000]},
		"Co57" :{1:[11000,200000], 2:[11000,200000], 3:[11000,200000]},
		"Mn54" :[],
		"Na22" :[],
	}	

	# source peak energies for plotting vlines
	peak_e = {
		"Am241": [ 26.34,  59.54],
		"Ba133": [ 30.85,  81.00],
		"Cd109": [ 22.1 ,  88.04],
		"Co57" : [122.06, 136.47],
		"Mn54" : [834.85        ],
		"Na22" : [511.0         ],
	}

	# conversion parameters
	# just guesses which should be close enough for optimization
	gamma = {
		1: 65835.2 / 60.0,
		2: 67874.3 / 60.0,
		3: 35789.0 / 30.0,
		4: 33384.1 / 60.0,
	}
	res_s   = {1:0.060, 2:0.080, 3:0.075}
	rp_rref = {1:0.035, 2:0.034, 3:0.038}
	rp_eref = {1:122  , 2:122  , 3:122  }

	# discard events with area in channel 4 (LYSO) greater than this
	# todo: by channel
	lyso_cut_hi = 14000.0
	
	lyso_branch = "area_3046_4"
	exp_area_pvs = "area_3046_{}"

	sim_edep_mev = "Edep_MeV_Si{}"
	sim_edep_kev = "e{}"
	sim_area_pvs = "a{}"

	# transformer resolution for energy and area
	xf_res_en = 200
	xf_res_ar = 200


	def vp(self, level, *args, **kwargs):
		"""verbosity print. print if verbosity >= level."""
		if self.verbosity >= level:
			print(*args, **kwargs)

	def __init__(self, ):
		
		self.rho_p = {c:(self.rp_eref[c] * self.rp_rref[c]**2) for c in self.ch_comp}

		# branches to load for energy and area
		self.branches_a = {self.exp_area_pvs.format(n) for n in self.ch_load}
		self.branches_e = {self.sim_edep_mev.format(n) for n in self.ch_comp}

		# compose lists and dicts of files to be loaded
		self.exp_files = {s:os.sep.join([self.dir_exp_data, self.exp_file.format(r)]) for s,r in self.exp_runs.items() if r}
		self.sim_files = {_:os.sep.join([self.dir_sim_data, self.sim_file.format(_)]) for _ in self.sim_sources}
		self.bg_files  = [os.sep.join([self.dir_exp_data, self.exp_file.format(r)]) for r in self.bg_runs]

		# compose lists of sources
		self.sim_sources = sorted(self.sim_sources)
		self.exp_sources = sorted(self.exp_runs.keys())
		self.all_sources = sorted(set(self.exp_sources) & set(self.sim_sources))
		self.any_sources = sorted(set(self.exp_sources) | set(self.sim_sources))
		self.source_index = {s:i for i,s in enumerate(self.any_sources)}

		# todo: accept args to determine currently hard-coded parameters

	def procure_data(self):
		"""load data and process into forms needed by
		the rest of the routine"""

		# calculate bin edges and midpoints
		# 
		# full ranges, which don't depend on channel (yet)
		self.en_edges_full = {}
		self.ar_edges_full = {}
		self.en_mids_full  = {}
		self.ar_mids_full  = {}
		# 
		# comparison ranges, which depend on source and channel
		self.en_edges = {_:{} for _ in self.sim_sources}
		self.ar_edges = {_:{} for _ in self.sim_sources}
		self.en_mids  = {_:{} for _ in self.sim_sources}
		self.ar_mids  = {_:{} for _ in self.sim_sources}
		for src in self.sim_sources:
			self.en_edges_full[src] = data.edges_lin(*self.src_energy_range_full[src],self.xf_res_en)
			self.ar_edges_full[src] = data.edges_lin(*self.src_area_range_full[  src],self.xf_res_ar)
			self.en_mids_full[src] = 0.5 * (self.en_edges_full[src][1:] + self.en_edges_full[src][:-1])
			self.ar_mids_full[src] = 0.5 * (self.ar_edges_full[src][1:] + self.ar_edges_full[src][:-1])
			for ch in self.ch_comp:
				self.ar_edges[src][ch] = data.edges_lin(*self.src_area_range.get(src,{}).get(ch,self.src_range_default),self.xf_res_ar)
				self.ar_mids[src][ch] = 0.5 * (self.ar_edges[src][ch][1:] + self.ar_edges[src][ch][:-1])
				self.en_edges[src][ch] = self.en_edges_full[src]
				self.en_mids[src][ch]  = self.en_mids_full[src]


		# create bud functions for converting MeV to KeV
		buds_mev_to_kev = [
			data.bud_function(
				self.sim_edep_mev.format(n),
				self.sim_edep_kev.format(n),
				lambda e:e*1000
			) for n in self.ch_comp
		]

		# create mask function for requiring that at least one scintillator
		# has energy deposited
		mask_any_activity = data.mask_any(*[
			data.cut(
				self.sim_edep_mev.format(n),
				0
			) for n in self.ch_comp
		])


		# load branches and create branch managers for experimental data
		self.bms_obs_src = {}
		for src,file in self.exp_files.items():
			branches = fileio.load_branches(file, self.branches_a)
			this_bm = data.BranchManager(branches)
			self.bms_obs_src[src] = this_bm

			# apply lyso veto
			this_bm.mask(data.cut(self.lyso_branch,hi=self.lyso_cut_hi),apply_mask=True)

		# load branches and create branch manager for experimental background spectra
		bg_branch_sets = []
		for file in self.bg_files:
			branches = fileio.load_branches(file, self.branches_a)
			bg_branch_sets.append(branches)
		combined_branches = {key:np.concatenate([_[key] for _ in bg_branch_sets]) for key in self.branches_a}
		self.bm_obs_bg = data.BranchManager(combined_branches)
		self.bm_obs_bg.mask(data.cut(self.lyso_branch,hi=self.lyso_cut_hi),apply_mask=True)

		# load branches and create branch managers for simulation data
		self.bms_sim_src = {}
		self.sim_e = {_:{} for _ in self.sim_sources}
		self.sim_bg = {}
		self.transformers = {_:{} for _ in self.sim_sources}
		for src in self.sim_sources:

			# load branches corresponding to energy deposits in each scint
			branches = fileio.load_branches(self.sim_files[src], self.branches_e)

			# create and store branch manager
			this_bm = data.BranchManager(branches)
			self.bms_sim_src[src] = this_bm

			# discard events with no energy deposit in any scintillator
			self.vp(1, "{}, n events, n events with any energy deposit".format(src))
			self.vp(1, this_bm[self.sim_edep_mev.format(self.ch_comp[0])].shape[0])
			this_bm.mask(mask_any_activity, apply_mask = True)
			self.vp(1, this_bm[self.sim_edep_mev.format(self.ch_comp[0])].shape[0])

			# make new branches to convert MeV to KeV for convenience
			this_bm.bud(buds_mev_to_kev)

			# make transformers
			for ch in self.ch_comp:
				this_sim_e  = positive(this_bm[self.sim_edep_kev.format(ch)])
				this_sim_bg = positive(self.bm_obs_bg[self.exp_area_pvs.format(ch)])

				self.sim_e[src][ch]  = this_sim_e
				self.sim_bg[ch]      = this_sim_bg

				t1=time.time()
				this_transformer = transformer_mixer(
					this_sim_e,
					self.en_edges[src][ch],
					this_sim_bg,
					self.ar_edges[src][ch],
				)
				t2=time.time()
				self.vp(2,"setting up transformer {},{} took {:.3f} ms".format(src,ch,(t2-t1)*1000))
				self.transformers[src][ch] = this_transformer
			self.vp(1,"")

	def plot_basic_check(self, sources=None, savefig="", ):
		"""test view of some of the loaded data"""
		# savefig = "sample_vs_spread_{}_tweak_counts"

		guess_c = 380
		guess_d = 1.5

		if sources is None:
			show_sources = self.bms_sim_src.keys()

		for src in show_sources:

			# plt.subplot(121)
			# plt.hist(bms_sim[src]["e1"], histtype='step', bins=400, label="ch 1")
			# plt.hist(bms_sim[src]["e2"], histtype='step', bins=400, label="ch 2")
			# plt.hist(bms_sim[src]["e3"], histtype='step', bins=400, label="ch 3")
			# # plt.hist(bms_sim[src]["e4"], histtype='step', bins=400, label="ch 4")
			# plt.yscale('log')
			# plt.xlabel('simulated energy deposits (KeV)')
			# plt.ylabel('counts')
			# plt.title(src)
			# plt.legend()
			# # plt.savefig("./figs/sim_{}_edep.png".format(src.lower()))
			# # plt.show()

			this_lo, this_hi = self.src_area_range_full[src]
			plt.subplots(nrows=1,ncols=len(self.ch_comp),sharex=True,sharey=True)

			for ich,ch in enumerate(self.ch_comp):

				this_transformer = self.transformers[src][ch]
				this_counts  = None
				this_ar_mids = None
				
				t1=time.time()
				this_xf_result = this_transformer(
					this_ar_mids,
					guess_c,
					guess_d,
					self.gamma[ch],
					self.res_s[ch],
					self.rho_p[ch],
					en_counts=this_counts
				)
				t2=time.time()
				self.vp(2,"calling transformer {},{} took {:.3f} ms".format(src,ch,(t2-t1)*1000))

				plt.subplot(1,len(self.ch_comp),ich+1)

				plt.hist(
					self.bms_obs_src[src][self.exp_area_pvs.format(ch)],
					this_transformer.ar_mids,
					label="observed",
					histtype='step',
				)
				plt.step(this_transformer.ar_mids, this_xf_result, where='mid', label="simulated")
				
				plt.yscale('log')
				plt.xlabel('area (pVs)')
				plt.ylabel('counts')
				plt.title("Channel {}".format(ch))
				plt.legend()

			self.vp(2,"")

			plt.ylim(1, plt.ylim()[1])
			plt.suptitle(src)
			if savefig:
				plt.savefig(savefig.format(src=src.lower()))
			plt.show()

	def evaluate(self, xdata, *parameters, incl_src=True, incl_bg=True):
		
		# unpack parameters
		# 
		# we get one value each of gamma, res_s, and rho_p, since these
		# are parameters of the channel, and do not vary by source
		# 
		# however, we get one value each, per source, of c and d
		# so we need to unpack these accounting for the sources present
		# 
		# the order of *cd is assumed to be c0,d0,c1,d1, etc, where the
		# ordering 0,1,... is the sorted order of source identifiers
		cd = parameters[:2*len(self.fit_sources)]
		gamma, *res_s, rho_p, ps0, ps1 = parameters[2*len(self.fit_sources):]
		sources = sorted(xdata.keys())
		amp_sim = {s:cd[2*i  ] for i,s in enumerate(sources)}
		amp_bg  = {s:cd[2*i+1] for i,s in enumerate(sources)}

		# calculate total size
		size = sum(_.size for _ in xdata.values())

		# array of modeled area spectrum 
		# we'll fill it in piecewise
		spec = np.zeros(size)

		# what channel is this
		ch = xdata.id

		# calculate modeled counts one source at a time
		istart = 0
		for isrc,src in enumerate(sources):

			this_ar_mids = xdata[src]
			this_transformer = self.transformers[src][ch]

			# C, D, gamma, res_s, rho_p = params
			this_mixed = this_transformer(
				this_ar_mids,
				amp_sim[src],
				amp_bg[src],
				gamma,
				*res_s,
				rho_p,
				ps0, ps1,
				incl_src = incl_src,
				incl_bg  = incl_bg ,
			)

			spec[istart:istart+this_mixed.size] = this_mixed
			istart += this_mixed.size

		return spec
			

			



	def fit(self, channels=None, sources=None, plot_test_fit=False, plot_best_fit=False):
		"""fit model to all sources' spectra at once"""

		if channels is None:
			channels = self.ch_comp
		if sources is None:
			sources = self.all_sources

		# this is important!
		# pieces of the routine rely on the list of source
		# identifiers being sorted.
		self.fit_sources  = sorted(sources)
		self.fit_channels = sorted(channels)
		del sources
		del channels

		# keep the results for later
		# todo: implement fit_result class
		#       have one dict of ch:fit_result
		#       assign this_result and use that in loop
		self.counts = {_:{} for _ in self.fit_sources}

		self.fit_npars = {}
		self.fit_xdata = {}
		self.fit_ydata = {}
		self.fit_yerr  = {}
		self.fit_ydata_pieces = {}
		self.fit_ydata_pos = {}
		
		self.fit_popt  = {}
		self.fit_perr  = {}
		self.fit_pcov  = {}

		self.fit_yopt     = {}
		self.fit_yopt_src = {}
		self.fit_yopt_bg  = {}
		self.fit_yerr_sim = {}
		self.fit_yerr_tot = {}
		self.fit_resid = {}
		self.fit_pulls = {}
		self.fit_chi2  = {}
		self.fit_ndof  = {}
		self.fit_rchi2 = {}
		
		self.fit_xflat = {}
		self.fit_xline = {}



		class idict(dict):
			"""dict with extra attribute"""
			def __init__(self, id_, *args, **kwargs):
				super(idict, self).__init__(*args, **kwargs)
				self.id = id_
			def flat(self):
				return np.concatenate([self[_] for _ in sorted(self.keys())], axis=0)

		# do the fit routine one channel at a time
		# since no parameters are shared between channels
		for ch in self.fit_channels:

			self.vp(1,"")
			self.vp(1,"performing fit routine on channel {}".format(ch))

			# xdata is a dict of src:ar_mids
			# which additionally has a channel identifier
			# which lets evaluate determine which channel to use
			self.fit_xdata[ch] = idict(ch)

			# ydata is an array with shape (n)
			# array counts in area histograms
			# it will be concatenated from each source in this channel
			self.fit_ydata_pieces[ch] = []

			# get the pieces, per source
			for src in self.fit_sources:
				self.fit_xdata[ch][src] = self.ar_mids[src][ch]

				this_counts, _ = np.histogram(
					self.bms_obs_src[src][self.exp_area_pvs.format(ch)],
					self.ar_edges[src][ch],
				)
				self.fit_ydata_pieces[ch].append(this_counts)
				self.counts[src][ch] = this_counts

			self.fit_ydata[ch] = np.concatenate(self.fit_ydata_pieces[ch], axis=0)


			# compose initial parameter guesses
			# if needed: better way for gamma, res_s, rho_p
			# if needed: nonzero guess for background contribution
			n_res_s_pars = 2
			res_s_basic = False

			p0_cd   = []
			if res_s_basic:
				p0_rest = [self.gamma[ch], self.res_s[ch], self.rho_p[ch]] + squish_p0
			else:
				p0_rest = [self.gamma[ch], *([0.0]*n_res_s_pars), self.rho_p[ch]] + squish_p0

			unit_evaluation = self.evaluate(self.fit_xdata[ch], *([1,0]*len(self.fit_sources) + p0_rest))
			istart = 0
			for isrc,src in enumerate(self.fit_sources):
				sum_ydata = self.fit_ydata_pieces[ch][isrc].sum()
				sum_unit  = unit_evaluation[istart:istart+self.fit_xdata[ch][src].size].sum()
				istart += self.fit_xdata[ch][src].size
				p0_cd += [sum_ydata / sum_unit, 0.0]
			p0 = p0_cd + p0_rest
			self.fit_npars[ch] = len(p0)

			# test the evaluation (fit) function with guess parameters
			# todo: this uses zero background contribution, so it isn't
			# useful for troubleshooting background contributions
			self.fit_xflat[ch] = self.fit_xdata[ch].flat()
			self.fit_xline[ch] = np.linspace(0,1,self.fit_xflat[ch].size)
			if plot_test_fit:
				y0 = self.evaluate(self.fit_xdata[ch],*p0)
				plt.plot(self.fit_xline[ch], self.fit_ydata[ch], label='data')
				plt.plot(self.fit_xline[ch], y0   , label='model, guess params')
				plt.yscale('log')
				plt.xlabel('bins (concanetated, unscaled)')
				plt.ylabel('counts')
				plt.legend()
				plt.show()


			# perform optimization and evaluate results
			self.vp(1, "performing curve_fit")
			self.fit_ydata_pos[ch] = (self.fit_ydata[ch] > 0)
			self.fit_yerr[ch] = np.sqrt(self.fit_ydata[ch])
			# todo: better way of handling zero-count bins
			self.fit_yerr[ch][self.fit_yerr[ch] <= 0] = 1.0
			self.fit_popt[ch], self.fit_pcov[ch] = opt.curve_fit(
				self.evaluate,
				self.fit_xdata[ch],
				self.fit_ydata[ch],
				p0=p0,
				sigma=self.fit_yerr[ch],
				absolute_sigma=True,
			)
			self.fit_perr[ch] = np.sqrt(np.diag(self.fit_pcov[ch]))
			self.fit_yopt[ch]     = self.evaluate(self.fit_xdata[ch],*self.fit_popt[ch])
			self.fit_yopt_src[ch] = self.evaluate(self.fit_xdata[ch],*self.fit_popt[ch], incl_bg=False)
			self.fit_yopt_bg[ch]  = self.evaluate(self.fit_xdata[ch],*self.fit_popt[ch], incl_src=False)

			self.fit_resid[ch] = self.fit_ydata[ch] - self.fit_yopt[ch]

			# self.fit_yerr_sim[ch] = np.sqrt(self.fit_yopt[ch])
			# todo: full error calculation
			# currently just include error contributions from c, d
			# as well as the poisson error on background bin counts
			# since these are the easiest components
			pieces_err_c  = []
			pieces_err_d  = []
			pieces_err_bg = []
			jstart = 0
			for isrc,src in enumerate(self.fit_sources):
				this_c     = self.fit_popt[ch][isrc*2  ]
				this_d     = self.fit_popt[ch][isrc*2+1]
				this_c_err = self.fit_perr[ch][isrc*2  ]
				this_d_err = self.fit_perr[ch][isrc*2+1]
				this_bg_counts = self.transformers[src][ch].bg_counts
				this_size = this_bg_counts.size
				pieces_err_bg.append(np.sqrt(this_bg_counts) * this_d)
				pieces_err_d.append(this_d_err * self.fit_yopt_bg[ ch][jstart:jstart+this_size] / this_d)
				pieces_err_c.append(this_c_err * self.fit_yopt_src[ch][jstart:jstart+this_size] / this_c)
				jstart += this_size
			err_c  = np.concatenate(pieces_err_c , axis=0)
			err_d  = np.concatenate(pieces_err_d , axis=0)
			err_bg = np.concatenate(pieces_err_bg, axis=0)

			self.fit_yerr_sim[ch] = np.sqrt(err_c**2 + err_d**2 + err_bg**2)
			self.fit_yerr_tot[ch] = np.sqrt(self.fit_yerr[ch]**2 + self.fit_yerr_sim[ch]**2)

			self.fit_pulls[ch] = self.fit_resid[ch] / self.fit_yerr_tot[ch]
			self.fit_chi2[ch]  = (self.fit_pulls[ch][self.fit_ydata_pos[ch]] ** 2).sum()
			self.fit_ndof[ch]  = self.fit_ydata_pos[ch].sum() - self.fit_npars[ch]
			self.fit_rchi2[ch] = self.fit_chi2[ch] / self.fit_ndof[ch]

			# print results
			pfe = "{:>12.4e}"
			pff = "{:>12.4f}"
			self.vp(1,"popt (top), perr (bottom)")
			self.vp(1," ".join([pff.format(_) for _ in self.fit_popt[ch]]))
			self.vp(1," ".join([pff.format(_) for _ in self.fit_perr[ch]]))
			self.vp(1,"")
			self.vp(2,"pcov")
			self.vp(2,"\n".join([" ".join([pfe.format(__) for __ in _]) for _ in self.fit_pcov[ch]]))
			self.vp(2,"")

			# plot best fit results
			if plot_best_fit:
				plt.plot(self.fit_xline[ch], self.fit_ydata[ch]   , label='data')
				plt.plot(self.fit_xline[ch], self.fit_yopt[ch]    , label='model, best fit')
				plt.plot(self.fit_xline[ch], self.fit_yopt_src[ch], label='source bit')
				plt.plot(self.fit_xline[ch], self.fit_yopt_bg[ch] , label='background bit')
				plt.yscale('log')
				plt.xlabel('bins (concatenated, unscaled)')
				plt.ylabel('counts')
				plt.legend()
				plt.show()

	def plot_best_fits(self, sharex=False, sharey=False):
		""""""

		for ch in self.fit_channels:

			# create figure and axes
			nrows = 1
			ncols = len(self.fit_sources)
			fig, ax = plt.subplots(nrows,ncols,sharex=sharex,sharey=sharey,figsize=(ncols*8-1,nrows*8-1))
			if nrows*ncols == 1:
				ax = [ax]
			src_ax = {s:ax[i] for i,s in enumerate(self.fit_sources)}

			istart = 0
			for isrc,src in enumerate(self.fit_sources):

				this_x = self.fit_xdata[ch][src]
				this_y_obs    = self.fit_ydata_pieces[ch][isrc]
				this_yerr_obs = self.fit_yerr[ch][istart:istart+this_x.shape[0]]
				this_y_sim    = self.fit_yopt[ch][istart:istart+this_x.shape[0]]
				this_yerr_sim = self.fit_yerr_sim[ch][istart:istart+this_x.shape[0]]
				this_y_sim_src = self.fit_yopt_src[ch][istart:istart+this_x.shape[0]]
				this_y_sim_bg  = self.fit_yopt_bg[ch][istart:istart+this_x.shape[0]]
				istart += this_x.shape[0]

				this_ax = src_ax[src]
				this_ax.fill_between(this_x, this_y_obs-this_yerr_obs, this_y_obs+this_yerr_obs, step='mid', color='r', alpha=0.25)
				this_ax.fill_between(this_x, this_y_sim-this_yerr_sim, this_y_sim+this_yerr_sim, step='mid', color='k', alpha=0.25)
				this_ax.step(this_x, this_y_sim_src, color='lightgreen', where='mid', label='sim piece')
				this_ax.step(this_x, this_y_sim_bg , color='turquoise' , where='mid', label='bg piece')
				this_ax.step(this_x, this_y_obs    , color='r'         , where='mid', label='observation')
				this_ax.step(this_x, this_y_sim    , color='k'         , where='mid', label='combined sim+bg')
				this_ax.set_xlabel("area (pVs)")
				this_ax.set_ylabel("counts")
				this_ax.set_title(src)
				this_ax.legend(handles=[
					lines.Line2D([],[],color='r'         ,ls='-',marker='',label='observation'),
					lines.Line2D([],[],color='k'         ,ls='-',marker='',label='combined sim+bg'),
					lines.Line2D([],[],color='lightgreen',ls='-',marker='',label='sim piece'),
					lines.Line2D([],[],color='turquoise' ,ls='-',marker='',label='bg piece'),
				])

			plt.suptitle("Channel {} - chi2/dof = {:.1f}/{} = {:.4f}".format(
				ch,
				self.fit_chi2[ch],
				self.fit_ndof[ch],
				self.fit_rchi2[ch],
			))

			# plt.yscale('log')
			plt.show()












def main():

	# todo: argparse

	rtn = routine()
	rtn.procure_data()
	# rtn.plot_basic_check()
	rtn.fit(plot_test_fit=False, plot_best_fit=False)
	rtn.plot_best_fits()

	sys.exit(0)

	# There was some code I developed for estimating the prevalence
	# of source and background data within a spectrum, using regions
	# where the spectrum was entirely background, etc.
	# the code has been removed, but it is still in previous versions
	# of this document. There may be need to add that kind of 
	# functionality in the future, but for now I'll just use zero for
	# the parameter guess for background contributions.

if __name__ == "__main__":
	main()
