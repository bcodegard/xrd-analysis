"""
"""

__author__ = "Brunel Odegard"
__version__ = "0.0"


import os
import sys
import math

import numpy as np

import utils.data as data
import utils.fit as fit
import utils.display as display

import matplotlib.pyplot as plt




# specify dataset

# list of sources to perform analysis on
SOURCES = [
	# "Cs137", # no sim file
	# "Mn54" , # DRS saturation
	# "Na22" , # DRS saturation
	"Am241",
	"Cd109",
	"Ba133",
	"Co57" ,
]

# ranges for splitting sim datasets based on primary vertex energy
# 
# The keys are values in KeV read off plots of simulation data/
# They are not values from some other reference.
# The keys will change if/when the simulation parameters are updated
# to better match reference values and precision.
# 
# These are also used to mark vertical lines on plots
PVE_KEV = {
	"Am241": {
		# "14.1":[ 13, 15], # too low energy to contribute many (any?) events
		"18.1":[  0, 19], # currently lumps in all peaks below the 20.5KeV peaks
		"20.5":[ 19, 22], 
		"26.5":[ 26, 27], 
		"59.5":[ 59, 60],
	},
	"Cd109": {
		"22.1":[ 21, 23],
		"25.1":[ 24, 26],
		"88.0":[ 86, 90],
	},
	"Ba133": {
		"31.1" :[ 30, 32], 
		"53.5" :[ 52, 54],
		"81.0" :[ 80, 82],
		"303.2":[302,304],
		"356.0":[355,357],
	},
	"Co57" : {
		"14.1" :[ 13, 15],
		"122.1":[121,123],
		"136.1":[135,137],
	},
	"Mn54" : {
		"835.0":[834,836],
	},
	"Na22" : {
		"511.0":[510,512],
	},
}

# # too low energy or too low statistics to contribute
del PVE_KEV["Co57" ]["14.1"]

# # remove these when not treating PMT saturation region
# del PVE_KEV["Co57" ]["136.1"]

# remove when not treating the corresponding energy region
# del PVE_KEV["Ba133"]["303.2"] # is reduced to exclude PMT saturation


# run numbers for experimental source spectra
RUNS_EXP_SRC = {
	# "Cs137":0,
	# "Mn54" :0,
	# "Na22" :0,
	"Am241":4291,
	"Ba133":4293,
	"Cd109":4292,
	"Co57" :4294,
}

# run numbers for background spectra
RUNS_EXP_BG = [4225, 4226]

# data directories and files
DIR_SIM_SRC = "/home/bode/Documents/GitHub/xrd-analysis/data/scint-simulation/np"
# DIR_EXP_SRC = "/home/bode/Documents/GitHub/xrd-analysis/data/root/scintillator"
# DIR_EXP_BG  = "/home/bode/Documents/GitHub/xrd-analysis/data/root/scintillator"
DIR_EXP_SRC = "/home/bode/Documents/GitHub/xrd-analysis/data/scint-experiment/np"
DIR_EXP_BG  = "/home/bode/Documents/GitHub/xrd-analysis/data/scint-experiment/np"

FILENAME_SIM_SRC = "{src}.npz"
FILENAME_SIM_SRC_BY = {
	# "Am241":"{src}x10.npz",
	# "Ba133":"{src}x10_with53mult.npz",
	# "Cd109":"{src}x100.npz",
	# "Co57" :"{src}x10.npz",
	"Am241":"{src}x10new.npz",
	"Ba133":"{src}x10new.npz",
	"Cd109":"{src}x100new.npz",
	"Co57" :"{src}x10new.npz",
}
FILES_SIM_SRC = {_:os.sep.join([DIR_SIM_SRC, FILENAME_SIM_SRC_BY.get(_,FILENAME_SIM_SRC).format(src=_)]) for _ in SOURCES}

# FILENAME_EXP_SRC = "Run{run}.root"
# FILENAME_EXP_BG  = "Run{run}.root"
FILENAME_EXP_SRC = "Run{run}.npz"
FILENAME_EXP_BG  = "Run{run}.npz"
FILE_EXP_SRC = os.sep.join([DIR_EXP_SRC, FILENAME_EXP_SRC])
FILE_EXP_BG  = os.sep.join([DIR_EXP_BG , FILENAME_EXP_BG ])
# FILES_SIM_SRC = {_:FILE_SIM_SRC.format(src=_)               for _ in SOURCES}
FILES_EXP_SRC = {_:FILE_EXP_SRC.format(run=RUNS_EXP_SRC[_]) for _ in SOURCES}
FILES_EXP_BG  = [FILE_EXP_SRC.format(run=_) for _ in RUNS_EXP_BG]

# channel number identification
CH_COMP = [1,2,3,4]
CH_LYSO = 4
CH_ALL  = CH_COMP + [CH_LYSO]

# names and lists of leaves
LEAF_SIM_PVE_MEV  = "GammaTracks.initialEnergy_MeV"
LEAF_SIM_EDEP_MEV = "Edep_MeV_Si{ch}"
LEAF_SIM_EDEP_KEV = "Edep_KeV_Si{ch}"
# LEAF_SIM_AREA_PVS = "area_pvs_{ch}"
LEAF_EXP_AREA_PVS = "area_{drs}_{ch}"
LEAF_EXP_AREA_NVS = "area_nVs_{drs}_{ch}"
DRS_ID = "3046"

LEAF_LYSO_AREA_PVS = LEAF_EXP_AREA_PVS.format(ch=CH_LYSO, drs=DRS_ID)
LYSO_AREA_PVS_CUT_HI = 14000.0

# sets of leaves to load for each dataset
# the rest of the leaves defined above are constructed, rather than loaded from the files.
LEAVES_LOAD_SIM_SRC = {LEAF_SIM_PVE_MEV} | {LEAF_SIM_EDEP_MEV.format(ch=_) for _ in CH_ALL}
LEAVES_LOAD_EXP_SRC = {LEAF_EXP_AREA_PVS.format(drs=DRS_ID, ch=_) for _ in CH_ALL}
LEAVES_LOAD_EXP_BG  = {LEAF_EXP_AREA_PVS.format(drs=DRS_ID, ch=_) for _ in CH_ALL}



# specify data ranges and binning

# Area range per source and channel for making
# comparisons between model and observation.

# # extended - includes PMT saturation region
# # DRS saturation in channels 1 2 3 begins after area (nVs) 280 310 310
# # These values will be used as caps on upper limits to fit regions
# # 
# # command used to test DRS saturation:
# # python3 fb-dev.py 4293 --fit A1 0 800 100 and 4293 --fit A1 0 800 100 --cut v1 . 949 and 4293 --fit A1 0 800 100 --cut v1 949
# AREA_RANGE_NVS = {
# 	"Am241":{1:[11, 90], 2:[11, 90], 3:[11, 90]},
# 	"Ba133":{1:[11,280], 2:[11,310], 3:[11,310]},
# 	"Cd109":{1:[11,130], 2:[11,130], 3:[11,130]},
# 	"Co57" :{1:[11,200], 2:[11,200], 3:[11,200]},
# 	# "Mn54" :[],
# 	# "Na22" :[],
# }

# # include PMT saturation but exclude poorly simulated low tails
# AREA_RANGE_NVS = {
# 	"Am241":{1:[ 38, 90], 2:[ 38, 90], 3:[ 39, 90]},
# 	"Ba133":{1:[ 11,280], 2:[ 11,310], 3:[ 11,310]},
# 	"Cd109":{1:[ 11,130], 2:[ 11,130], 3:[ 11,130]},
# 	"Co57" :{1:[106,200], 2:[107,200], 3:[112,200]},
# }


# idential ranges for all sources; include PMT saturation
AREA_RANGE_NVS = {
	"Am241":{1:[12,150], 2:[12,150], 3:[12,165], 4:[0,300]},
	"Ba133":{1:[12,150], 2:[12,150], 3:[12,165], 4:[0,300]},
	"Cd109":{1:[12,150], 2:[12,150], 3:[12,165], 4:[0,300]},
	"Co57" :{1:[12,150], 2:[12,150], 3:[12,165], 4:[0,300]},
}
# # Everything but DRS saturation; limit for DRS saturation slightly conservative
# AREA_RANGE_NVS = {
# 	"Am241":{1:[13,290], 2:[13,312], 3:[13,322], 4:[0,300]},
# 	"Ba133":{1:[12,290], 2:[12,312], 3:[12,322], 4:[0,300]},
# 	"Cd109":{1:[12,290], 2:[12,312], 3:[12,322], 4:[0,300]},
# 	"Co57" :{1:[12,290], 2:[12,312], 3:[12,322], 4:[0,300]},
# }
# # include DRS saturation to see what happens
# AREA_RANGE_NVS = {
# 	"Am241":{1:[13,400], 2:[13,400], 3:[13,400], 4:[0,300]},
# 	"Ba133":{1:[12,400], 2:[12,400], 3:[12,400], 4:[0,300]},
# 	"Cd109":{1:[12,400], 2:[12,400], 3:[12,400], 4:[0,300]},
# 	"Co57" :{1:[12,400], 2:[12,400], 3:[12,400], 4:[0,300]},
# }


# # include PMT saturation but exclude poorly simulated low tails
# # also increase upper range of some all sources to DRS limit
# AREA_RANGE_NVS = {
# 	"Am241":{1:[ 38,280], 2:[ 38,310], 3:[ 39,310]},
# 	"Ba133":{1:[ 11,280], 2:[ 11,310], 3:[ 11,310]},
# 	"Cd109":{1:[ 11,280], 2:[ 11,310], 3:[ 11,310]},
# 	"Co57" :{1:[106,280], 2:[107,310], 3:[112,310]},
# }

# # semi restrictive - excludes PMT saturation but includes Am241 low peak
# AREA_RANGE_NVS = {
# 	"Am241":{1:[11, 80], 2:[11, 80], 3:[11, 80]},
# 	"Ba133":{1:[11, 80], 2:[11, 80], 3:[11, 80]},
# 	"Cd109":{1:[11, 42], 2:[11, 42], 3:[11, 42]},
# 	"Co57" :{1:[11, 80], 2:[11, 80], 3:[11, 80]},
# }

# # restrictive - excludes PMT saturation and AM241 low peak
# AREA_RANGE_NVS = {
# 	"Am241":{1:[40, 80], 2:[40, 80], 3:[40, 80]},
# 	"Ba133":{1:[11, 80], 2:[11, 80], 3:[11, 80]},
# 	"Cd109":{1:[11, 42], 2:[11, 42], 3:[11, 42]},
# 	"Co57" :{1:[11, 80], 2:[11, 80], 3:[11, 80]},
# }

# resolution for projections (unitless; number of bins)
RES_A = 400
RES_E = 200

# Source peak energies for plotting vlines.
# These differ from sim component values, so they're tracked separately.
SOURCE_PEAK_ENERGY_KEV = {
	"Am241": [ 26.34,  59.54],
	"Ba133": [ 30.85,  81.00],
	"Cd109": [ 22.1 ,  88.04],
	"Co57" : [122.06, 136.47],
	"Mn54" : [834.85        ],
	"Na22" : [511.0         ],
}


# parameter guesses, based on some initial fit results

# # cubic, params scaled to 50 nVs, include the two HE peaks
# PARAM_GUESS = {
# 	1:{
# 		"gamma"         : 1.135e+00,
# 		"res_s_a"       : 5.726e-02,
# 		"rho_p"         : 3.433e-03,
# 		"xf_a"          : 8.575e-17,
# 		"xf_c"          : 3.792e-02,
# 		"xf_d"          : 6.208e-04,
# 		"n_bg_Am241"    : 3.505e-01,
# 		"n_59.5_Am241"  : 7.609e-03,
# 		"n_bg_Cd109"    : 4.662e-01,
# 		"n_22.1_Cd109"  : 1.376e+00,
# 		"n_25.1_Cd109"  : 4.460e-01,
# 		"n_88.0_Cd109"  : 9.376e-02,
# 		"n_bg_Ba133"    : 4.142e-01,
# 		"n_303.2_Ba133" : 2.790e-02,
# 		"n_31.1_Ba133"  : 4.512e-01,
# 		"n_356.0_Ba133" : 3.967e-02,
# 		"n_81.0_Ba133"  : 1.030e-01,
# 		"n_bg_Co57"     : 3.104e-01,
# 		"n_122.1_Co57"  : 1.763e-02,
# 		"n_136.1_Co57"  : 1.111e-02,
# 	},
# 	2:{
# 		"gamma"         : 1.169e+00,
# 		"res_s_a"       : 4.915e-02,
# 		"rho_p"         : 5.708e-03,
# 		"xf_a"          : 2.455e-16,
# 		"xf_c"          : 4.523e-02,
# 		"xf_d"          : 7.591e-14,
# 		"n_bg_Am241"    : 3.288e-01,
# 		"n_59.5_Am241"  : 1.628e-03,
# 		"n_bg_Cd109"    : 4.983e-01,
# 		"n_22.1_Cd109"  : 2.708e-01,
# 		"n_25.1_Cd109"  : 9.964e-02,
# 		"n_88.0_Cd109"  : 2.282e-02,
# 		"n_bg_Ba133"    : 5.630e-01,
# 		"n_303.2_Ba133" : 7.617e-03,
# 		"n_31.1_Ba133"  : 1.249e-01,
# 		"n_356.0_Ba133" : 4.324e-03,
# 		"n_81.0_Ba133"  : 2.991e-02,
# 		"n_bg_Co57"     : 3.096e-01,
# 		"n_122.1_Co57"  : 4.518e-03,
# 		"n_136.1_Co57"  : 3.304e-03,
# 	},
# 	3:{
# 		"gamma"         : 1.229e+00,
# 		"res_s_a"       : 4.401e-02,
# 		"rho_p"         : 6.749e-03,
# 		"xf_a"          : 1.336e-14,
# 		"xf_c"          : 4.207e-02,
# 		"xf_d"          : 2.325e-16,
# 		"n_bg_Am241"    : 3.381e-01,
# 		"n_59.5_Am241"  : 7.783e-04,
# 		"n_bg_Cd109"    : 5.471e-01,
# 		"n_22.1_Cd109"  : 1.391e-01,
# 		"n_25.1_Cd109"  : 3.993e-02,
# 		"n_88.0_Cd109"  : 1.001e-02,
# 		"n_bg_Ba133"    : 5.312e-01,
# 		"n_303.2_Ba133" : 1.873e-03,
# 		"n_31.1_Ba133"  : 5.697e-02,
# 		"n_356.0_Ba133" : 4.330e-03,
# 		"n_81.0_Ba133"  : 1.440e-02,
# 		"n_bg_Co57"     : 3.295e-01,
# 		"n_122.1_Co57"  : 1.955e-03,
# 		"n_136.1_Co57"  : 1.562e-03,
# 	}
# }

# sim v3, res A,E = 200,100, energy range everything but DRS saturation
PARAM_GUESS = {
	1:{
		"gamma"        :1.152e+00,
		"res_s_a"      :5.520e-02,
		"res_s_b"      :8.011e-04,
		"rho_p"        :3.774e-03,
		"xf_c"         :4.422e-02,
		"n_bg_Am241"   :3.390e-01,
		"n_18.1_Am241" :5.684e-01,
		"n_20.5_Am241" :1.570e+00,
		"n_26.5_Am241" :2.556e-01,
		"n_59.5_Am241" :5.084e-02,
		"n_bg_Cd109"   :4.480e-01,
		"n_22.1_Cd109" :2.004e-01,
		"n_25.1_Cd109" :4.611e-02,
		"n_88.0_Cd109" :1.006e-02,
		"n_bg_Ba133"   :3.579e-01,
		"n_303.2_Ba133":3.060e-02,
		"n_31.1_Ba133" :3.630e-01,
		"n_356.0_Ba133":4.074e-02,
		"n_53.5_Ba133" :7.398e-02,
		"n_81.0_Ba133" :1.054e-01,
		"n_bg_Co57"    :3.832e-01,
		"n_122.1_Co57" :4.953e-02,
		"n_136.1_Co57" :2.585e-02,
	},
	2:{

	},
	3:{

	}
}


# convenience and utility functions

def pprint(stuff, fmt=False, join=False, pre=False, sep=''):
	lines = []
	for _ in stuff:
		if join:
			line = join.join(map(str, [fmt.format(__) if fmt else __ for __ in _]))
		else:
			line = fmt.format(_) if fmt else _
		if pre:
			print(pre, line)
		else:
			print(line)

def dict_reduce(method, *dicts):
	"""make new dict by performing reduce per key,value on list of dicts"""

	# no action if one dict; error if zero.
	if len(dicts) < 2:
		return dicts[0]

	# assume all dicts have identical keys
	new_dict = {}
	for key,value in dicts[0].items():
		new_value = value
		for dict_ in dicts[1:]:
			new_value = method(value, dict_[key])
		new_dict[key] = new_value

	return new_dict

def normed(arr, axis=None):
	return arr / np.sum(arr, axis)

def normed_if(arr, axis=None, do=False):
	return normed(arr,axis) if do else arr

def summary(arr, ):
	return ", ".join(map(str,[arr[0], arr[1], arr[2], "...", arr[-1]]))

def fn_indent(fn):
	def wrapped(inst, *args, **kwargs):
		# f_inc()
		inst.vp(1, '\n')
		inst.vp(1, fn.__name__)
		inst.inc_indent()
		ans = fn(inst, *args, **kwargs)
		inst.dec_indent()
		return ans
	return wrapped




class routine(object):
	
	indent = 0
	def inc_indent(self):
		self.indent += 1
		# print(self.indent)
	def dec_indent(self):
		self.indent = max([0, self.indent - 1])
		# print(self.indent)

	vb = 2
	def vp(self,lv=1,*args,**kwargs):
		kwargs['sep']=kwargs['sep'] if 'sep' in kwargs else ''
		if self.vb >= lv:
			print('\t'*self.indent, *args, **kwargs)
	def vpp(self,lv=1,*args,**kwargs):
		kwargs['sep']=kwargs['sep'] if 'sep' in kwargs else ''
		if self.vb >= lv:
			pprint(*args,**kwargs)


	def __init__(self):
		self.load_data()
		self.prepare_data()
		# self.setup_model()


	@fn_indent
	def load_data(self):
		"""load arrays from specified files and initialize branch managers"""

		# load simulated source data
		# {src:{"E":manager for component E}}
		self.sim_src = {}
		eks = sorted(LEAVES_LOAD_SIM_SRC)[0]
		self.vp(1, "")
		self.vp(1, "loading simulation source data")
		self.vp(2, "leaves: {}".format(LEAVES_LOAD_SIM_SRC))
		for src,file in FILES_SIM_SRC.items():
			self.vp(2, "source {}, file {}".format(src,file))
			arrays = np.load(file)
			self.vp(2, "leaf {} has shape {}".format(eks, arrays[eks].shape))
			self.vpp(3, [(k,list(v[:10])) for k,v in arrays.items()], join=" = ")
			arrays = {k:v for k,v in arrays.items() if k in LEAVES_LOAD_SIM_SRC}

			# separate simulated source spectra by primary vertex energy
			# "E":manager where "E" is primary vertex energy
			self.sim_src[src] = {}
			for label_e_kev, (lo_kev,hi_kev) in PVE_KEV[src].items():
				self.vp(2, "finding events for Epv={} KeV, accepting ({},{})".format(
					label_e_kev, lo_kev, hi_kev))

				# find events within the specified energy range
				this_ftr = np.logical_and(
					arrays[LEAF_SIM_PVE_MEV] > (lo_kev / 1000),
					arrays[LEAF_SIM_PVE_MEV] < (hi_kev / 1000))
				this_n_pass = this_ftr.sum()
				self.vp(2, "{} / {} pass".format(this_n_pass, this_ftr.size))

				# todo: check the there are energy deposits
				#       for now, I've just disabled the 14.1KeV ones,
				#       which are the only ones with no deposits.

				# setup branch manager for this component
				if this_n_pass:
					self.sim_src[src][label_e_kev] = data.BranchManager(
						{k:v[this_ftr] for k,v in arrays.items()},
						import_copies=False, export_copies=False)
				else:
					self.vp(2, "discarding this component since it has no events")

				del this_ftr
			del arrays


		# load experimental source data
		# {src:manager}
		self.exp_src = {}
		eke = sorted(LEAVES_LOAD_EXP_SRC)[0]
		self.vp(1, "")
		self.vp(1, "loading experimental source data")
		self.vp(2, "leaves: {}".format(LEAVES_LOAD_EXP_SRC))
		for src,file in FILES_EXP_SRC.items():
			self.vp(2, "source {}, file {}".format(src,file))
			arrays = np.load(file)
			self.vp(2, "leaf {} has shape {}".format(eke, arrays[eke].shape))
			self.vpp(3, [(k,list(v[:10])) for k,v in arrays.items()], join=" = ")
			self.exp_src[src] = data.BranchManager(
				{k:v for k,v in arrays.items() if k in LEAVES_LOAD_EXP_SRC},
				export_copies=False, import_copies=False)
			del arrays


		# load experimental background data
		exp_bg_pieces = []
		self.vp(1, "")
		self.vp(1, "loading experimental background data")
		self.vp(2, "leaves: {}".format(LEAVES_LOAD_EXP_BG))
		for file in FILES_EXP_BG:
			self.vp(2, "file {}".format(file))
			arrays = np.load(file)
			self.vp(2, "leaf {} has shape {}".format(eke, arrays[eke].shape))
			self.vpp(3, [(k,list(v[:10])) for k,v in arrays.items()], join=" = ")
			exp_bg_pieces.append({k:v for k,v in arrays.items() if k in LEAVES_LOAD_EXP_SRC})
			del arrays
		self.vp(1, "concatenating background datasets")
		exp_bg = dict_reduce(lambda a,b: np.concatenate((a,b), axis=0), *exp_bg_pieces)
		del exp_bg_pieces
		self.vp(2, "leaf {} has shape {}".format(eke, exp_bg[eke].shape))
		self.exp_bg = data.BranchManager(exp_bg, export_copies=False, import_copies=False)
		del exp_bg


	@fn_indent
	def prepare_data(self):
		"""process loaded data, creating forms needed by analysis routines"""

		# process experimental source data
		# buds for converting pVs area measurements to nVs
		buds_pvs_to_nvs = [
			data.bud_function(
				LEAF_EXP_AREA_PVS.format(ch=_,drs=DRS_ID),
				LEAF_EXP_AREA_NVS.format(ch=_,drs=DRS_ID),
				lambda a:a/1000
			) for _ in CH_COMP
		]
		# mask for LYSO activity
		mask_lyso_cut = data.cut(LEAF_LYSO_AREA_PVS,hi=LYSO_AREA_PVS_CUT_HI)
		# apply masks and buds
		self.vp(1, "")
		self.vp(1, "preparing experimental source data")
		self.vp(2, "cutting on LYSO area (pVs) < {} and converting pVs -> nVs".format(LYSO_AREA_PVS_CUT_HI))
		for src,manager in self.exp_src.items():
			self.vp(3, src)
			manager.mask(mask_lyso_cut, apply_mask=True)
			manager.bud(buds_pvs_to_nvs)
			self.vp(3, "before : {}".format(list(manager[LEAF_EXP_AREA_PVS.format(ch=CH_COMP[0],drs=DRS_ID)][:10])))
			self.vp(3, "after  : {}".format(list(manager[LEAF_EXP_AREA_NVS.format(ch=CH_COMP[0],drs=DRS_ID)][:10])))


		# process experimental background data
		self.vp(1, "")
		self.vp(1, "preparing experimental background data")
		self.vp(2, "cutting on LYSO area (pVs) < {} and converting pVs -> nVs".format(LYSO_AREA_PVS_CUT_HI))
		self.exp_bg.mask(mask_lyso_cut, apply_mask=True)
		self.exp_bg.bud(buds_pvs_to_nvs)
		self.vp(3, "before : {}".format(list(self.exp_bg[LEAF_EXP_AREA_PVS.format(ch=CH_COMP[0],drs=DRS_ID)][:10])))
		self.vp(3, "after  : {}".format(list(self.exp_bg[LEAF_EXP_AREA_NVS.format(ch=CH_COMP[0],drs=DRS_ID)][:10])))


		# process simulated source data
		# buds for converting MeV energy deposits to KeV
		buds_mev_to_kev = [
			data.bud_function(
				LEAF_SIM_EDEP_MEV.format(ch=_),
				LEAF_SIM_EDEP_KEV.format(ch=_),
				lambda e:e*1000
			) for _ in CH_COMP
		]
		# apply buds
		self.vp(1, "")
		self.vp(1, "preparing simulated source data")
		self.vp(2, "converting MeV -> KeV")
		for src,managers in self.sim_src.items():
			for e_kev,manager in managers.items():
				self.vp(3, "{}, {}".format(src, e_kev))
				kev = manager.bud(buds_mev_to_kev)
				self.vp(3, "before : {}".format(list(manager[LEAF_SIM_EDEP_MEV.format(ch=CH_COMP[0])][:20])))
				self.vp(3, "after  : {}".format(list(manager[LEAF_SIM_EDEP_KEV.format(ch=CH_COMP[0])][:20])))


		# calculate bins and bin data in to spectra
		self.vp(1, "")
		self.vp(1, "binning datasets")

		# {channel:{source:{component:edges}}}
		self.edges_e_kev = {}
		self.mids_e_kev  = {}
		
		# {channel:{source:edges}}
		self.edges_a_nvs = {}
		self.mids_a_nvs  = {}
		
		# {channel:{source:{component:spectrum}}}
		self.spec_sim_src_kev     = {}
		self.spec_sim_src_kev_err = {}

		# {channel:{source:spectrum}}
		self.spec_exp_src_nvs     = {}
		self.spec_exp_src_nvs_err = {}

		# {channel:{source:spectrum}}
		self.spec_exp_bg_nvs      = {}
		self.spec_exp_bg_nvs_err  = {}
		
		for ch in CH_COMP:
			self.vp(2, "channel {}".format(ch))

			self.edges_a_nvs[ch]      = {}
			self.edges_e_kev[ch]      = {}
			self.mids_a_nvs[ch]       = {}
			self.mids_e_kev[ch]       = {}
			self.spec_sim_src_kev[ch] = {}
			self.spec_exp_src_nvs[ch] = {}
			self.spec_exp_bg_nvs[ch]  = {}
			self.spec_sim_src_kev_err[ch] = {}
			self.spec_exp_src_nvs_err[ch] = {}
			self.spec_exp_bg_nvs_err[ch]  = {}

			for src in SOURCES:
				self.vp(2, "")
				self.vp(2, "source {}".format(src))

				# populate edges_a_nvs
				this_data = self.exp_src[src][LEAF_EXP_AREA_NVS.format(ch=ch,drs=DRS_ID)]
				this_data_inrange = this_data[np.logical_and(this_data>AREA_RANGE_NVS[src][ch][0], this_data<AREA_RANGE_NVS[src][ch][1])]
				this_ndata = this_data_inrange.size

				this_nbins = min([math.floor(this_ndata / 20), RES_A])				
				this_edges = data.edges_lin(*AREA_RANGE_NVS[src][ch], RES_A)
				# this_edges = data.edges_equal_count(RES_A, this_data, *AREA_RANGE_NVS[src][ch])
				
				self.edges_a_nvs[ch][src] = this_edges
				self.mids_a_nvs[ch][src] = 0.5*(this_edges[1:]+this_edges[:-1])
				self.vp(2, "area (nVs), {} data, {} edges {}".format(this_ndata, this_nbins, summary(this_edges)))

				# populate spec_exp_src_nvs
				this_hist = np.histogram(this_data, this_edges)[0]
				self.spec_exp_src_nvs[ch][src] = this_hist
				self.spec_exp_src_nvs_err[ch][src] = np.sqrt(this_hist)
				del this_data

				# populate spec_exp_bg_nvs
				this_data = self.exp_bg[LEAF_EXP_AREA_NVS.format(ch=ch,drs=DRS_ID)]
				this_hist = np.histogram(this_data, this_edges)[0]
				# this_sum  = this_hist.sum()
				this_sum  = 1 # don't normalize counts
				self.spec_exp_bg_nvs[ch][src] = this_hist / this_sum
				self.spec_exp_bg_nvs_err[ch][src] = np.sqrt(this_hist) / this_sum
				del this_data, this_edges, this_hist, this_sum

				# populate edges_e_kev and spec_sim_src_kev
				self.edges_e_kev[ch][src] = {}
				self.mids_e_kev[ch][src] = {}
				self.spec_sim_src_kev[ch][src] = {}
				self.spec_sim_src_kev_err[ch][src] = {}
				self.vp(2, "components of sim spectrum")
				for e_kev,manager in self.sim_src[src].items():
					this_data = manager[LEAF_SIM_EDEP_KEV.format(ch=ch,drs=DRS_ID)]
					this_data_nonzero = this_data[this_data>0]

					this_nbins = min([math.floor(this_data_nonzero.size / 20), RES_E])
					this_edges = data.edges_lin(this_data[this_data>0].min(), this_data.max(), RES_E)
					# this_edges = data.edges_equal_count(this_nbins, this_data, this_data_nonzero.min(), this_data.max())

					self.edges_e_kev[ch][src][e_kev] = this_edges
					self.mids_e_kev[ch][src][e_kev] = 0.5*(this_edges[1:]+this_edges[:-1])
					this_hist = np.histogram(this_data,this_edges)[0]
					# this_sum = np.sum(this_hist)
					this_sum = 1 # don't normalize counts
					self.spec_sim_src_kev[ch][src][e_kev] = this_hist / this_sum
					self.spec_sim_src_kev_err[ch][src][e_kev] = np.sqrt(this_hist) / this_sum
					self.vp(2, "{} KeV, {} data, {} edges {}".format(e_kev, this_data_nonzero.size, this_nbins, summary(this_edges)))
					# self.vp(2, "energy (KeV) edges {}".format())
					del this_data, this_edges


		# set up concatenated experimental spectra,
		# and objects, methods, and information for convenience,
		# packing and unpacking
		# 
		# Order alphabetically by source so that consistency can
		# be achieved withouth storing the order used.
		# 
		# {channel:concatenated_source_spectra}
		self.spec_exp_src_nvs_flat     = {}
		self.spec_exp_src_nvs_err_flat = {}
		src_sorted = sorted(SOURCES)
		self.vp(1, "")
		self.vp(1, "concatenating experimental source spectra")
		self.vp(1, "source order is {}".format(src_sorted))
		for ch in CH_COMP:
			this_pieces = [self.spec_exp_src_nvs[ch][_] for _ in src_sorted]
			self.spec_exp_src_nvs_flat[ch] = np.concatenate(this_pieces, axis=0)
			self.spec_exp_src_nvs_err_flat[ch] = np.sqrt(self.spec_exp_src_nvs_flat[ch])
			self.vp(2, "channel {}, size {}".format(ch, self.spec_exp_src_nvs_flat[ch].size))


	def unflatten_sources(self, data_flat, ch=None):
		if ch is None:
			ch = CH_COMP[0]
		pieces = {}
		istart = 0
		for src in sorted(SOURCES):
			this_size = self.spec_exp_src_nvs[ch][src].shape[0]
			pieces[src] = data_flat[istart:istart+this_size]
			istart += this_size
		return pieces


	def show_spectra(self):
		# one channel at a time
		for ch in CH_COMP:
			nsrc = len(SOURCES)
			nr = 2
			nc = nsrc # math.ceil(nsrc/2)
			fig,ax = plt.subplots(nr,nc,sharex=False,sharey=False)
			fig.subplots_adjust(
			    top=0.90,
			    bottom=0.10,
			    left=0.06,
			    right=0.94,
			    hspace=0.2,
			    wspace=0.2,
			)
			plt.suptitle("observed area and simulated energy spectra, channel {}".format(ch))
			fig.set_size_inches(20,10)
			fig.set_dpi(100)

			for i,src in enumerate(SOURCES):

				plt.subplot(nr,nc,i+1)
				# plt.fill_between(self.mids_a_nvs[ch][src], self.spec_exp_src_nvs[ch][src], step='mid', alpha=0.5)#facecolor=(1,0.4,0.4,0.4), edgecolor=(1,0.4,0.4,1))
				mids   = self.mids_a_nvs[ch][src]
				edges  = self.edges_a_nvs[ch][src]
				widths = edges[1:] - edges[:-1]
				plt.fill_between(
					mids,
					(self.spec_exp_src_nvs[ch][src] + self.spec_exp_src_nvs_err[ch][src]) / widths,
					(self.spec_exp_src_nvs[ch][src] - self.spec_exp_src_nvs_err[ch][src]) / widths,
					color='k',
					step='mid',
					alpha=0.2,
					label='source (observed)'
				)
				plt.step(mids, self.spec_exp_src_nvs[ch][src] / widths, color='k', where='mid')
				
				plt.fill_between(
					mids,
					(self.spec_exp_bg_nvs[ch][src] + self.spec_exp_bg_nvs_err[ch][src]) / widths,
					(self.spec_exp_bg_nvs[ch][src] - self.spec_exp_bg_nvs_err[ch][src]) / widths,
					color='g',
					step='mid',
					alpha=0.4,
					label='background (observed)'
				)
				plt.step(mids, self.spec_exp_bg_nvs[ch][src] / widths, color='g', where='mid')
				
				plt.title("observed area spectrum, {}".format(src))
				plt.xlabel('Area (nVs)')
				plt.legend()


				plt.subplot(nr,nc,i+1+nsrc)
				for e,spec in self.spec_sim_src_kev[ch][src].items():
					mids   = self.mids_e_kev[ch][src][e]
					edges  = self.edges_e_kev[ch][src][e]
					widths = edges[1:] - edges[:-1]
					plt.fill_between(mids, spec/widths, step='mid', alpha=0.3, label="{} KeV".format(e))
					plt.step(mids, spec/widths, where='mid')
				plt.xlabel('Energy (KeV)')
				plt.title("simulated energy deposits, {}".format(src))
				plt.yscale('log')
				plt.legend()

			plt.show()

	def show_evaluate(self, pm, ch, xdata=None, incl_exp=True, incl_sep=True, incl_sim_e=False, incl_err=True):

		ev     = self.evaluate(xdata, pm, ch)
		model     = self.unflatten_sources(ev)

		if incl_err:
			ev_err = self.parametrizer.vector_num_error_p_only(pm, self.evaluate, xdata, [ch])
			model_err = self.unflatten_sources(ev_err)

		if incl_sim_e:
			pm.rho_p   *= 1e-1
			pm.res_s_a *= 1e-1
			pm.res_s_b *= 1e-1
			ev_zres = self.evaluate(xdata, pm, ch)
			pm.rho_p   *= 1e+1
			pm.res_s_a *= 1e+1
			pm.res_s_b *= 1e+1
			model_zres = self.unflatten_sources(ev_zres)


		nc = len(model)
		fig,ax = plt.subplots(1,nc,sharex=False,sharey=False)
		fig.subplots_adjust(
		    top    = 0.9 ,
		    bottom = 0.08,
		    left   = 0.03,
		    right  = 0.99,
		    hspace = 0.2 ,
		    wspace = 0.2 ,
		)
		fig.set_size_inches(nc*5,7)
		fig.set_dpi(120)

		for i,src in enumerate(sorted(model.keys())):

			this_edges = self.edges_a_nvs[ch][src]
			this_w = this_edges[1:] - this_edges[:-1]
			this_x = self.mids_a_nvs[ch][src]
			this_model     = model[src]
			plt.subplot(1,nc,i+1)

			if incl_exp:
				this_exp = self.spec_exp_src_nvs[ch][src]
				this_exp_err = np.sqrt(this_exp)
				plt.step(this_x, this_exp/this_w, 'k', where='mid', label='observed')
				plt.fill_between(
					x  = this_x,
					y1 = (this_exp + this_exp_err) / this_w,
					y2 = (this_exp - this_exp_err) / this_w,
					step='mid',
					alpha=0.2,
					color='k',
				)

				if incl_err:
					this_model_err = model_err[src]
					plt.step(this_x, this_model/this_w, color='b', where='mid', label='best fit')
					plt.fill_between(
						x  = this_x,
						y1 = (this_model+this_model_err) / this_w,
						y2 = (this_model-this_model_err) / this_w,
						step='mid',
						alpha=0.3,
						color='b',
					)

			if incl_sep:
				# plot background component
				plt.step(
					this_x,
					(self.spec_exp_bg_nvs[ch][src] * pm["n_bg_{src}".format(src=src)]) / this_w,
					color='c',
					where='mid',
					label='model bg',
				)
				# plot simulation components
				for epv, projector in self.projectors[ch][src].items():
					plt.step(
						this_x,
						(normed_if(projector(pm)) * pm["n_{epv}_{src}".format(epv=epv, src=src)]) / this_w,
						where='mid',
						label='model {} KeV'.format(epv),
					)

			if incl_sim_e:
				
				# plot source emission peaks converted to area with zero resolution
				for k,v in PVE_KEV[src].items():
					pve = float(k)
					mu_a_pve, _ = self.E_to_mu_sigma(pve, pm)
					plt.axvline(mu_a_pve, label='{} KeV'.format(pve))

				# plot full source spectra with zero resolution, but including saturation xf
				this_model_zres = model_zres[src]
				plt.step(this_x, this_model_zres/this_w, color='c', where='mid', label='0 resolution')
				# plt.fill_between(
				# 	x  = this_x,
				# 	y1 = (this_model+this_model_err) / this_w,
				# 	y2 = (this_model-this_model_err) / this_w,
				# 	step='mid',
				# 	alpha=0.3,
				# 	color='b',
				# )

			plt.title(src)
			plt.xlabel('area (nVs)')
			# plt.yscale('log')
			plt.legend()

		if pm.chi2 is not None:
			plt.suptitle('channel {} - chi2/dof = {:.1f}/{} = {:.3f}'.format(
				ch, pm.chi2, pm.ndof, pm.rchi2))
		else:
			plt.suptitle('channel {}'.format(ch))
		plt.show()


	def show_model(self, pm, ch, r_energy, r_area, ):
		"""show the model (point spread function) from E to A"""

		# use E_to_mu_sigma
		# and xf_and_dxf
		# which have already been complied into one PSF, as
		# self.psf_EA = fit.transformed_gaus_spread(E_to_mu_sigma, xf_and_dxf)
		# 
		# which is function of (energy, area_prime, *p) -> probability_density
		# energy determines (mu, sigma) of the gaussian; area_prime is the axis
		# over wich the gaussian is calculated.
		# integrating over area_prime should yield one. 

		# we want to make a 2d plot with axes (energy, area) and value equal
		# to the probability density that an event with that energy yields a
		# measurement of that area.


		# make flat 2d array of A,E
		xres = 200 # area
		yres = 200 # energy
		l_area   = np.linspace(*r_area  , xres)
		l_energy = np.linspace(*r_energy, yres)

		g_area = np.broadcast_to(
			l_area.reshape([xres,1]),
			[xres, yres],
		)
		g_energy = np.broadcast_to(
			l_energy.reshape([1,yres]),
			[xres, yres],
		)

		# calculate probability density
		density = self.psf_EA(g_energy, g_area, pm)
		# # cutoff
		density[np.logical_not(density > 1e-6)] = 0
		# sampling (n events per energy)
		print(density.shape, density.min(), density.max())
		n_samples_per_e = 10000
		counts = np.random.poisson((density * n_samples_per_e))

		# plt.imshow(density)
		# plt.show()

		display.display2d(
			xdata=None, ydata=None, 
			xbins=l_area, ybins=l_energy,
			xlog=False, ylog=False,
			
			counts = counts,
			xlabel = "Energy (KeV)",
			ylabel = "Area (nVs)",	
			# norm = None,

		)
		plt.show()




	def show_evaluate_p0(self, ch, src_spec=True, ea_model=False):
		if src_spec:
			self.show_evaluate(self.parametrizer.get_p0(), ch, incl_err=False)
		if ea_model:
			self.show_model(self.parametrizer.get_p0(), ch, (0, 800), (0, 1000))


	@fn_indent
	def setup_model(self, ch=None):
		"""initialize functions and objects for performing model calculation"""


		# initialize parametrizer class so that parameters can be
		# added at time of defining each stage
		self.parametrizer = fit.parametrizer()

		# do these in a module-level function and just access it here
		#  define transformation for E -> mu,sigma
		#  define transformation for A* -> A
		#  create point spread function for E -> (A* density)

		# todo: automatically get set of desired parameters

		def E_to_res(E, pm):
			# res_s = pm.res_s_a # constant res_s
			# res_s = pm.res_s_a + (pm.res_s_b * 50) / E 
			res_s = pm.res_s_a - (pm.res_s_b / 50) * E
			# res_s = pm.res_s_a + (pm.res_s_b) * np.sqrt(50/E)
			res_p_squared = (pm.rho_p * 50) / E
			return np.sqrt(res_s**2 + res_p_squared)
		self.E_to_res = E_to_res

		def E_to_mu_sigma(E, pm):
			mu = E * pm.gamma
			sigma = mu * E_to_res(E, pm)
			return mu, sigma
		self.E_to_mu_sigma = E_to_mu_sigma

		# parameters for E -> mu, sigma
		self.parametrizer.add_parameters({
			"gamma"  : 1.195, 
			"rho_p"  : 0.007,
			"res_s_a": 0.044,
			"res_s_b": 0.000,
		})


		# # trivial transformation for testing
		# def xf_and_dxf(A, pm):
		# 	return A, 1

		# channels for which a cubic term should be included
		CH_CUBIC = []
		if ch in CH_CUBIC:
			# cubic polynomial a + A + cA^2 + dA^3
			def xf_and_dxf(A, pm):
				# scale params to 50 nVs
				# a = pm.xf_a * 50
				c = pm.xf_c / 50
				d = pm.xf_d / (50**2)
				A2 = A**2
				A3 = A**3
				# return a + A + c*A2 + d*A3, 1 + 2*A*c + 3*A2*d
				return A + c*A2 + d*A3, 1 + 2*A*c + 3*A2*d
			self.parametrizer.add_parameters({
				# "xf_a": 0.0,
				# "xf_b":0.0,
				"xf_c": 0.0,
				"xf_d": 0.0,
			})

		else:
			# quadratic with just A + cA^2
			def xf_and_dxf(A, pm):
				c = pm.xf_c / 50
				return A + c*(A**2), 1 + 2*A*c
			self.parametrizer.add_parameters({
				# "xf_a": 0.0,
				# "xf_b":0.0,
				"xf_c": 0.0,
				# "xf_d": 0.0,
			})

		self.xf_and_dxf = xf_and_dxf

		# # quadratic polynomial (a?) + A + c A**2
		# def xf_and_dxf(A, pm):
		# 	a = pm.xf_a * 50
		# 	c = pm.xf_c / 50
		# 	return a + A + (A**2)*c, 1 + 2*A*c



		# make binned projectors for each piece of each source spectrum
		# using the point spread function as above
		# 
		# note: normalize the spectrum components to each have unit amplitude
		#       before giving them to the projectors, so that the parameter
		#       for component magnitude is a true "number of events" measure.
		# 
		# {ch:{src:{Epv:projector}}}
		self.projectors = {}
		ch_setup = CH_COMP if ch is None else [ch]
		for ch in ch_setup:
			self.projectors[ch] = {}
			for src in SOURCES:
				self.projectors[ch][src] = {}
				for e in self.sim_src[src].keys():#PVE_KEV[src].keys():

					# todo: calculate 2d bins once and give references
					#       if memory use becomes too large
					# test_emus = lambda E,pm:(E*pm.gamma, E*pm.gamma*pm.res)
					this_e_mids  = self.mids_e_kev[ch][src][e]
					this_e_edges = self.edges_e_kev[ch][src][e]
					this_e_width = this_e_edges[1:]-this_e_edges[:-1]

					this_a_mids  = self.mids_a_nvs[ch][src]
					this_a_edges = self.edges_a_nvs[ch][src]
					this_a_width = this_a_edges[1:] + this_a_edges[:-1]

					# psf_EA = fit.gaus_spread(E_to_mu_sigma)
					self.psf_EA = fit.transformed_gaus_spread(E_to_mu_sigma, xf_and_dxf)

					this_projector = fit.binned_projector(
						func = self.psf_EA,
						xMids  = this_e_mids,
						xWidth = this_e_width,
						yMids  = this_a_mids,
						yWidth = this_a_width,
						xSpec  = self.spec_sim_src_kev[ch][src][e],
					)
					self.projectors[ch][src][e] = this_projector


		# parameters for background contribution
		for src in SOURCES:
			self.parametrizer.add_parameters({"n_bg_{}".format(src):0.0})
			self.parametrizer.add_parameters({"n_{}_{}".format(_,src):0.0 for _ in PVE_KEV[src].keys()})


	def get_xdata_flat(self, ch):
		pieces = []
		for src in sorted(SOURCES):
			pieces.append(self.spec_exp_bg_nvs[ch][src])
			for e in self.sim_src[src].keys():#PVE_KEV[src].keys():
				pieces.append(self.spec_sim_src_kev[ch][src][e])
		return np.concatenate(pieces, axis=0)

	def get_xdata_err_flat(self, ch):
		pieces = []
		for src in sorted(SOURCES):
			pieces.append(self.spec_exp_bg_nvs_err[ch][src])
			for e in self.sim_src[src].keys():#PVE_KEV[src].keys():
				pieces.append(self.spec_sim_src_kev_err[ch][src][e])
		return np.concatenate(pieces, axis=0)

	def evaluate(self, xdata, pm, ch):
		""""""

		# xdata is not needed since all the data used in modeling the spectra
		# is contained within the class instance. However, it's necessary to
		# accept it as a parameter, since curve_fit expects it to be there. We
		# can just pass None to the fitter and ignore the parameter here.


		# Unfortunately, curve_fit is the only acceptable option here.
		# We need to get parameter covariance, and we also need to specify
		# sigma and set absolute_sigma = True.
		# Neither leastsq nor least_squares behaves properly with regards
		# to sigma, and minimize cannot calculate covariance.
		# 
		# So, to make curve_fit work properly, we need to concatenate all the
		# observed and modeled source spectra.
		# 
		# The concatenation of the experimental sepctra is supplied to the minimizer;
		# it does not need to be known here.
		# 
		# This function needs to return the concatenation of all of the modeled spectra.
		# Crucially, these must be in the same order as the experimetnal spectra.
		# To achieve this without memory, the order will be alphabetical by source.


		# list of spectra, not concatenated yet.
		model_src_spectra = []

		# for tracking position in xdata if supplied
		istart = 0

		# calculate spectra for each source, sorted
		for src in sorted(SOURCES):

			# start with background component
			if xdata is None:
				this_spectrum = self.spec_exp_bg_nvs[ch][src] * pm["n_bg_{src}".format(src=src)]
			else:
				this_size = self.spec_exp_bg_nvs[ch][src].size
				this_spectrum = xdata[istart:istart+this_size] * pm["n_bg_{src}".format(src=src)]
				istart += this_size

			# add each simulation component
			for epv, projector in self.projectors[ch][src].items():

				if xdata is None:
					xSpec = None
				else:
					this_size = self.spec_sim_src_kev[ch][src][epv].size
					xSpec = xdata[istart:istart+this_size]
					# xSpec = None
					istart += this_size
				
				# Since projection to an incomplete range doesn't preserve norm,
				# we have to normalize here again if we want the meaning of the
				# number parameters to stay the same.
				# 
				# However, for the purpose of understanding the effects of perturbing
				# the spectrum data, it's necessary disable this feature.
				this_spectrum += normed_if(projector(pm, xSpec)) * pm["n_{epv}_{src}".format(epv=epv, src=src)]

			# append this source's modeled spectrum to the list
			model_src_spectra.append(this_spectrum)
			del this_spectrum

		# concatenate list of spectra and return result
		return np.concatenate(model_src_spectra, axis=0)


	def optimize(self, ch):

		pm0 = self.parametrizer.get_p0()
		xdata_flat = self.get_xdata_flat(ch)
		xerr_flat  = self.get_xdata_err_flat(ch)
		# xdata_flat = None
		# xerr_flat  = None

		# plt.plot(np.linspace(0,1,xdata_flat.size), xdata_flat+xerr_flat, 'r-')
		# plt.plot(np.linspace(0,1,xdata_flat.size), xdata_flat-xerr_flat, 'r-')
		# plt.plot(np.linspace(0,1,xdata_flat.size), xdata_flat, 'k-')
		# plt.show()
		# sys.exit(0)

		bounds = {_:[0,np.inf] for _ in pm0.v_names}
		# del bounds["res_s_b"]

		ydata_flat = self.spec_exp_src_nvs_flat[ch]
		yerr_flat  = self.spec_exp_src_nvs_err_flat[ch]

		pm_opt, ym_opt, ym_err = self.parametrizer.curve_fit(
			xdata  = xdata_flat,
			xerr   = xerr_flat,
			ydata  = ydata_flat,
			yerr   = yerr_flat,
			f      = self.evaluate,
			f_args = [ch],
			bounds = bounds,
			p0     = PARAM_GUESS.get(ch, None),
		)
		for ip,p in enumerate(pm_opt.v_names):
			print("{:>2} {:>16} = {:>10.3e} \xb1 {:>10.6e}".format(ip, p, pm_opt.v_opt[ip], pm_opt.v_err[ip]))
		
		self.show_evaluate(pm_opt, ch, xdata=xdata_flat, incl_sep=True, incl_sim_e=False)

		# x data (shape != shape of ydata)
		# x      = xdata_flat
		# x_err  = xerr_flat 

		# area bins
		ar_e = self.edges_a_nvs[ch]
		ar_m = self.mids_a_nvs[ch]
		
		# experimental y data
		ye     = self.unflatten_sources(ydata_flat, ch=ch)
		ye_err = self.unflatten_sources(yerr_flat , ch=ch)

		# modeled y data
		ym     = self.unflatten_sources(ym_opt, ch=ch)
		ym_err = self.unflatten_sources(ym_err, ch=ch)

		return pm_opt, (ar_e, ar_m), (ye,ye_err), (ym,ym_err)








def main():
	rtn = routine()

	# rtn.show_spectra()
	# sys.exit(0)

	# rtn.setup_model(1)
	# rtn.show_evaluate_p0(1)
	# sys.exit(0)

	# rtn.setup_model(1)
	# rtn.show_evaluate_p0(1, False, True)
	# sys.exit(0)

	# ch_fit = CH_COMP
	ch_fit = [1]
	# ch_fit = [2,3]
	# ch_fit = [1,2,3]

	for ch in ch_fit:
		rtn.setup_model(ch)
		pm_opt, area, ye, ym = rtn.optimize(ch)

		show_optimized_model = True
		if show_optimized_model:
			...

		show_spectra = False
		if show_spectra:
			for src in SOURCES:

				plt.fill_between(
					x  = area[1][src],
					y1 = (ye[0][src]+ye[1][src]),
					y2 = (ye[0][src]-ye[1][src]),
					step='mid',
					alpha=0.3,
					color='k',
				)
				plt.fill_between(
					x  = area[1][src],
					y1 = (ym[0][src]+ym[1][src]),
					y2 = (ym[0][src]-ym[1][src]),
					step='mid',
					alpha=0.3,
					color='b',
				)

				plt.step(
					area[1][src],
					ye[0][src],
					color='k',
					where='mid',
					label='experiment'
				)
				plt.step(
					area[1][src],
					ym[0][src],
					color='b',
					where='mid',
					label='model'
				)

				plt.title(src)
				plt.legend()
				plt.xlabel('area (nVs)')
				plt.show()

		save_spectra = False
		if save_spectra:

			arrays = {}
			arrays |= {"a_edges_{}".format(k):v for k,v in area[0].items()}
			arrays |= {"a_mids_{}".format( k):v for k,v in area[1].items()}
			arrays |= {"ye_{}".format(     k):v for k,v in ye[0].items()}
			arrays |= {"ye_err_{}".format( k):v for k,v in ye[1].items()}
			arrays |= {"ym_{}".format(     k):v for k,v in ym[0].items()}
			arrays |= {"ym_err_{}".format( k):v for k,v in ym[1].items()}

			# arrays |= rtn.spec_exp_src_nvs[ch]
			for src in SOURCES:
				arrays |= {"exp_{}_{}".format(k,src):rtn.exp_src[src][k] for k in rtn.exp_src[src].keys}
			
			sim_version = 2
			file = "./data/spectra/simv{}_ch{}.npz".format(sim_version, ch)
			np.savez(file, **arrays)



if __name__ == "__main__":
	# todo: argparse for control
	main()

